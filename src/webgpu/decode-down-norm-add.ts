import { loadDecodeMlpPleFixture } from "../model/decode-mlp-ple-fixture";
import { getWebGpuDevice } from "./device";

const WORKGROUP_SIZE = 256;
const WORKGROUP_COUNT = 384;
const OUTPUT_FEATURES = 1536;

export interface DecodeDownNormAddResult {
  sourceOperator: "com.xenova.gemma4.DecodeDownNormAddFused";
  sourceVariant: "codes-fixed-subgroup-32";
  workgroupSize: 256;
  workgroupCount: 384;
  hiddenMaximumAbsoluteError: number;
  hiddenMaximumRelativeError: number;
  hiddenBitMismatches: number;
  gpuBufferAllocations: number;
  allocationsPerDispatch: 0;
}

const pipelineCache = new WeakMap<GPUDevice, Promise<GPUComputePipeline>>();

export async function runDecodeDownNormAdd(): Promise<DecodeDownNormAddResult> {
  const [fixture, device] = await Promise.all([loadDecodeMlpPleFixture(), getWebGpuDevice()]);
  if (!device.features.has("subgroups") || !device.features.has("shader-f16")) {
    throw new Error("DecodeDownNormAddFused requires WebGPU subgroups and shader-f16");
  }
  const pipelinePromise = pipelineCache.get(device) ?? compilePipeline(device);
  if (!pipelineCache.has(device)) pipelineCache.set(device, pipelinePromise);
  const pipeline = await pipelinePromise;

  const input = createBuffer(device, "Down input codes", fixture.expectedGateUpBits.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const bits = createBuffer(device, "Down packed weights", fixture.downBits.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const partials = createBuffer(device, "Down atomic partials", (OUTPUT_FEATURES + 1) * 4, GPUBufferUsage.STORAGE);
  const scales = createBuffer(device, "Down row scales", fixture.downScales.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const hidden = createBuffer(device, "Down residual hidden", fixture.hiddenBeforeDown.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  const norm = createBuffer(device, "Down norm weights", fixture.postFfNorm.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const params = createBuffer(device, "Down parameters", 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const readback = createBuffer(device, "Down readback", fixture.expectedHiddenAfterDown.byteLength, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const buffers = [input, bits, partials, scales, hidden, norm, params, readback];
  try {
    device.queue.writeBuffer(input, 0, fixture.expectedGateUpBits);
    device.queue.writeBuffer(bits, 0, fixture.downBits);
    device.queue.writeBuffer(scales, 0, fixture.downScales);
    device.queue.writeBuffer(hidden, 0, fixture.hiddenBeforeDown);
    device.queue.writeBuffer(norm, 0, fixture.postFfNorm);
    device.queue.writeBuffer(params, 0, new Float32Array([27.842519760131836, 16.64207649230957, 0, 0]));
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: bits } },
        { binding: 2, resource: { buffer: partials } },
        { binding: 3, resource: { buffer: scales } },
        { binding: 4, resource: { buffer: hidden } },
        { binding: 5, resource: { buffer: norm } },
        { binding: 6, resource: { buffer: params } },
      ],
    });
    const encoder = device.createCommandEncoder({ label: "DecodeDownNormAddFused dispatch" });
    const pass = encoder.beginComputePass({ label: "DecodeDownNormAddFused" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(WORKGROUP_COUNT);
    pass.end();
    encoder.copyBufferToBuffer(hidden, 0, readback, 0, fixture.expectedHiddenAfterDown.byteLength);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const actual = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    const errors = measureErrors(actual, fixture.expectedHiddenAfterDown);
    return {
      sourceOperator: "com.xenova.gemma4.DecodeDownNormAddFused",
      sourceVariant: "codes-fixed-subgroup-32",
      workgroupSize: WORKGROUP_SIZE,
      workgroupCount: WORKGROUP_COUNT,
      ...errors,
      gpuBufferAllocations: buffers.length,
      allocationsPerDispatch: 0,
    };
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
}

async function compilePipeline(device: GPUDevice): Promise<GPUComputePipeline> {
  return device.createComputePipelineAsync({
    label: "DecodeDownNormAddFused fixed subgroup 32",
    layout: "auto",
    compute: { module: device.createShaderModule({ code: createDecodeDownNormAddShader() }), entryPoint: "main" },
  });
}

export function createDecodeDownNormAddShader(bits: 2 | 4 = 4): string {
  const int2 = bits === 2;
  const loadAndAccumulate = int2
    ? `let activation0 = vec4<f32>(a[word * 4u]);
    let activation1 = vec4<f32>(a[word * 4u + 1u]);
    let activation2 = vec4<f32>(a[word * 4u + 2u]);
    let activation3 = vec4<f32>(a[word * 4u + 3u]);
    activation_sum = activation_sum +
      (activation0.x + activation0.y + activation0.z + activation0.w) +
      (activation1.x + activation1.y + activation1.z + activation1.w) +
      (activation2.x + activation2.y + activation2.z + activation2.w) +
      (activation3.x + activation3.y + activation3.z + activation3.w);`
    : `let activation0 = vec4<f32>(a[word * 2u]);
    let activation1 = vec4<f32>(a[word * 2u + 1u]);
    activation_sum = activation_sum +
      (activation0.x + activation0.y + activation0.z + activation0.w) +
      (activation1.x + activation1.y + activation1.z + activation1.w);`;
  const unpackAndDot = int2
    ? `let plane0 = unpack4x8unorm(packed & 0x03030303u);
      let plane1 = unpack4x8unorm((packed >> 2u) & 0x03030303u);
      let plane2 = unpack4x8unorm((packed >> 4u) & 0x03030303u);
      let plane3 = unpack4x8unorm((packed >> 6u) & 0x03030303u);
      sums[row] = sums[row] +
        ((dot(vec4<f32>(plane0.x, plane1.x, plane2.x, plane3.x), activation0) +
        dot(vec4<f32>(plane0.y, plane1.y, plane2.y, plane3.y), activation1)) +
        (dot(vec4<f32>(plane0.z, plane1.z, plane2.z, plane3.z), activation2) +
        dot(vec4<f32>(plane0.w, plane1.w, plane2.w, plane3.w), activation3)));`
    : `let lo = unpack4x8unorm(packed & 0x0f0f0f0fu);
      let hi = unpack4x8unorm((packed >> 4u) & 0x0f0f0f0fu);
      sums[row] = sums[row] +
        dot(vec4<f32>(lo.x, hi.x, lo.y, hi.y), activation0) +
        dot(vec4<f32>(lo.z, hi.z, lo.w, hi.w), activation1);`;
  return `enable f16;
enable subgroups;

struct Params { inScale: f32, outScale: f32 }
@group(0) @binding(0) var<storage, read> a: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> bits_buf: array<u32>;
@group(0) @binding(2) var<storage, read_write> pp: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> scale: array<f32>;
@group(0) @binding(4) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(5) var<storage, read> nw: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

const OUT_F: u32 = 1536u;
const WORDS_PER_ROW: u32 = 768u;
const ZP: f32 = ${int2 ? "2.0" : "8.0"};
const WG: u32 = 256u;
const N_ROWS: u32 = 4u;
const TOTAL_WGS: u32 = 384u;
const EPS: f32 = 0.000001;
const COUNTER_IDX: u32 = OUT_F;

var<workgroup> dsh: array<f32, 1536>;
var<workgroup> sgq: array<vec4<f32>, 8>;
var<workgroup> sgs: array<f32, 8>;
var<workgroup> lastFlag: u32;

fn srq(value: f32, value_scale: f32) -> f32 {
  if (value_scale == 0.0) { return value; }
  return clamp(round(value / value_scale), -128.0, 127.0) * value_scale;
}

fn sg_sum(value: f32) -> f32 { return subgroupAdd(value); }
fn sg_sum_v4(value: vec4<f32>) -> vec4<f32> { return subgroupAdd(value); }

fn reduce_sum(value: f32, tid: u32) -> f32 {
  let subgroup_sum = sg_sum(value);
  if ((tid & 31u) == 0u) { sgs[tid >> 5u] = subgroup_sum; }
  workgroupBarrier();
  var total = 0.0;
  for (var index = 0u; index < 8u; index = index + 1u) { total = total + sgs[index]; }
  workgroupBarrier();
  return total;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
  let tid = local_id.x;
  let row_base = workgroup_id.x * N_ROWS;
  var sums = vec4<f32>(0.0);
  var activation_sum = 0.0;
  var word = tid;
  loop {
    if (word >= WORDS_PER_ROW) { break; }
    ${loadAndAccumulate}
    for (var row = 0u; row < N_ROWS; row = row + 1u) {
      let output_row = row_base + row;
      let packed = bits_buf[output_row * WORDS_PER_ROW + word];
      ${unpackAndDot}
    }
    word = word + WG;
  }

  let reduced = sg_sum_v4(sums);
  let reduced_activation = sg_sum(activation_sum);
  if ((tid & 31u) == 0u) {
    sgq[tid >> 5u] = reduced;
    sgs[tid >> 5u] = reduced_activation;
  }
  workgroupBarrier();
  if (tid == 0u) {
    var total = vec4<f32>(0.0);
    var total_activation = 0.0;
    for (var index = 0u; index < 8u; index = index + 1u) {
      total = total + sgq[index];
      total_activation = total_activation + sgs[index];
    }
    let zero_point_sum = ZP * total_activation;
    for (var row = 0u; row < N_ROWS; row = row + 1u) {
      let output_row = row_base + row;
      let down = srq(
        scale[output_row] * (params.inScale * fma(total[row], 255.0, -zero_point_sum)),
        params.outScale,
      );
      atomicStore(&pp[output_row], bitcast<u32>(down));
    }
  }
  storageBarrier();

  if (tid == 0u) {
    let ticket = atomicAdd(&pp[COUNTER_IDX], 1u);
    lastFlag = select(0u, 1u, ticket == TOTAL_WGS - 1u);
  }
  if (workgroupUniformLoad(&lastFlag) != 1u) { return; }
  if (tid == 0u) { atomicStore(&pp[COUNTER_IDX], 0u); }

  var accumulator = 0.0;
  var output = tid;
  loop {
    if (output >= OUT_F) { break; }
    let down = bitcast<f32>(atomicLoad(&pp[output]));
    dsh[output] = down;
    accumulator = accumulator + down * down;
    output = output + WG;
  }
  let rms = inverseSqrt(reduce_sum(accumulator, tid) / f32(OUT_F) + EPS);
  output = tid;
  loop {
    if (output >= OUT_F) { break; }
    hidden[output] = hidden[output] + dsh[output] * rms * nw[output];
    output = output + WG;
  }
}`;
}

function createBuffer(device: GPUDevice, label: string, size: number, usage: GPUBufferUsageFlags): GPUBuffer {
  return device.createBuffer({ label, size, usage });
}

function measureErrors(actual: Float32Array, expected: Float32Array) {
  let hiddenMaximumAbsoluteError = 0;
  let hiddenMaximumRelativeError = 0;
  let hiddenBitMismatches = 0;
  const actualBits = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const expectedBits = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    const absolute = Math.abs(actual[index] - expected[index]);
    hiddenMaximumAbsoluteError = Math.max(hiddenMaximumAbsoluteError, absolute);
    hiddenMaximumRelativeError = Math.max(hiddenMaximumRelativeError, absolute / Math.max(Math.abs(expected[index]), 1e-7));
    if (actualBits[index] !== expectedBits[index]) hiddenBitMismatches += 1;
  }
  return { hiddenMaximumAbsoluteError, hiddenMaximumRelativeError, hiddenBitMismatches };
}