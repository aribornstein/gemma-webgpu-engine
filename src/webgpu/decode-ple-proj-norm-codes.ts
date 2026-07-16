import { loadDecodeMlpPleFixture } from "../model/decode-mlp-ple-fixture";
import { getWebGpuDevice } from "./device";

const WORKGROUP_SIZE = 256;
const WORKGROUP_COUNT = 96;
const OUTPUT_FEATURES = 1536;

export interface DecodePleProjNormCodesResult {
  sourceOperator: "com.xenova.gemma4.DecodePleProjNormCodes";
  sourceVariant: "codes-fixed-subgroup-32";
  workgroupSize: 256;
  workgroupCount: 96;
  hiddenMaximumAbsoluteError: number;
  hiddenMaximumRelativeError: number;
  hiddenBitMismatches: number;
  nextInputMaximumAbsoluteError: number;
  nextInputMaximumRelativeError: number;
  nextInputBitMismatches: number;
  nextInputSignedZeroBitMismatches: number;
  nextInputNonzeroBitMismatches: number;
  nextSumMaximumAbsoluteError: number;
  nextSumMaximumRelativeError: number;
  nextSumBitMismatches: number;
  gpuBufferAllocations: number;
  allocationsPerDispatch: 0;
}

const pipelineCache = new WeakMap<GPUDevice, Promise<GPUComputePipeline>>();

export async function runDecodePleProjNormCodes(): Promise<DecodePleProjNormCodesResult> {
  const [fixture, device] = await Promise.all([loadDecodeMlpPleFixture(), getWebGpuDevice()]);
  if (!device.features.has("subgroups")) {
    throw new Error("DecodePleProjNormCodes requires WebGPU subgroups");
  }
  const pipelinePromise = pipelineCache.get(device) ?? compilePipeline(device);
  if (!pipelineCache.has(device)) pipelineCache.set(device, pipelinePromise);
  const pipeline = await pipelinePromise;
  const input = createBuffer(device, "PLE projection input", fixture.expectedPleGateOutput.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const codes = createBuffer(device, "PLE projection codes", fixture.pleProjectionWeights.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const scales = createBuffer(device, "PLE projection row scales", fixture.pleProjectionRowScales.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const partials = createBuffer(device, "PLE projection atomic partials", (OUTPUT_FEATURES + 1) * 4, GPUBufferUsage.STORAGE);
  const hidden = createBuffer(device, "PLE residual hidden", fixture.expectedHiddenAfterDown.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  const normWeights = createBuffer(device, "PLE packed norm weights", fixture.pleNormWeights.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const nextInput = createBuffer(device, "Next layer input", fixture.expectedNextLayerInput.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const nextSum = createBuffer(device, "Next layer input sum", fixture.expectedNextLayerSum.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const params = createBuffer(device, "PLE projection parameters", 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const readbackSize = fixture.expectedHiddenAfterPle.byteLength + fixture.expectedNextLayerInput.byteLength + fixture.expectedNextLayerSum.byteLength;
  const readback = createBuffer(device, "PLE projection readback", readbackSize, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const buffers = [input, codes, scales, partials, hidden, normWeights, nextInput, nextSum, params, readback];
  try {
    device.queue.writeBuffer(input, 0, fixture.expectedPleGateOutput);
    device.queue.writeBuffer(codes, 0, fixture.pleProjectionWeights);
    device.queue.writeBuffer(scales, 0, fixture.pleProjectionRowScales);
    device.queue.writeBuffer(hidden, 0, fixture.expectedHiddenAfterDown);
    device.queue.writeBuffer(normWeights, 0, fixture.pleNormWeights);
    device.queue.writeBuffer(params, 0, new Float32Array([0.4842597544193268, 0.03764764964580536, 0.03129800781607628, 0]));
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: codes } },
        { binding: 2, resource: { buffer: scales } },
        { binding: 3, resource: { buffer: partials } },
        { binding: 4, resource: { buffer: hidden } },
        { binding: 5, resource: { buffer: normWeights } },
        { binding: 6, resource: { buffer: nextInput } },
        { binding: 7, resource: { buffer: nextSum } },
        { binding: 8, resource: { buffer: params } },
      ],
    });
    const encoder = device.createCommandEncoder({ label: "DecodePleProjNormCodes dispatch" });
    const pass = encoder.beginComputePass({ label: "DecodePleProjNormCodes" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(WORKGROUP_COUNT);
    pass.end();
    const hiddenBytes = fixture.expectedHiddenAfterPle.byteLength;
    const nextInputBytes = fixture.expectedNextLayerInput.byteLength;
    encoder.copyBufferToBuffer(hidden, 0, readback, 0, hiddenBytes);
    encoder.copyBufferToBuffer(nextInput, 0, readback, hiddenBytes, nextInputBytes);
    encoder.copyBufferToBuffer(nextSum, 0, readback, hiddenBytes + nextInputBytes, fixture.expectedNextLayerSum.byteLength);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const mapped = readback.getMappedRange();
    const actualHidden = new Float32Array(mapped.slice(0, hiddenBytes));
    const actualNextInput = new Float32Array(mapped.slice(hiddenBytes, hiddenBytes + nextInputBytes));
    const actualNextSum = new Float32Array(mapped.slice(hiddenBytes + nextInputBytes));
    readback.unmap();
    const hiddenErrors = measureErrors(actualHidden, fixture.expectedHiddenAfterPle);
    const nextInputErrors = measureErrors(actualNextInput, fixture.expectedNextLayerInput);
    const nextSumErrors = measureErrors(actualNextSum, fixture.expectedNextLayerSum);
    return {
      sourceOperator: "com.xenova.gemma4.DecodePleProjNormCodes",
      sourceVariant: "codes-fixed-subgroup-32",
      workgroupSize: WORKGROUP_SIZE,
      workgroupCount: WORKGROUP_COUNT,
      hiddenMaximumAbsoluteError: hiddenErrors.maximumAbsoluteError,
      hiddenMaximumRelativeError: hiddenErrors.maximumRelativeError,
      hiddenBitMismatches: hiddenErrors.bitMismatches,
      nextInputMaximumAbsoluteError: nextInputErrors.maximumAbsoluteError,
      nextInputMaximumRelativeError: nextInputErrors.maximumRelativeError,
      nextInputBitMismatches: nextInputErrors.bitMismatches,
      nextInputSignedZeroBitMismatches: nextInputErrors.signedZeroBitMismatches,
      nextInputNonzeroBitMismatches: nextInputErrors.nonzeroBitMismatches,
      nextSumMaximumAbsoluteError: nextSumErrors.maximumAbsoluteError,
      nextSumMaximumRelativeError: nextSumErrors.maximumRelativeError,
      nextSumBitMismatches: nextSumErrors.bitMismatches,
      gpuBufferAllocations: buffers.length,
      allocationsPerDispatch: 0,
    };
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
}

async function compilePipeline(device: GPUDevice): Promise<GPUComputePipeline> {
  return device.createComputePipelineAsync({
    label: "DecodePleProjNormCodes fixed subgroup 32",
    layout: "auto",
    compute: { module: device.createShaderModule({ code: createDecodePleProjNormCodesShader() }), entryPoint: "main" },
  });
}

export function createDecodePleProjNormCodesShader(): string {
  return `enable subgroups;

struct Params { inScale: f32, projInScale: f32, projOutScale: f32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> codes: array<u32>;
@group(0) @binding(2) var<storage, read> row_scale: array<f32>;
@group(0) @binding(3) var<storage, read_write> pp: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(5) var<storage, read> w12s: array<f32>;
@group(0) @binding(6) var<storage, read_write> y2: array<f32>;
@group(0) @binding(7) var<storage, read_write> sum2: array<f32>;
@group(0) @binding(8) var<uniform> params: Params;

const OUT_F: u32 = 1536u;
const KV4: u32 = 64u;
const K_ITER: u32 = 2u;
const WG: u32 = 256u;
const SG_ROWS: u32 = 2u;
const ROWS_PER_WG: u32 = 16u;
const TOTAL_WGS: u32 = 96u;
const EPS: f32 = 0.000001;

var<workgroup> lastFlag: u32;
var<workgroup> sgp: array<f32, 8>;

fn sg_sum(value: f32) -> f32 { return subgroupAdd(value); }
fn reduce_sum(value: f32, tid: u32) -> f32 {
  let subgroup_sum = sg_sum(value);
  if ((tid & 31u) == 0u) { sgp[tid >> 5u] = subgroup_sum; }
  workgroupBarrier();
  var total = 0.0;
  for (var index = 0u; index < 8u; index = index + 1u) { total = total + sgp[index]; }
  workgroupBarrier();
  return total;
}
fn srq(value: f32, scale: f32) -> f32 {
  if (scale == 0.0) { return value; }
  return clamp(round(value / scale), -128.0, 127.0) * scale;
}
fn srq4(value: vec4<f32>, scale: f32) -> vec4<f32> {
  if (scale == 0.0) { return value; }
  return clamp(round(value / scale), vec4<f32>(-128.0), vec4<f32>(127.0)) * scale;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
  let tid = local_id.x;
  let subgroup_id = tid / 32u;
  let lane = tid & 31u;
  let row_base = workgroup_id.x * ROWS_PER_WG + subgroup_id * SG_ROWS;

  var activation: array<vec4<f32>, 2>;
  var activation_accumulator = 0.0;
  for (var iteration = 0u; iteration < K_ITER; iteration = iteration + 1u) {
    let word = lane + iteration * 32u;
    let base = word * 4u;
    activation[iteration] = srq4(vec4<f32>(a[base], a[base + 1u], a[base + 2u], a[base + 3u]), params.projInScale);
    activation_accumulator = activation_accumulator +
      (activation[iteration].x + activation[iteration].y) +
      (activation[iteration].z + activation[iteration].w);
  }
  var accumulators: array<f32, 2>;
  for (var row = 0u; row < SG_ROWS; row = row + 1u) {
    let output_row = row_base + row;
    var accumulator = 0.0;
    if (output_row < OUT_F) {
      for (var iteration = 0u; iteration < K_ITER; iteration = iteration + 1u) {
        let word = lane + iteration * 32u;
        accumulator = accumulator + dot(unpack4x8unorm(codes[output_row * KV4 + word]), activation[iteration]);
      }
    }
    accumulators[row] = accumulator;
  }
  let activation_sum = sg_sum(activation_accumulator);
  for (var row = 0u; row < SG_ROWS; row = row + 1u) {
    let sum = sg_sum(accumulators[row]);
    let output_row = row_base + row;
    if (lane == 0u && output_row < OUT_F) {
      atomicStore(&pp[output_row], bitcast<u32>(srq(
        row_scale[output_row] * fma(sum, 255.0, -128.0 * activation_sum),
        params.projOutScale,
      )));
    }
  }
  storageBarrier();

  if (tid == 0u) {
    let ticket = atomicAdd(&pp[OUT_F], 1u);
    lastFlag = select(0u, 1u, ticket == TOTAL_WGS - 1u);
  }
  if (workgroupUniformLoad(&lastFlag) != 1u) { return; }
  if (tid == 0u) { atomicStore(&pp[OUT_F], 0u); }
  let residual_scale = w12s[2u * OUT_F];

  var first_accumulator = 0.0;
  var index = tid;
  loop {
    if (index >= OUT_F) { break; }
    let value = bitcast<f32>(atomicLoad(&pp[index]));
    first_accumulator = first_accumulator + value * value;
    index = index + WG;
  }
  let first_rms = inverseSqrt(reduce_sum(first_accumulator, tid) / f32(OUT_F) + EPS);

  var local_hidden: array<f32, 6>;
  var second_accumulator = 0.0;
  var element = 0u;
  index = tid;
  loop {
    if (index >= OUT_F) { break; }
    let normalized = bitcast<f32>(atomicLoad(&pp[index])) * first_rms * w12s[index];
    let hidden_value = (hidden[index] + normalized) * residual_scale;
    hidden[index] = hidden_value;
    local_hidden[element] = hidden_value;
    second_accumulator = second_accumulator + hidden_value * hidden_value;
    index = index + WG;
    element = element + 1u;
  }
  let second_rms = inverseSqrt(reduce_sum(second_accumulator, tid) / f32(OUT_F) + EPS);

  var quantized_sum = 0.0;
  index = tid;
  element = 0u;
  loop {
    if (index >= OUT_F) { break; }
    let normalized = local_hidden[element] * second_rms * w12s[OUT_F + index];
    let quantized = srq(normalized, params.inScale);
    y2[index] = quantized;
    quantized_sum = quantized_sum + quantized;
    index = index + WG;
    element = element + 1u;
  }
  let total = reduce_sum(quantized_sum, tid);
  if (tid == 0u) { sum2[0] = total; }
}`;
}

function createBuffer(device: GPUDevice, label: string, size: number, usage: GPUBufferUsageFlags): GPUBuffer {
  return device.createBuffer({ label, size, usage });
}

function measureErrors(actual: Float32Array, expected: Float32Array) {
  let maximumAbsoluteError = 0;
  let maximumRelativeError = 0;
  let bitMismatches = 0;
  let signedZeroBitMismatches = 0;
  let nonzeroBitMismatches = 0;
  const actualBits = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const expectedBits = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    const absolute = Math.abs(actual[index] - expected[index]);
    maximumAbsoluteError = Math.max(maximumAbsoluteError, absolute);
    maximumRelativeError = Math.max(maximumRelativeError, absolute / Math.max(Math.abs(expected[index]), 1e-7));
    if (actualBits[index] !== expectedBits[index]) {
      bitMismatches += 1;
      if ((actualBits[index] & 0x7fffffff) === 0 && (expectedBits[index] & 0x7fffffff) === 0) {
        signedZeroBitMismatches += 1;
      } else {
        nonzeroBitMismatches += 1;
      }
    }
  }
  return { maximumAbsoluteError, maximumRelativeError, bitMismatches, signedZeroBitMismatches, nonzeroBitMismatches };
}