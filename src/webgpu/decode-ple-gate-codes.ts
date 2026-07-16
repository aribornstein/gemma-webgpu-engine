import { loadDecodeMlpPleFixture } from "../model/decode-mlp-ple-fixture";
import { getWebGpuDevice } from "./device";

const WORKGROUP_SIZE = 32;
const WORKGROUP_COUNT = 256;

export interface DecodePleGateCodesResult {
  sourceOperator: "com.xenova.gemma4.DecodePleGateCodes";
  sourceVariant: "codes-fixed-subgroup-32";
  workgroupSize: 32;
  workgroupCount: 256;
  outputMaximumAbsoluteError: number;
  outputMaximumRelativeError: number;
  outputBitMismatches: number;
  signedZeroBitMismatches: number;
  nonzeroBitMismatches: number;
  gpuBufferAllocations: number;
  allocationsPerDispatch: 0;
}

const pipelineCache = new WeakMap<GPUDevice, Promise<GPUComputePipeline>>();

export async function runDecodePleGateCodes(): Promise<DecodePleGateCodesResult> {
  const [fixture, device] = await Promise.all([loadDecodeMlpPleFixture(), getWebGpuDevice()]);
  if (!device.features.has("subgroups")) {
    throw new Error("DecodePleGateCodes requires WebGPU subgroups");
  }
  const pipelinePromise = pipelineCache.get(device) ?? compilePipeline(device);
  if (!pipelineCache.has(device)) pipelineCache.set(device, pipelinePromise);
  const pipeline = await pipelinePromise;
  const input = createBuffer(device, "PLE gate input", fixture.expectedHiddenAfterDown.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const codes = createBuffer(device, "PLE gate codes", fixture.pleGateWeights.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const scales = createBuffer(device, "PLE gate row scales", fixture.pleGateRowScales.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const ple = createBuffer(device, "PLE gate multiplier", fixture.pleInput.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const output = createBuffer(device, "PLE gate output", fixture.expectedPleGateOutput.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const lut = createBuffer(device, "PLE GELU lookup", fixture.pleGeluLut.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const params = createBuffer(device, "PLE gate parameters", 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const readback = createBuffer(device, "PLE gate readback", fixture.expectedPleGateOutput.byteLength, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const buffers = [input, codes, scales, ple, output, lut, params, readback];
  try {
    device.queue.writeBuffer(input, 0, fixture.expectedHiddenAfterDown);
    device.queue.writeBuffer(codes, 0, fixture.pleGateWeights);
    device.queue.writeBuffer(scales, 0, fixture.pleGateRowScales);
    device.queue.writeBuffer(ple, 0, fixture.pleInput);
    device.queue.writeBuffer(lut, 0, fixture.pleGeluLut);
    device.queue.writeBuffer(params, 0, new Float32Array([3.334678888320923, 0.01857776567339897, 0, 0]));
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: codes } },
        { binding: 2, resource: { buffer: scales } },
        { binding: 3, resource: { buffer: ple } },
        { binding: 4, resource: { buffer: output } },
        { binding: 5, resource: { buffer: lut } },
        { binding: 6, resource: { buffer: params } },
      ],
    });
    const encoder = device.createCommandEncoder({ label: "DecodePleGateCodes dispatch" });
    const pass = encoder.beginComputePass({ label: "DecodePleGateCodes" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(WORKGROUP_COUNT);
    pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, fixture.expectedPleGateOutput.byteLength);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const actual = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    return {
      sourceOperator: "com.xenova.gemma4.DecodePleGateCodes",
      sourceVariant: "codes-fixed-subgroup-32",
      workgroupSize: WORKGROUP_SIZE,
      workgroupCount: WORKGROUP_COUNT,
      ...measureErrors(actual, fixture.expectedPleGateOutput),
      gpuBufferAllocations: buffers.length,
      allocationsPerDispatch: 0,
    };
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
}

async function compilePipeline(device: GPUDevice): Promise<GPUComputePipeline> {
  return device.createComputePipelineAsync({
    label: "DecodePleGateCodes fixed subgroup 32",
    layout: "auto",
    compute: { module: device.createShaderModule({ code: createDecodePleGateCodesShader() }), entryPoint: "main" },
  });
}

export function createDecodePleGateCodesShader(): string {
  return `enable subgroups;

struct Params { inScale: f32, linOutScale: f32, pleOffset: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> codes: array<u32>;
@group(0) @binding(2) var<storage, read> row_scale: array<f32>;
@group(0) @binding(3) var<storage, read> ple: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read> gelu_lut: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

const IN_FEATURES: u32 = 1536u;
const OUT_FEATURES: u32 = 256u;
const WPR: u32 = 384u;

fn reduce(value: f32) -> f32 { return subgroupAdd(value); }
fn srq(value: f32, scale: f32) -> f32 {
  if (scale == 0.0) { return value; }
  return clamp(round(value / scale), -128.0, 127.0) * scale;
}
fn srq4(value: vec4<f32>, scale: f32) -> vec4<f32> {
  if (scale == 0.0) { return value; }
  return clamp(round(value / scale), vec4<f32>(-128.0), vec4<f32>(127.0)) * scale;
}
fn gelu_grid(value: f32, scale: f32) -> f32 {
  return gelu_lut[u32(clamp(round(value / scale), -128.0, 127.0) + 128.0)];
}

@compute @workgroup_size(32, 1, 1)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
  let output_row = workgroup_id.x;
  let lane = local_id.x;
  var accumulator = 0.0;
  var activation_accumulator = 0.0;
  var word = lane;
  loop {
    if (word >= WPR) { break; }
    let input_base = word * 4u;
    let activation = srq4(vec4<f32>(
      a[input_base], a[input_base + 1u], a[input_base + 2u], a[input_base + 3u]
    ), params.inScale);
    activation_accumulator = activation_accumulator +
      (activation.x + activation.y) + (activation.z + activation.w);
    accumulator = accumulator + dot(unpack4x8unorm(codes[output_row * WPR + word]), activation);
    word = word + 32u;
  }
  let activation_sum = reduce(activation_accumulator);
  let sum = reduce(accumulator);
  if (lane == 0u) {
    let linear = row_scale[output_row] * fma(sum, 255.0, -128.0 * activation_sum);
    out[output_row] = gelu_grid(srq(linear, params.linOutScale), params.linOutScale) *
      ple[params.pleOffset + output_row];
  }
}`;
}

function createBuffer(device: GPUDevice, label: string, size: number, usage: GPUBufferUsageFlags): GPUBuffer {
  return device.createBuffer({ label, size, usage });
}

function measureErrors(actual: Float32Array, expected: Float32Array) {
  let outputMaximumAbsoluteError = 0;
  let outputMaximumRelativeError = 0;
  let outputBitMismatches = 0;
  let signedZeroBitMismatches = 0;
  let nonzeroBitMismatches = 0;
  const actualBits = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const expectedBits = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    const absolute = Math.abs(actual[index] - expected[index]);
    outputMaximumAbsoluteError = Math.max(outputMaximumAbsoluteError, absolute);
    outputMaximumRelativeError = Math.max(outputMaximumRelativeError, absolute / Math.max(Math.abs(expected[index]), 1e-7));
    if (actualBits[index] !== expectedBits[index]) {
      outputBitMismatches += 1;
      if ((actualBits[index] & 0x7fffffff) === 0 && (expectedBits[index] & 0x7fffffff) === 0) {
        signedZeroBitMismatches += 1;
      } else {
        nonzeroBitMismatches += 1;
      }
    }
  }
  return { outputMaximumAbsoluteError, outputMaximumRelativeError, outputBitMismatches, signedZeroBitMismatches, nonzeroBitMismatches };
}