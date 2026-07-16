import { loadDecodeMlpPleFixture } from "../model/decode-mlp-ple-fixture";
import { getWebGpuDevice } from "./device";
import { measureGpuDispatches } from "./gpu-timestamp";

const WORKGROUP_SIZE = 64;
const WORKGROUP_COUNT = 768;
const OUTPUT_FEATURES = 6144;

export interface DecodeGateUpPresrqResult {
  sourceOperator: "com.xenova.gemma4.DecodeGateUpNormPresrq";
  sourceVariant: "presrq-codes-fixed-subgroup-32";
  sourceMetadataSha256: string;
  sourceTensorsSha256: string;
  workgroupSize: 64;
  workgroupCount: 768;
  outputElements: 6144;
  capturedBufferElements: 12288;
  outputBitMismatches: number;
  signedZeroBitMismatches: number;
  codeBitMismatches: number;
  actualNegativeZeros: number;
  expectedNegativeZeros: number;
  positiveToNegativeZeroMismatches: number;
  negativeToPositiveZeroMismatches: number;
  firstMismatchIndex: number | null;
  firstMismatchActual: number | null;
  firstMismatchExpected: number | null;
  gpuBufferAllocations: number;
  bytesAllocated: number;
  allocationsPerDispatch: 0;
  gpuKernelDispatchesPerSample: number | null;
  gpuKernelSamplesMs: number[] | null;
  gpuKernelMedianMs: number | null;
  gpuKernelP95Ms: number | null;
}

interface Resources {
  bindGroup: GPUBindGroup;
  outputBuffer: GPUBuffer;
  readBuffer: GPUBuffer;
  buffers: GPUBuffer[];
  bytesAllocated: number;
}

const pipelineCache = new WeakMap<GPUDevice, Promise<GPUComputePipeline>>();

export async function runDecodeGateUpPresrq(): Promise<DecodeGateUpPresrqResult> {
  const [fixture, device] = await Promise.all([
    loadDecodeMlpPleFixture(),
    getWebGpuDevice(),
  ]);
  if (!device.features.has("subgroups") || !device.features.has("shader-f16")) {
    throw new Error("DecodeGateUpNormPresrq requires WebGPU subgroups and shader-f16");
  }

  const pipelinePromise = pipelineCache.get(device) ?? compilePipeline(device);
  if (!pipelineCache.has(device)) pipelineCache.set(device, pipelinePromise);
  let pipeline: GPUComputePipeline;
  try {
    pipeline = await pipelinePromise;
  } catch (error) {
    pipelineCache.delete(device);
    throw error;
  }

  const resources = createResources(device, pipeline, fixture);
  try {
    const encoder = device.createCommandEncoder({ label: "DecodeGateUpNormPresrq dispatch" });
    const pass = encoder.beginComputePass({ label: "DecodeGateUpNormPresrq" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, resources.bindGroup);
    pass.dispatchWorkgroups(WORKGROUP_COUNT);
    pass.end();
    encoder.copyBufferToBuffer(
      resources.outputBuffer,
      0,
      resources.readBuffer,
      0,
      fixture.expectedGateUpBits.byteLength,
    );
    device.queue.submit([encoder.finish()]);
    await resources.readBuffer.mapAsync(GPUMapMode.READ);
    const actual = new Uint16Array(resources.readBuffer.getMappedRange().slice(0));
    resources.readBuffer.unmap();

    let outputBitMismatches = 0;
    let signedZeroBitMismatches = 0;
    let codeBitMismatches = 0;
    let actualNegativeZeros = 0;
    let expectedNegativeZeros = 0;
    let positiveToNegativeZeroMismatches = 0;
    let negativeToPositiveZeroMismatches = 0;
    let firstMismatchIndex: number | null = null;
    let firstMismatchActual: number | null = null;
    let firstMismatchExpected: number | null = null;
    for (let index = 0; index < fixture.expectedGateUpBits.length; index += 1) {
      if (actual[index] === 0x8000) actualNegativeZeros += 1;
      if (fixture.expectedGateUpBits[index] === 0x8000) expectedNegativeZeros += 1;
      if (actual[index] !== fixture.expectedGateUpBits[index]) {
        outputBitMismatches += 1;
        if ((actual[index] & 0x7fff) === 0 && (fixture.expectedGateUpBits[index] & 0x7fff) === 0) {
          signedZeroBitMismatches += 1;
          if (actual[index] === 0x8000) positiveToNegativeZeroMismatches += 1;
          else negativeToPositiveZeroMismatches += 1;
        } else {
          codeBitMismatches += 1;
        }
        if (firstMismatchIndex === null) {
          firstMismatchIndex = index;
          firstMismatchActual = actual[index];
          firstMismatchExpected = fixture.expectedGateUpBits[index];
        }
      }
    }
    const gpuSamples = await measureGpuDispatches(
      device,
      "DecodeGateUpNormPresrq",
      20,
      (pass) => {
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, resources.bindGroup);
        pass.dispatchWorkgroups(WORKGROUP_COUNT);
      },
    );
    return {
      sourceOperator: "com.xenova.gemma4.DecodeGateUpNormPresrq",
      sourceVariant: "presrq-codes-fixed-subgroup-32",
      sourceMetadataSha256: fixture.metadataSha256,
      sourceTensorsSha256: fixture.tensorFileSha256,
      workgroupSize: WORKGROUP_SIZE,
      workgroupCount: WORKGROUP_COUNT,
      outputElements: OUTPUT_FEATURES,
      capturedBufferElements: fixture.expectedGateUpBits.length as 12288,
      outputBitMismatches,
      signedZeroBitMismatches,
      codeBitMismatches,
      actualNegativeZeros,
      expectedNegativeZeros,
      positiveToNegativeZeroMismatches,
      negativeToPositiveZeroMismatches,
      firstMismatchIndex,
      firstMismatchActual,
      firstMismatchExpected,
      gpuBufferAllocations: resources.buffers.length,
      bytesAllocated: resources.bytesAllocated,
      allocationsPerDispatch: 0,
      gpuKernelDispatchesPerSample: gpuSamples ? 20 : null,
      gpuKernelSamplesMs: gpuSamples?.map(round) ?? null,
      gpuKernelMedianMs: gpuSamples ? round(percentile(gpuSamples, 0.5)) : null,
      gpuKernelP95Ms: gpuSamples ? round(percentile(gpuSamples, 0.95)) : null,
    };
  } finally {
    for (const buffer of resources.buffers) buffer.destroy();
  }
}

async function compilePipeline(device: GPUDevice): Promise<GPUComputePipeline> {
  const module = device.createShaderModule({ code: createDecodeGateUpPresrqShader() });
  return device.createComputePipelineAsync({
    label: "DecodeGateUpNormPresrq fixed subgroup 32",
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
}

export function createDecodeGateUpPresrqShader(bits: 2 | 4 = 4): string {
  const int2 = bits === 2;
  const unpackAndAccumulate = int2
    ? `let gate0 = unpack4x8unorm(packed_gate & 0x03030303u);
      let gate1 = unpack4x8unorm((packed_gate >> 2u) & 0x03030303u);
      let gate2 = unpack4x8unorm((packed_gate >> 4u) & 0x03030303u);
      let gate3 = unpack4x8unorm((packed_gate >> 6u) & 0x03030303u);
      gate_accumulator[row] = gate_accumulator[row] +
        ((dot(vec4<f32>(gate0.x, gate1.x, gate2.x, gate3.x), activation[0]) +
        dot(vec4<f32>(gate0.y, gate1.y, gate2.y, gate3.y), activation[1])) +
        (dot(vec4<f32>(gate0.z, gate1.z, gate2.z, gate3.z), activation[2]) +
        dot(vec4<f32>(gate0.w, gate1.w, gate2.w, gate3.w), activation[3])));

      let packed_up = up_bits[output_row * WPR + word];
      let up0 = unpack4x8unorm(packed_up & 0x03030303u);
      let up1 = unpack4x8unorm((packed_up >> 2u) & 0x03030303u);
      let up2 = unpack4x8unorm((packed_up >> 4u) & 0x03030303u);
      let up3 = unpack4x8unorm((packed_up >> 6u) & 0x03030303u);
      up_accumulator[row] = up_accumulator[row] +
        ((dot(vec4<f32>(up0.x, up1.x, up2.x, up3.x), activation[0]) +
        dot(vec4<f32>(up0.y, up1.y, up2.y, up3.y), activation[1])) +
        (dot(vec4<f32>(up0.z, up1.z, up2.z, up3.z), activation[2]) +
        dot(vec4<f32>(up0.w, up1.w, up2.w, up3.w), activation[3])));`
    : `let gate_lo = unpack4x8unorm(packed_gate & 0x0f0f0f0fu);
      let gate_hi = unpack4x8unorm((packed_gate >> 4u) & 0x0f0f0f0fu);
      gate_accumulator[row] = gate_accumulator[row] +
        (dot(vec4<f32>(gate_lo.x, gate_hi.x, gate_lo.y, gate_hi.y), activation[0]) +
        dot(vec4<f32>(gate_lo.z, gate_hi.z, gate_lo.w, gate_hi.w), activation[1]));

      let packed_up = up_bits[output_row * WPR + word];
      let up_lo = unpack4x8unorm(packed_up & 0x0f0f0f0fu);
      let up_hi = unpack4x8unorm((packed_up >> 4u) & 0x0f0f0f0fu);
      up_accumulator[row] = up_accumulator[row] +
        (dot(vec4<f32>(up_lo.x, up_hi.x, up_lo.y, up_hi.y), activation[0]) +
        dot(vec4<f32>(up_lo.z, up_hi.z, up_lo.w, up_hi.w), activation[1]));`;
  return `enable f16;
enable subgroups;

struct Params {
  gateOutScale: f32,
  upOutScale: f32,
  outQuantScale: f32,
}

@group(0) @binding(0) var<storage, read> hidden: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> gate_bits: array<u32>;
@group(0) @binding(2) var<storage, read> gate_scale: array<f32>;
@group(0) @binding(3) var<storage, read> up_bits: array<u32>;
@group(0) @binding(4) var<storage, read> up_scale: array<f32>;
@group(0) @binding(5) var<storage, read> sum_a: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f16>;
@group(0) @binding(7) var<storage, read> gelu_lut: array<f32>;
@group(0) @binding(8) var<uniform> params: Params;

const INTER: u32 = ${int2 ? 12288 : 6144}u;
const WPR: u32 = ${int2 ? 96 : 192}u;
const ZP: f32 = ${int2 ? "2.0" : "8.0"};
const SG_COUNT: u32 = 2u;
const N_ROWS: u32 = ${int2 ? 2 : 4}u;

fn reduce_sum(value: f32, local_index: u32) -> f32 {
  return subgroupAdd(value);
}

fn srq(value: f32, scale: f32) -> f32 {
  if (scale == 0.0) { return value; }
  return clamp(round(value / scale), -128.0, 127.0) * scale;
}

fn gelu_grid(value: f32, scale: f32) -> f32 {
  return gelu_lut[u32(clamp(round(value / scale), -128.0, 127.0) + 128.0)];
}

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let subgroup_id = local_id.x / 32u;
  let lane = local_id.x & 31u;
  let workgroup = workgroup_id.x * SG_COUNT + subgroup_id;
  let row_base = workgroup * N_ROWS;
  var gate_accumulator: array<f32, 4>;
  var up_accumulator: array<f32, 4>;
  for (var row = 0u; row < N_ROWS; row = row + 1u) {
    gate_accumulator[row] = 0.0;
    up_accumulator[row] = 0.0;
  }

  var word = lane;
  loop {
    if (word >= WPR) { break; }
    var activation_f16: array<vec4<f16>, ${int2 ? 4 : 2}>;
    for (var chunk = 0u; chunk < ${int2 ? 4 : 2}u; chunk = chunk + 1u) {
      activation_f16[chunk] = hidden[word * ${int2 ? 4 : 2}u + chunk];
    }
    var activation: array<vec4<f32>, ${int2 ? 4 : 2}>;
    for (var chunk = 0u; chunk < ${int2 ? 4 : 2}u; chunk = chunk + 1u) {
      activation[chunk] = vec4<f32>(activation_f16[chunk]);
    }
    for (var row = 0u; row < N_ROWS; row = row + 1u) {
      let output_row = row_base + row;
      let packed_gate = gate_bits[output_row * WPR + word];
      ${unpackAndAccumulate}
    }
    word = word + 32u;
  }

  let activation_sum = sum_a[0];
  for (var row = 0u; row < N_ROWS; row = row + 1u) {
    let gate_sum = reduce_sum(gate_accumulator[row], local_id.x);
    let up_sum = reduce_sum(up_accumulator[row], local_id.x);
    if (lane == 0u) {
      let output_row = row_base + row;
      let gate = srq(
        gate_scale[output_row] * fma(gate_sum, 255.0, -(ZP * activation_sum)),
        params.gateOutScale,
      );
      let up = srq(
        up_scale[output_row] * fma(up_sum, 255.0, -(ZP * activation_sum)),
        params.upOutScale,
      );
      let value = gelu_grid(gate, params.gateOutScale) * up;
      let quant_scale = params.outQuantScale;
      var code: f32;
      if (quant_scale == 0.0) {
        code = value;
      } else {
        code = clamp(round(value / quant_scale), -128.0, 127.0);
      }
      out[output_row] = f16(code);
    }
  }
}`;
}

function createResources(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  fixture: Awaited<ReturnType<typeof loadDecodeMlpPleFixture>>,
): Resources {
  const inputBuffer = createBuffer(device, "Gate/up presrq input", fixture.preMlpInputBits.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const gateBitsBuffer = createBuffer(device, "Gate packed weights", fixture.gateBits.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const gateScalesBuffer = createBuffer(device, "Gate row scales", fixture.gateScales.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const upBitsBuffer = createBuffer(device, "Up packed weights", fixture.upBits.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const upScalesBuffer = createBuffer(device, "Up row scales", fixture.upScales.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const sumBuffer = createBuffer(device, "Gate/up activation sum", fixture.preMlpSum.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const lutBuffer = createBuffer(device, "Gate GELU lookup", fixture.gateGeluLut.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const outputBuffer = createBuffer(device, "Gate/up output codes", fixture.expectedGateUpBits.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const paramsBuffer = createBuffer(device, "Gate/up parameters", 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const readBuffer = createBuffer(device, "Gate/up readback", fixture.expectedGateUpBits.byteLength, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const buffers = [inputBuffer, gateBitsBuffer, gateScalesBuffer, upBitsBuffer, upScalesBuffer, sumBuffer, lutBuffer, outputBuffer, paramsBuffer, readBuffer];

  device.queue.writeBuffer(inputBuffer, 0, fixture.preMlpInputBits);
  device.queue.writeBuffer(gateBitsBuffer, 0, fixture.gateBits);
  device.queue.writeBuffer(gateScalesBuffer, 0, fixture.gateScales);
  device.queue.writeBuffer(upBitsBuffer, 0, fixture.upBits);
  device.queue.writeBuffer(upScalesBuffer, 0, fixture.upScales);
  device.queue.writeBuffer(sumBuffer, 0, fixture.preMlpSum);
  device.queue.writeBuffer(lutBuffer, 0, fixture.gateGeluLut);
  device.queue.writeBuffer(paramsBuffer, 0, new Float32Array([
    0.6181102395057678,
    0.6181102395057678,
    27.842519760131836,
    0,
  ]));

  return {
    bindGroup: device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: gateBitsBuffer } },
        { binding: 2, resource: { buffer: gateScalesBuffer } },
        { binding: 3, resource: { buffer: upBitsBuffer } },
        { binding: 4, resource: { buffer: upScalesBuffer } },
        { binding: 5, resource: { buffer: sumBuffer } },
        { binding: 6, resource: { buffer: outputBuffer } },
        { binding: 7, resource: { buffer: lutBuffer } },
        { binding: 8, resource: { buffer: paramsBuffer } },
      ],
    }),
    outputBuffer,
    readBuffer,
    buffers,
    bytesAllocated: buffers.reduce((sum, buffer) => sum + buffer.size, 0),
  };
}

function createBuffer(
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
): GPUBuffer {
  return device.createBuffer({ label, size, usage });
}

function percentile(sortedValues: number[], quantile: number): number {
  const index = (sortedValues.length - 1) * quantile;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  return sortedValues[lower] +
    (sortedValues[upper] - sortedValues[lower]) * (index - lower);
}

function round(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}