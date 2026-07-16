import type { GemmaRotaryRow } from "../model/gemma-rope";

export interface GemmaPrefillRopePipeline {
  headDimension: 256 | 512;
  pipeline: GPUComputePipeline;
}

export interface GemmaPrefillRopeResources {
  bindGroup: GPUBindGroup;
  cosine: GPUBuffer;
  sine: GPUBuffer;
  parameters: GPUBuffer;
  rowCapacity: number;
  heads: number;
  ownedBuffers: GPUBuffer[];
}

const pipelineCache = new WeakMap<
  GPUDevice,
  Map<256 | 512, Promise<GemmaPrefillRopePipeline>>
>();

export function getGemmaPrefillRopePipeline(
  device: GPUDevice,
  headDimension: 256 | 512,
): Promise<GemmaPrefillRopePipeline> {
  let devicePipelines = pipelineCache.get(device);
  if (!devicePipelines) {
    devicePipelines = new Map();
    pipelineCache.set(device, devicePipelines);
  }
  const cached = devicePipelines.get(headDimension);
  if (cached) return cached;
  const compiled = compileGemmaPrefillRopePipeline(device, headDimension).catch((error) => {
    devicePipelines?.delete(headDimension);
    throw error;
  });
  devicePipelines.set(headDimension, compiled);
  return compiled;
}

export async function compileGemmaPrefillRopePipeline(
  device: GPUDevice,
  headDimension: 256 | 512,
): Promise<GemmaPrefillRopePipeline> {
  const pipeline = await device.createComputePipelineAsync({
    label: `Gemma prefill exact RoPE ${headDimension}`,
    layout: "auto",
    compute: {
      module: device.createShaderModule({
        code: createGemmaPrefillRopeShader(headDimension),
      }),
      entryPoint: "main",
    },
  });
  return { headDimension, pipeline };
}

export function createGemmaPrefillRopeResources(
  device: GPUDevice,
  compiled: GemmaPrefillRopePipeline,
  activations: GPUBuffer,
  rows: number,
  heads: number,
  rotary: GemmaRotaryRow,
): GemmaPrefillRopeResources {
  validateInputs(compiled, activations, rows, rows, heads, rotary);
  const halfDimension = compiled.headDimension / 2;
  const storageUpload = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const cosine = device.createBuffer({
    label: "Gemma prefill RoPE cosine",
    size: rows * halfDimension * 4,
    usage: storageUpload,
  });
  const sine = device.createBuffer({
    label: "Gemma prefill RoPE sine",
    size: rows * halfDimension * 4,
    usage: storageUpload,
  });
  const parameters = device.createBuffer({
    label: "Gemma prefill RoPE parameters",
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const resources = {
    bindGroup: device.createBindGroup({
      layout: compiled.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: activations } },
        { binding: 1, resource: { buffer: cosine } },
        { binding: 2, resource: { buffer: sine } },
        { binding: 3, resource: { buffer: parameters } },
      ],
    }),
    cosine,
    sine,
    parameters,
    rowCapacity: rows,
    heads,
    ownedBuffers: [cosine, sine, parameters],
  };
  updateGemmaPrefillRope(device, compiled, resources, rows, rotary);
  return resources;
}

export function updateGemmaPrefillRope(
  device: GPUDevice,
  compiled: GemmaPrefillRopePipeline,
  resources: GemmaPrefillRopeResources,
  rows: number,
  rotary: GemmaRotaryRow,
): void {
  validateRotary(compiled.headDimension, resources.rowCapacity, rows, rotary);
  device.queue.writeBuffer(resources.cosine, 0, rotary.cosine);
  device.queue.writeBuffer(resources.sine, 0, rotary.sine);
  device.queue.writeBuffer(
    resources.parameters,
    0,
    new Uint32Array([rows, resources.heads, 0, 0]),
  );
}

export function encodeGemmaPrefillRope(
  encoder: GPUCommandEncoder,
  compiled: GemmaPrefillRopePipeline,
  resources: GemmaPrefillRopeResources,
  rows: number,
): void {
  if (!Number.isInteger(rows) || rows < 1 || rows > resources.rowCapacity) {
    throw new Error("Gemma prefill RoPE dispatch rows exceed resource capacity");
  }
  const pass = encoder.beginComputePass({ label: "Gemma prefill exact RoPE" });
  pass.setPipeline(compiled.pipeline);
  pass.setBindGroup(0, resources.bindGroup);
  pass.dispatchWorkgroups(rows, resources.heads);
  pass.end();
}

export function destroyGemmaPrefillRopeResources(
  resources: GemmaPrefillRopeResources,
): void {
  for (const buffer of resources.ownedBuffers) buffer.destroy();
}

export function createGemmaPrefillRopeShader(
  headDimension: 256 | 512,
): string {
  const halfDimension = headDimension / 2;
  return `struct Parameters {
  sequence: u32,
  heads: u32,
}

@group(0) @binding(0) var<storage, read_write> activations: array<f32>;
@group(0) @binding(1) var<storage, read> cosine: array<f32>;
@group(0) @binding(2) var<storage, read> sine: array<f32>;
@group(0) @binding(3) var<uniform> parameters: Parameters;

const HEAD_DIMENSION: u32 = ${headDimension}u;
const HALF_DIMENSION: u32 = ${halfDimension}u;

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let token = workgroupId.x;
  let head = workgroupId.y;
  if (token >= parameters.sequence || head >= parameters.heads) { return; }
  let activationBase = (token * parameters.heads + head) * HEAD_DIMENSION;
  let rotaryBase = token * HALF_DIMENSION;
  var pair = lane;
  loop {
    if (pair >= HALF_DIMENSION) { break; }
    let cosineValue = cosine[rotaryBase + pair];
    let sineValue = sine[rotaryBase + pair];
    let low = activations[activationBase + pair];
    let high = activations[activationBase + pair + HALF_DIMENSION];
    activations[activationBase + pair] =
      fma(low, cosineValue, 0.0) + fma(-high, sineValue, 0.0);
    activations[activationBase + pair + HALF_DIMENSION] =
      fma(high, cosineValue, 0.0) + fma(low, sineValue, 0.0);
    pair = pair + 64u;
  }
}`;
}

function validateInputs(
  compiled: GemmaPrefillRopePipeline,
  activations: GPUBuffer,
  rowCapacity: number,
  rows: number,
  heads: number,
  rotary: GemmaRotaryRow,
): void {
  if (!Number.isInteger(rowCapacity) || rowCapacity < 1 ||
      !Number.isInteger(heads) || heads < 1 ||
      activations.size < rowCapacity * heads * compiled.headDimension * 4) {
    throw new Error("Gemma prefill RoPE activation geometry is invalid");
  }
  validateRotary(compiled.headDimension, rowCapacity, rows, rotary);
}

function validateRotary(
  headDimension: number,
  rowCapacity: number,
  rows: number,
  rotary: GemmaRotaryRow,
): void {
  const expected = rows * headDimension / 2;
  if (!Number.isInteger(rows) || rows < 1 || rows > rowCapacity ||
      rotary.cosine.length !== expected || rotary.sine.length !== expected) {
    throw new Error("Gemma prefill RoPE table geometry is invalid");
  }
}