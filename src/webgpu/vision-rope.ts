import type { GemmaVisionRotaryTable } from "../model/gemma-vision-rope";

const HEAD_DIMENSION = 64;
const PAIRS_PER_HEAD = 32;

export interface GemmaVisionRopeResources {
  bindGroup: GPUBindGroup;
  cosine: GPUBuffer;
  sine: GPUBuffer;
  parameters: GPUBuffer;
  rowCapacity: number;
  heads: number;
  ownedBuffers: GPUBuffer[];
}

let pipelineCache = new WeakMap<GPUDevice, Promise<GPUComputePipeline>>();

export function getGemmaVisionRopePipeline(device: GPUDevice): Promise<GPUComputePipeline> {
  const cached = pipelineCache.get(device);
  if (cached) return cached;
  const compiled = device.createComputePipelineAsync({
    label: "Gemma vision multidimensional RoPE",
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: createGemmaVisionRopeShader() }),
      entryPoint: "main",
    },
  }).catch((error) => {
    pipelineCache.delete(device);
    throw error;
  });
  pipelineCache.set(device, compiled);
  return compiled;
}

export function createGemmaVisionRopeResources(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  activations: GPUBuffer,
  rowCapacity: number,
  heads: number,
): GemmaVisionRopeResources {
  if (!Number.isInteger(rowCapacity) || rowCapacity < 1 ||
      !Number.isInteger(heads) || heads < 1 ||
      activations.size < rowCapacity * heads * HEAD_DIMENSION * 4) {
    throw new Error("Gemma vision RoPE activation geometry is invalid");
  }
  const storageUpload = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const cosine = device.createBuffer({
    size: rowCapacity * PAIRS_PER_HEAD * 4,
    usage: storageUpload,
  });
  const sine = device.createBuffer({
    size: rowCapacity * PAIRS_PER_HEAD * 4,
    usage: storageUpload,
  });
  const parameters = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  return {
    bindGroup: device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
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
    rowCapacity,
    heads,
    ownedBuffers: [cosine, sine, parameters],
  };
}

export function updateGemmaVisionRope(
  device: GPUDevice,
  resources: GemmaVisionRopeResources,
  table: GemmaVisionRotaryTable,
): void {
  if (table.rows > resources.rowCapacity ||
      table.cosine.length !== table.rows * PAIRS_PER_HEAD ||
      table.sine.length !== table.cosine.length) {
    throw new Error("Gemma vision RoPE table geometry is invalid");
  }
  device.queue.writeBuffer(resources.cosine, 0, table.cosine);
  device.queue.writeBuffer(resources.sine, 0, table.sine);
  device.queue.writeBuffer(
    resources.parameters,
    0,
    new Uint32Array([table.rows, resources.heads, 0, 0]),
  );
}

export function encodeGemmaVisionRope(
  encoder: GPUCommandEncoder,
  pipeline: GPUComputePipeline,
  resources: GemmaVisionRopeResources,
  rows: number,
): void {
  if (!Number.isInteger(rows) || rows < 1 || rows > resources.rowCapacity) {
    throw new Error("Gemma vision RoPE dispatch rows are invalid");
  }
  const pass = encoder.beginComputePass({ label: "Gemma vision multidimensional RoPE" });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, resources.bindGroup);
  pass.dispatchWorkgroups(rows, resources.heads);
  pass.end();
}

export function createGemmaVisionRopeShader(): string {
  return `struct Parameters { rows: u32, heads: u32, padding0: u32, padding1: u32 }
@group(0) @binding(0) var<storage, read_write> activations: array<f32>;
@group(0) @binding(1) var<storage, read> cosine: array<f32>;
@group(0) @binding(2) var<storage, read> sine: array<f32>;
@group(0) @binding(3) var<uniform> parameters: Parameters;

@compute @workgroup_size(32)
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) pair: u32,
) {
  let row = workgroupId.x;
  let head = workgroupId.y;
  if (row >= parameters.rows || head >= parameters.heads) { return; }
  let axis = pair / 16u;
  let axisPair = pair % 16u;
  let lowDimension = axis * 32u + axisPair;
  let highDimension = lowDimension + 16u;
  let base = (row * parameters.heads + head) * 64u;
  let rotaryIndex = row * 32u + pair;
  let c = cosine[rotaryIndex];
  let s = sine[rotaryIndex];
  let low = activations[base + lowDimension];
  let high = activations[base + highDimension];
  activations[base + lowDimension] = low * c - high * s;
  activations[base + highDimension] = high * c + low * s;
}`;
}