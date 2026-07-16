const PATCH_DIMENSION = 768;
const HIDDEN_SIZE = 768;
const POSITION_SIZE = 10_240;
const WORKGROUP_SIZE = 256;

export interface GemmaVisionPatchEmbedResources {
  bindGroup: GPUBindGroup;
  output: GPUBuffer;
  parameters: GPUBuffer;
  rowCapacity: number;
  ownedBuffers: GPUBuffer[];
}

const pipelineCache = new WeakMap<GPUDevice, Promise<GPUComputePipeline>>();

export function getGemmaVisionPatchEmbedPipeline(device: GPUDevice): Promise<GPUComputePipeline> {
  const cached = pipelineCache.get(device);
  if (cached) return cached;
  const compiled = device.createComputePipelineAsync({
    label: "Gemma vision BF16 patch and position embedder",
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: createGemmaVisionPatchEmbedShader() }),
      entryPoint: "main",
    },
  }).catch((error) => {
    pipelineCache.delete(device);
    throw error;
  });
  pipelineCache.set(device, compiled);
  return compiled;
}

export function createGemmaVisionPatchEmbedResources(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  patches: GPUBuffer,
  positions: GPUBuffer,
  projection: GPUBuffer,
  positionEmbeddings: GPUBuffer,
  rowCapacity: number,
  output?: GPUBuffer,
): GemmaVisionPatchEmbedResources {
  if (!Number.isInteger(rowCapacity) || rowCapacity < 1 ||
      patches.size < rowCapacity * PATCH_DIMENSION * 4 ||
      positions.size < rowCapacity * 2 * 4 ||
      projection.size < HIDDEN_SIZE * PATCH_DIMENSION * 4 ||
      positionEmbeddings.size < 2 * POSITION_SIZE * HIDDEN_SIZE * 4) {
    throw new Error("Gemma vision patch embed buffers do not match model geometry");
  }
  const ownedBuffers: GPUBuffer[] = [];
  const outputBuffer = output ?? device.createBuffer({
    label: "Gemma vision patch embeddings",
    size: rowCapacity * HIDDEN_SIZE * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  if (!output) ownedBuffers.push(outputBuffer);
  const parameters = device.createBuffer({
    label: "Gemma vision patch embed parameters",
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  ownedBuffers.push(parameters);
  return {
    bindGroup: device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        binding(0, patches),
        binding(1, positions),
        binding(2, projection),
        binding(3, positionEmbeddings),
        binding(4, outputBuffer),
        binding(5, parameters),
      ],
    }),
    output: outputBuffer,
    parameters,
    rowCapacity,
    ownedBuffers,
  };
}

export function updateGemmaVisionPatchEmbed(
  device: GPUDevice,
  resources: GemmaVisionPatchEmbedResources,
  rows: number,
): void {
  if (!Number.isInteger(rows) || rows < 1 || rows > resources.rowCapacity) {
    throw new Error("Gemma vision patch embed row count is invalid");
  }
  device.queue.writeBuffer(resources.parameters, 0, new Uint32Array([rows, 0, 0, 0]));
}

export function encodeGemmaVisionPatchEmbed(
  encoder: GPUCommandEncoder,
  pipeline: GPUComputePipeline,
  resources: GemmaVisionPatchEmbedResources,
  rows: number,
): void {
  const pass = encoder.beginComputePass({ label: "Gemma vision patch embed" });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, resources.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(rows * HIDDEN_SIZE / WORKGROUP_SIZE));
  pass.end();
}

export function createGemmaVisionPatchEmbedShader(): string {
  return `
struct Parameters { rows: u32, padding0: u32, padding1: u32, padding2: u32 }
@group(0) @binding(0) var<storage, read> patches: array<f32>;
@group(0) @binding(1) var<storage, read> positions: array<i32>;
@group(0) @binding(2) var<storage, read> projection: array<f32>;
@group(0) @binding(3) var<storage, read> positionEmbeddings: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> parameters: Parameters;

fn bf16(value: f32) -> f32 {
  let bits = bitcast<u32>(value);
  let rounding = 0x7fffu + ((bits >> 16u) & 1u);
  return bitcast<f32>((bits + rounding) & 0xffff0000u);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  if (index >= parameters.rows * ${HIDDEN_SIZE}u) { return; }
  let row = index / ${HIDDEN_SIZE}u;
  let feature = index % ${HIDDEN_SIZE}u;
  var accumulator = 0.0;
  for (var column = 0u; column < ${PATCH_DIMENSION}u; column = column + 1u) {
    let pixel = bf16(2.0 * (patches[row * ${PATCH_DIMENSION}u + column] - 0.5));
    accumulator = accumulator + pixel * projection[feature * ${PATCH_DIMENSION}u + column];
  }
  let x = positions[row * 2u];
  let y = positions[row * 2u + 1u];
  if (x >= 0 && y >= 0) {
    accumulator = accumulator + positionEmbeddings[u32(x) * ${HIDDEN_SIZE}u + feature];
    accumulator = accumulator + positionEmbeddings[
      (${POSITION_SIZE}u + u32(y)) * ${HIDDEN_SIZE}u + feature
    ];
  }
  output[index] = accumulator;
}`;
}

function binding(bindingIndex: number, buffer: GPUBuffer): GPUBindGroupEntry {
  return { binding: bindingIndex, resource: { buffer } };
}