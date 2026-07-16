const HIDDEN_SIZE = 768;
const POOL_SIZE = 3;
const WORKGROUP_SIZE = 256;

export interface GemmaVisionPoolResources {
  bindGroup: GPUBindGroup;
  output: GPUBuffer;
  parameters: GPUBuffer;
  patchRows: number;
  patchColumns: number;
  outputRows: number;
  ownedBuffers: GPUBuffer[];
}

let pipelineCache = new WeakMap<GPUDevice, Promise<GPUComputePipeline>>();

export function getGemmaVisionPoolPipeline(device: GPUDevice): Promise<GPUComputePipeline> {
  const cached = pipelineCache.get(device);
  if (cached) return cached;
  const compiled = device.createComputePipelineAsync({
    label: "Gemma vision 3x3 position pool",
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: createGemmaVisionPoolShader() }),
      entryPoint: "main",
    },
  }).catch((error) => {
    pipelineCache.delete(device);
    throw error;
  });
  pipelineCache.set(device, compiled);
  return compiled;
}

export function createGemmaVisionPoolResources(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  input: GPUBuffer,
  patchRows: number,
  patchColumns: number,
  output?: GPUBuffer,
): GemmaVisionPoolResources {
  if (!Number.isInteger(patchRows) || patchRows < POOL_SIZE || patchRows % POOL_SIZE !== 0 ||
      !Number.isInteger(patchColumns) || patchColumns < POOL_SIZE ||
      patchColumns % POOL_SIZE !== 0 ||
      input.size < patchRows * patchColumns * HIDDEN_SIZE * 4) {
    throw new Error("Gemma vision pool geometry is invalid");
  }
  const outputRows = patchRows * patchColumns / (POOL_SIZE * POOL_SIZE);
  const outputBytes = outputRows * HIDDEN_SIZE * 4;
  if (output && output.size < outputBytes) {
    throw new Error("Gemma vision pool output is too small");
  }
  const ownedBuffers: GPUBuffer[] = [];
  const outputBuffer = output ?? device.createBuffer({
    label: "Gemma vision pooled patches",
    size: outputBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  if (!output) ownedBuffers.push(outputBuffer);
  const parameters = device.createBuffer({
    label: "Gemma vision pool parameters",
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  ownedBuffers.push(parameters);
  device.queue.writeBuffer(
    parameters,
    0,
    new Uint32Array([patchRows, patchColumns, outputRows, 0]),
  );
  return {
    bindGroup: device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: parameters } },
      ],
    }),
    output: outputBuffer,
    parameters,
    patchRows,
    patchColumns,
    outputRows,
    ownedBuffers,
  };
}

export function encodeGemmaVisionPool(
  encoder: GPUCommandEncoder,
  pipeline: GPUComputePipeline,
  resources: GemmaVisionPoolResources,
): void {
  const pass = encoder.beginComputePass({ label: "Gemma vision 3x3 position pool" });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, resources.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(resources.outputRows * HIDDEN_SIZE / WORKGROUP_SIZE));
  pass.end();
}

export function createGemmaVisionPoolShader(): string {
  return `struct Parameters {
  patchRows: u32,
  patchColumns: u32,
  outputRows: u32,
  padding: u32,
}
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> parameters: Parameters;

const HIDDEN_SIZE: u32 = 768u;
const POOLED_COLUMNS: u32 = 3u;
const SCALE: f32 = ${Math.fround(Math.sqrt(HIDDEN_SIZE) / 9)};

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= parameters.outputRows * HIDDEN_SIZE) { return; }
  let outputRow = index / HIDDEN_SIZE;
  let feature = index % HIDDEN_SIZE;
  let pooledWidth = parameters.patchColumns / POOLED_COLUMNS;
  let pooledY = outputRow / pooledWidth;
  let pooledX = outputRow % pooledWidth;
  var sum = 0.0;
  for (var y = 0u; y < 3u; y = y + 1u) {
    for (var x = 0u; x < 3u; x = x + 1u) {
      let sourceRow = (pooledY * 3u + y) * parameters.patchColumns +
        pooledX * 3u + x;
      sum = sum + input[sourceRow * HIDDEN_SIZE + feature];
    }
  }
  output[index] = sum * SCALE;
}`;
}