export interface GemmaVisionF32DensePipeline {
  rows: number;
  inFeatures: number;
  outFeatures: number;
  pipeline: GPUComputePipeline;
}

export interface GemmaVisionF32DenseResources {
  bindGroup: GPUBindGroup;
  output: GPUBuffer;
  ownedBuffers: GPUBuffer[];
}

const pipelineCache = new WeakMap<GPUDevice, Map<string, Promise<GemmaVisionF32DensePipeline>>>();

export function getGemmaVisionF32DensePipeline(
  device: GPUDevice,
  rows: number,
  inFeatures: number,
  outFeatures: number,
): Promise<GemmaVisionF32DensePipeline> {
  validateGeometry(rows, inFeatures, outFeatures);
  let pipelines = pipelineCache.get(device);
  if (!pipelines) {
    pipelines = new Map();
    pipelineCache.set(device, pipelines);
  }
  const key = `${rows}:${inFeatures}:${outFeatures}`;
  const cached = pipelines.get(key);
  if (cached) return cached;
  const compiled = device.createComputePipelineAsync({
    label: `Gemma vision F32 dense ${inFeatures}x${outFeatures}`,
    layout: "auto",
    compute: {
      module: device.createShaderModule({
        code: createGemmaVisionF32DenseShader(rows, inFeatures, outFeatures),
      }),
      entryPoint: "main",
    },
  }).then((pipeline) => ({ rows, inFeatures, outFeatures, pipeline })).catch((error) => {
    pipelines?.delete(key);
    throw error;
  });
  pipelines.set(key, compiled);
  return compiled;
}

export function createGemmaVisionF32DenseResources(
  device: GPUDevice,
  compiled: GemmaVisionF32DensePipeline,
  input: GPUBuffer,
  weights: GPUBuffer,
  output?: GPUBuffer,
): GemmaVisionF32DenseResources {
  const outputBytes = compiled.rows * compiled.outFeatures * 4;
  if (input.size < compiled.rows * compiled.inFeatures * 4 ||
      weights.size < compiled.outFeatures * compiled.inFeatures * 4 ||
      (output && output.size < outputBytes)) {
    throw new Error("Gemma vision F32 dense buffers do not match geometry");
  }
  const ownedBuffers: GPUBuffer[] = [];
  const outputBuffer = output ?? device.createBuffer({
    label: "Gemma vision F32 dense output",
    size: outputBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  if (!output) ownedBuffers.push(outputBuffer);
  return {
    bindGroup: device.createBindGroup({
      layout: compiled.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: weights } },
        { binding: 2, resource: { buffer: outputBuffer } },
      ],
    }),
    output: outputBuffer,
    ownedBuffers,
  };
}

export function encodeGemmaVisionF32Dense(
  encoder: GPUCommandEncoder,
  compiled: GemmaVisionF32DensePipeline,
  resources: GemmaVisionF32DenseResources,
): void {
  const pass = encoder.beginComputePass({ label: compiled.pipeline.label });
  pass.setPipeline(compiled.pipeline);
  pass.setBindGroup(0, resources.bindGroup);
  pass.dispatchWorkgroups(compiled.outFeatures, compiled.rows);
  pass.end();
}

export function createGemmaVisionF32DenseShader(
  rows: number,
  inFeatures: number,
  outFeatures: number,
): string {
  validateGeometry(rows, inFeatures, outFeatures);
  return `enable subgroups;
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> weights: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const ROWS: u32 = ${rows}u;
const IN_VECTORS: u32 = ${inFeatures / 4}u;
const OUT_FEATURES: u32 = ${outFeatures}u;

@compute @workgroup_size(32)
fn main(
  @builtin(workgroup_id) workgroup: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let outputFeature = workgroup.x;
  let row = workgroup.y;
  if (outputFeature >= OUT_FEATURES || row >= ROWS) { return; }
  var sum = 0.0;
  for (var vector = lane; vector < IN_VECTORS; vector = vector + 32u) {
    sum = sum + dot(
      input[row * IN_VECTORS + vector],
      weights[outputFeature * IN_VECTORS + vector],
    );
  }
  sum = subgroupAdd(sum);
  if (lane == 0u) {
    output[row * OUT_FEATURES + outputFeature] = sum;
  }
}`;
}

function validateGeometry(rows: number, inFeatures: number, outFeatures: number): void {
  if (!Number.isInteger(rows) || rows < 1 || rows > 65535 ||
      !Number.isInteger(inFeatures) || inFeatures < 4 || inFeatures % 4 !== 0 ||
      !Number.isInteger(outFeatures) || outFeatures < 1 || outFeatures > 65535) {
    throw new Error("Gemma vision F32 dense geometry is invalid");
  }
}