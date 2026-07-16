import {
  createGemmaPrefillParameter,
  gemmaPrefillParameterBinding,
  writeGemmaPrefillParameter,
  type GemmaPrefillParameterArena,
} from "./prefill-parameter-arena";

export interface GemmaPrefillPleDenseGeometry {
  rows: number;
  inFeatures: number;
  outFeatures: number;
}

export interface GemmaPrefillPleDensePipeline extends GemmaPrefillPleDenseGeometry {
  outputRowsPerWorkgroup: 1 | 2;
  pipeline: GPUComputePipeline;
}

export interface GemmaPrefillPleDenseWeights {
  codes: GPUBuffer;
  rowScales: GPUBuffer;
  inputScale: number;
  outputScale: number;
}

export interface GemmaPrefillPleDenseResources {
  bindGroup: GPUBindGroup;
  output: GPUBuffer;
  workgroupCountX: number;
  workgroupCountY: number;
  ownedBuffers: GPUBuffer[];
}

const pipelineCache = new WeakMap<GPUDevice, Map<string, Promise<GemmaPrefillPleDensePipeline>>>();

export function getGemmaPrefillPleDensePipeline(
  device: GPUDevice,
  geometry: GemmaPrefillPleDenseGeometry,
): Promise<GemmaPrefillPleDensePipeline> {
  validateGeometry(geometry);
  let devicePipelines = pipelineCache.get(device);
  if (!devicePipelines) {
    devicePipelines = new Map();
    pipelineCache.set(device, devicePipelines);
  }
  const key = `${geometry.rows}:${geometry.inFeatures}:${geometry.outFeatures}`;
  const cached = devicePipelines.get(key);
  if (cached) return cached;
  const compiled = compileGemmaPrefillPleDensePipeline(device, geometry).catch((error) => {
    devicePipelines?.delete(key);
    throw error;
  });
  devicePipelines.set(key, compiled);
  return compiled;
}

export async function compileGemmaPrefillPleDensePipeline(
  device: GPUDevice,
  geometry: GemmaPrefillPleDenseGeometry,
): Promise<GemmaPrefillPleDensePipeline> {
  validateGeometry(geometry);
  const outputRowsPerWorkgroup = geometry.outFeatures >= 1024 ? 2 : 1;
  const pipeline = await device.createComputePipelineAsync({
    label: `Gemma prefill PLE dense ${geometry.inFeatures}x${geometry.outFeatures}`,
    layout: "auto",
    compute: {
      module: device.createShaderModule({
        code: createGemmaPrefillPleDenseShader(geometry, outputRowsPerWorkgroup),
      }),
      entryPoint: "main",
    },
  });
  return { ...geometry, outputRowsPerWorkgroup, pipeline };
}

export function createGemmaPrefillPleDenseResources(
  device: GPUDevice,
  compiled: GemmaPrefillPleDensePipeline,
  activation: GPUBuffer,
  weights: GemmaPrefillPleDenseWeights,
  output?: GPUBuffer,
  parameterArena?: GemmaPrefillParameterArena,
): GemmaPrefillPleDenseResources {
  const activationBytes = compiled.rows * compiled.inFeatures * 4;
  const codeBytes = compiled.outFeatures * compiled.inFeatures;
  const outputBytes = compiled.rows * compiled.outFeatures * 4;
  if (activation.size < activationBytes || weights.codes.size < codeBytes ||
      weights.rowScales.size < compiled.outFeatures * 4 ||
      (output && output.size < outputBytes)) {
    throw new Error("Gemma prefill PLE dense buffers do not match pipeline geometry");
  }
  if (!Number.isFinite(weights.inputScale) || !Number.isFinite(weights.outputScale)) {
    throw new Error("Gemma prefill PLE dense scales must be finite");
  }
  const ownedBuffers: GPUBuffer[] = [];
  const outputBuffer = output ?? device.createBuffer({
    label: "Gemma prefill PLE dense output",
    size: outputBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  if (!output) ownedBuffers.push(outputBuffer);
  const parameters = createGemmaPrefillParameter(
    device,
    16,
    "Gemma prefill PLE dense parameters",
    parameterArena,
  );
  ownedBuffers.push(...parameters.ownedBuffers);
  writeGemmaPrefillParameter(
    device,
    parameters.slice,
    new Float32Array([weights.inputScale, weights.outputScale, 0, 0]),
  );
  const groups = Math.ceil(compiled.outFeatures / compiled.outputRowsPerWorkgroup);
  const workgroupCountX = Math.min(groups, 65535);
  return {
    bindGroup: device.createBindGroup({
      layout: compiled.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: activation } },
        { binding: 1, resource: { buffer: weights.codes } },
        { binding: 2, resource: { buffer: weights.rowScales } },
        { binding: 3, resource: { buffer: outputBuffer } },
        gemmaPrefillParameterBinding(4, parameters.slice),
      ],
    }),
    output: outputBuffer,
    workgroupCountX,
    workgroupCountY: Math.ceil(groups / workgroupCountX),
    ownedBuffers,
  };
}

export function encodeGemmaPrefillPleDense(
  encoder: GPUCommandEncoder,
  compiled: GemmaPrefillPleDensePipeline,
  resources: GemmaPrefillPleDenseResources,
): void {
  const pass = encoder.beginComputePass({ label: compiled.pipeline.label });
  encodeGemmaPrefillPleDensePass(pass, compiled, resources);
  pass.end();
}

export function encodeGemmaPrefillPleDensePass(
  pass: GPUComputePassEncoder,
  compiled: GemmaPrefillPleDensePipeline,
  resources: GemmaPrefillPleDenseResources,
): void {
  pass.setPipeline(compiled.pipeline);
  pass.setBindGroup(0, resources.bindGroup);
  pass.dispatchWorkgroups(resources.workgroupCountX, resources.workgroupCountY);
}

export function destroyGemmaPrefillPleDenseResources(
  resources: GemmaPrefillPleDenseResources,
): void {
  for (const buffer of resources.ownedBuffers) buffer.destroy();
}

export function createGemmaPrefillPleDenseShader(
  geometry: GemmaPrefillPleDenseGeometry,
  outputRowsPerWorkgroup = geometry.outFeatures >= 1024 ? 2 : 1,
): string {
  validateGeometry(geometry);
  if (outputRowsPerWorkgroup !== 1 && outputRowsPerWorkgroup !== 2) {
    throw new Error("Gemma prefill PLE dense output tile is unsupported");
  }
  return `enable subgroups;

struct Parameters {
  inputScale: f32,
  outputScale: f32,
}

@group(0) @binding(0) var<storage, read> activation: array<f32>;
@group(0) @binding(1) var<storage, read> codes: array<u32>;
@group(0) @binding(2) var<storage, read> rowScales: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> parameters: Parameters;

const ROWS: u32 = ${geometry.rows}u;
const IN_FEATURES: u32 = ${geometry.inFeatures}u;
const OUT_FEATURES: u32 = ${geometry.outFeatures}u;
const OUTPUT_ROWS: u32 = ${outputRowsPerWorkgroup}u;
const WORDS_PER_ROW: u32 = IN_FEATURES / 4u;
const GRID_X: u32 = ${Math.min(Math.ceil(geometry.outFeatures / outputRowsPerWorkgroup), 65535)}u;

fn srq4(value: vec4<f32>, scale: f32) -> vec4<f32> {
  if (scale == 0.0) { return value; }
  return clamp(round(value / scale), vec4<f32>(-128.0), vec4<f32>(127.0)) * scale;
}

fn srq(value: f32, scale: f32) -> f32 {
  if (scale == 0.0) { return value; }
  return clamp(round(value / scale), -128.0, 127.0) * scale;
}

fn signedWeights(word: u32, scale: f32) -> vec4<f32> {
  return vec4<f32>(
    f32(word & 255u) - 128.0,
    f32((word >> 8u) & 255u) - 128.0,
    f32((word >> 16u) & 255u) - 128.0,
    f32((word >> 24u) & 255u) - 128.0,
  ) * scale;
}

@compute @workgroup_size(32, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let group = workgroup.y * GRID_X + workgroup.x;
  let outputBase = group * OUTPUT_ROWS;
  if (outputBase >= OUT_FEATURES) { return; }

  for (var row = 0u; row < ROWS; row = row + 1u) {
    var accumulators: array<f32, OUTPUT_ROWS>;
    for (var outputRow = 0u; outputRow < OUTPUT_ROWS; outputRow = outputRow + 1u) {
      accumulators[outputRow] = 0.0;
    }
    var word = lane;
    loop {
      if (word >= WORDS_PER_ROW) { break; }
      let inputBase = row * IN_FEATURES + word * 4u;
      let values = srq4(vec4<f32>(
        activation[inputBase],
        activation[inputBase + 1u],
        activation[inputBase + 2u],
        activation[inputBase + 3u],
      ), parameters.inputScale);
      for (var outputRow = 0u; outputRow < OUTPUT_ROWS; outputRow = outputRow + 1u) {
        let outputIndex = outputBase + outputRow;
        if (outputIndex < OUT_FEATURES) {
          let packed = codes[outputIndex * WORDS_PER_ROW + word];
          accumulators[outputRow] = accumulators[outputRow] + dot(
            signedWeights(packed, rowScales[outputIndex]),
            values,
          );
        }
      }
      word = word + 32u;
    }
    for (var outputRow = 0u; outputRow < OUTPUT_ROWS; outputRow = outputRow + 1u) {
      let sum = subgroupAdd(accumulators[outputRow]);
      let outputIndex = outputBase + outputRow;
      if (lane == 0u && outputIndex < OUT_FEATURES) {
        output[row * OUT_FEATURES + outputIndex] = srq(sum, parameters.outputScale);
      }
    }
  }
}`;
}

function validateGeometry(geometry: GemmaPrefillPleDenseGeometry): void {
  if (!Number.isInteger(geometry.rows) || geometry.rows < 1 ||
      !Number.isInteger(geometry.inFeatures) || geometry.inFeatures < 4 ||
      geometry.inFeatures % 4 !== 0 ||
      !Number.isInteger(geometry.outFeatures) || geometry.outFeatures < 1) {
    throw new Error("Gemma prefill PLE dense geometry is invalid");
  }
}