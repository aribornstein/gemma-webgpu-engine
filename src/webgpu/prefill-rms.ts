import {
  createGemmaPrefillParameter,
  gemmaPrefillParameterBinding,
  writeGemmaPrefillParameter,
  type GemmaPrefillParameterArena,
} from "./prefill-parameter-arena";

export interface GemmaPrefillRmsPipeline {
  dimension: number;
  weighted: boolean;
  pipeline: GPUComputePipeline;
}

export interface GemmaPrefillRmsResources {
  bindGroup: GPUBindGroup;
  output: GPUBuffer;
  rows: number;
  ownedBuffers: GPUBuffer[];
}

export interface GemmaPrefillRmsResidualResources {
  bindGroup: GPUBindGroup;
  rows: number;
  ownedBuffers: GPUBuffer[];
}

export type GemmaPrefillRmsBufferSlice = GPUBuffer | {
  buffer: GPUBuffer;
  offset: number;
  size: number;
};

const pipelineCache = new WeakMap<
  GPUDevice,
  Map<string, Promise<GemmaPrefillRmsPipeline>>
>();
const residualPipelineCache = new WeakMap<
  GPUDevice,
  Map<string, Promise<GemmaPrefillRmsPipeline>>
>();

export function getGemmaPrefillRmsPipeline(
  device: GPUDevice,
  dimension: number,
  weighted: boolean,
): Promise<GemmaPrefillRmsPipeline> {
  validateDimension(dimension);
  let devicePipelines = pipelineCache.get(device);
  if (!devicePipelines) {
    devicePipelines = new Map();
    pipelineCache.set(device, devicePipelines);
  }
  const key = `${dimension}:${weighted}`;
  const cached = devicePipelines.get(key);
  if (cached) return cached;
  const compiled = compileGemmaPrefillRmsPipeline(device, dimension, weighted).catch(
    (error) => {
      devicePipelines?.delete(key);
      throw error;
    },
  );
  devicePipelines.set(key, compiled);
  return compiled;
}

export async function compileGemmaPrefillRmsPipeline(
  device: GPUDevice,
  dimension: number,
  weighted: boolean,
): Promise<GemmaPrefillRmsPipeline> {
  validateDimension(dimension);
  const pipeline = await device.createComputePipelineAsync({
    label: `Gemma prefill exact RMS ${dimension}`,
    layout: "auto",
    compute: {
      module: device.createShaderModule({
        code: createGemmaPrefillRmsShader(dimension, weighted),
      }),
      entryPoint: "main",
    },
  });
  return { dimension, weighted, pipeline };
}

export function getGemmaPrefillRmsResidualPipeline(
  device: GPUDevice,
  dimension: number,
  scaled: boolean,
): Promise<GemmaPrefillRmsPipeline> {
  validateDimension(dimension);
  let devicePipelines = residualPipelineCache.get(device);
  if (!devicePipelines) {
    devicePipelines = new Map();
    residualPipelineCache.set(device, devicePipelines);
  }
  const key = `${dimension}:${scaled}`;
  const cached = devicePipelines.get(key);
  if (cached) return cached;
  const compiled = compileGemmaPrefillRmsResidualPipeline(device, dimension, scaled).catch(
    (error) => {
      devicePipelines?.delete(key);
      throw error;
    },
  );
  devicePipelines.set(key, compiled);
  return compiled;
}

export async function compileGemmaPrefillRmsResidualPipeline(
  device: GPUDevice,
  dimension: number,
  scaled: boolean,
): Promise<GemmaPrefillRmsPipeline> {
  validateDimension(dimension);
  const pipeline = await device.createComputePipelineAsync({
    label: scaled
      ? `Gemma prefill exact RMS residual scale ${dimension}`
      : `Gemma prefill exact RMS residual ${dimension}`,
    layout: "auto",
    compute: {
      module: device.createShaderModule({
        code: createGemmaPrefillRmsShader(
          dimension,
          true,
          scaled ? "residual-scale" : "residual",
        ),
      }),
      entryPoint: "main",
    },
  });
  return { dimension, weighted: true, pipeline };
}

export function createGemmaPrefillRmsResources(
  device: GPUDevice,
  compiled: GemmaPrefillRmsPipeline,
  rows: number,
  input: GPUBuffer,
  weight: GemmaPrefillRmsBufferSlice | null,
  output?: GPUBuffer,
  parameterArena?: GemmaPrefillParameterArena,
): GemmaPrefillRmsResources {
  if (!Number.isInteger(rows) || rows < 1 || rows > 65535) {
    throw new Error("Gemma prefill RMS row count must be a positive integer below 65536");
  }
  const dataBytes = rows * compiled.dimension * 4;
  if (input.size < dataBytes || (output && output.size < dataBytes) ||
      (compiled.weighted && (!weight || sliceSize(weight) < compiled.dimension * 4)) ||
      (!compiled.weighted && weight)) {
    throw new Error("Gemma prefill RMS buffers do not match pipeline geometry");
  }
  const ownedBuffers: GPUBuffer[] = [];
  const make = (label: string, size: number, usage: GPUBufferUsageFlags) => {
    const buffer = device.createBuffer({ label, size, usage });
    ownedBuffers.push(buffer);
    return buffer;
  };
  const outputBuffer = output ?? make(
    "Gemma prefill exact RMS output",
    dataBytes,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const parameterAllocation = createGemmaPrefillParameter(
    device,
    16,
    "Gemma prefill exact RMS parameters",
    parameterArena,
  );
  ownedBuffers.push(...parameterAllocation.ownedBuffers);
  writeGemmaPrefillParameter(
    device,
    parameterAllocation.slice,
    new Uint32Array([rows, rows, 0, 0]),
  );
  const entries = [binding(0, input)];
  if (compiled.weighted) entries.push(sliceBinding(1, weight!));
  entries.push(binding(compiled.weighted ? 2 : 1, outputBuffer));
  entries.push(gemmaPrefillParameterBinding(
    compiled.weighted ? 3 : 2,
    parameterAllocation.slice,
  ));
  return {
    bindGroup: device.createBindGroup({
      layout: compiled.pipeline.getBindGroupLayout(0),
      entries,
    }),
    output: outputBuffer,
    rows,
    ownedBuffers,
  };
}

export function encodeGemmaPrefillRms(
  encoder: GPUCommandEncoder,
  compiled: GemmaPrefillRmsPipeline,
  resources: GemmaPrefillRmsResources,
): void {
  const pass = encoder.beginComputePass({ label: "Gemma prefill exact RMS" });
  encodeGemmaPrefillRmsPass(pass, compiled, resources);
  pass.end();
}

export function encodeGemmaPrefillRmsPass(
  pass: GPUComputePassEncoder,
  compiled: GemmaPrefillRmsPipeline,
  resources: GemmaPrefillRmsResources,
): void {
  pass.setPipeline(compiled.pipeline);
  pass.setBindGroup(0, resources.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(resources.rows / 64));
}

export function destroyGemmaPrefillRmsResources(
  resources: GemmaPrefillRmsResources,
): void {
  for (const buffer of resources.ownedBuffers) buffer.destroy();
}

export function createGemmaPrefillRmsResidualResources(
  device: GPUDevice,
  compiled: GemmaPrefillRmsPipeline,
  rows: number,
  input: GPUBuffer,
  weight: GemmaPrefillRmsBufferSlice,
  residual: GPUBuffer,
  factors: GPUBuffer | null,
  factorIndex: number,
  parameterArena?: GemmaPrefillParameterArena,
): GemmaPrefillRmsResidualResources {
  if (!compiled.weighted || !Number.isInteger(rows) || rows < 1 || rows > 65535) {
    throw new Error("Gemma prefill RMS residual geometry is invalid");
  }
  const dataBytes = rows * compiled.dimension * 4;
  if (input.size < dataBytes || residual.size < dataBytes ||
      sliceSize(weight) < compiled.dimension * 4 ||
      (factors && (!Number.isInteger(factorIndex) || factorIndex < 0 ||
        factors.size < (factorIndex + 1) * 4)) ||
      (!factors && factorIndex !== 0)) {
    throw new Error("Gemma prefill RMS residual buffers do not match pipeline geometry");
  }
  const parameterAllocation = createGemmaPrefillParameter(
    device,
    16,
    "Gemma prefill exact RMS residual parameters",
    parameterArena,
  );
  writeGemmaPrefillParameter(
    device,
    parameterAllocation.slice,
    new Uint32Array([rows, rows, factorIndex, 0]),
  );
  const entries: GPUBindGroupEntry[] = [
    binding(0, input),
    sliceBinding(1, weight),
    binding(2, residual),
  ];
  if (factors) entries.push(binding(3, factors));
  entries.push(gemmaPrefillParameterBinding(factors ? 4 : 3, parameterAllocation.slice));
  return {
    bindGroup: device.createBindGroup({
      layout: compiled.pipeline.getBindGroupLayout(0),
      entries,
    }),
    rows,
    ownedBuffers: parameterAllocation.ownedBuffers,
  };
}

export function encodeGemmaPrefillRmsResidual(
  encoder: GPUCommandEncoder,
  compiled: GemmaPrefillRmsPipeline,
  resources: GemmaPrefillRmsResidualResources,
): void {
  const pass = encoder.beginComputePass({ label: compiled.pipeline.label });
  encodeGemmaPrefillRmsResidualPass(pass, compiled, resources);
  pass.end();
}

export function encodeGemmaPrefillRmsResidualPass(
  pass: GPUComputePassEncoder,
  compiled: GemmaPrefillRmsPipeline,
  resources: GemmaPrefillRmsResidualResources,
): void {
  pass.setPipeline(compiled.pipeline);
  pass.setBindGroup(0, resources.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(resources.rows / 64));
}

export function destroyGemmaPrefillRmsResidualResources(
  resources: GemmaPrefillRmsResidualResources,
): void {
  for (const buffer of resources.ownedBuffers) buffer.destroy();
}

export function createGemmaPrefillRmsShader(
  dimension: number,
  weighted: boolean,
  epilogue: "output" | "residual" | "residual-scale" = "output",
): string {
  validateDimension(dimension);
  if (!weighted && epilogue !== "output") {
    throw new Error("Gemma prefill RMS residual epilogues require weights");
  }
  const inverseDimension = Math.fround(1 / dimension);
  const weightBinding = weighted
    ? "@group(0) @binding(1) var<storage, read> weight: array<vec4<f32>>;"
    : "";
  const outputBinding = weighted ? 2 : 1;
  const paramsBinding = weighted ? 3 : 2;
  const outputDeclaration = epilogue === "output"
    ? `@group(0) @binding(${outputBinding}) var<storage, read_write> output: array<vec4<f32>>;`
    : `@group(0) @binding(2) var<storage, read_write> residual: array<vec4<f32>>;${
      epilogue === "residual-scale"
        ? "\n@group(0) @binding(3) var<storage, read> factors: array<f32>;"
        : ""
    }`;
  const parameterBinding = epilogue === "residual-scale"
    ? 4
    : epilogue === "residual" ? 3 : paramsBinding;
  const outputExpression = epilogue === "residual-scale"
    ? `    let weighted = fma(normalized, weight[vector], vec4<f32>(0.0));
    let added = fma(residual[base + vector], vec4<f32>(1.0), weighted);
    residual[base + vector] = added * vec4<f32>(factors[params.factorIndex]);`
    : epilogue === "residual"
      ? `    let weighted = fma(normalized, weight[vector], vec4<f32>(0.0));
    residual[base + vector] = fma(
      residual[base + vector],
      vec4<f32>(1.0),
      weighted,
    );`
      : weighted
        ? "    output[base + vector] = normalized * weight[vector];"
        : "    output[base + vector] = normalized;";

  return `struct Params {
  rows: u32,
  rowStride: u32,
  factorIndex: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
${weightBinding}
${outputDeclaration}
@group(0) @binding(${parameterBinding}) var<uniform> params: Params;

const DIMENSION: u32 = ${dimension}u;
const DIMENSION_F: f32 = ${dimension}.0;
const INVERSE_DIMENSION: f32 = ${inverseDimension};
const EPSILON: f32 = 0.000001;
const VECTOR_COUNT: u32 = DIMENSION / 4u;
const ILP_COUNT: u32 = VECTOR_COUNT / 4u;

fn divExact(value: f32, divisor: f32, reciprocal: f32) -> f32 {
  let quotient = value * reciprocal;
  let remainder = fma(-divisor, quotient, value);
  return fma(remainder, reciprocal, quotient);
}

fn reciprocalExact(value: f32) -> f32 {
  let first = 1.0 / value;
  let second = fma(fma(-value, first, 1.0), first, first);
  return fma(fma(-value, second, 1.0), second, second);
}

fn sqrtExact(value: f32) -> f32 {
  var inverse = inverseSqrt(value);
  var root = value * inverse;
  var halfInverse = 0.5 * inverse;
  inverse = fma(-root, halfInverse, 0.5);
  root = fma(root, inverse, root);
  halfInverse = fma(halfInverse, inverse, halfInverse);
  inverse = fma(-root, halfInverse, 1.5);
  halfInverse = halfInverse + halfInverse;
  halfInverse = halfInverse * inverse;
  root = halfInverse * value;
  inverse = fma(halfInverse, value, -root);
  var correction = fma(-halfInverse, root, 1.0);
  correction = fma(-halfInverse, inverse, correction);
  halfInverse = 0.5 * root;
  halfInverse = fma(halfInverse, correction, inverse);
  return halfInverse + root;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let row = invocation.x;
  if (row >= params.rows) { return; }
  let base = row * VECTOR_COUNT;
  var level0_0 = vec4<f32>(0.0);
  var level0_1 = vec4<f32>(0.0);
  var level0_2 = vec4<f32>(0.0);
  var level0_3 = vec4<f32>(0.0);
  var level1_0 = vec4<f32>(0.0);
  var level1_1 = vec4<f32>(0.0);
  var level1_2 = vec4<f32>(0.0);
  var level1_3 = vec4<f32>(0.0);
  var level2_0 = vec4<f32>(0.0);
  var level2_1 = vec4<f32>(0.0);
  var level2_2 = vec4<f32>(0.0);
  var level2_3 = vec4<f32>(0.0);
  var level3_0 = vec4<f32>(0.0);
  var level3_1 = vec4<f32>(0.0);
  var level3_2 = vec4<f32>(0.0);
  var level3_3 = vec4<f32>(0.0);

  var index = 0u;
  loop {
    if (index + 16u > ILP_COUNT) { break; }
    for (var block = 0u; block < 16u; block = block + 1u) {
      let vectorBase = base + (index + block) * 4u;
      let value0 = input[vectorBase];
      let value1 = input[vectorBase + 1u];
      let value2 = input[vectorBase + 2u];
      let value3 = input[vectorBase + 3u];
      level0_0 = fma(fma(value0, value0, vec4<f32>(0.0)), vec4<f32>(1.0), level0_0);
      level0_1 = fma(fma(value1, value1, vec4<f32>(0.0)), vec4<f32>(1.0), level0_1);
      level0_2 = fma(fma(value2, value2, vec4<f32>(0.0)), vec4<f32>(1.0), level0_2);
      level0_3 = fma(fma(value3, value3, vec4<f32>(0.0)), vec4<f32>(1.0), level0_3);
    }
    index = index + 16u;
    level1_0 = fma(level0_0, vec4<f32>(1.0), level1_0);
    level1_1 = fma(level0_1, vec4<f32>(1.0), level1_1);
    level1_2 = fma(level0_2, vec4<f32>(1.0), level1_2);
    level1_3 = fma(level0_3, vec4<f32>(1.0), level1_3);
    level0_0 = vec4<f32>(0.0);
    level0_1 = vec4<f32>(0.0);
    level0_2 = vec4<f32>(0.0);
    level0_3 = vec4<f32>(0.0);
    if ((index & 0xf0u) == 0u) {
      level2_0 = fma(level1_0, vec4<f32>(1.0), level2_0);
      level2_1 = fma(level1_1, vec4<f32>(1.0), level2_1);
      level2_2 = fma(level1_2, vec4<f32>(1.0), level2_2);
      level2_3 = fma(level1_3, vec4<f32>(1.0), level2_3);
      level1_0 = vec4<f32>(0.0);
      level1_1 = vec4<f32>(0.0);
      level1_2 = vec4<f32>(0.0);
      level1_3 = vec4<f32>(0.0);
      if ((index & 0xf00u) == 0u) {
        level3_0 = fma(level2_0, vec4<f32>(1.0), level3_0);
        level3_1 = fma(level2_1, vec4<f32>(1.0), level3_1);
        level3_2 = fma(level2_2, vec4<f32>(1.0), level3_2);
        level3_3 = fma(level2_3, vec4<f32>(1.0), level3_3);
        level2_0 = vec4<f32>(0.0);
        level2_1 = vec4<f32>(0.0);
        level2_2 = vec4<f32>(0.0);
        level2_3 = vec4<f32>(0.0);
      }
    }
  }
  loop {
    if (index >= ILP_COUNT) { break; }
    let vectorBase = base + index * 4u;
    let value0 = input[vectorBase];
    let value1 = input[vectorBase + 1u];
    let value2 = input[vectorBase + 2u];
    let value3 = input[vectorBase + 3u];
    level0_0 = fma(fma(value0, value0, vec4<f32>(0.0)), vec4<f32>(1.0), level0_0);
    level0_1 = fma(fma(value1, value1, vec4<f32>(0.0)), vec4<f32>(1.0), level0_1);
    level0_2 = fma(fma(value2, value2, vec4<f32>(0.0)), vec4<f32>(1.0), level0_2);
    level0_3 = fma(fma(value3, value3, vec4<f32>(0.0)), vec4<f32>(1.0), level0_3);
    index = index + 1u;
  }
  level0_0 = fma(level1_0, vec4<f32>(1.0), level0_0);
  level0_1 = fma(level1_1, vec4<f32>(1.0), level0_1);
  level0_2 = fma(level1_2, vec4<f32>(1.0), level0_2);
  level0_3 = fma(level1_3, vec4<f32>(1.0), level0_3);
  level0_0 = fma(level2_0, vec4<f32>(1.0), level0_0);
  level0_1 = fma(level2_1, vec4<f32>(1.0), level0_1);
  level0_2 = fma(level2_2, vec4<f32>(1.0), level0_2);
  level0_3 = fma(level2_3, vec4<f32>(1.0), level0_3);
  level0_0 = fma(level3_0, vec4<f32>(1.0), level0_0);
  level0_1 = fma(level3_1, vec4<f32>(1.0), level0_1);
  level0_2 = fma(level3_2, vec4<f32>(1.0), level0_2);
  level0_3 = fma(level3_3, vec4<f32>(1.0), level0_3);
  var vector = ILP_COUNT * 4u;
  loop {
    if (vector >= VECTOR_COUNT) { break; }
    let value = input[base + vector];
    level0_0 = fma(fma(value, value, vec4<f32>(0.0)), vec4<f32>(1.0), level0_0);
    vector = vector + 1u;
  }
  level0_0 = fma(level0_1, vec4<f32>(1.0), level0_0);
  level0_0 = fma(level0_2, vec4<f32>(1.0), level0_0);
  level0_0 = fma(level0_3, vec4<f32>(1.0), level0_0);
  var squareSum = level0_0.x;
  squareSum = fma(level0_0.y, 1.0, squareSum);
  squareSum = fma(level0_0.z, 1.0, squareSum);
  squareSum = fma(level0_0.w, 1.0, squareSum);
  let meanSquare = divExact(squareSum, DIMENSION_F, INVERSE_DIMENSION) + EPSILON;
  let inverseRms = reciprocalExact(sqrtExact(meanSquare));
  for (vector = 0u; vector < VECTOR_COUNT; vector = vector + 1u) {
    let normalized = input[base + vector] * vec4<f32>(inverseRms);
${outputExpression}
  }
}`;
}

function binding(bindingIndex: number, buffer: GPUBuffer): GPUBindGroupEntry {
  return { binding: bindingIndex, resource: { buffer } };
}

function sliceBinding(
  bindingIndex: number,
  slice: GemmaPrefillRmsBufferSlice,
): GPUBindGroupEntry {
  return slice instanceof GPUBuffer
    ? binding(bindingIndex, slice)
    : {
        binding: bindingIndex,
        resource: { buffer: slice.buffer, offset: slice.offset, size: slice.size },
      };
}

function sliceSize(slice: GemmaPrefillRmsBufferSlice): number {
  return slice instanceof GPUBuffer ? slice.size : slice.size;
}

function validateDimension(dimension: number): void {
  if (!Number.isInteger(dimension) || dimension < 4 ||
      dimension >= 4_194_304 || dimension % 4 !== 0) {
    throw new Error("Gemma prefill RMS dimension must be a multiple of four below 4194304");
  }
}