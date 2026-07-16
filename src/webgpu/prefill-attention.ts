export interface GemmaPrefillAttentionPipeline {
  headDimension: 64 | 256 | 512;
  queryTile: 16 | 32;
  pipeline: GPUComputePipeline;
}

export interface GemmaPrefillAttentionParameters {
  sequence: number;
  keyLength: number;
  queryOffset: number;
  queryHeads: number;
  kvHeads: number;
  window: number;
  causal?: boolean;
}

export interface GemmaPrefillAttentionResources {
  bindGroup: GPUBindGroup;
  output: GPUBuffer;
  parameters: GPUBuffer;
  sequenceCapacity: number;
  queryHeads: number;
  cacheCapacity: number;
  ownedBuffers: GPUBuffer[];
}

const pipelineCache = new WeakMap<
  GPUDevice,
  Map<64 | 256 | 512, Promise<GemmaPrefillAttentionPipeline>>
>();

export function getGemmaPrefillAttentionPipeline(
  device: GPUDevice,
  headDimension: 64 | 256 | 512,
): Promise<GemmaPrefillAttentionPipeline> {
  let devicePipelines = pipelineCache.get(device);
  if (!devicePipelines) {
    devicePipelines = new Map();
    pipelineCache.set(device, devicePipelines);
  }
  const cached = devicePipelines.get(headDimension);
  if (cached) return cached;
  const compiled = compileGemmaPrefillAttentionPipeline(device, headDimension).catch(
    (error) => {
      devicePipelines?.delete(headDimension);
      throw error;
    },
  );
  devicePipelines.set(headDimension, compiled);
  return compiled;
}

export async function compileGemmaPrefillAttentionPipeline(
  device: GPUDevice,
  headDimension: 64 | 256 | 512,
): Promise<GemmaPrefillAttentionPipeline> {
  const queryTile = headDimension === 512 ? 32 : 16;
  const pipeline = await device.createComputePipelineAsync({
    label: `Gemma prefill tiled attention ${headDimension}`,
    layout: "auto",
    compute: {
      module: device.createShaderModule({
        code: createGemmaPrefillAttentionShader(headDimension),
      }),
      entryPoint: "main",
    },
  });
  return { headDimension, queryTile, pipeline };
}

export function createGemmaPrefillAttentionResources(
  device: GPUDevice,
  compiled: GemmaPrefillAttentionPipeline,
  query: GPUBuffer,
  key: GPUBuffer,
  value: GPUBuffer,
  sequenceCapacity: number,
  cacheCapacity: number,
  parameters: GemmaPrefillAttentionParameters,
  output?: GPUBuffer,
): GemmaPrefillAttentionResources {
  validateParameters(parameters, sequenceCapacity, cacheCapacity);
  const queryBytes = sequenceCapacity * parameters.queryHeads * compiled.headDimension * 4;
  const cacheBytes = cacheCapacity * parameters.kvHeads * compiled.headDimension * 4;
  if (query.size < queryBytes || key.size < cacheBytes || value.size < cacheBytes ||
      (output && output.size < queryBytes)) {
    throw new Error("Gemma prefill attention buffers do not match pipeline geometry");
  }
  const ownedBuffers: GPUBuffer[] = [];
  const outputBuffer = output ?? device.createBuffer({
    label: "Gemma prefill attention output",
    size: queryBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  if (!output) ownedBuffers.push(outputBuffer);
  const parameterBuffer = device.createBuffer({
    label: "Gemma prefill attention parameters",
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  ownedBuffers.push(parameterBuffer);
  const resources = {
    bindGroup: device.createBindGroup({
      layout: compiled.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: query } },
        { binding: 1, resource: { buffer: key } },
        { binding: 2, resource: { buffer: value } },
        { binding: 3, resource: { buffer: outputBuffer } },
        { binding: 4, resource: { buffer: parameterBuffer } },
      ],
    }),
    output: outputBuffer,
    parameters: parameterBuffer,
    sequenceCapacity,
    queryHeads: parameters.queryHeads,
    cacheCapacity,
    ownedBuffers,
  };
  updateGemmaPrefillAttention(device, resources, cacheCapacity, parameters);
  return resources;
}

export function updateGemmaPrefillAttention(
  device: GPUDevice,
  resources: GemmaPrefillAttentionResources,
  cacheCapacity: number,
  parameters: GemmaPrefillAttentionParameters,
): void {
  validateParameters(parameters, resources.sequenceCapacity, cacheCapacity);
  if (parameters.queryHeads !== resources.queryHeads) {
    throw new Error("Gemma prefill attention query head count cannot change after binding");
  }
  device.queue.writeBuffer(resources.parameters, 0, new Uint32Array([
    parameters.sequence,
    parameters.keyLength,
    parameters.queryOffset,
    parameters.queryHeads,
    parameters.kvHeads,
    parameters.window,
    resources.cacheCapacity,
    parameters.causal === false ? 0 : 1,
  ]));
}

export function encodeGemmaPrefillAttention(
  encoder: GPUCommandEncoder,
  compiled: GemmaPrefillAttentionPipeline,
  resources: GemmaPrefillAttentionResources,
  sequence: number,
): void {
  const pass = encoder.beginComputePass({ label: "Gemma prefill tiled attention" });
  encodeGemmaPrefillAttentionPass(pass, compiled, resources, sequence);
  pass.end();
}

export function encodeGemmaPrefillAttentionPass(
  pass: GPUComputePassEncoder,
  compiled: GemmaPrefillAttentionPipeline,
  resources: GemmaPrefillAttentionResources,
  sequence: number,
): void {
  if (!Number.isInteger(sequence) || sequence < 1 || sequence > resources.sequenceCapacity) {
    throw new Error("Gemma prefill attention dispatch exceeds sequence capacity");
  }
  pass.setPipeline(compiled.pipeline);
  pass.setBindGroup(0, resources.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(sequence / compiled.queryTile), resources.queryHeads);
}

export function destroyGemmaPrefillAttentionResources(
  resources: GemmaPrefillAttentionResources,
): void {
  for (const buffer of resources.ownedBuffers) buffer.destroy();
}

export function createGemmaPrefillAttentionShader(
  headDimension: 64 | 256 | 512,
): string {
  const queryTile = headDimension === 512 ? 32 : 16;
  const slice = headDimension / 32;
  const stagedType = headDimension === 512 ? "f16" : "f32";
  const scores = Array.from({ length: 8 }, (_, key) => `    var score${key}: f32 = NEGATIVE_INFINITY;
    {
      let keyIndex = keyStart + ${key}u;
      var partial: f32 = 0.0;
      let keyBase = ${key}u * (HEAD_DIMENSION / 4u) + clusterLane * SLICE;
      for (var component = 0u; component < SLICE; component = component + 1u) {
        partial = partial + dot(querySlice[component], vec4<f32>(keyTile[keyBase + component]));
      }
      partial = partial + subgroupShuffleXor(partial, 1u);
      partial = partial + subgroupShuffleXor(partial, 2u);
      partial = partial + subgroupShuffleXor(partial, 4u);
      if (keyIndex >= minimumKey && keyIndex < maximumKey) {
        score${key} = partial;
      }
    }`).join("\n");
  const tileMax = Array.from({ length: 7 }, (_, index) =>
    `    tileMaximum = max(tileMaximum, score${index + 1});`).join("\n");
  const probabilities = Array.from({ length: 8 }, (_, key) =>
    `    let probability${key} = select(0.0, exp(score${key} - newMaximum), ` +
    `score${key} != NEGATIVE_INFINITY);`).join("\n");
  const probabilitySum = Array.from({ length: 8 }, (_, key) => `probability${key}`).join(" + ");
  const valueAccumulation = Array.from({ length: 8 }, (_, key) =>
    `      accumulator = accumulator + probability${key} * vec4<f32>(` +
    `valueTile[${key}u * (HEAD_DIMENSION / 4u) + clusterLane * SLICE + component]);`,
  ).join("\n");

  return `${headDimension === 512 ? "enable f16;\n" : ""}enable subgroups;

struct Parameters {
  sequence: u32,
  keyLength: u32,
  queryOffset: u32,
  queryHeads: u32,
  kvHeads: u32,
  window: u32,
  cacheCapacity: u32,
  causal: u32,
}

@group(0) @binding(0) var<storage, read> query: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> key: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> value: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> parameters: Parameters;

const HEAD_DIMENSION: u32 = ${headDimension}u;
const QUERY_TILE: u32 = ${queryTile}u;
const LANES_PER_QUERY: u32 = 8u;
const SLICE: u32 = ${slice}u;
const KEY_TILE: u32 = 8u;
const NEGATIVE_INFINITY: f32 = -3.4028234663852886e38;

var<workgroup> keyTile: array<vec4<${stagedType}>, KEY_TILE * (HEAD_DIMENSION / 4u)>;
var<workgroup> valueTile: array<vec4<${stagedType}>, KEY_TILE * (HEAD_DIMENSION / 4u)>;

@compute @workgroup_size(${queryTile * 8}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) thread: u32,
) {
  let queryHead = workgroupId.y;
  let queryInTile = thread / LANES_PER_QUERY;
  let clusterLane = thread % LANES_PER_QUERY;
  let queryIndex = workgroupId.x * QUERY_TILE + queryInTile;
  let queryValid = queryIndex < parameters.sequence && queryHead < parameters.queryHeads;
  let groupSize = parameters.queryHeads / parameters.kvHeads;
  let kvHead = queryHead / groupSize;
  let queryPosition = parameters.queryOffset + min(queryIndex, parameters.sequence - 1u);
  let queryBase = (min(queryIndex, parameters.sequence - 1u) * parameters.queryHeads +
    queryHead) * (HEAD_DIMENSION / 4u) + clusterLane * SLICE;
  var querySlice: array<vec4<f32>, SLICE>;
  var outputSlice: array<vec4<f32>, SLICE>;
  for (var component = 0u; component < SLICE; component = component + 1u) {
    querySlice[component] = query[queryBase + component];
    outputSlice[component] = vec4<f32>(0.0);
  }
  var runningMaximum = NEGATIVE_INFINITY;
  var runningDenominator = 0.0;
  var maximumKey = parameters.keyLength;
  if (parameters.causal != 0u) {
    maximumKey = min(parameters.keyLength, queryPosition + 1u);
  }
  var minimumKey = 0u;
  if (parameters.window > 0u && queryPosition + 1u > parameters.window) {
    minimumKey = queryPosition + 1u - parameters.window;
  }
  let lastQueryPosition = parameters.queryOffset + min(
    workgroupId.x * QUERY_TILE + QUERY_TILE - 1u,
    parameters.sequence - 1u,
  );
  var workgroupEnd = parameters.keyLength;
  if (parameters.causal != 0u) {
    workgroupEnd = min(parameters.keyLength, lastQueryPosition + 1u);
  }
  let firstQueryPosition = parameters.queryOffset + workgroupId.x * QUERY_TILE;
  var workgroupStart = 0u;
  if (parameters.window > 0u && firstQueryPosition + 1u > parameters.window) {
    workgroupStart = firstQueryPosition + 1u - parameters.window;
  }

  var keyStart = workgroupStart;
  loop {
    if (keyStart >= workgroupEnd) { break; }
    workgroupBarrier();
    for (var index = thread; index < KEY_TILE * (HEAD_DIMENSION / 4u);
        index = index + ${queryTile * 8}u) {
      let slot = index / (HEAD_DIMENSION / 4u);
      let dimension = index % (HEAD_DIMENSION / 4u);
      let keyIndex = keyStart + slot;
      let physicalKey = keyIndex % parameters.cacheCapacity;
      let cacheBase = (physicalKey * parameters.kvHeads + kvHead) *
        (HEAD_DIMENSION / 4u) + dimension;
      if (keyIndex < workgroupEnd) {
        keyTile[index] = vec4<${stagedType}>(key[cacheBase]);
        valueTile[index] = vec4<${stagedType}>(value[cacheBase]);
      } else {
        keyTile[index] = vec4<${stagedType}>(0.0);
        valueTile[index] = vec4<${stagedType}>(0.0);
      }
    }
    workgroupBarrier();
${scores}
    var tileMaximum = score0;
${tileMax}
    let newMaximum = max(runningMaximum, tileMaximum);
    let correction = select(
      exp(runningMaximum - newMaximum),
      0.0,
      runningMaximum == NEGATIVE_INFINITY,
    );
${probabilities}
    runningDenominator = runningDenominator * correction + (${probabilitySum});
    for (var component = 0u; component < SLICE; component = component + 1u) {
      var accumulator = outputSlice[component] * correction;
${valueAccumulation}
      outputSlice[component] = accumulator;
    }
    runningMaximum = newMaximum;
    keyStart = keyStart + KEY_TILE;
  }

  if (queryValid) {
    let outputBase = (queryIndex * parameters.queryHeads + queryHead) *
      (HEAD_DIMENSION / 4u) + clusterLane * SLICE;
    let inverseDenominator = 1.0 / runningDenominator;
    for (var component = 0u; component < SLICE; component = component + 1u) {
      output[outputBase + component] = outputSlice[component] * inverseDenominator;
    }
  }
}`;
}

function validateParameters(
  parameters: GemmaPrefillAttentionParameters,
  sequenceCapacity: number,
  cacheCapacity: number,
): void {
  const integers = [
    parameters.sequence,
    parameters.keyLength,
    parameters.queryOffset,
    parameters.queryHeads,
    parameters.kvHeads,
    parameters.window,
  ];
  if (integers.some((value) => !Number.isInteger(value) || value < 0) ||
      (parameters.causal !== undefined && typeof parameters.causal !== "boolean") ||
      parameters.sequence < 1 || parameters.sequence > sequenceCapacity ||
      parameters.keyLength < 1 || parameters.keyLength > cacheCapacity ||
      parameters.queryOffset + parameters.sequence > parameters.keyLength ||
      parameters.queryHeads < 1 || parameters.kvHeads < 1 ||
      parameters.queryHeads % parameters.kvHeads !== 0) {
    throw new Error("Gemma prefill attention parameters are invalid");
  }
}