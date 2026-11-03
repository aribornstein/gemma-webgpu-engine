const VOCAB_SIZE = 262144;
const WORKGROUP_SIZE = 256;
const SORT_SIZE = 1024;
const LIST_COUNT = VOCAB_SIZE / SORT_SIZE;
const ENTRY_BYTES = 8;
const TAIL_RESULT_BYTES = 8;

/** Largest top-k the GPU candidate path supports. */
export const GEMMA_MAX_GPU_TOP_K = 256;

export interface GemmaTopCandidatePipelines {
  listLength: number;
  select: GPUComputePipeline;
  merge: GPUComputePipeline;
  tail: GPUComputePipeline;
  finish: GPUComputePipeline;
}

interface MergeLevel {
  bindGroup: GPUBindGroup;
  workgroups: number;
}

export interface GemmaTopCandidateResources {
  pipelines: GemmaTopCandidatePipelines;
  params: GPUBuffer;
  selectBindGroup: GPUBindGroup;
  mergeLevels: MergeLevel[];
  tailBindGroup: GPUBindGroup;
  finishBindGroup: GPUBindGroup;
  finalList: GPUBuffer;
  tailResult: GPUBuffer;
  readback: GPUBuffer;
  buffers: GPUBuffer[];
}

export interface GemmaTopCandidates {
  /** Token IDs ranked by descending raw logit, ties by ascending ID. Length is the list length. */
  tokenIds: Uint32Array;
  /** Raw (pre-softcap) logits of `tokenIds`. */
  logits: Float32Array;
  /**
   * `sum(exp(scaled - scaledMax))` over every token outside the first `k` candidates, where
   * `scaled = softcap(logit) / temperature` is evaluated in f32 on the GPU.
   */
  tailMass: number;
  /** Count of NaN or infinite logits; callers must fall back to full logits when non-zero. */
  nonFiniteCount: number;
}

/** Candidate list length used for a requested top-k: a power of two from 16 to 256. */
export function gemmaTopCandidateListLength(topK: number): number {
  if (!Number.isInteger(topK) || topK < 1 || topK > GEMMA_MAX_GPU_TOP_K) {
    throw new Error(`GPU top-k must be an integer from 1 to ${GEMMA_MAX_GPU_TOP_K}`);
  }
  let length = 16;
  while (length < topK) length *= 2;
  return length;
}

const pipelineCache = new WeakMap<GPUDevice, Map<number, Promise<GemmaTopCandidatePipelines>>>();

export function getGemmaTopCandidatePipelines(
  device: GPUDevice,
  listLength: number,
): Promise<GemmaTopCandidatePipelines> {
  validateListLength(listLength);
  let devicePipelines = pipelineCache.get(device);
  if (!devicePipelines) {
    devicePipelines = new Map();
    pipelineCache.set(device, devicePipelines);
  }
  const cached = devicePipelines.get(listLength);
  if (cached) return cached;
  const compiled = compilePipelines(device, listLength).catch((error) => {
    devicePipelines?.delete(listLength);
    throw error;
  });
  devicePipelines.set(listLength, compiled);
  return compiled;
}

async function compilePipelines(
  device: GPUDevice,
  listLength: number,
): Promise<GemmaTopCandidatePipelines> {
  const module = device.createShaderModule({
    label: `Gemma top-${listLength} candidates`,
    code: createGemmaTopCandidateShader(listLength),
  });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(({ type }) => type === "error");
  if (errors.length > 0) {
    throw new Error(errors.map(({ lineNum, linePos, message }) =>
      `${lineNum}:${linePos} ${message}`).join("\n"));
  }
  const pipeline = (entryPoint: string) => device.createComputePipelineAsync({
    label: `Gemma top-${listLength} candidates ${entryPoint}`,
    layout: "auto",
    compute: { module, entryPoint },
  });
  const [select, merge, tail, finish] = await Promise.all([
    pipeline("selectTop"),
    pipeline("mergeTop"),
    pipeline("tailMass"),
    pipeline("finishTail"),
  ]);
  return { listLength, select, merge, tail, finish };
}

export function createGemmaTopCandidateResources(
  device: GPUDevice,
  pipelines: GemmaTopCandidatePipelines,
  logits: GPUBuffer,
): GemmaTopCandidateResources {
  const { listLength } = pipelines;
  const listsPerMerge = SORT_SIZE / listLength;
  const buffers: GPUBuffer[] = [];
  const createBuffer = (descriptor: GPUBufferDescriptor) => {
    const buffer = device.createBuffer(descriptor);
    buffers.push(buffer);
    return buffer;
  };
  const params = createBuffer({
    label: "Gemma top candidates params",
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const lists = [0, 1].map((index) => createBuffer({
    label: `Gemma top candidates lists ${index}`,
    size: LIST_COUNT * listLength * ENTRY_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  }));
  const tailPartials = createBuffer({
    label: "Gemma top candidates tail partials",
    size: LIST_COUNT * 8,
    usage: GPUBufferUsage.STORAGE,
  });
  const tailResult = createBuffer({
    label: "Gemma top candidates tail result",
    size: TAIL_RESULT_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readback = createBuffer({
    label: "Gemma top candidates readback",
    size: listLength * ENTRY_BYTES + TAIL_RESULT_BYTES,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const entry = (binding: number, buffer: GPUBuffer): GPUBindGroupEntry => ({
    binding,
    resource: { buffer },
  });
  const selectBindGroup = device.createBindGroup({
    layout: pipelines.select.getBindGroupLayout(0),
    entries: [entry(1, logits), entry(3, lists[0])],
  });
  const mergeLevels: MergeLevel[] = [];
  let listCount = LIST_COUNT;
  let current = 0;
  while (listCount > 1) {
    const workgroups = Math.ceil(listCount / listsPerMerge);
    const levelParams = createBuffer({
      label: `Gemma top candidates merge level ${mergeLevels.length}`,
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(levelParams, 0, new Uint32Array([listCount, 0, 0, 0]));
    mergeLevels.push({
      workgroups,
      bindGroup: device.createBindGroup({
        layout: pipelines.merge.getBindGroupLayout(0),
        entries: [entry(2, lists[current]), entry(3, lists[1 - current]), entry(5, levelParams)],
      }),
    });
    current = 1 - current;
    listCount = workgroups;
  }
  const finalList = lists[current];
  return {
    pipelines,
    params,
    selectBindGroup,
    mergeLevels,
    tailBindGroup: device.createBindGroup({
      layout: pipelines.tail.getBindGroupLayout(0),
      entries: [entry(0, params), entry(1, logits), entry(2, finalList), entry(4, tailPartials)],
    }),
    finishBindGroup: device.createBindGroup({
      layout: pipelines.finish.getBindGroupLayout(0),
      entries: [entry(4, tailPartials), entry(6, tailResult)],
    }),
    finalList,
    tailResult,
    readback,
    buffers,
  };
}

/**
 * Sets the requested top-k, the softcap (0 disables it), and the temperature used for the tail
 * mass. Must be called before encoding.
 */
export function updateGemmaTopCandidates(
  device: GPUDevice,
  resources: GemmaTopCandidateResources,
  topK: number,
  softcap: number,
  temperature: number,
): void {
  if (!Number.isInteger(topK) || topK < 1 || topK > resources.pipelines.listLength) {
    throw new Error(`Top-k ${topK} does not fit candidate list length ${resources.pipelines.listLength}`);
  }
  if (!(temperature > 0) || !Number.isFinite(temperature)) {
    throw new Error("GPU top candidates require a positive finite temperature");
  }
  if (!(softcap >= 0) || !Number.isFinite(softcap)) throw new Error("Softcap must be finite and non-negative");
  const data = new ArrayBuffer(16);
  new Uint32Array(data, 0, 1)[0] = topK;
  new Float32Array(data, 4, 2).set([softcap, 1 / temperature]);
  device.queue.writeBuffer(resources.params, 0, data);
}

export function encodeGemmaTopCandidatesPass(
  pass: GPUComputePassEncoder,
  resources: GemmaTopCandidateResources,
): void {
  const { pipelines } = resources;
  pass.setPipeline(pipelines.select);
  pass.setBindGroup(0, resources.selectBindGroup);
  pass.dispatchWorkgroups(LIST_COUNT);
  pass.setPipeline(pipelines.merge);
  for (const level of resources.mergeLevels) {
    pass.setBindGroup(0, level.bindGroup);
    pass.dispatchWorkgroups(level.workgroups);
  }
  pass.setPipeline(pipelines.tail);
  pass.setBindGroup(0, resources.tailBindGroup);
  pass.dispatchWorkgroups(LIST_COUNT);
  pass.setPipeline(pipelines.finish);
  pass.setBindGroup(0, resources.finishBindGroup);
  pass.dispatchWorkgroups(1);
}

export function encodeGemmaTopCandidatesCopy(
  encoder: GPUCommandEncoder,
  resources: GemmaTopCandidateResources,
): void {
  const listBytes = resources.pipelines.listLength * ENTRY_BYTES;
  encoder.copyBufferToBuffer(resources.finalList, 0, resources.readback, 0, listBytes);
  encoder.copyBufferToBuffer(resources.tailResult, 0, resources.readback, listBytes, TAIL_RESULT_BYTES);
}

export async function readGemmaTopCandidates(
  resources: GemmaTopCandidateResources,
): Promise<GemmaTopCandidates> {
  const { listLength } = resources.pipelines;
  await resources.readback.mapAsync(GPUMapMode.READ);
  const bytes = resources.readback.getMappedRange().slice(0);
  resources.readback.unmap();
  const words = new Uint32Array(bytes);
  const floats = new Float32Array(bytes);
  const tokenIds = new Uint32Array(listLength);
  const logits = new Float32Array(listLength);
  for (let index = 0; index < listLength; index += 1) {
    logits[index] = floats[2 * index];
    tokenIds[index] = words[2 * index + 1];
  }
  return {
    tokenIds,
    logits,
    tailMass: floats[2 * listLength],
    nonFiniteCount: words[2 * listLength + 1],
  };
}

export function destroyGemmaTopCandidateResources(resources: GemmaTopCandidateResources): void {
  for (const buffer of resources.buffers) buffer.destroy();
}

function validateListLength(listLength: number): void {
  if (![16, 32, 64, 128, 256].includes(listLength)) {
    throw new Error("Top candidate list length must be a power of two from 16 to 256");
  }
}

export function createGemmaTopCandidateShader(listLength: number): string {
  validateListLength(listLength);
  return /* wgsl */ `
struct Entry {
  value: f32,
  id: u32,
}

struct SamplingParams {
  k: u32,
  softcap: f32,
  inverseTemperature: f32,
  unused: u32,
}

struct LevelParams {
  listCount: u32,
  unused0: u32,
  unused1: u32,
  unused2: u32,
}

const SORT_SIZE = ${SORT_SIZE}u;
const LIST_LENGTH = ${listLength}u;
const LISTS_PER_MERGE = ${SORT_SIZE / listLength}u;
const PADDING_ID = 0xffffffffu;

@group(0) @binding(0) var<uniform> params: SamplingParams;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read> listsIn: array<Entry>;
@group(0) @binding(3) var<storage, read_write> listsOut: array<Entry>;
@group(0) @binding(4) var<storage, read_write> tailPartials: array<vec2<u32>>;
@group(0) @binding(5) var<uniform> mergeLevel: LevelParams;
@group(0) @binding(6) var<storage, read_write> tailResult: array<u32, 2>;

var<workgroup> keys: array<f32, ${SORT_SIZE}>;
var<workgroup> ids: array<u32, ${SORT_SIZE}>;
var<workgroup> sums: array<f32, ${WORKGROUP_SIZE}>;
var<workgroup> counts: array<u32, ${WORKGROUP_SIZE}>;

// Rank order: larger value first, ties by smaller token ID. Signed zeros compare equal.
fn ranksBefore(leftValue: f32, leftId: u32, rightValue: f32, rightId: u32) -> bool {
  return leftValue > rightValue || (leftValue == rightValue && leftId < rightId);
}

fn bitonicSort(lane: u32) {
  for (var size = 2u; size <= SORT_SIZE; size = size << 1u) {
    for (var stride = size >> 1u; stride > 0u; stride = stride >> 1u) {
      for (var pair = lane; pair < SORT_SIZE / 2u; pair += ${WORKGROUP_SIZE}u) {
        let left = (pair / stride) * 2u * stride + (pair % stride);
        let right = left + stride;
        let rankAscending = (left & size) == 0u;
        let rightFirst = ranksBefore(keys[right], ids[right], keys[left], ids[left]);
        let leftFirst = ranksBefore(keys[left], ids[left], keys[right], ids[right]);
        if (select(leftFirst, rightFirst, rankAscending)) {
          let value = keys[left];
          let id = ids[left];
          keys[left] = keys[right];
          ids[left] = ids[right];
          keys[right] = value;
          ids[right] = id;
        }
      }
      workgroupBarrier();
    }
  }
}

// Padding ranks after every finite logit (ties lose on PADDING_ID). Non-finite logits make the
// caller fall back to full logits, so padding never needs to rank after them.
const PADDING_VALUE = -0x1.fffffep+127f;

fn isNonFinite(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u;
}

fn scaled(value: f32) -> f32 {
  var capped = value;
  if (params.softcap > 0.0) {
    // Clamping keeps tanh away from implementations that overflow exp; tanh(15) rounds to 1.
    capped = params.softcap * tanh(clamp(value / params.softcap, -15.0, 15.0));
  }
  return capped * params.inverseTemperature;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn selectTop(
  @builtin(workgroup_id) workgroup: vec3<u32>,
  @builtin(local_invocation_id) local: vec3<u32>,
) {
  let lane = local.x;
  let base = workgroup.x * SORT_SIZE;
  for (var index = lane; index < SORT_SIZE; index += ${WORKGROUP_SIZE}u) {
    keys[index] = logits[base + index];
    ids[index] = base + index;
  }
  workgroupBarrier();
  bitonicSort(lane);
  for (var index = lane; index < LIST_LENGTH; index += ${WORKGROUP_SIZE}u) {
    listsOut[workgroup.x * LIST_LENGTH + index] = Entry(keys[index], ids[index]);
  }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn mergeTop(
  @builtin(workgroup_id) workgroup: vec3<u32>,
  @builtin(local_invocation_id) local: vec3<u32>,
) {
  let lane = local.x;
  let firstList = workgroup.x * LISTS_PER_MERGE;
  for (var index = lane; index < SORT_SIZE; index += ${WORKGROUP_SIZE}u) {
    if (firstList + index / LIST_LENGTH < mergeLevel.listCount) {
      let entry = listsIn[firstList * LIST_LENGTH + index];
      keys[index] = entry.value;
      ids[index] = entry.id;
    } else {
      keys[index] = PADDING_VALUE;
      ids[index] = PADDING_ID;
    }
  }
  workgroupBarrier();
  bitonicSort(lane);
  for (var index = lane; index < LIST_LENGTH; index += ${WORKGROUP_SIZE}u) {
    listsOut[workgroup.x * LIST_LENGTH + index] = Entry(keys[index], ids[index]);
  }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn tailMass(
  @builtin(workgroup_id) workgroup: vec3<u32>,
  @builtin(local_invocation_id) local: vec3<u32>,
) {
  let lane = local.x;
  let last = listsIn[params.k - 1u];
  let maximum = scaled(listsIn[0].value);
  var sum = 0.0;
  var nonFinite = 0u;
  for (var offset = 0u; offset < SORT_SIZE; offset += ${WORKGROUP_SIZE}u) {
    let token = workgroup.x * SORT_SIZE + offset + lane;
    let value = logits[token];
    if (isNonFinite(value)) {
      nonFinite += 1u;
    } else if (!ranksBefore(value, token, last.value, last.id) && token != last.id) {
      sum += exp(scaled(value) - maximum);
    }
  }
  sums[lane] = sum;
  counts[lane] = nonFinite;
  workgroupBarrier();
  for (var stride = ${WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride >> 1u) {
    if (lane < stride) {
      sums[lane] += sums[lane + stride];
      counts[lane] += counts[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    tailPartials[workgroup.x] = vec2<u32>(bitcast<u32>(sums[0]), counts[0]);
  }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn finishTail(@builtin(local_invocation_id) local: vec3<u32>) {
  let lane = local.x;
  let partial = tailPartials[lane];
  sums[lane] = bitcast<f32>(partial.x);
  counts[lane] = partial.y;
  workgroupBarrier();
  for (var stride = ${WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride >> 1u) {
    if (lane < stride) {
      sums[lane] += sums[lane + stride];
      counts[lane] += counts[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    tailResult[0] = bitcast<u32>(sums[0]);
    tailResult[1] = counts[0];
  }
}
`;
}
