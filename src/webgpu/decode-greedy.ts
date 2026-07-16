const WORKGROUP_SIZE = 256;
const VALUES_PER_THREAD = 4;
const VALUES_PER_WORKGROUP = WORKGROUP_SIZE * VALUES_PER_THREAD;
const VOCAB_SIZE = 262144;

export interface GemmaGreedyPipelines {
  partial: GPUComputePipeline;
  final: GPUComputePipeline;
  inputCount: number;
  partialCount: number;
}

export interface GemmaGreedyResources {
  partialBindGroup: GPUBindGroup;
  finalBindGroup: GPUBindGroup;
  result: GPUBuffer;
  readback: GPUBuffer;
  buffers: GPUBuffer[];
}

export interface GemmaGreedyResult {
  token: number;
  logit: number;
}

const pipelineCache = new WeakMap<GPUDevice, Map<number, Promise<GemmaGreedyPipelines>>>();

export function getGemmaGreedyPipelines(
  device: GPUDevice,
  inputCount = VOCAB_SIZE,
): Promise<GemmaGreedyPipelines> {
  validateInputCount(inputCount);
  let devicePipelines = pipelineCache.get(device);
  if (!devicePipelines) {
    devicePipelines = new Map();
    pipelineCache.set(device, devicePipelines);
  }
  const cached = devicePipelines.get(inputCount);
  if (cached) return cached;
  const compiled = compileGemmaGreedyPipelines(device, inputCount).catch((error) => {
    devicePipelines?.delete(inputCount);
    throw error;
  });
  devicePipelines.set(inputCount, compiled);
  return compiled;
}

export async function compileGemmaGreedyPipelines(
  device: GPUDevice,
  inputCount = VOCAB_SIZE,
): Promise<GemmaGreedyPipelines> {
  validateInputCount(inputCount);
  const partialCount = Math.ceil(inputCount / VALUES_PER_WORKGROUP);
  if (partialCount > WORKGROUP_SIZE) {
    throw new Error(`Gemma greedy reduction supports at most ${VOCAB_SIZE} logits`);
  }
  const module = device.createShaderModule({
    code: createGemmaGreedyShader(inputCount, partialCount),
  });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(({ type }) => type === "error");
  if (errors.length > 0) {
    throw new Error(errors.map(({ lineNum, linePos, message }) =>
      `${lineNum}:${linePos} ${message}`).join("\n"));
  }
  const [partial, final] = await Promise.all([
    device.createComputePipelineAsync({
      label: "Gemma greedy partial reduction",
      layout: "auto",
      compute: { module, entryPoint: "partial" },
    }),
    device.createComputePipelineAsync({
      label: "Gemma greedy final reduction",
      layout: "auto",
      compute: { module, entryPoint: "finish" },
    }),
  ]);
  return { partial, final, inputCount, partialCount };
}

export function createGemmaGreedyResources(
  device: GPUDevice,
  pipelines: GemmaGreedyPipelines,
  logits: GPUBuffer,
): GemmaGreedyResources {
  const partialValues = device.createBuffer({
    label: "Gemma greedy partial values",
    size: pipelines.partialCount * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE,
  });
  const partialTokens = device.createBuffer({
    label: "Gemma greedy partial tokens",
    size: pipelines.partialCount * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE,
  });
  const result = device.createBuffer({
    label: "Gemma greedy result",
    size: 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    label: "Gemma greedy readback",
    size: 8,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const buffers = [partialValues, partialTokens, result, readback];
  const entry = (binding: number, buffer: GPUBuffer): GPUBindGroupEntry => ({
    binding,
    resource: { buffer },
  });
  return {
    partialBindGroup: device.createBindGroup({
      layout: pipelines.partial.getBindGroupLayout(0),
      entries: [entry(0, logits), entry(1, partialValues), entry(2, partialTokens)],
    }),
    finalBindGroup: device.createBindGroup({
      layout: pipelines.final.getBindGroupLayout(0),
      entries: [entry(1, partialValues), entry(2, partialTokens), entry(3, result)],
    }),
    result,
    readback,
    buffers,
  };
}

export function encodeGemmaGreedy(
  encoder: GPUCommandEncoder,
  pipelines: GemmaGreedyPipelines,
  resources: GemmaGreedyResources,
  copyResult = false,
): void {
  const partialPass = encoder.beginComputePass({ label: "Gemma greedy partial reduction" });
  partialPass.setPipeline(pipelines.partial);
  partialPass.setBindGroup(0, resources.partialBindGroup);
  partialPass.dispatchWorkgroups(pipelines.partialCount);
  partialPass.end();
  const finalPass = encoder.beginComputePass({ label: "Gemma greedy final reduction" });
  finalPass.setPipeline(pipelines.final);
  finalPass.setBindGroup(0, resources.finalBindGroup);
  finalPass.dispatchWorkgroups(1);
  finalPass.end();
  if (copyResult) {
    encoder.copyBufferToBuffer(resources.result, 0, resources.readback, 0, 8);
  }
}

export async function readGemmaGreedyResult(
  resources: GemmaGreedyResources,
): Promise<GemmaGreedyResult> {
  await resources.readback.mapAsync(GPUMapMode.READ);
  const bytes = resources.readback.getMappedRange();
  const view = new DataView(bytes);
  const result = {
    token: view.getUint32(0, true),
    logit: view.getFloat32(4, true),
  };
  resources.readback.unmap();
  return result;
}

export function destroyGemmaGreedyResources(resources: GemmaGreedyResources): void {
  for (const buffer of resources.buffers) buffer.destroy();
}

export function createGemmaGreedyShader(inputCount: number, partialCount: number): string {
  return `
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> partialValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> partialTokens: array<u32>;
@group(0) @binding(3) var<storage, read_write> result: array<u32>;

var<workgroup> scratchValues: array<f32, ${WORKGROUP_SIZE}>;
var<workgroup> scratchTokens: array<u32, ${WORKGROUP_SIZE}>;

fn shouldReplace(currentValue: f32, currentToken: u32, candidateValue: f32, candidateToken: u32) -> bool {
  return candidateValue > currentValue ||
    (candidateValue == currentValue && candidateToken < currentToken);
}

fn reduce(lane: u32) {
  var stride = ${WORKGROUP_SIZE / 2}u;
  loop {
    if (lane < stride && shouldReplace(
      scratchValues[lane],
      scratchTokens[lane],
      scratchValues[lane + stride],
      scratchTokens[lane + stride],
    )) {
      scratchValues[lane] = scratchValues[lane + stride];
      scratchTokens[lane] = scratchTokens[lane + stride];
    }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride = stride >> 1u;
  }
}

@compute @workgroup_size(${WORKGROUP_SIZE}, 1, 1)
fn partial(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let base = workgroupId.x * ${VALUES_PER_WORKGROUP}u;
  var bestValue = logits[base];
  var bestToken = base;
  for (var offset = lane; offset < ${VALUES_PER_WORKGROUP}u; offset = offset + ${WORKGROUP_SIZE}u) {
    let token = base + offset;
    if (token < ${inputCount}u) {
      let value = logits[token];
      if (shouldReplace(bestValue, bestToken, value, token)) {
        bestValue = value;
        bestToken = token;
      }
    }
  }
  scratchValues[lane] = bestValue;
  scratchTokens[lane] = bestToken;
  workgroupBarrier();
  reduce(lane);
  if (lane == 0u) {
    partialValues[workgroupId.x] = scratchValues[0];
    partialTokens[workgroupId.x] = scratchTokens[0];
  }
}

@compute @workgroup_size(${WORKGROUP_SIZE}, 1, 1)
fn finish(@builtin(local_invocation_index) lane: u32) {
  var value = partialValues[0];
  var token = partialTokens[0];
  if (lane < ${partialCount}u) {
    value = partialValues[lane];
    token = partialTokens[lane];
  }
  scratchValues[lane] = value;
  scratchTokens[lane] = token;
  workgroupBarrier();
  reduce(lane);
  if (lane == 0u) {
    result[0] = scratchTokens[0];
    result[1] = bitcast<u32>(scratchValues[0]);
  }
}`;
}

function validateInputCount(inputCount: number): void {
  if (!Number.isInteger(inputCount) || inputCount < 1) {
    throw new Error("Gemma greedy inputCount must be a positive integer");
  }
}