const HIDDEN_SIZE = 1024;
const HEAD_COUNT = 8;
const HEAD_DIMENSION = 128;
const CONTEXT_LENGTH = 13;
const WORKGROUP_SIZE = 128;
const QUERY_SCALE = (HEAD_DIMENSION ** -0.5) / Math.log(2);
const KEY_SCALE = Math.log(1 + Math.E) / Math.log(2);
const LOGIT_CAP = 50;

export interface GemmaAudioAttentionResources {
  bindGroup: GPUBindGroup;
  output: GPUBuffer;
  rows: number;
  ownedBuffers: GPUBuffer[];
}

const pipelineCache = new WeakMap<GPUDevice, Promise<GPUComputePipeline>>();

export function getGemmaAudioAttentionPipeline(device: GPUDevice): Promise<GPUComputePipeline> {
  const cached = pipelineCache.get(device);
  if (cached) return cached;
  const pending = device.createComputePipelineAsync({
    label: "Gemma audio blocked relative attention",
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: createGemmaAudioAttentionShader() }),
      entryPoint: "main",
    },
  }).catch((error) => {
    pipelineCache.delete(device);
    throw error;
  });
  pipelineCache.set(device, pending);
  return pending;
}

export function createGemmaAudioAttentionResources(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  query: GPUBuffer,
  key: GPUBuffer,
  value: GPUBuffer,
  relativeKeys: GPUBuffer,
  perDimensionScale: GPUBuffer,
  mask: GPUBuffer,
  rows: number,
  output?: GPUBuffer,
): GemmaAudioAttentionResources {
  const hiddenBytes = rows * HIDDEN_SIZE * 4;
  if (!Number.isInteger(rows) || rows < 1 || rows > 750 ||
      query.size < hiddenBytes || key.size < hiddenBytes || value.size < hiddenBytes ||
      relativeKeys.size < CONTEXT_LENGTH * HIDDEN_SIZE * 4 ||
      perDimensionScale.size < HEAD_DIMENSION * 4 || mask.size < rows * 4 ||
      (output && output.size < hiddenBytes)) {
    throw new Error("Gemma audio attention buffers do not match model geometry");
  }
  const ownedBuffers: GPUBuffer[] = [];
  const outputBuffer = output ?? device.createBuffer({
    label: "Gemma audio attention output",
    size: hiddenBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  if (!output) ownedBuffers.push(outputBuffer);
  const parameters = device.createBuffer({
    label: "Gemma audio attention parameters",
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(parameters, 0, new Uint32Array([rows, 0, 0, 0]));
  ownedBuffers.push(parameters);
  return {
    bindGroup: device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        binding(0, query),
        binding(1, key),
        binding(2, value),
        binding(3, relativeKeys),
        binding(4, perDimensionScale),
        binding(5, mask),
        binding(6, outputBuffer),
        binding(7, parameters),
      ],
    }),
    output: outputBuffer,
    rows,
    ownedBuffers,
  };
}

export function encodeGemmaAudioAttention(
  encoder: GPUCommandEncoder,
  pipeline: GPUComputePipeline,
  resources: GemmaAudioAttentionResources,
): void {
  const pass = encoder.beginComputePass({ label: "Gemma audio blocked relative attention" });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, resources.bindGroup);
  pass.dispatchWorkgroups(HEAD_COUNT, resources.rows);
  pass.end();
}

export function destroyGemmaAudioAttentionResources(
  resources: GemmaAudioAttentionResources,
): void {
  for (const buffer of resources.ownedBuffers) buffer.destroy();
}

export function createGemmaAudioRelativePositions(): Float32Array {
  const output = new Float32Array(CONTEXT_LENGTH * HIDDEN_SIZE);
  const increment = Math.log(10_000) / (HIDDEN_SIZE / 2 - 1);
  for (let row = 0; row < CONTEXT_LENGTH; row += 1) {
    const position = CONTEXT_LENGTH - 1 - row;
    for (let dimension = 0; dimension < HIDDEN_SIZE / 2; dimension += 1) {
      const scaled = position * Math.exp(-dimension * increment);
      output[row * HIDDEN_SIZE + dimension] = Math.fround(Math.sin(scaled));
      output[row * HIDDEN_SIZE + HIDDEN_SIZE / 2 + dimension] =
        Math.fround(Math.cos(scaled));
    }
  }
  return output;
}

export function createGemmaAudioAttentionShader(): string {
  return `struct Parameters { rows: u32, padding0: u32, padding1: u32, padding2: u32 }
@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read> relativeKeys: array<f32>;
@group(0) @binding(4) var<storage, read> perDimensionScale: array<f32>;
@group(0) @binding(5) var<storage, read> mask: array<u32>;
@group(0) @binding(6) var<storage, read_write> output: array<f32>;
@group(0) @binding(7) var<uniform> parameters: Parameters;
var<workgroup> logits: array<f32, ${CONTEXT_LENGTH}>;

fn softplus(value: f32) -> f32 {
  return max(value, 0.0) + log(1.0 + exp(-abs(value)));
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) groupId: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let head = groupId.x;
  let queryRow = groupId.y;
  if (head >= ${HEAD_COUNT}u || queryRow >= parameters.rows) { return; }
  let headOffset = head * ${HEAD_DIMENSION}u;
  if (lane < ${CONTEXT_LENGTH}u) {
    let relativeRow = lane;
    let keyRow = i32(queryRow) + i32(relativeRow) - ${CONTEXT_LENGTH - 1};
    var logit = -1e9;
    if (mask[queryRow] != 0u && keyRow >= 0 && keyRow < i32(parameters.rows) &&
        mask[u32(keyRow)] != 0u) {
      var dotProduct = 0.0;
      for (var dimension = 0u; dimension < ${HEAD_DIMENSION}u; dimension++) {
        let q = query[queryRow * ${HIDDEN_SIZE}u + headOffset + dimension] *
          ${QUERY_SCALE} * softplus(perDimensionScale[dimension]);
        let k = key[u32(keyRow) * ${HIDDEN_SIZE}u + headOffset + dimension] * ${KEY_SCALE};
        let relative = relativeKeys[relativeRow * ${HIDDEN_SIZE}u + headOffset + dimension];
        dotProduct = fma(q, k + relative, dotProduct);
      }
      logit = tanh(dotProduct / ${LOGIT_CAP}.0) * ${LOGIT_CAP}.0;
    }
    logits[relativeRow] = logit;
  }
  workgroupBarrier();
  if (lane == 0u) {
    var maximum = logits[0];
    for (var index = 1u; index < ${CONTEXT_LENGTH}u; index++) {
      maximum = max(maximum, logits[index]);
    }
    var denominator = 0.0;
    for (var index = 0u; index < ${CONTEXT_LENGTH}u; index++) {
      logits[index] = exp(logits[index] - maximum);
      denominator += logits[index];
    }
    for (var index = 0u; index < ${CONTEXT_LENGTH}u; index++) {
      logits[index] /= denominator;
    }
  }
  workgroupBarrier();
  var result = 0.0;
  for (var relativeRow = 0u; relativeRow < ${CONTEXT_LENGTH}u; relativeRow++) {
    let keyRow = i32(queryRow) + i32(relativeRow) - ${CONTEXT_LENGTH - 1};
    if (keyRow >= 0 && keyRow < i32(parameters.rows)) {
      result = fma(
        logits[relativeRow],
        value[u32(keyRow) * ${HIDDEN_SIZE}u + headOffset + lane],
        result,
      );
    }
  }
  output[queryRow * ${HIDDEN_SIZE}u + headOffset + lane] =
    select(result, 0.0, mask[queryRow] == 0u);
}`;
}

function binding(bindingIndex: number, buffer: GPUBuffer): GPUBindGroupEntry {
  return { binding: bindingIndex, resource: { buffer } };
}