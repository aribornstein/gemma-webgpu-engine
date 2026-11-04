import type { GemmaResearchPatch } from "../research/engine-types";
import type { ResearchResidualEditTargets } from "./research-residual-edit";

export interface ResearchActivationPatch {
  layerIndex: number;
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  buffers: GPUBuffer[];
  stage: NonNullable<GemmaResearchPatch["stage"]>;
}

const pipelines = new WeakMap<GPUDevice, Map<string, Promise<GPUComputePipeline>>>();

export function getResearchActivationPatchPipeline(
  device: GPUDevice,
  stage: NonNullable<GemmaResearchPatch["stage"]> = "layer-output",
): Promise<GPUComputePipeline> {
  let devicePipelines = pipelines.get(device);
  if (!devicePipelines) { devicePipelines = new Map(); pipelines.set(device, devicePipelines); }
  const cached = devicePipelines.get(stage);
  if (cached) return cached;
  const compiled = device.createComputePipelineAsync({
    label: "Research token-local activation patch",
    layout: "auto",
    compute: { module: device.createShaderModule({ code: createShader(stage) }), entryPoint: "main" },
  }).catch((error) => {
    devicePipelines?.delete(stage);
    throw error;
  });
  devicePipelines.set(stage, compiled);
  return compiled;
}

export function createResearchActivationPatch(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  targets: ResearchResidualEditTargets,
  patch: GemmaResearchPatch,
): ResearchActivationPatch {
  if (patch.vector.length !== 1536 || !patch.vector.every(Number.isFinite) ||
      !Number.isFinite(patch.amount) || Math.abs(patch.amount) > 2 ||
      !["replace", "add"].includes(patch.mode)) throw new Error("Invalid activation patch");
  const vector = device.createBuffer({ label: "Research donor vector", size: 1536 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const params = device.createBuffer({ label: "Research patch parameters", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const stage = patch.stage ?? "layer-output";
  try {
    device.queue.writeBuffer(vector, 0, Float32Array.from(patch.vector));
    device.queue.writeBuffer(params, 0, new Float32Array([patch.amount, targets.inScale, patch.mode === "replace" ? 1 : 0, 0]));
    const buffers = [targets.hidden, targets.normWeightsAndScale, targets.nextInput, targets.nextSum, vector, params];
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.flatMap((buffer, binding) => stage === "mlp" && [1, 2, 3].includes(binding)
        ? [] : [{ binding, resource: { buffer } }]),
    });
    return { layerIndex: patch.layer, pipeline, bindGroup, buffers: [vector, params], stage };
  } catch (error) {
    vector.destroy();
    params.destroy();
    throw error;
  }
}

export function encodeResearchActivationPatch(pass: GPUComputePassEncoder, patch: ResearchActivationPatch): void {
  pass.setPipeline(patch.pipeline);
  pass.setBindGroup(0, patch.bindGroup);
  pass.dispatchWorkgroups(1);
}

function createShader(stage: NonNullable<GemmaResearchPatch["stage"]>): string {
  return /* wgsl */ `enable subgroups;
${stage === "attention" ? "enable f16;" : ""}
struct Params { amount: f32, inScale: f32, replace: f32, unused: f32 }
@group(0) @binding(0) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> norms: array<f32>;
@group(0) @binding(2) var<storage, read_write> nextInput: array<${stage === "attention" ? "f16" : "f32"}>;
@group(0) @binding(3) var<storage, read_write> nextSum: array<f32>;
@group(0) @binding(4) var<storage, read> donor: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;
var<workgroup> partial: array<f32, 8>;
fn reduce(value: f32, tid: u32) -> f32 {
  let sub = subgroupAdd(value);
  if ((tid & 31u) == 0u) { partial[tid >> 5u] = sub; }
  workgroupBarrier();
  var total = 0.0;
  for (var i = 0u; i < 8u; i++) { total += partial[i]; }
  workgroupBarrier();
  return total;
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) id: vec3<u32>) {
  if (params.amount == 0.0) { return; }
  let tid = id.x;
  var local: array<f32, 6>;
  var squares = 0.0;
  for (var k = 0u; k < 6u; k++) {
    let i = tid + k * 256u;
    var value = hidden[i] + params.amount * donor[i];
    if (params.replace == 1.0) {
      value = hidden[i] + params.amount * (donor[i] - hidden[i]);
      if (params.amount == 1.0) { value = donor[i]; }
    }
    hidden[i] = value;
    local[k] = value;
    squares += value * value;
  }
  ${stage === "mlp" ? "" : `
  let rms = inverseSqrt(reduce(squares, tid) / 1536.0 + 0.000001);
  var sum = 0.0;
  for (var k = 0u; k < 6u; k++) {
    let i = tid + k * 256u;
    var value = local[k] * rms * norms[1536u + i];
    ${stage === "attention" ? "value = f32(f16(value));" : ""}
    if (params.inScale != 0.0) {
      value = clamp(round(value / params.inScale), -128.0, 127.0) * params.inScale;
    }
    ${stage === "attention" ? "value = f32(f16(value)); nextInput[i] = f16(value);" : "nextInput[i] = value;"}
    sum += value;
  }
  let total = reduce(sum, tid);
  if (tid == 0u) { nextSum[0] = total; }`}
}`;
}
