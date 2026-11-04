import type { GemmaResearchPatch } from "../research/engine-types";
import type { ResearchResidualEditTargets } from "./research-residual-edit";

export interface ResearchActivationPatch {
  layerIndex: number;
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  buffers: GPUBuffer[];
}

const pipelines = new WeakMap<GPUDevice, Promise<GPUComputePipeline>>();

export function getResearchActivationPatchPipeline(device: GPUDevice): Promise<GPUComputePipeline> {
  const cached = pipelines.get(device);
  if (cached) return cached;
  const compiled = device.createComputePipelineAsync({
    label: "Research token-local activation patch",
    layout: "auto",
    compute: { module: device.createShaderModule({ code: shader }), entryPoint: "main" },
  }).catch((error) => {
    pipelines.delete(device);
    throw error;
  });
  pipelines.set(device, compiled);
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
  try {
    device.queue.writeBuffer(vector, 0, Float32Array.from(patch.vector));
    device.queue.writeBuffer(params, 0, new Float32Array([patch.amount, targets.inScale, patch.mode === "replace" ? 1 : 0, 0]));
    const buffers = [targets.hidden, targets.normWeightsAndScale, targets.nextInput, targets.nextSum, vector, params];
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    return { layerIndex: patch.layer, pipeline, bindGroup, buffers: [vector, params] };
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

const shader = /* wgsl */ `enable subgroups;
struct Params { amount: f32, inScale: f32, replace: f32, unused: f32 }
@group(0) @binding(0) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> norms: array<f32>;
@group(0) @binding(2) var<storage, read_write> nextInput: array<f32>;
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
  let rms = inverseSqrt(reduce(squares, tid) / 1536.0 + 0.000001);
  var sum = 0.0;
  for (var k = 0u; k < 6u; k++) {
    let i = tid + k * 256u;
    var value = local[k] * rms * norms[1536u + i];
    if (params.inScale != 0.0) {
      value = clamp(round(value / params.inScale), -128.0, 127.0) * params.inScale;
    }
    nextInput[i] = value;
    sum += value;
  }
  let total = reduce(sum, tid);
  if (tid == 0u) { nextSum[0] = total; }
}`;
