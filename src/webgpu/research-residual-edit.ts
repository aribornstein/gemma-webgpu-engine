import type { GemmaResearchStage } from "../research/engine-types";

/**
 * Research-only residual edit applied after one decoder layer:
 *
 *   h' = h + coefficient * b * dot(a, h)
 *
 * followed by recomputing the next layer's normalized SRQ input and its sum exactly as the layer's
 * PLE projection kernel does (RMS with epsilon 1e-6, next-layer norm weights, SRQ to `inScale`).
 * This is the effective-weight edit W_eff = W + coefficient * b a^T on the residual write of the
 * layer; packed weights are never modified. A coefficient of exactly 0 returns before any write,
 * so an identity overlay is bit-identical to unmodified inference.
 */
const HIDDEN = 1536;
const WORKGROUP = 256;

export interface ResearchResidualEditPipeline {
  pipeline: GPUComputePipeline;
}

export interface ResearchResidualEditResources {
  layerIndex: number;
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  vectors: GPUBuffer;
  params: GPUBuffer;
  coefficient: number;
}

export interface ResearchResidualEditTargets {
  hidden: GPUBuffer;
  normWeightsAndScale: GPUBuffer;
  nextInput: GPUBuffer;
  nextSum: GPUBuffer;
  inScale: number;
}

const pipelineCache = new WeakMap<GPUDevice, Promise<GPUComputePipeline>>();

export function getResearchResidualEditPipeline(device: GPUDevice): Promise<GPUComputePipeline> {
  let cached = pipelineCache.get(device);
  if (!cached) {
    cached = device.createComputePipelineAsync({
      label: "Research residual rank-1 edit",
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: createResearchResidualEditShader() }),
        entryPoint: "main",
      },
    });
    pipelineCache.set(device, cached);
  }
  return cached;
}

export function createResearchResidualEditResources(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  layerIndex: number,
  targets: ResearchResidualEditTargets,
  a: Float32Array,
  b: Float32Array,
  coefficient: number,
): ResearchResidualEditResources {
  if (a.length !== HIDDEN || b.length !== HIDDEN) {
    throw new Error(`Research residual edit vectors must have ${HIDDEN} elements`);
  }
  if (!Number.isFinite(coefficient)) throw new Error("Research residual edit coefficient must be finite");
  for (const value of a) if (!Number.isFinite(value)) throw new Error("Research edit vector a is not finite");
  for (const value of b) if (!Number.isFinite(value)) throw new Error("Research edit vector b is not finite");
  const vectors = device.createBuffer({
    label: "Research residual edit vectors",
    size: 2 * HIDDEN * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const packed = new Float32Array(2 * HIDDEN);
  packed.set(a, 0);
  packed.set(b, HIDDEN);
  device.queue.writeBuffer(vectors, 0, packed);
  const params = device.createBuffer({
    label: "Research residual edit params",
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(params, 0, new Float32Array([coefficient, targets.inScale, 0, 0]));
  const entry = (binding: number, buffer: GPUBuffer): GPUBindGroupEntry => ({ binding, resource: { buffer } });
  const bindGroup = device.createBindGroup({
    label: "Research residual edit bind group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      entry(0, targets.hidden),
      entry(1, targets.normWeightsAndScale),
      entry(2, targets.nextInput),
      entry(3, targets.nextSum),
      entry(4, vectors),
      entry(5, params),
    ],
  });
  return { layerIndex, pipeline, bindGroup, vectors, params, coefficient };
}

export function encodeResearchResidualEditPass(
  pass: GPUComputePassEncoder,
  resources: ResearchResidualEditResources,
): void {
  pass.setPipeline(resources.pipeline);
  pass.setBindGroup(0, resources.bindGroup);
  pass.dispatchWorkgroups(1);
}

export function destroyResearchResidualEditResources(resources: ResearchResidualEditResources): void {
  resources.vectors.destroy();
  resources.params.destroy();
}

export function createResearchResidualEditShader(): string {
  return /* wgsl */ `
struct Params { coefficient: f32, inScale: f32, unused0: f32, unused1: f32 }
@group(0) @binding(0) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> w12s: array<f32>;
@group(0) @binding(2) var<storage, read_write> nextInput: array<f32>;
@group(0) @binding(3) var<storage, read_write> nextSum: array<f32>;
@group(0) @binding(4) var<storage, read> vectors: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

const OUT_F: u32 = ${HIDDEN}u;
const WG: u32 = ${WORKGROUP}u;
const EPS: f32 = 0.000001;
var<workgroup> partial: array<f32, ${WORKGROUP}>;

fn srq(value: f32, scale: f32) -> f32 {
  if (scale == 0.0) { return value; }
  return clamp(round(value / scale), -128.0, 127.0) * scale;
}

fn reduce(value: f32, tid: u32) -> f32 {
  partial[tid] = value;
  workgroupBarrier();
  for (var stride = WG / 2u; stride > 0u; stride = stride >> 1u) {
    if (tid < stride) { partial[tid] = partial[tid] + partial[tid + stride]; }
    workgroupBarrier();
  }
  let total = partial[0];
  workgroupBarrier();
  return total;
}

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(local_invocation_id) local: vec3<u32>) {
  let tid = local.x;
  if (params.coefficient == 0.0) { return; }
  var projection = 0.0;
  for (var index = tid; index < OUT_F; index += WG) {
    projection = projection + vectors[index] * hidden[index];
  }
  let dotAH = reduce(projection, tid);
  let scale = params.coefficient * dotAH;
  var squares = 0.0;
  for (var index = tid; index < OUT_F; index += WG) {
    let value = hidden[index] + scale * vectors[OUT_F + index];
    hidden[index] = value;
    squares = squares + value * value;
  }
  let rms = inverseSqrt(reduce(squares, tid) / f32(OUT_F) + EPS);
  var quantizedSum = 0.0;
  for (var index = tid; index < OUT_F; index += WG) {
    let quantized = srq(hidden[index] * rms * w12s[OUT_F + index], params.inScale);
    nextInput[index] = quantized;
    quantizedSum = quantizedSum + quantized;
  }
  let total = reduce(quantizedSum, tid);
  if (tid == 0u) { nextSum[0] = total; }
}
`;
}

/** Research-only copy of the residual stream after one layer into a dedicated capture buffer. */
export interface ResearchResidualCapture {
  layerIndex: number;
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  buffer: GPUBuffer;
  stage?: GemmaResearchStage;
}

const capturePipelineCache = new WeakMap<GPUDevice, Promise<GPUComputePipeline>>();

export function getResearchResidualCapturePipeline(device: GPUDevice): Promise<GPUComputePipeline> {
  let cached = capturePipelineCache.get(device);
  if (!cached) {
    cached = device.createComputePipelineAsync({
      label: "Research residual capture",
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          code: /* wgsl */ `
@group(0) @binding(0) var<storage, read> hidden: array<f32>;
@group(0) @binding(1) var<storage, read_write> capture: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < ${HIDDEN}u) { capture[id.x] = hidden[id.x]; }
}`,
        }),
        entryPoint: "main",
      },
    });
    capturePipelineCache.set(device, cached);
  }
  return cached;
}

export function createResearchResidualCapture(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  layerIndex: number,
  hidden: GPUBuffer,
  stage?: GemmaResearchStage,
): ResearchResidualCapture {
  const buffer = device.createBuffer({
    label: `Research residual capture after layer ${layerIndex}`,
    size: HIDDEN * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: hidden } },
      { binding: 1, resource: { buffer } },
    ],
  });
  return { layerIndex, pipeline, bindGroup, buffer, stage };
}

export function encodeResearchResidualCapturePass(
  pass: GPUComputePassEncoder,
  capture: ResearchResidualCapture,
): void {
  pass.setPipeline(capture.pipeline);
  pass.setBindGroup(0, capture.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(HIDDEN / WORKGROUP));
}
