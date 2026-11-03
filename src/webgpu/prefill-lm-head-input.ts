import {
  createGemmaPrefillParameter,
  gemmaPrefillParameterBinding,
  writeGemmaPrefillParameter,
  type GemmaPrefillParameterArena,
  type GemmaPrefillParameterSlice,
} from "./prefill-parameter-arena";

const HIDDEN_SIZE = 1536;

/**
 * Hands the final prefill row to the decode LM head. The decode LM head consumes the final
 * normalized activation and its sum, and may store its weights in a decode-specific layout
 * (block-major), so prefill must reuse that kernel rather than read its weights directly.
 * The sum uses the same per-thread stride and subgroup reduction as the decode epilogue.
 */
export interface GemmaPrefillLmHeadInputResources {
  bindGroup: GPUBindGroup;
  parameters: GemmaPrefillParameterSlice;
  rowCapacity: number;
  ownedBuffers: GPUBuffer[];
}

const pipelineCache = new WeakMap<GPUDevice, Promise<GPUComputePipeline>>();

export function getGemmaPrefillLmHeadInputPipeline(
  device: GPUDevice,
): Promise<GPUComputePipeline> {
  const cached = pipelineCache.get(device);
  if (cached) return cached;
  const compiled = device.createComputePipelineAsync({
    label: "Gemma prefill LM head input",
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: createGemmaPrefillLmHeadInputShader() }),
      entryPoint: "main",
    },
  }).catch((error) => {
    pipelineCache.delete(device);
    throw error;
  });
  pipelineCache.set(device, compiled);
  return compiled;
}

export function createGemmaPrefillLmHeadInputResources(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  source: GPUBuffer,
  rowCapacity: number,
  activation: GPUBuffer,
  activationSum: GPUBuffer,
  parameterArena?: GemmaPrefillParameterArena,
): GemmaPrefillLmHeadInputResources {
  if (!Number.isInteger(rowCapacity) || rowCapacity < 1 ||
      source.size < rowCapacity * HIDDEN_SIZE * 4 ||
      activation.size < HIDDEN_SIZE * 4 || activationSum.size < 4) {
    throw new Error("Gemma prefill LM head input buffers do not match model geometry");
  }
  const parameters = createGemmaPrefillParameter(
    device,
    16,
    "Gemma prefill LM head input parameters",
    parameterArena,
  );
  const resources = {
    bindGroup: device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: source } },
        { binding: 1, resource: { buffer: activation } },
        { binding: 2, resource: { buffer: activationSum } },
        gemmaPrefillParameterBinding(3, parameters.slice),
      ],
    }),
    parameters: parameters.slice,
    rowCapacity,
    ownedBuffers: [...parameters.ownedBuffers],
  };
  updateGemmaPrefillLmHeadInput(device, resources, rowCapacity - 1);
  return resources;
}

export function updateGemmaPrefillLmHeadInput(
  device: GPUDevice,
  resources: GemmaPrefillLmHeadInputResources,
  row: number,
): void {
  if (!Number.isInteger(row) || row < 0 || row >= resources.rowCapacity) {
    throw new Error("Gemma prefill LM head input row is outside the block");
  }
  writeGemmaPrefillParameter(
    device,
    resources.parameters,
    new Uint32Array([row * HIDDEN_SIZE, 0, 0, 0]),
  );
}

export function encodeGemmaPrefillLmHeadInputPass(
  pass: GPUComputePassEncoder,
  pipeline: GPUComputePipeline,
  resources: GemmaPrefillLmHeadInputResources,
): void {
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, resources.bindGroup);
  pass.dispatchWorkgroups(1);
}

export function createGemmaPrefillLmHeadInputShader(): string {
  return `enable subgroups;

struct Parameters {
  sourceStart: u32,
}

@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> activation: array<f32>;
@group(0) @binding(2) var<storage, read_write> activationSum: array<f32>;
@group(0) @binding(3) var<uniform> parameters: Parameters;

const HIDDEN: u32 = ${HIDDEN_SIZE}u;
const WG: u32 = 256u;

var<workgroup> partials: array<f32, 8>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) localId: vec3<u32>) {
  let tid = localId.x;
  var accumulator = 0.0;
  var index = tid;
  loop {
    if (index >= HIDDEN) { break; }
    let value = source[parameters.sourceStart + index];
    activation[index] = value;
    accumulator = accumulator + value;
    index = index + WG;
  }
  let subgroupTotal = subgroupAdd(accumulator);
  if ((tid & 31u) == 0u) { partials[tid >> 5u] = subgroupTotal; }
  workgroupBarrier();
  if (tid == 0u) {
    var total = 0.0;
    for (var part = 0u; part < 8u; part = part + 1u) { total = total + partials[part]; }
    activationSum[0] = total;
  }
}`;
}
