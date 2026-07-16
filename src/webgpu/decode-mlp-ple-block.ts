import { loadDecodeMlpPleFixture } from "../model/decode-mlp-ple-fixture";
import { createGemmaGeluLut } from "../model/gemma-gelu-lut";
import type { MaterializedGemmaLayer } from "../model/gemma-layer-materializer";
import { getWebGpuDevice } from "./device";
import { createDecodeDownNormAddShader } from "./decode-down-norm-add";
import { createDecodeGateUpPresrqShader } from "./decode-gate-up-presrq";
import { createDecodePleGateCodesShader } from "./decode-ple-gate-codes";
import { createDecodePleProjNormCodesShader } from "./decode-ple-proj-norm-codes";

export interface DecodeMlpPleBlockPipelines {
  bitWidth: 2 | 4;
  intermediateFeatures: 6144 | 12288;
  gateUpWorkgroupCount: 768 | 3072;
  gateUp: GPUComputePipeline;
  down: GPUComputePipeline;
  pleGate: GPUComputePipeline;
  pleProjection: GPUComputePipeline;
}

export interface DecodeMlpPleBlockResult {
  sourceOperators: [
    "com.xenova.gemma4.DecodeGateUpNormPresrq",
    "com.xenova.gemma4.DecodeDownNormAddFused",
    "com.xenova.gemma4.DecodePleGateCodes",
    "com.xenova.gemma4.DecodePleProjNormCodes",
  ];
  implementation: "shared-storage-four-dispatch";
  dispatchesPerToken: 4;
  hiddenMaximumAbsoluteError: number;
  hiddenMaximumRelativeError: number;
  hiddenBitMismatches: number;
  nextInputMaximumAbsoluteError: number;
  nextInputMaximumRelativeError: number;
  nextInputNonzeroBitMismatches: number;
  nextInputSignedZeroBitMismatches: number;
  nextSumMaximumAbsoluteError: number;
  nextSumMaximumRelativeError: number;
  nextSumBitMismatches: number;
  gpuBufferAllocations: number;
  bytesAllocated: number;
  allocationsPerDispatch: 0;
  cpuReadbacksBetweenKernels: 0;
  gpuCopiesBetweenKernels: 0;
}

const pipelineCache = new WeakMap<GPUDevice, Map<2 | 4, Promise<DecodeMlpPleBlockPipelines>>>();

export interface DecodeMlpPleSharedBuffers {
  preMlpInput: GPUBuffer;
  preMlpSum: GPUBuffer;
  hidden: GPUBuffer;
  pleInput?: GPUBuffer;
  pleInputOffset?: number;
}

export interface DecodeMlpPleBlockResources {
  gateUpBindGroup: GPUBindGroup;
  downBindGroup: GPUBindGroup;
  pleGateBindGroup: GPUBindGroup;
  pleProjectionBindGroup: GPUBindGroup;
  gateOutput: GPUBuffer;
  hidden: GPUBuffer;
  nextInput: GPUBuffer;
  nextSum: GPUBuffer;
  readback: GPUBuffer;
  modelWeights: DecodeMlpPleModelWeightBuffers;
  modelScales: DecodeMlpPleModelScales;
  buffers: GPUBuffer[];
}

export interface DecodeMlpPleModelWeightBuffers {
  gatePacked: GPUBuffer;
  gateRowScales: GPUBuffer;
  upPacked: GPUBuffer;
  upRowScales: GPUBuffer;
  gateGeluLut: GPUBuffer;
  downPacked: GPUBuffer;
  downRowScales: GPUBuffer;
  postFeedforwardNorm: GPUBuffer;
  pleGatePacked: GPUBuffer;
  pleGateRowScales: GPUBuffer;
  pleGeluLut: GPUBuffer;
  pleProjectionPacked: GPUBuffer;
  pleProjectionRowScales: GPUBuffer;
  postPleNextInputNormAndLayerScale: GPUBuffer;
}

export interface DecodeMlpPleModelScales {
  gateInput: number;
  gateOutput: number;
  upOutput: number;
  downInput: number;
  downOutput: number;
  pleGateInput: number;
  pleGateOutput: number;
  pleProjectionInput: number;
  pleProjectionOutput: number;
  nextInput: number;
}

export interface DecodeMlpPleMaterializedWeights {
  layer: MaterializedGemmaLayer;
  pleNormWeights: Float32Array;
  nextInputScale: number;
}

export async function runDecodeMlpPleBlock(): Promise<DecodeMlpPleBlockResult> {
  const [fixture, device] = await Promise.all([loadDecodeMlpPleFixture(), getWebGpuDevice()]);
  if (!device.features.has("subgroups") || !device.features.has("shader-f16")) {
    throw new Error("The layer-0 MLP/PLE block requires WebGPU subgroups and shader-f16");
  }
  let devicePipelines = pipelineCache.get(device);
  if (!devicePipelines) {
    devicePipelines = new Map();
    pipelineCache.set(device, devicePipelines);
  }
  const pipelinesPromise = devicePipelines.get(4) ?? compileDecodeMlpPleBlockPipelines(device, 4);
  if (!devicePipelines.has(4)) devicePipelines.set(4, pipelinesPromise);
  const pipelines = await pipelinesPromise;
  const resources = createDecodeMlpPleBlockResources(device, pipelines, fixture);
  try {
    const encoder = device.createCommandEncoder({ label: "Layer-0 MLP/PLE block" });
    encodeDecodeMlpPleBlock(encoder, pipelines, resources);

    const hiddenBytes = fixture.expectedHiddenAfterPle.byteLength;
    const nextInputBytes = fixture.expectedNextLayerInput.byteLength;
    encoder.copyBufferToBuffer(resources.hidden, 0, resources.readback, 0, hiddenBytes);
    encoder.copyBufferToBuffer(resources.nextInput, 0, resources.readback, hiddenBytes, nextInputBytes);
    encoder.copyBufferToBuffer(resources.nextSum, 0, resources.readback, hiddenBytes + nextInputBytes, 4);
    device.queue.submit([encoder.finish()]);
    await resources.readback.mapAsync(GPUMapMode.READ);
    const mapped = resources.readback.getMappedRange();
    const actualHidden = new Float32Array(mapped.slice(0, hiddenBytes));
    const actualNextInput = new Float32Array(mapped.slice(hiddenBytes, hiddenBytes + nextInputBytes));
    const actualNextSum = new Float32Array(mapped.slice(hiddenBytes + nextInputBytes));
    resources.readback.unmap();
    const hiddenErrors = measureErrors(actualHidden, fixture.expectedHiddenAfterPle);
    const nextInputErrors = measureErrors(actualNextInput, fixture.expectedNextLayerInput);
    const nextSumErrors = measureErrors(actualNextSum, fixture.expectedNextLayerSum);
    return {
      sourceOperators: [
        "com.xenova.gemma4.DecodeGateUpNormPresrq",
        "com.xenova.gemma4.DecodeDownNormAddFused",
        "com.xenova.gemma4.DecodePleGateCodes",
        "com.xenova.gemma4.DecodePleProjNormCodes",
      ],
      implementation: "shared-storage-four-dispatch",
      dispatchesPerToken: 4,
      hiddenMaximumAbsoluteError: hiddenErrors.maximumAbsoluteError,
      hiddenMaximumRelativeError: hiddenErrors.maximumRelativeError,
      hiddenBitMismatches: hiddenErrors.bitMismatches,
      nextInputMaximumAbsoluteError: nextInputErrors.maximumAbsoluteError,
      nextInputMaximumRelativeError: nextInputErrors.maximumRelativeError,
      nextInputNonzeroBitMismatches: nextInputErrors.nonzeroBitMismatches,
      nextInputSignedZeroBitMismatches: nextInputErrors.signedZeroBitMismatches,
      nextSumMaximumAbsoluteError: nextSumErrors.maximumAbsoluteError,
      nextSumMaximumRelativeError: nextSumErrors.maximumRelativeError,
      nextSumBitMismatches: nextSumErrors.bitMismatches,
      gpuBufferAllocations: resources.buffers.length,
      bytesAllocated: resources.buffers.reduce((sum, buffer) => sum + buffer.size, 0),
      allocationsPerDispatch: 0,
      cpuReadbacksBetweenKernels: 0,
      gpuCopiesBetweenKernels: 0,
    };
  } finally {
    for (const buffer of resources.buffers) buffer.destroy();
  }
}

export async function compileDecodeMlpPleBlockPipelines(
  device: GPUDevice,
  profile: MaterializedGemmaLayer["profile"] | 2 | 4 = 4,
): Promise<DecodeMlpPleBlockPipelines> {
  const bitWidth = typeof profile === "number"
    ? profile
    : profile.endsWith("int2") ? 2 : 4;
  const compile = (label: string, code: string) => device.createComputePipelineAsync({
    label,
    layout: "auto",
    compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
  });
  const [gateUp, down, pleGate, pleProjection] = await Promise.all([
    compile(`MLP/PLE int${bitWidth} gate/up`, createDecodeGateUpPresrqShader(bitWidth)),
    compile(`MLP/PLE int${bitWidth} down`, createDecodeDownNormAddShader(bitWidth)),
    compile("MLP/PLE input gate", createDecodePleGateCodesShader()),
    compile("MLP/PLE projection", createDecodePleProjNormCodesShader()),
  ]);
  return {
    bitWidth,
    intermediateFeatures: bitWidth === 2 ? 12288 : 6144,
    gateUpWorkgroupCount: bitWidth === 2 ? 3072 : 768,
    gateUp,
    down,
    pleGate,
    pleProjection,
  };
}

export function createDecodeMlpPleBlockResources(
  device: GPUDevice,
  pipelines: DecodeMlpPleBlockPipelines,
  fixture: Awaited<ReturnType<typeof loadDecodeMlpPleFixture>>,
  sharedBuffers?: DecodeMlpPleSharedBuffers,
  materialized?: DecodeMlpPleMaterializedWeights,
): DecodeMlpPleBlockResources {
  const materializedBitWidth = materialized?.layer.profile.endsWith("int2") ? 2 : 4;
  if (materialized && materializedBitWidth !== pipelines.bitWidth) {
    throw new Error(
      `The int${pipelines.bitWidth} MLP/PLE pipelines do not match ${materialized.layer.profile} weights`,
    );
  }
  const layer = materialized?.layer;
  const buffers: GPUBuffer[] = [];
  const make = (label: string, size: number, usage: GPUBufferUsageFlags) => {
    const buffer = device.createBuffer({ label, size, usage });
    buffers.push(buffer);
    return buffer;
  };
  const upload = (label: string, data: ArrayBufferView, usage = GPUBufferUsage.STORAGE) => {
    const buffer = make(label, data.byteLength, usage | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  const uniform = (label: string, values: number[]) => upload(label, new Float32Array(values), GPUBufferUsage.UNIFORM);

  const preMlpInput = sharedBuffers?.preMlpInput ?? upload("Block pre-MLP input", fixture.preMlpInputBits);
  const preMlpSum = sharedBuffers?.preMlpSum ?? upload("Block pre-MLP sum", fixture.preMlpSum);
  const gateBits = upload("Block gate bits", layer?.mlp.gate.packedWeights ?? fixture.gateBits);
  const gateScales = upload("Block gate scales", layer?.mlp.gate.rowScales ?? fixture.gateScales);
  const upBits = upload("Block up bits", layer?.mlp.up.packedWeights ?? fixture.upBits);
  const upScales = upload("Block up scales", layer?.mlp.up.rowScales ?? fixture.upScales);
  const gateLut = upload(
    "Block gate LUT",
    layer ? createGemmaGeluLut(layer.mlp.gate.outputScale) : fixture.gateGeluLut,
  );
  const gateOutput = make(
    "Block gate/up output",
    pipelines.intermediateFeatures * Uint16Array.BYTES_PER_ELEMENT,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const gateParams = uniform("Block gate/up parameters", [
    layer?.mlp.gate.outputScale ?? 0.6181102395057678,
    layer?.mlp.up.outputScale ?? 0.6181102395057678,
    layer?.mlp.down.inputScale ?? 27.842519760131836,
    0,
  ]);

  const downBits = upload("Block down bits", layer?.mlp.down.packedWeights ?? fixture.downBits);
  const downScales = upload("Block down scales", layer?.mlp.down.rowScales ?? fixture.downScales);
  const downPartials = make("Block down partials", 1537 * 4, GPUBufferUsage.STORAGE);
  const hidden = sharedBuffers?.hidden ?? upload("Block hidden", fixture.hiddenBeforeDown, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const postFfNorm = upload("Block post-FFN norm", layer?.norms.postFeedforward ?? fixture.postFfNorm);
  const downParams = uniform("Block down parameters", [
    layer?.mlp.down.inputScale ?? 27.842519760131836,
    layer?.mlp.down.outputScale ?? 16.64207649230957,
    0,
    0,
  ]);

  const pleGateCodes = upload("Block PLE gate codes", layer?.ple.inputGate.packedWeights ?? fixture.pleGateWeights);
  const pleGateScales = upload("Block PLE gate scales", layer?.ple.inputGate.rowScales ?? fixture.pleGateRowScales);
  const pleInput = sharedBuffers?.pleInput ?? upload("Block PLE multiplier", fixture.pleInput);
  const pleLut = upload(
    "Block PLE LUT",
    layer ? createGemmaGeluLut(layer.ple.inputGate.outputScale) : fixture.pleGeluLut,
  );
  const pleOutput = make("Block PLE gate output", fixture.expectedPleGateOutput.byteLength, GPUBufferUsage.STORAGE);
  const pleGateParams = uniform("Block PLE gate parameters", [
    layer?.ple.inputGate.inputScale ?? 3.334678888320923,
    layer?.ple.inputGate.outputScale ?? 0.01857776567339897,
    0,
    0,
  ]);

  const pleProjectionCodes = upload("Block PLE projection codes", layer?.ple.projection.packedWeights ?? fixture.pleProjectionWeights);
  const pleProjectionScales = upload("Block PLE projection scales", layer?.ple.projection.rowScales ?? fixture.pleProjectionRowScales);
  const pleProjectionPartials = make("Block PLE projection partials", 1537 * 4, GPUBufferUsage.STORAGE);
  const pleNormWeights = upload("Block PLE norm weights", materialized?.pleNormWeights ?? fixture.pleNormWeights);
  const nextInput = make("Block next-layer input", fixture.expectedNextLayerInput.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const nextSum = make("Block next-layer sum", 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const pleProjectionParams = uniform("Block PLE projection parameters", [
    materialized?.nextInputScale ?? 0.4842597544193268,
    layer?.ple.projection.inputScale ?? 0.03764764964580536,
    layer?.ple.projection.outputScale ?? 0.03129800781607628,
    0,
  ]);
  const readback = make("Block final readback", fixture.expectedHiddenAfterPle.byteLength + fixture.expectedNextLayerInput.byteLength + 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);

  const bind = (pipeline: GPUComputePipeline, entries: GPUBindGroupEntry[]) => device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
  const resource = (
    binding: number,
    buffer: GPUBuffer,
    offset = 0,
    size?: number,
  ): GPUBindGroupEntry => ({
    binding,
    resource: { buffer, offset, ...(size === undefined ? {} : { size }) },
  });
  return {
    gateUpBindGroup: bind(pipelines.gateUp, [resource(0, preMlpInput), resource(1, gateBits), resource(2, gateScales), resource(3, upBits), resource(4, upScales), resource(5, preMlpSum), resource(6, gateOutput), resource(7, gateLut), resource(8, gateParams)]),
    downBindGroup: bind(pipelines.down, [resource(0, gateOutput), resource(1, downBits), resource(2, downPartials), resource(3, downScales), resource(4, hidden), resource(5, postFfNorm), resource(6, downParams)]),
    pleGateBindGroup: bind(pipelines.pleGate, [resource(0, hidden), resource(1, pleGateCodes), resource(2, pleGateScales), resource(3, pleInput, sharedBuffers?.pleInputOffset ?? 0, fixture.pleInput.byteLength), resource(4, pleOutput), resource(5, pleLut), resource(6, pleGateParams)]),
    pleProjectionBindGroup: bind(pipelines.pleProjection, [resource(0, pleOutput), resource(1, pleProjectionCodes), resource(2, pleProjectionScales), resource(3, pleProjectionPartials), resource(4, hidden), resource(5, pleNormWeights), resource(6, nextInput), resource(7, nextSum), resource(8, pleProjectionParams)]),
    gateOutput,
    hidden,
    nextInput,
    nextSum,
    readback,
    modelWeights: {
      gatePacked: gateBits,
      gateRowScales: gateScales,
      upPacked: upBits,
      upRowScales: upScales,
      gateGeluLut: gateLut,
      downPacked: downBits,
      downRowScales: downScales,
      postFeedforwardNorm: postFfNorm,
      pleGatePacked: pleGateCodes,
      pleGateRowScales: pleGateScales,
      pleGeluLut: pleLut,
      pleProjectionPacked: pleProjectionCodes,
      pleProjectionRowScales: pleProjectionScales,
      postPleNextInputNormAndLayerScale: pleNormWeights,
    },
    modelScales: {
      gateInput: layer?.mlp.gate.inputScale ?? 3.334678888320923,
      gateOutput: layer?.mlp.gate.outputScale ?? 0.6181102395057678,
      upOutput: layer?.mlp.up.outputScale ?? 0.6181102395057678,
      downInput: layer?.mlp.down.inputScale ?? 27.842519760131836,
      downOutput: layer?.mlp.down.outputScale ?? 16.64207649230957,
      pleGateInput: layer?.ple.inputGate.inputScale ?? 3.334678888320923,
      pleGateOutput: layer?.ple.inputGate.outputScale ?? 0.01857776567339897,
      pleProjectionInput: layer?.ple.projection.inputScale ?? 0.03764764964580536,
      pleProjectionOutput: layer?.ple.projection.outputScale ?? 0.03129800781607628,
      nextInput: materialized?.nextInputScale ?? 0.4842597544193268,
    },
    buffers,
  };
}

export function encodeDecodeMlpPleBlock(
  encoder: GPUCommandEncoder,
  pipelines: DecodeMlpPleBlockPipelines,
  resources: DecodeMlpPleBlockResources,
): void {
  const pass = encoder.beginComputePass({ label: "Layer-0 MLP/PLE four-dispatch block" });
  encodeDecodeMlpPleBlockPass(pass, pipelines, resources);
  pass.end();
}

export function encodeDecodeMlpPleBlockPass(
  pass: GPUComputePassEncoder,
  pipelines: DecodeMlpPleBlockPipelines,
  resources: DecodeMlpPleBlockResources,
): void {
  pass.setPipeline(pipelines.gateUp);
  pass.setBindGroup(0, resources.gateUpBindGroup);
  pass.dispatchWorkgroups(pipelines.gateUpWorkgroupCount);
  pass.setPipeline(pipelines.down);
  pass.setBindGroup(0, resources.downBindGroup);
  pass.dispatchWorkgroups(384);
  pass.setPipeline(pipelines.pleGate);
  pass.setBindGroup(0, resources.pleGateBindGroup);
  pass.dispatchWorkgroups(256);
  pass.setPipeline(pipelines.pleProjection);
  pass.setBindGroup(0, resources.pleProjectionBindGroup);
  pass.dispatchWorkgroups(96);
}

export function destroyDecodeMlpPleBlockResources(
  resources: DecodeMlpPleBlockResources,
): void {
  for (const buffer of resources.buffers) buffer.destroy();
}

function measureErrors(actual: Float32Array, expected: Float32Array) {
  let maximumAbsoluteError = 0;
  let maximumRelativeError = 0;
  let bitMismatches = 0;
  let signedZeroBitMismatches = 0;
  let nonzeroBitMismatches = 0;
  const actualBits = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const expectedBits = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    const absolute = Math.abs(actual[index] - expected[index]);
    maximumAbsoluteError = Math.max(maximumAbsoluteError, absolute);
    maximumRelativeError = Math.max(maximumRelativeError, absolute / Math.max(Math.abs(expected[index]), 1e-7));
    if (actualBits[index] !== expectedBits[index]) {
      bitMismatches += 1;
      if ((actualBits[index] & 0x7fffffff) === 0 && (expectedBits[index] & 0x7fffffff) === 0) signedZeroBitMismatches += 1;
      else nonzeroBitMismatches += 1;
    }
  }
  return { maximumAbsoluteError, maximumRelativeError, bitMismatches, signedZeroBitMismatches, nonzeroBitMismatches };
}