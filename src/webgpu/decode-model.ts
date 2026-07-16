import type { DecodeMlpPleFixture } from "../model/decode-mlp-ple-fixture";
import {
  loadGemmaInputWeights,
  type GemmaInputTensorSource,
} from "../model/gemma-input-weights";
import { loadGemmaOutputWeights } from "../model/gemma-output-weights";
import {
  createGemmaDecodeInputResources,
  destroyGemmaDecodeInputResources,
  encodeGemmaDecodeInputPass,
  getGemmaDecodeInputPipeline,
  type GemmaDecodeInputPipeline,
  type GemmaDecodeInputResources,
} from "./decode-input";
import {
  createGemmaGreedyResources,
  destroyGemmaGreedyResources,
  encodeGemmaGreedyPass,
  getGemmaGreedyPipelines,
  readGemmaGreedyResult,
  type GemmaGreedyPipelines,
  type GemmaGreedyResources,
  type GemmaGreedyResult,
} from "./decode-greedy";
import {
  createGemmaLmHeadResources,
  destroyGemmaLmHeadResources,
  encodeGemmaLmHeadPass,
  getGemmaLmHeadPipeline,
  type GemmaLmHeadPipeline,
  type GemmaLmHeadMode,
  type GemmaLmHeadResources,
} from "./decode-lm-head";
import {
  commitGemmaDecodeStackCaches,
  destroyGemmaDecodeStackResources,
  encodeGemmaDecodeStackPass,
  loadGemmaDecodeStackResources,
  type GemmaDecodeStackResources,
  type GemmaDecodeStackRuntime,
} from "./decode-stack";
import type { DecodeOprojNormMode } from "./decode-oproj-norm";

const VOCAB_SIZE = 262144;

export type GemmaDecodeModelRuntime = Omit<
  GemmaDecodeStackRuntime,
  "finalInputNorm" | "finalInputScale"
>;

export interface GemmaDecodeModelResources {
  inputPipeline: GemmaDecodeInputPipeline;
  input: GemmaDecodeInputResources;
  stack: GemmaDecodeStackResources;
  lmHeadPipeline: GemmaLmHeadPipeline;
  lmHead: GemmaLmHeadResources;
  greedyPipelines: GemmaGreedyPipelines;
  greedy: GemmaGreedyResources;
  logits: GPUBuffer;
  dispatchesPerToken: number;
}

export type GemmaModelOutputMode = "none" | "greedy" | "logits";

export interface GemmaModelOutput {
  prediction: GemmaGreedyResult | null;
  logits: Float32Array | null;
  logitsReadbackMs: number;
}

export async function loadGemmaDecodeModelResources(
  device: GPUDevice,
  source: GemmaInputTensorSource,
  fixture: DecodeMlpPleFixture,
  runtime: GemmaDecodeModelRuntime,
  lmHeadMode: GemmaLmHeadMode = "block-major-columns",
  oprojMode: DecodeOprojNormMode = "subgroup-rows",
): Promise<GemmaDecodeModelResources> {
  const [inputWeights, outputWeights, inputPipeline, lmHeadPipeline, greedyPipelines] =
    await Promise.all([
    loadGemmaInputWeights(source),
    loadGemmaOutputWeights(source),
    getGemmaDecodeInputPipeline(device),
    getGemmaLmHeadPipeline(device, VOCAB_SIZE, lmHeadMode),
    getGemmaGreedyPipelines(device, VOCAB_SIZE),
  ]);
  const input = createGemmaDecodeInputResources(device, inputPipeline, inputWeights);
  let stack: GemmaDecodeStackResources | null = null;
  let lmHead: GemmaLmHeadResources | null = null;
  try {
    stack = await loadGemmaDecodeStackResources(device, source, fixture, {
      ...runtime,
      hiddenBuffer: input.hidden,
      perLayerInputsBuffer: input.perLayerInputs,
      finalInputNorm: outputWeights.finalNorm,
      finalInputScale: outputWeights.inputScale,
    }, oprojMode);
    lmHead = createGemmaLmHeadResources(
      device,
      lmHeadPipeline,
      {
        activation: stack.finalInput,
        activationSum: stack.finalSum,
      },
      outputWeights,
    );
    const greedy = createGemmaGreedyResources(device, greedyPipelines, lmHead.logits);
    return {
      inputPipeline,
      input,
      stack,
      lmHeadPipeline,
      lmHead,
      greedyPipelines,
      greedy,
      logits: lmHead.logits,
      dispatchesPerToken: stack.dispatchesPerToken + 7,
    };
  } catch (error) {
    if (lmHead) destroyGemmaLmHeadResources(lmHead);
    if (stack) destroyGemmaDecodeStackResources(stack);
    destroyGemmaDecodeInputResources(input);
    throw error;
  }
}

export function encodeGemmaDecodeModel(
  encoder: GPUCommandEncoder,
  resources: GemmaDecodeModelResources,
  outputMode: GemmaModelOutputMode = "greedy",
  logitsReadback?: GPUBuffer,
): void {
  const pass = encoder.beginComputePass({ label: "Gemma complete decode token" });
  encodeGemmaDecodeInputPass(pass, resources.inputPipeline, resources.input);
  encodeGemmaDecodeStackPass(pass, resources.stack);
  if (outputMode !== "none") {
    encodeGemmaLmHeadPass(pass, resources.lmHeadPipeline, resources.lmHead);
    if (outputMode === "greedy") {
      encodeGemmaGreedyPass(pass, resources.greedyPipelines, resources.greedy);
    }
  }
  pass.end();
  if (outputMode === "greedy") {
    encoder.copyBufferToBuffer(resources.greedy.result, 0, resources.greedy.readback, 0, 8);
  } else if (outputMode === "logits") {
    if (!logitsReadback || logitsReadback.size < resources.logits.size) {
      throw new Error("Gemma logits output requires a matching readback buffer");
    }
    encoder.copyBufferToBuffer(resources.logits, 0, logitsReadback, 0, resources.logits.size);
  }
}

export async function submitGemmaDecodeModel(
  device: GPUDevice,
  resources: GemmaDecodeModelResources,
  outputMode: GemmaModelOutputMode = "greedy",
  logitsReadback?: GPUBuffer,
): Promise<GemmaModelOutput> {
  const encoder = device.createCommandEncoder({ label: "Gemma complete decode token" });
  encodeGemmaDecodeModel(encoder, resources, outputMode, logitsReadback);
  device.queue.submit([encoder.finish()]);
  if (outputMode === "none") await device.queue.onSubmittedWorkDone();
  commitGemmaDecodeStackCaches(resources.stack);
  if (outputMode === "greedy") {
    return {
      prediction: await readGemmaGreedyResult(resources.greedy),
      logits: null,
      logitsReadbackMs: 0,
    };
  }
  if (outputMode === "logits") {
    const startedAt = performance.now();
    const logits = await readGemmaLogits(logitsReadback!, resources.logits.size);
    return {
      prediction: null,
      logits,
      logitsReadbackMs: performance.now() - startedAt,
    };
  }
  return { prediction: null, logits: null, logitsReadbackMs: 0 };
}

async function readGemmaLogits(readback: GPUBuffer, byteLength: number): Promise<Float32Array> {
  await readback.mapAsync(GPUMapMode.READ);
  const logits = new Float32Array(readback.getMappedRange(0, byteLength).slice(0));
  readback.unmap();
  return logits;
}

export function destroyGemmaDecodeModelResources(
  resources: GemmaDecodeModelResources,
): void {
  destroyGemmaGreedyResources(resources.greedy);
  destroyGemmaLmHeadResources(resources.lmHead);
  destroyGemmaDecodeStackResources(resources.stack);
  destroyGemmaDecodeInputResources(resources.input);
}