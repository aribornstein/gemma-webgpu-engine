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
import {
  createGemmaTopCandidateResources,
  destroyGemmaTopCandidateResources,
  encodeGemmaTopCandidatesCopy,
  encodeGemmaTopCandidatesPass,
  getGemmaTopCandidatePipelines,
  readGemmaTopCandidates,
  type GemmaTopCandidateResources,
  type GemmaTopCandidates,
} from "./decode-top-candidates";

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
  /** Created on demand by `prepareGemmaTopCandidates`, keyed by candidate list length. */
  topCandidates: Map<number, GemmaTopCandidateResources>;
}

/**
 * `candidates` selects the top-ranked logits and the softmax tail mass on the GPU and reads back
 * only those (see `decode-top-candidates.ts`); `logits` reads back the full vocabulary.
 */
export type GemmaModelOutputMode = "none" | "greedy" | "logits" | "candidates";

export interface GemmaModelOutput {
  prediction: GemmaGreedyResult | null;
  logits: Float32Array | null;
  candidates?: GemmaTopCandidates | null;
  logitsReadbackMs: number;
}

export async function loadGemmaDecodeModelResources(
  device: GPUDevice,
  source: GemmaInputTensorSource,
  fixture: DecodeMlpPleFixture,
  runtime: GemmaDecodeModelRuntime,
  lmHeadMode: GemmaLmHeadMode = "block-major-columns",
  // subgroup-rows-x4 is bit-identical to subgroup-rows and measured faster (median and p95).
  oprojMode: DecodeOprojNormMode = "subgroup-rows-x4",
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
      topCandidates: new Map(),
    };
  } catch (error) {
    if (lmHead) destroyGemmaLmHeadResources(lmHead);
    if (stack) destroyGemmaDecodeStackResources(stack);
    destroyGemmaDecodeInputResources(input);
    throw error;
  }
}

/** Compiles and allocates the GPU top-candidate stage for one candidate list length. */
export async function prepareGemmaTopCandidates(
  device: GPUDevice,
  resources: GemmaDecodeModelResources,
  listLength: number,
): Promise<GemmaTopCandidateResources> {
  const existing = resources.topCandidates.get(listLength);
  if (existing) return existing;
  const pipelines = await getGemmaTopCandidatePipelines(device, listLength);
  const raced = resources.topCandidates.get(listLength);
  if (raced) return raced;
  const created = createGemmaTopCandidateResources(device, pipelines, resources.logits);
  resources.topCandidates.set(listLength, created);
  return created;
}

export function encodeGemmaDecodeModel(
  encoder: GPUCommandEncoder,
  resources: GemmaDecodeModelResources,
  outputMode: GemmaModelOutputMode = "greedy",
  logitsReadback?: GPUBuffer,
  topCandidates?: GemmaTopCandidateResources,
): void {
  if (outputMode === "candidates" && !topCandidates) {
    throw new Error("Gemma candidates output requires prepared top-candidate resources");
  }
  const pass = encoder.beginComputePass({ label: "Gemma complete decode token" });
  encodeGemmaDecodeInputPass(pass, resources.inputPipeline, resources.input);
  encodeGemmaDecodeStackPass(pass, resources.stack);
  if (outputMode !== "none") {
    encodeGemmaLmHeadPass(pass, resources.lmHeadPipeline, resources.lmHead);
    if (outputMode === "greedy") {
      encodeGemmaGreedyPass(pass, resources.greedyPipelines, resources.greedy);
    } else if (outputMode === "candidates") {
      encodeGemmaTopCandidatesPass(pass, topCandidates!);
    }
  }
  pass.end();
  if (outputMode === "candidates") {
    encodeGemmaTopCandidatesCopy(encoder, topCandidates!);
  } else if (outputMode === "greedy") {
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
  topCandidates?: GemmaTopCandidateResources,
): Promise<GemmaModelOutput> {
  const encoder = device.createCommandEncoder({ label: "Gemma complete decode token" });
  encodeGemmaDecodeModel(encoder, resources, outputMode, logitsReadback, topCandidates);
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
  if (outputMode === "candidates") {
    const startedAt = performance.now();
    const candidates = await readGemmaTopCandidates(topCandidates!);
    return {
      prediction: null,
      logits: null,
      candidates,
      logitsReadbackMs: performance.now() - startedAt,
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

/** Reads the logits of the most recent evaluation, which stay in `resources.logits`. */
export async function readGemmaCurrentLogits(
  device: GPUDevice,
  resources: GemmaDecodeModelResources,
  logitsReadback: GPUBuffer,
): Promise<Float32Array> {
  const encoder = device.createCommandEncoder({ label: "Gemma current logits readback" });
  encoder.copyBufferToBuffer(resources.logits, 0, logitsReadback, 0, resources.logits.size);
  device.queue.submit([encoder.finish()]);
  return readGemmaLogits(logitsReadback, resources.logits.size);
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
  for (const candidates of resources.topCandidates.values()) {
    destroyGemmaTopCandidateResources(candidates);
  }
  resources.topCandidates.clear();
  destroyGemmaGreedyResources(resources.greedy);
  destroyGemmaLmHeadResources(resources.lmHead);
  destroyGemmaDecodeStackResources(resources.stack);
  destroyGemmaDecodeInputResources(resources.input);
}