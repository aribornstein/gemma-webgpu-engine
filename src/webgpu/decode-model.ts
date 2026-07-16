import type { DecodeMlpPleFixture } from "../model/decode-mlp-ple-fixture";
import {
  loadGemmaInputWeights,
  type GemmaInputTensorSource,
} from "../model/gemma-input-weights";
import { loadGemmaOutputWeights } from "../model/gemma-output-weights";
import {
  createGemmaDecodeInputResources,
  destroyGemmaDecodeInputResources,
  encodeGemmaDecodeInput,
  getGemmaDecodeInputPipeline,
  type GemmaDecodeInputPipeline,
  type GemmaDecodeInputResources,
} from "./decode-input";
import {
  createGemmaGreedyResources,
  destroyGemmaGreedyResources,
  encodeGemmaGreedy,
  getGemmaGreedyPipelines,
  readGemmaGreedyResult,
  type GemmaGreedyPipelines,
  type GemmaGreedyResources,
  type GemmaGreedyResult,
} from "./decode-greedy";
import {
  createGemmaLmHeadResources,
  destroyGemmaLmHeadResources,
  encodeGemmaLmHead,
  getGemmaLmHeadPipeline,
  type GemmaLmHeadPipeline,
  type GemmaLmHeadResources,
} from "./decode-lm-head";
import {
  commitGemmaDecodeStackCaches,
  destroyGemmaDecodeStackResources,
  encodeGemmaDecodeStack,
  loadGemmaDecodeStackResources,
  type GemmaDecodeStackResources,
  type GemmaDecodeStackRuntime,
} from "./decode-stack";

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

export async function loadGemmaDecodeModelResources(
  device: GPUDevice,
  source: GemmaInputTensorSource,
  fixture: DecodeMlpPleFixture,
  runtime: GemmaDecodeModelRuntime,
): Promise<GemmaDecodeModelResources> {
  const [inputWeights, outputWeights, inputPipeline, lmHeadPipeline, greedyPipelines] =
    await Promise.all([
    loadGemmaInputWeights(source),
    loadGemmaOutputWeights(source),
    getGemmaDecodeInputPipeline(device),
    getGemmaLmHeadPipeline(device, VOCAB_SIZE),
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
    });
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
): void {
  encodeGemmaDecodeInput(encoder, resources.inputPipeline, resources.input);
  encodeGemmaDecodeStack(encoder, resources.stack);
  encodeGemmaLmHead(encoder, resources.lmHeadPipeline, resources.lmHead);
  encodeGemmaGreedy(encoder, resources.greedyPipelines, resources.greedy);
}

export async function submitGemmaDecodeModel(
  device: GPUDevice,
  resources: GemmaDecodeModelResources,
): Promise<GemmaGreedyResult> {
  const encoder = device.createCommandEncoder({ label: "Gemma complete decode token" });
  encodeGemmaDecodeInput(encoder, resources.inputPipeline, resources.input);
  encodeGemmaDecodeStack(encoder, resources.stack);
  encodeGemmaLmHead(encoder, resources.lmHeadPipeline, resources.lmHead);
  encodeGemmaGreedy(encoder, resources.greedyPipelines, resources.greedy, true);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  commitGemmaDecodeStackCaches(resources.stack);
  return readGemmaGreedyResult(resources.greedy);
}

export function destroyGemmaDecodeModelResources(
  resources: GemmaDecodeModelResources,
): void {
  destroyGemmaGreedyResources(resources.greedy);
  destroyGemmaLmHeadResources(resources.lmHead);
  destroyGemmaDecodeStackResources(resources.stack);
  destroyGemmaDecodeInputResources(resources.input);
}