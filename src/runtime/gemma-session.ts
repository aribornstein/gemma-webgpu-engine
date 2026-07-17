import { ReadonlySafetensorsCache } from "../model/cached-safetensors";
import { PinnedSafetensorsSource } from "../model/pinned-safetensors";
import { GemmaVisionWeightCache } from "../model/gemma-vision-weights";
import { loadDecodeMlpPleFixture } from "../model/decode-mlp-ple-fixture";
import {
  loadGemmaTokenInputBatch,
  loadGemmaTokenInputs,
  type GemmaInputTensorSource,
  type GemmaTokenInputs,
} from "../model/gemma-input-weights";
import { createGemmaRotaryBlock, createGemmaRotaryRows } from "../model/gemma-rope";
import { getWebGpuDevice } from "../webgpu/device";
import type { GemmaLmHeadMode } from "../webgpu/decode-lm-head";
import type { DecodeOprojNormMode } from "../webgpu/decode-oproj-norm";
import {
  uploadGemmaTokenInputBatch,
  uploadGemmaTokenInputs,
} from "../webgpu/decode-input";
import {
  destroyGemmaDecodeModelResources,
  loadGemmaDecodeModelResources,
  submitGemmaDecodeModel,
  type GemmaDecodeModelResources,
  type GemmaModelOutput,
  type GemmaModelOutputMode,
} from "../webgpu/decode-model";
import { updateGemmaDecodeStackToken } from "../webgpu/decode-stack";
import {
  destroyGemmaVisionImageResources,
  encodeGemmaVisionImage,
  type GemmaVisionImageResources,
} from "../webgpu/vision-image";
import { GEMMA_VISION_LAYER_COUNT } from "../webgpu/vision-encoder";
import {
  createGemmaFixedPrefillResources,
  destroyGemmaFixedPrefillResources,
  GEMMA_FIXED_PREFILL_ROWS,
  submitGemmaFixedPrefill,
  updateGemmaFixedPrefill,
  type GemmaFixedPrefillGpuProfile,
  type GemmaFixedPrefillResources,
  type GemmaPrefillGateUpMode,
  type GemmaPrefillPleInputMode,
  type GemmaPrefillQkvSrqMode,
  type GemmaPrefillRmsEpilogueMode,
} from "../webgpu/prefill-model";
import {
  resolveGemmaGenerationConfig,
  usesGemmaGpuGreedy,
  type GemmaGenerationOptions,
} from "./generation-config";
import type { DecodingConfig } from "./decoding";
import {
  compileGenerationConstraint,
  maskConstraintLogits,
  type CompiledGenerationConstraint,
} from "./constraints";
import {
  emitGemmaGenerationUpdate,
  throwIfGemmaGenerationAborted,
} from "./generation-control";
import {
  loadGemmaTokenizer,
  type GemmaGenerationInput,
  type GemmaMultimodalGenerationInput,
  type GemmaTokenizer,
} from "./gemma-tokenizer";
import {
  GEMMA_VISION_MAX_SOFT_TOKENS,
  prepareGemmaVisionImage,
  validateGemmaVisionTokenBudget,
  type GemmaVisionInput,
} from "./gemma-vision-input";
import { sampleToken, SeededRandom } from "./sampling";
import {
  countGemmaReasoningTokens,
  parseGemmaResponse,
  parseGemmaToolCalls,
  type GemmaParsedToolCall,
} from "./gemma-response";
import { TokenByteTrie } from "./token-byte-trie";
import {
  assertGemmaContextSupported,
  availableGemmaOutputTokens,
  GEMMA_VALIDATED_CONTEXT_CAPACITY,
} from "./gemma-context";
import {
  isFinalGemmaPrefillSegment,
  planGemmaPrefillSegments,
  type GemmaPrefillMode,
  type GemmaPrefillStrategy,
} from "./gemma-prefill-plan";
import {
  reusableGemmaPromptPrefixLength,
  sameGemmaMultimodalIdentity,
} from "./gemma-prompt-cache";

export type { GemmaGenerationOptions } from "./generation-config";
export type { GemmaGenerationUpdate } from "./generation-control";
export type { GemmaPrefillMode, GemmaPrefillStrategy } from "./gemma-prefill-plan";

export type GemmaGenerationStopReason = "end-token" | "stop-token" | "length";

export interface GemmaGenerationResult {
  text: string;
  reasoning: string;
  reasoningTokenCount: number;
  rawText: string;
  toolCalls: readonly GemmaParsedToolCall[];
  promptTokenIds: number[];
  generatedTokenIds: number[];
  decodingConfig: DecodingConfig;
  stoppedOnEndToken: boolean;
  stoppedOnStopToken: boolean;
  stopReason: GemmaGenerationStopReason;
}

export interface GemmaGenerationTiming {
  requestSetupMs: number;
  visionPreprocessMs: number;
  visionEncodeMs: number;
  visionWeightLoadMs: number;
  visionPatchEmbedMs: number;
  visionLayerSetupMs: number;
  visionLayerExecutionMs: number;
  visionPostprocessMs: number;
  cacheResetMs: number;
  promptTokensReused: number;
  prefillMs: number;
  prefillMode: GemmaPrefillMode;
  prefillGpuProfiles: readonly GemmaFixedPrefillGpuProfile[] | null;
  timeToFirstTokenMs: number;
  decodeTokenMs: readonly number[];
  interTokenLatencyMs: readonly number[];
  timePerOutputTokenMs: number | null;
  logitsReadbackMs: number;
  callbackMs: number;
  totalMs: number;
}

export interface MeasuredGemmaGenerationResult {
  result: GemmaGenerationResult;
  timing: GemmaGenerationTiming;
}

export interface GemmaMeasuredGenerationOptions extends GemmaGenerationOptions {
  profilePrefillStages?: boolean;
}

interface MutableGemmaGenerationTiming {
  startedAt: number;
  requestSetupMs: number;
  visionPreprocessMs: number;
  visionEncodeMs: number;
  visionWeightLoadMs: number;
  visionPatchEmbedMs: number;
  visionLayerSetupMs: number;
  visionLayerExecutionMs: number;
  visionPostprocessMs: number;
  cacheResetMs: number;
  promptTokensReused: number;
  prefillMs: number;
  prefillMode: GemmaPrefillMode;
  profilePrefillStages: boolean;
  prefillGpuProfiles: GemmaFixedPrefillGpuProfile[];
  timeToFirstTokenMs: number | null;
  decodeTokenMs: number[];
  interTokenLatencyMs: number[];
  lastTokenEmittedAt: number | null;
  logitsReadbackMs: number;
  callbackMs: number;
}

export interface GemmaSessionLoadOptions {
  cacheCapacity?: number;
  sourceUrl?: string;
  lmHeadMode?: GemmaLmHeadMode;
  oprojMode?: DecodeOprojNormMode;
  prefillStrategy?: GemmaPrefillStrategy;
  prefillGateUpMode?: GemmaPrefillGateUpMode;
  prefillRmsEpilogueMode?: GemmaPrefillRmsEpilogueMode;
  prefillQkvSrqMode?: GemmaPrefillQkvSrqMode;
  prefillPleInputMode?: GemmaPrefillPleInputMode;
  prefillMaxInFlightBlocks?: 1 | 2 | 4 | 8;
}

interface GemmaSessionTensorSource extends GemmaInputTensorSource {
  close(): void;
}

export interface GemmaSessionMemoryEstimate {
  gpuBufferCount: number;
  gpuBufferBytes: number;
  visionWeightEntryCount: number;
  visionWeightSourceBytes: number;
  visionWeightMaterializedBytes: number;
  scope: "retained-resource-graph";
}

interface PreparedGemmaPrompt {
  tokenIds: number[];
  visionInputs: GemmaVisionInput[];
  visionStarts: number[];
  visionIdentities: string[];
  visionPreprocessMs: number;
}

interface GemmaSoftTokenSource {
  buffer: GPUBuffer;
  byteOffset: number;
}

const GEMMA_PAD_TOKEN_ID = 0;
const GEMMA_TEXT_HIDDEN_SIZE = 1536;
const GEMMA_TOKEN_INPUT_CACHE_CAPACITY = 256;
const GEMMA_DEFAULT_MAX_IN_FLIGHT_PREFILL_BLOCKS = 4;

export class GemmaGenerationSession {
  private position = 0;
  private readonly evaluatedTokenIds: number[] = [];
  private destroyed = false;
  private generating = false;
  private readonly device: GPUDevice;
  private readonly cache: GemmaSessionTensorSource;
  private readonly tokenizer: GemmaTokenizer;
  private readonly resources: GemmaDecodeModelResources;
  private readonly prefill: GemmaFixedPrefillResources | null;
  private readonly prefillStrategy: GemmaPrefillStrategy;
  private readonly prefillMaxInFlightBlocks: 1 | 2 | 4 | 8;
  private readonly logitsReadback: GPUBuffer;
  private readonly tokenInputCache = new Map<number, GemmaTokenInputs>();
  private readonly visionWeightCache = new GemmaVisionWeightCache();
  private readonly evaluatedVisionIdentities: string[] = [];
  private tokenByteTrie: TokenByteTrie | null = null;
  private activeTiming: MutableGemmaGenerationTiming | null = null;
  readonly cacheCapacity: number;

  private constructor(
    device: GPUDevice,
    cache: GemmaSessionTensorSource,
    tokenizer: GemmaTokenizer,
    resources: GemmaDecodeModelResources,
    prefill: GemmaFixedPrefillResources | null,
    prefillStrategy: GemmaPrefillStrategy,
    prefillMaxInFlightBlocks: 1 | 2 | 4 | 8,
    logitsReadback: GPUBuffer,
    cacheCapacity: number,
  ) {
    this.device = device;
    this.cache = cache;
    this.tokenizer = tokenizer;
    this.resources = resources;
    this.prefill = prefill;
    this.prefillStrategy = prefillStrategy;
    this.prefillMaxInFlightBlocks = prefillMaxInFlightBlocks;
    this.logitsReadback = logitsReadback;
    this.cacheCapacity = cacheCapacity;
  }

  static async load(options: GemmaSessionLoadOptions = {}): Promise<GemmaGenerationSession> {
    const cacheCapacity = options.cacheCapacity ?? GEMMA_VALIDATED_CONTEXT_CAPACITY;
    const prefillStrategy = options.prefillStrategy ?? "auto";
    const prefillGateUpMode = options.prefillGateUpMode ?? "fused";
    const prefillRmsEpilogueMode = options.prefillRmsEpilogueMode ?? "fused";
    const prefillQkvSrqMode = options.prefillQkvSrqMode ?? "separate";
    const prefillPleInputMode = options.prefillPleInputMode ?? "copied";
    const prefillMaxInFlightBlocks = options.prefillMaxInFlightBlocks ??
      GEMMA_DEFAULT_MAX_IN_FLIGHT_PREFILL_BLOCKS;
    if (!Number.isInteger(cacheCapacity) || cacheCapacity < 1) {
      throw new Error("Gemma cache capacity must be a positive integer");
    }
    if (prefillStrategy !== "auto" && prefillStrategy !== "fixed-32" &&
        prefillStrategy !== "chunked-32" &&
        prefillStrategy !== "sequential") {
      throw new Error(
        "Gemma prefill strategy must be auto, fixed-32, chunked-32, or sequential",
      );
    }
    if (prefillGateUpMode !== "fused-activated" && prefillGateUpMode !== "fused" &&
        prefillGateUpMode !== "separate") {
      throw new Error("Gemma prefill gate/up mode is invalid");
    }
    if (prefillRmsEpilogueMode !== "fused" && prefillRmsEpilogueMode !== "separate") {
      throw new Error("Gemma prefill RMS epilogue mode is invalid");
    }
    if (prefillQkvSrqMode !== "shared" && prefillQkvSrqMode !== "separate") {
      throw new Error("Gemma prefill QKV SRQ mode is invalid");
    }
    if (prefillPleInputMode !== "direct" && prefillPleInputMode !== "copied") {
      throw new Error("Gemma prefill PLE input mode is invalid");
    }
    if (prefillMaxInFlightBlocks !== 1 && prefillMaxInFlightBlocks !== 2 &&
        prefillMaxInFlightBlocks !== 4 && prefillMaxInFlightBlocks !== 8) {
      throw new Error("Gemma prefill in-flight block count must be 1, 2, 4, or 8");
    }
    const [device, cache, tokenizer, fixture] = await Promise.all([
      getWebGpuDevice(),
      options.sourceUrl
        ? PinnedSafetensorsSource.open(options.sourceUrl)
        : ReadonlySafetensorsCache.open(),
      loadGemmaTokenizer(),
      loadDecodeMlpPleFixture(),
    ]);
    if (!device.features.has("subgroups") || !device.features.has("shader-f16")) {
      cache.close();
      throw new Error("Gemma generation requires WebGPU subgroups and shader-f16");
    }
    assertGemmaContextSupported(cacheCapacity, device.limits);
    const rotary = createGemmaRotaryRows(0);
    try {
      const resources = await loadGemmaDecodeModelResources(device, cache, fixture, {
        hidden: new Float32Array(1536),
        keyLength: 1,
        queryOffset: 0,
        cacheCapacity,
        slidingRotary: rotary.sliding,
        fullRotary: rotary.full,
      }, options.lmHeadMode, options.oprojMode);
      let prefill: GemmaFixedPrefillResources | null = null;
      try {
        if (prefillStrategy !== "sequential" &&
          cacheCapacity >= GEMMA_FIXED_PREFILL_ROWS) {
          prefill = await createGemmaFixedPrefillResources(
            device,
            resources,
            prefillGateUpMode,
            prefillRmsEpilogueMode,
            prefillQkvSrqMode,
            prefillPleInputMode,
          );
        }
      } catch (error) {
        destroyGemmaDecodeModelResources(resources);
        throw error;
      }
      let logitsReadback: GPUBuffer;
      try {
        logitsReadback = device.createBuffer({
          label: "Gemma sampled logits readback",
          size: resources.logits.size,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
      } catch (error) {
        if (prefill) destroyGemmaFixedPrefillResources(prefill);
        destroyGemmaDecodeModelResources(resources);
        throw error;
      }
      return new GemmaGenerationSession(
        device,
        cache,
        tokenizer,
        resources,
        prefill,
        prefillStrategy,
        prefillMaxInFlightBlocks,
        logitsReadback,
        cacheCapacity,
      );
    } catch (error) {
      cache.close();
      throw error;
    }
  }

  async generate(
    input: GemmaGenerationInput,
    options: GemmaGenerationOptions = {},
  ): Promise<GemmaGenerationResult> {
    this.assertAlive();
    if (this.generating) throw new Error("Gemma generation is already in progress");
    const setupStartedAt = performance.now();
    const {
      signal,
      onToken,
      onVisionProgress,
      onPrefillProgress,
      constraint,
      reusePromptCache = true,
      ...decodingOptions
    } = options;
    throwIfGemmaGenerationAborted(signal);
    const config = resolveGemmaGenerationConfig(decodingOptions);
    const compiledConstraint = constraint
      ? compileGenerationConstraint(constraint)
      : null;
    const tokenByteTrie = compiledConstraint ? this.getTokenByteTrie() : null;
    const outputMode: GemmaModelOutputMode = compiledConstraint || !usesGemmaGpuGreedy(config)
      ? "logits"
      : "greedy";
    const maxNewTokens = config.maxNewTokens;
    this.generating = true;
    const visionResources: GemmaVisionImageResources[] = [];
    try {
      const preparedPrompt = await this.prepareGenerationInput(
        input,
        signal,
        onVisionProgress,
      );
      const promptTokenIds = preparedPrompt.tokenIds;
      if (this.activeTiming) {
        this.activeTiming.requestSetupMs = performance.now() - setupStartedAt;
        this.activeTiming.visionPreprocessMs = preparedPrompt.visionPreprocessMs;
      }
      if (maxNewTokens > availableGemmaOutputTokens(promptTokenIds.length, this.cacheCapacity)) {
        throw new Error(
          `Gemma prompt and output require ${promptTokenIds.length + maxNewTokens - 1} ` +
          `cache positions, exceeding capacity ${this.cacheCapacity}`,
        );
      }

      const resetStartedAt = performance.now();
      const promptTokensReused = this.preparePromptCache(
        promptTokenIds,
        reusePromptCache,
        preparedPrompt.visionIdentities,
      );
      if (this.activeTiming) {
        this.activeTiming.cacheResetMs = performance.now() - resetStartedAt;
        this.activeTiming.promptTokensReused = promptTokensReused;
      }
      throwIfGemmaGenerationAborted(signal);

      const softTokens = new Map<number, GemmaSoftTokenSource>();
      for (let imageIndex = 0; imageIndex < preparedPrompt.visionInputs.length; imageIndex += 1) {
        const promptStart = preparedPrompt.visionStarts[imageIndex];
        const visionInput = preparedPrompt.visionInputs[imageIndex];
        if (promptStart + visionInput.softTokenCount <= promptTokensReused) continue;
        throwIfGemmaGenerationAborted(signal);
        onVisionProgress?.({
          imageIndex,
          imageCount: preparedPrompt.visionInputs.length,
          phase: "encoding",
          completedLayers: 0,
          totalLayers: GEMMA_VISION_LAYER_COUNT,
        });
        const visionEncodeStartedAt = performance.now();
        const encoded = await encodeGemmaVisionImage(
          this.device,
          this.cache,
          visionInput,
          (progress) => {
            throwIfGemmaGenerationAborted(signal);
            onVisionProgress?.({
              imageIndex,
              imageCount: preparedPrompt.visionInputs.length,
              phase: "encoding",
              completedLayers: progress.completedLayers,
              totalLayers: GEMMA_VISION_LAYER_COUNT,
            });
          },
          signal,
          this.visionWeightCache,
        );
        if (this.activeTiming) {
          this.activeTiming.visionEncodeMs += performance.now() - visionEncodeStartedAt;
          this.activeTiming.visionWeightLoadMs += encoded.timing.weightLoadMilliseconds;
          this.activeTiming.visionPatchEmbedMs += encoded.timing.patchEmbedMilliseconds;
          this.activeTiming.visionLayerSetupMs += encoded.timing.layerSetupMilliseconds;
          this.activeTiming.visionLayerExecutionMs += encoded.timing.layerExecutionMilliseconds;
          this.activeTiming.visionPostprocessMs += encoded.timing.postprocessMilliseconds;
        }
        throwIfGemmaGenerationAborted(signal);
        visionResources.push(encoded);
        for (let tokenIndex = 0; tokenIndex < encoded.softTokenCount; tokenIndex += 1) {
          softTokens.set(promptStart + tokenIndex, {
            buffer: encoded.output,
            byteOffset: tokenIndex * GEMMA_TEXT_HIDDEN_SIZE * 4,
          });
        }
      }
      let modelOutput: GemmaModelOutput | null = null;
      const prefillStartedAt = performance.now();
      const pendingPromptTokenIds = promptTokenIds.slice(promptTokensReused);
      const prefillSegments = planGemmaPrefillSegments(
        this.position,
        pendingPromptTokenIds.length,
        this.cacheCapacity,
        this.prefillStrategy,
        this.prefill !== null,
        GEMMA_FIXED_PREFILL_ROWS,
      );
      const usesFixedPrefill = prefillSegments.some(({ mode }) => mode === "fixed-32");
      if (this.activeTiming) {
        this.activeTiming.prefillMode = usesFixedPrefill
          ? pendingPromptTokenIds.length <= GEMMA_FIXED_PREFILL_ROWS
            ? "fixed-32"
            : "chunked-32"
          : "sequential";
      }
      const prefillMode = this.activeTiming?.prefillMode ?? (
        usesFixedPrefill ? "chunked-32" : "sequential"
      );
      onPrefillProgress?.({
        completedPromptTokens: promptTokensReused,
        totalPromptTokens: promptTokenIds.length,
        reusedPromptTokens: promptTokensReused,
        mode: prefillMode,
      });
      let inFlightPrefillBlocks = 0;
      for (const segment of prefillSegments) {
        if (segment.mode === "fixed-32") {
          throwIfGemmaGenerationAborted(signal);
          const isFinalSegment = isFinalGemmaPrefillSegment(
            segment,
            pendingPromptTokenIds.length,
          );
          const blockOutput = await this.evaluatePromptBlock(
            pendingPromptTokenIds.slice(segment.start, segment.start + segment.rows),
            promptTokensReused + segment.start,
            softTokens,
            isFinalSegment ? outputMode : "none",
          );
          if (blockOutput.prediction || blockOutput.logits) modelOutput = blockOutput;
          if (!isFinalSegment) {
            inFlightPrefillBlocks += 1;
            if (inFlightPrefillBlocks >= this.prefillMaxInFlightBlocks) {
              await this.device.queue.onSubmittedWorkDone();
              inFlightPrefillBlocks = 0;
            }
          } else {
            inFlightPrefillBlocks = 0;
          }
          throwIfGemmaGenerationAborted(signal);
          onPrefillProgress?.({
            completedPromptTokens: promptTokensReused + segment.start + segment.rows,
            totalPromptTokens: promptTokenIds.length,
            reusedPromptTokens: promptTokensReused,
            mode: prefillMode,
          });
          continue;
        }
        if (inFlightPrefillBlocks > 0) {
          await this.device.queue.onSubmittedWorkDone();
          inFlightPrefillBlocks = 0;
        }
        for (let index = segment.start; index < segment.start + segment.rows; index += 1) {
          throwIfGemmaGenerationAborted(signal);
          const tokenOutput = await this.evaluateToken(
            pendingPromptTokenIds[index],
            softTokens.get(promptTokensReused + index),
            index === pendingPromptTokenIds.length - 1 ? outputMode : "none",
          );
          if (tokenOutput.prediction || tokenOutput.logits) modelOutput = tokenOutput;
          throwIfGemmaGenerationAborted(signal);
        }
        onPrefillProgress?.({
          completedPromptTokens: promptTokensReused + segment.start + segment.rows,
          totalPromptTokens: promptTokenIds.length,
          reusedPromptTokens: promptTokensReused,
          mode: prefillMode,
        });
      }
      if (this.activeTiming) {
        this.activeTiming.prefillMs = performance.now() - prefillStartedAt;
      }
      if (!modelOutput) throw new Error("Gemma prompt produced no model output");

      const generatedTokenIds: number[] = [];
      const history = [...promptTokenIds];
      const random = new SeededRandom(config.seed);
      const customStopTokens = new Set(config.stopTokenIds);
      let stoppedOnEndToken = false;
      let stoppedOnStopToken = false;
      for (let index = 0; index < maxNewTokens; index += 1) {
        throwIfGemmaGenerationAborted(signal);
        let token = modelOutput.prediction?.token ?? 0;
        if (outputMode === "greedy" && !modelOutput.prediction) {
          throw new Error("Gemma greedy mode produced no prediction");
        }
        if (compiledConstraint && tokenByteTrie) {
          const logits = requiredGemmaLogits(modelOutput);
          throwIfGemmaGenerationAborted(signal);
          token = this.selectConstrainedToken(
            logits,
            history,
            config,
            random,
            compiledConstraint,
            tokenByteTrie,
            customStopTokens,
          );
        } else if (!usesGemmaGpuGreedy(config)) {
          const logits = requiredGemmaLogits(modelOutput);
          throwIfGemmaGenerationAborted(signal);
          token = sampleToken(logits, history, config, () => random.next());
        }
        if (this.activeTiming && this.activeTiming.timeToFirstTokenMs === null) {
          this.activeTiming.timeToFirstTokenMs = performance.now() - this.activeTiming.startedAt;
        }
        if (this.tokenizer.isEndToken(token)) {
          stoppedOnEndToken = true;
          break;
        }
        if (customStopTokens.has(token)) {
          stoppedOnStopToken = true;
          break;
        }
        if (compiledConstraint) {
          const bytes = this.tokenizer.tokenBytes(token);
          if (!bytes) throw new Error(`Constraint selected non-text token ${token}`);
          compiledConstraint.acceptToken(bytes);
        }
        generatedTokenIds.push(token);
        history.push(token);
        const tokenEmittedAt = performance.now();
        if (this.activeTiming?.lastTokenEmittedAt !== null && this.activeTiming) {
          this.activeTiming.interTokenLatencyMs.push(
            tokenEmittedAt - this.activeTiming.lastTokenEmittedAt,
          );
        }
        if (this.activeTiming) this.activeTiming.lastTokenEmittedAt = tokenEmittedAt;
        const callbackStartedAt = performance.now();
        await emitGemmaGenerationUpdate(
          token,
          generatedTokenIds,
          (tokenIds) => this.tokenizer.decodeTokens(tokenIds),
          onToken,
          (tokenIds) => this.tokenizer.decodeRawTokens(tokenIds),
        );
        if (this.activeTiming) {
          this.activeTiming.callbackMs += performance.now() - callbackStartedAt;
        }
        throwIfGemmaGenerationAborted(signal);
        if (index + 1 < maxNewTokens) {
          const decodeStartedAt = performance.now();
          modelOutput = await this.evaluateToken(token, undefined, outputMode);
          if (this.activeTiming) {
            this.activeTiming.decodeTokenMs.push(performance.now() - decodeStartedAt);
          }
          throwIfGemmaGenerationAborted(signal);
        }
      }
      const decodedText = this.tokenizer.decodeTokens(generatedTokenIds);
      const rawText = this.tokenizer.decodeRawTokens(generatedTokenIds);
      const parsedResponse = parseGemmaResponse(rawText, decodedText);
      const reasoningTokenCount = countGemmaReasoningTokens(
        generatedTokenIds.map((tokenId) => this.tokenizer.decodeRawTokens([tokenId])),
      );
      const toolCalls = parseGemmaToolCalls(rawText);
      compiledConstraint?.validateFinal(parsedResponse.text);
      return {
        text: parsedResponse.text,
        reasoning: parsedResponse.reasoning,
        reasoningTokenCount,
        rawText,
        toolCalls,
        promptTokenIds,
        generatedTokenIds,
        decodingConfig: config,
        stoppedOnEndToken,
        stoppedOnStopToken,
        stopReason: stoppedOnEndToken
          ? "end-token"
          : stoppedOnStopToken ? "stop-token" : "length",
      };
    } finally {
      for (const resources of visionResources.toReversed()) {
        destroyGemmaVisionImageResources(resources);
      }
      this.generating = false;
    }
  }

  promptTokenCount(input: GemmaGenerationInput): number {
    this.assertAlive();
    const tokenIds = this.tokenizer.encodeInput(input);
    if (!isMultimodalGenerationInput(input)) return tokenIds.length;
    const markerCount = tokenIds.filter((tokenId) => tokenId === this.tokenizer.imageTokenId).length;
    if (markerCount !== input.images.length) {
      throw new Error("Gemma image parts and image sources must have equal counts");
    }
    const visionTokenBudget = input.visionTokenBudget ?? GEMMA_VISION_MAX_SOFT_TOKENS;
    validateGemmaVisionTokenBudget(visionTokenBudget);
    return tokenIds.length - markerCount + markerCount * visionTokenBudget;
  }

  async generateMeasured(
    input: GemmaGenerationInput,
    options: GemmaMeasuredGenerationOptions = {},
  ): Promise<MeasuredGemmaGenerationResult> {
    if (this.activeTiming) throw new Error("Gemma generation timing is already active");
    const timing: MutableGemmaGenerationTiming = {
      startedAt: performance.now(),
      requestSetupMs: 0,
      visionPreprocessMs: 0,
      visionEncodeMs: 0,
      visionWeightLoadMs: 0,
      visionPatchEmbedMs: 0,
      visionLayerSetupMs: 0,
      visionLayerExecutionMs: 0,
      visionPostprocessMs: 0,
      cacheResetMs: 0,
      promptTokensReused: 0,
      prefillMs: 0,
      prefillMode: "sequential",
      profilePrefillStages: options.profilePrefillStages === true,
      prefillGpuProfiles: [],
      timeToFirstTokenMs: null,
      decodeTokenMs: [],
      interTokenLatencyMs: [],
      lastTokenEmittedAt: null,
      logitsReadbackMs: 0,
      callbackMs: 0,
    };
    this.activeTiming = timing;
    try {
      const result = await this.generate(input, options);
      const totalMs = performance.now() - timing.startedAt;
      const timePerOutputTokenMs = averageGemmaLatency(timing.interTokenLatencyMs);
      return {
        result,
        timing: {
          requestSetupMs: timing.requestSetupMs,
          visionPreprocessMs: timing.visionPreprocessMs,
          visionEncodeMs: timing.visionEncodeMs,
          visionWeightLoadMs: timing.visionWeightLoadMs,
          visionPatchEmbedMs: timing.visionPatchEmbedMs,
          visionLayerSetupMs: timing.visionLayerSetupMs,
          visionLayerExecutionMs: timing.visionLayerExecutionMs,
          visionPostprocessMs: timing.visionPostprocessMs,
          cacheResetMs: timing.cacheResetMs,
          promptTokensReused: timing.promptTokensReused,
          prefillMs: timing.prefillMs,
          prefillMode: timing.prefillMode,
          prefillGpuProfiles: timing.profilePrefillStages
            ? Object.freeze([...timing.prefillGpuProfiles])
            : null,
          timeToFirstTokenMs: timing.timeToFirstTokenMs ?? totalMs,
          decodeTokenMs: Object.freeze([...timing.decodeTokenMs]),
          interTokenLatencyMs: Object.freeze([...timing.interTokenLatencyMs]),
          timePerOutputTokenMs,
          logitsReadbackMs: timing.logitsReadbackMs,
          callbackMs: timing.callbackMs,
          totalMs,
        },
      };
    } finally {
      this.activeTiming = null;
    }
  }

  estimateRetainedGpuMemory(): GemmaSessionMemoryEstimate {
    const buffers = new Set<GPUBuffer>();
    const seen = new Set<object>();
    const visit = (value: unknown): void => {
      if ((typeof value !== "object" && typeof value !== "function") || value === null ||
          seen.has(value)) return;
      seen.add(value);
      if (isGpuBuffer(value)) {
        buffers.add(value);
        return;
      }
      if (value instanceof Map || value instanceof Set) {
        for (const item of value.values()) visit(item);
        return;
      }
      for (const item of Object.values(value)) visit(item);
    };
    visit(this.resources);
    visit(this.prefill);
    visit(this.logitsReadback);
    const visionWeights = this.visionWeightCache.estimateRetainedMemory();
    return {
      gpuBufferCount: buffers.size,
      gpuBufferBytes: Array.from(buffers, (buffer) => buffer.size)
        .reduce((sum, size) => sum + size, 0),
      visionWeightEntryCount: visionWeights.loadedEntryCount,
      visionWeightSourceBytes: visionWeights.sourceBytes,
      visionWeightMaterializedBytes: visionWeights.materializedBytes,
      scope: "retained-resource-graph",
    };
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    if (this.prefill) destroyGemmaFixedPrefillResources(this.prefill);
    this.logitsReadback.destroy();
    destroyGemmaDecodeModelResources(this.resources);
    this.tokenInputCache.clear();
    this.visionWeightCache.clear();
    this.cache.close();
  }

  private reset(): void {
    const encoder = this.device.createCommandEncoder({ label: "Reset Gemma K/V caches" });
    for (const cache of this.resources.stack.ownerCaches.values()) cache.encodeClear(encoder);
    this.device.queue.submit([encoder.finish()]);
    this.position = 0;
    this.evaluatedTokenIds.length = 0;
    this.evaluatedVisionIdentities.length = 0;
  }

  private preparePromptCache(
    promptTokenIds: readonly number[],
    allowReuse = true,
    visionIdentities: readonly string[] = [],
  ): number {
    if (!allowReuse || !sameGemmaMultimodalIdentity(
      visionIdentities,
      this.evaluatedVisionIdentities,
    )) {
      this.reset();
      this.evaluatedVisionIdentities.push(...visionIdentities);
      return 0;
    }
    const prefixLength = reusableGemmaPromptPrefixLength(
      promptTokenIds,
      this.evaluatedTokenIds,
    );
    const caches = Array.from(this.resources.stack.ownerCaches.values());
    if (prefixLength > 0 && caches.every((cache) => cache.canRetainPrefix(prefixLength))) {
      for (const cache of caches) cache.truncate(prefixLength);
      this.position = prefixLength;
      this.evaluatedTokenIds.length = prefixLength;
      return prefixLength;
    }
    this.reset();
    this.evaluatedVisionIdentities.push(...visionIdentities);
    return 0;
  }

  private async evaluateToken(
    tokenId: number,
    softToken?: GemmaSoftTokenSource,
    outputMode: GemmaModelOutputMode = "greedy",
  ) {
    const inputs = await this.loadTokenInputs(softToken ? GEMMA_PAD_TOKEN_ID : tokenId);
    uploadGemmaTokenInputs(this.device, this.resources.input, inputs);
    if (softToken) this.copySoftTokens(this.resources.input.hiddenUpload, [[0, softToken]]);
    const rotary = createGemmaRotaryRows(this.position);
    updateGemmaDecodeStackToken(
      this.device,
      this.resources.stack,
      this.position,
      rotary.sliding,
      rotary.full,
    );
    const output = await submitGemmaDecodeModel(
      this.device,
      this.resources,
      outputMode,
      outputMode === "logits" ? this.logitsReadback : undefined,
    );
    if (this.activeTiming) this.activeTiming.logitsReadbackMs += output.logitsReadbackMs;
    this.position += 1;
    this.evaluatedTokenIds.push(tokenId);
    return output;
  }

  private async loadTokenInputs(tokenId: number): Promise<GemmaTokenInputs> {
    const cached = this.tokenInputCache.get(tokenId);
    if (cached) {
      this.tokenInputCache.delete(tokenId);
      this.tokenInputCache.set(tokenId, cached);
      return cached;
    }
    const inputs = await loadGemmaTokenInputs(this.cache, tokenId);
    this.tokenInputCache.set(tokenId, inputs);
    if (this.tokenInputCache.size > GEMMA_TOKEN_INPUT_CACHE_CAPACITY) {
      const oldest = this.tokenInputCache.keys().next().value;
      if (oldest !== undefined) this.tokenInputCache.delete(oldest);
    }
    return inputs;
  }

  private async evaluatePromptBlock(
    tokenIds: readonly number[],
    promptStart = 0,
    softTokens: ReadonlyMap<number, GemmaSoftTokenSource> = new Map(),
    outputMode: GemmaModelOutputMode = "greedy",
  ) {
    if (!this.prefill || tokenIds.length < 1 || tokenIds.length > GEMMA_FIXED_PREFILL_ROWS) {
      throw new Error("Gemma fixed prefill token block is invalid");
    }
    const modelTokenIds = tokenIds.map((tokenId, row) =>
      softTokens.has(promptStart + row) ? GEMMA_PAD_TOKEN_ID : tokenId);
    const inputs = await loadGemmaTokenInputBatch(
      this.cache,
      modelTokenIds,
      GEMMA_FIXED_PREFILL_ROWS,
    );
    uploadGemmaTokenInputBatch(this.device, this.prefill.input, inputs);
    const blockSoftTokens: [number, GemmaSoftTokenSource][] = [];
    for (let row = 0; row < tokenIds.length; row += 1) {
      const source = softTokens.get(promptStart + row);
      if (source) blockSoftTokens.push([row, source]);
    }
    if (blockSoftTokens.length > 0) {
      this.copySoftTokens(this.prefill.input.hiddenUpload, blockSoftTokens);
    }
    const rotary = createGemmaRotaryBlock(this.position, GEMMA_FIXED_PREFILL_ROWS);
    updateGemmaFixedPrefill(
      this.device,
      this.prefill,
      this.position,
      tokenIds.length,
      rotary,
    );
    const output = await submitGemmaFixedPrefill(
      this.device,
      this.prefill,
      this.position,
      tokenIds.length,
      outputMode,
      outputMode === "logits" ? this.logitsReadback : undefined,
      this.activeTiming?.profilePrefillStages === true,
    );
    if (output.gpuProfile) this.activeTiming?.prefillGpuProfiles.push(output.gpuProfile);
    if (this.activeTiming) this.activeTiming.logitsReadbackMs += output.logitsReadbackMs;
    this.position += tokenIds.length;
    this.evaluatedTokenIds.push(...tokenIds);
    return output;
  }

  private copySoftTokens(
    destination: GPUBuffer,
    tokens: readonly (readonly [number, GemmaSoftTokenSource])[],
  ): void {
    const encoder = this.device.createCommandEncoder({
      label: "Copy Gemma vision soft tokens into language input",
    });
    for (const [row, source] of tokens) {
      encoder.copyBufferToBuffer(
        source.buffer,
        source.byteOffset,
        destination,
        row * GEMMA_TEXT_HIDDEN_SIZE * 4,
        GEMMA_TEXT_HIDDEN_SIZE * 4,
      );
    }
    this.device.queue.submit([encoder.finish()]);
  }

  private async prepareGenerationInput(
    input: GemmaGenerationInput,
    signal?: AbortSignal,
    onVisionProgress?: GemmaGenerationOptions["onVisionProgress"],
  ): Promise<PreparedGemmaPrompt> {
    const rawTokenIds = this.tokenizer.encodeInput(input);
    if (!isMultimodalGenerationInput(input)) {
      if (rawTokenIds.includes(this.tokenizer.imageTokenId)) {
        throw new Error("Gemma image markers require structured image sources");
      }
      return {
        tokenIds: rawTokenIds,
        visionInputs: [],
        visionStarts: [],
        visionIdentities: [],
        visionPreprocessMs: 0,
      };
    }
    const visionInputs: GemmaVisionInput[] = [];
    const visionTokenBudget = input.visionTokenBudget ?? GEMMA_VISION_MAX_SOFT_TOKENS;
    validateGemmaVisionTokenBudget(visionTokenBudget);
    let visionPreprocessMs = 0;
    for (let imageIndex = 0; imageIndex < input.images.length; imageIndex += 1) {
      throwIfGemmaGenerationAborted(signal);
      onVisionProgress?.({
        imageIndex,
        imageCount: input.images.length,
        phase: "preprocessing",
        completedLayers: 0,
        totalLayers: GEMMA_VISION_LAYER_COUNT,
      });
      const visionPreprocessStartedAt = performance.now();
      visionInputs.push(await prepareGemmaVisionImage(
        input.images[imageIndex],
        signal,
        visionTokenBudget,
      ));
      visionPreprocessMs += performance.now() - visionPreprocessStartedAt;
      throwIfGemmaGenerationAborted(signal);
    }
    const markerCount = rawTokenIds.filter(
      (tokenId) => tokenId === this.tokenizer.imageTokenId,
    ).length;
    if (markerCount !== visionInputs.length) {
      throw new Error("Gemma image parts and image sources must have equal counts");
    }
    const tokenIds: number[] = [];
    const visionStarts: number[] = [];
    let imageIndex = 0;
    for (const tokenId of rawTokenIds) {
      if (tokenId !== this.tokenizer.imageTokenId) {
        tokenIds.push(tokenId);
        continue;
      }
      const visionInput = visionInputs[imageIndex++];
      visionStarts.push(tokenIds.length);
      tokenIds.push(...new Array<number>(visionInput.softTokenCount).fill(tokenId));
    }
    const visionIdentities = visionInputs.map(({ identity }, index) => {
      if (!identity) throw new Error(`Gemma vision image ${index} has no content identity`);
      return identity;
    });
    return { tokenIds, visionInputs, visionStarts, visionIdentities, visionPreprocessMs };
  }

  private getTokenByteTrie(): TokenByteTrie {
    this.tokenByteTrie ??= new TokenByteTrie(this.tokenizer);
    return this.tokenByteTrie;
  }

  private selectConstrainedToken(
    logits: Float32Array,
    history: readonly number[],
    config: DecodingConfig,
    random: SeededRandom,
    constraint: CompiledGenerationConstraint,
    trie: TokenByteTrie,
    customStopTokens: ReadonlySet<number>,
  ): number {
    const terminationTokens = new Set([
      ...this.tokenizer.endTokenIds,
      ...customStopTokens,
    ]);
    const legalTokens = constraint.legalTokenIds(trie).filter(
      (tokenId) => !terminationTokens.has(tokenId),
    );
    const legalTerminationTokens = constraint.accepting ? [...terminationTokens] : [];
    if (legalTokens.length === 0 && legalTerminationTokens.length === 0) {
      throw new Error("Generation constraint reached a tokenization dead end");
    }
    const masked = maskConstraintLogits(logits, legalTokens, legalTerminationTokens);
    return sampleToken(masked, history, config, () => random.next());
  }

  private assertAlive(): void {
    if (this.destroyed) throw new Error("Gemma generation session has been destroyed");
  }
}

export function loadGemmaGenerationSession(
  options?: GemmaSessionLoadOptions,
): Promise<GemmaGenerationSession> {
  return GemmaGenerationSession.load(options);
}

function requiredGemmaLogits(output: GemmaModelOutput): Float32Array {
  if (!output.logits) throw new Error("Gemma model output is missing logits");
  return output.logits;
}

function averageGemmaLatency(samples: readonly number[]): number | null {
  if (samples.length === 0) return null;
  return samples.reduce((sum, sample) => sum + sample, 0) / samples.length;
}

function isGpuBuffer(value: object): value is GPUBuffer {
  const candidate = value as Partial<GPUBuffer>;
  return typeof candidate.size === "number" &&
    typeof candidate.mapAsync === "function" &&
    typeof candidate.getMappedRange === "function" &&
    typeof candidate.destroy === "function";
}

function isMultimodalGenerationInput(
  input: GemmaGenerationInput,
): input is GemmaMultimodalGenerationInput {
  return typeof input === "object" && !Array.isArray(input) &&
    "messages" in input && "images" in input && Array.isArray(input.images);
}