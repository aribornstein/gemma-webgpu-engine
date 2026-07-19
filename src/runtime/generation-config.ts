import {
  DEFAULT_DECODING_CONFIG,
  validateDecodingConfig,
  type DecodingConfig,
} from "./decoding";
import type { GemmaGenerationTokenHandler } from "./generation-control";
import type { GenerationConstraint } from "./constraints";

const DEFAULT_MAX_NEW_TOKENS = 256;
const VOCAB_SIZE = 262144;

export interface GemmaVisionGenerationProgress {
  imageIndex: number;
  imageCount: number;
  phase: "preprocessing" | "encoding";
  completedLayers: number;
  totalLayers: number;
}

export interface GemmaAudioGenerationProgress {
  audioIndex: number;
  audioCount: number;
  phase: "preprocessing" | "encoding";
  completedLayers: number;
  totalLayers: number;
}

export interface GemmaPrefillGenerationProgress {
  completedPromptTokens: number;
  totalPromptTokens: number;
  reusedPromptTokens: number;
  mode: "sequential" | "fixed-32" | "chunked-32";
}

export type GemmaGenerationOptions = Partial<DecodingConfig> & {
  signal?: AbortSignal;
  onToken?: GemmaGenerationTokenHandler;
  onVisionProgress?: (progress: GemmaVisionGenerationProgress) => void;
  onAudioProgress?: (progress: GemmaAudioGenerationProgress) => void;
  onPrefillProgress?: (progress: GemmaPrefillGenerationProgress) => void;
  constraint?: GenerationConstraint;
  requireReasoning?: boolean;
  reusePromptCache?: boolean;
  captureTokenLogProbabilities?: boolean;
};

export const DEFAULT_GENERATION_CONFIG: DecodingConfig = {
  ...DEFAULT_DECODING_CONFIG,
  temperature: 0,
  topK: 0,
  topP: 1,
  minP: 0,
  typicalP: 1,
  repetitionPenalty: 1,
  repetitionWindow: 0,
  frequencyPenalty: 0,
  presencePenalty: 0,
  maxNewTokens: DEFAULT_MAX_NEW_TOKENS,
  stopTokenIds: [],
};

export function resolveGemmaGenerationConfig(options: GemmaGenerationOptions): DecodingConfig {
  const config = { ...DEFAULT_GENERATION_CONFIG, ...options };
  const result = validateDecodingConfig(config);
  if (!result.ok) throw new Error(`Invalid Gemma decoding config: ${result.errors.join("; ")}`);
  if (config.stopTokenIds.some((tokenId) => tokenId >= VOCAB_SIZE)) {
    throw new Error(`Gemma stop token IDs must be below ${VOCAB_SIZE}`);
  }
  return { ...config, stopTokenIds: [...config.stopTokenIds] };
}

export function usesGemmaGpuGreedy(config: DecodingConfig): boolean {
  return config.temperature === 0 &&
    config.repetitionPenalty === 1 &&
    config.frequencyPenalty === 0 &&
    config.presencePenalty === 0;
}