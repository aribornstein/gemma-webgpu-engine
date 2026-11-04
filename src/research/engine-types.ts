/** Research-only rank-1 effective-weight overlay; see GemmaGenerationSession.setResearchOverlay. */
export interface GemmaResearchOverlay {
  site: "final" | { layer: number };
  a: Float32Array;
  b: Float32Array;
  /** gate(metadata) * alpha. Exactly 0 is an identity overlay. */
  coefficient: number;
}

/**
 * A decoder layer index (its next-layer input), `{ residual: L }` (the residual stream after layer
 * L, taken after any research edit at L), the LM head input (`final`), or the final residual.
 */
export type GemmaResearchCaptureSite = number | "final" | "residual" | { residual: number };

export interface GemmaResearchContinuationScore {
  tokenIds: readonly number[];
  tokenLogProbs: readonly number[];
  sumLogProb: number;
  meanLogProb: number;
  tokenCount: number;
}

export interface GemmaResearchActivationCapture {
  site: GemmaResearchCaptureSite;
  tokenPosition: number;
  values: Float32Array;
  stage?: GemmaResearchStage;
}

export const GEMMA_RESEARCH_STAGES = ["layer-input", "attention", "mlp", "layer-output"] as const;
export type GemmaResearchStage = typeof GEMMA_RESEARCH_STAGES[number];

/** Request-local activation diagnostic, not a pretrained weight repair. */
export interface GemmaResearchPatch {
  layer: number;
  tokenPosition: number;
  mode: "replace" | "add";
  amount: number;
  vector: Float32Array;
  stage?: Exclude<GemmaResearchStage, "layer-input">;
}

export interface GemmaResearchEvaluationRequest {
  promptTokenIds: readonly number[];
  continuationTokenIds?: readonly number[];
  generate?: { maxNewTokens: number; temperature?: number; seed?: number };
  capture?: { positions: readonly number[]; layers: readonly number[]; stages?: readonly GemmaResearchStage[] };
  patch?: GemmaResearchPatch;
  /** Historical residual steering, run request-locally and never treated as a native weight repair. */
  overlays?: readonly GemmaResearchOverlay[];
  signal?: AbortSignal;
  onProgress?: (progress: { evaluatedPositions: number; generatedTokens: number }) => void;
}

export interface GemmaResearchEvaluationResult {
  promptTokenIds: number[];
  tokenIds: number[];
  tokenLogProbs: number[];
  text: string;
  stopReason: "forced" | "end-token" | "length";
  captures: GemmaResearchActivationCapture[];
  firstDecision: { tokenId: number; text: string; logProbability: number; rawLogit: number }[];
  evaluatedPositions: number;
  milliseconds: number;
}
