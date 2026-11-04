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
}
