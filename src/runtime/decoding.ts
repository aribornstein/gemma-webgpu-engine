export interface DecodingConfig {
  temperature: number;
  topK: number;
  topP: number;
  minP: number;
  typicalP: number;
  repetitionPenalty: number;
  repetitionWindow: number;
  frequencyPenalty: number;
  presencePenalty: number;
  maxNewTokens: number;
  seed: number;
  stopTokenIds: number[];
}

export const DEFAULT_DECODING_CONFIG: DecodingConfig = {
  temperature: 0.8,
  topK: 40,
  topP: 0.95,
  minP: 0,
  typicalP: 1,
  repetitionPenalty: 1.05,
  repetitionWindow: 64,
  frequencyPenalty: 0,
  presencePenalty: 0,
  maxNewTokens: 256,
  seed: 42,
  stopTokenIds: [],
};

export function validateDecodingConfig(input: Partial<DecodingConfig>): { ok: boolean; errors: string[] } {
  const value = { ...DEFAULT_DECODING_CONFIG, ...input };
  const errors: string[] = [];
  if (!Number.isFinite(value.temperature) || value.temperature < 0) errors.push("temperature must be >= 0");
  if (!Number.isInteger(value.topK) || value.topK < 0) errors.push("topK must be a non-negative integer");
  for (const key of ["topP", "minP", "typicalP"] as const) {
    if (!Number.isFinite(value[key]) || value[key] < 0 || value[key] > 1) {
      errors.push(`${key} must be between 0 and 1`);
    }
  }
  if (!Number.isFinite(value.repetitionPenalty) || value.repetitionPenalty <= 0) errors.push("repetitionPenalty must be > 0");
  if (!Number.isInteger(value.repetitionWindow) || value.repetitionWindow < 0) {
    errors.push("repetitionWindow must be a non-negative integer");
  }
  if (!Number.isFinite(value.frequencyPenalty) || value.frequencyPenalty < 0) {
    errors.push("frequencyPenalty must be finite and non-negative");
  }
  if (!Number.isFinite(value.presencePenalty) || value.presencePenalty < 0) {
    errors.push("presencePenalty must be finite and non-negative");
  }
  if (!Number.isInteger(value.maxNewTokens) || value.maxNewTokens < 1) errors.push("maxNewTokens must be a positive integer");
  if (!Number.isInteger(value.seed)) errors.push("seed must be an integer");
  if (!Array.isArray(value.stopTokenIds) || value.stopTokenIds.some(
    (tokenId) => !Number.isInteger(tokenId) || tokenId < 0,
  )) {
    errors.push("stopTokenIds must contain non-negative integers");
  }
  return { ok: errors.length === 0, errors };
}
