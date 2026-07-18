import type {
  BenchmarkCase,
  CorrectnessResult,
  GenerationResult,
} from "./types";

export function validateGeneration(
  testCase: BenchmarkCase,
  result: GenerationResult,
): { correctness: CorrectnessResult; equalWorkEligible: boolean; exclusionReasons: string[] } {
  const outputBytes = new TextEncoder().encode(result.text).byteLength;
  const earlyTerminated = result.outputTokens < testCase.targetOutputTokens;
  const materiallyDifferent = Math.abs(result.outputTokens - testCase.targetOutputTokens) >
    Math.max(1, testCase.targetOutputTokens * 0.02);
  const invalidOutput = result.text.trim().length === 0 || result.error !== undefined;
  const repeatedOutput = hasPathologicalRepetition(result.text);
  const exclusionReasons: string[] = [];
  if (materiallyDifferent) exclusionReasons.push("output-token-count-differs-materially");
  if (invalidOutput) exclusionReasons.push("invalid-output");
  if (result.error) exclusionReasons.push("runtime-error");
  return {
    correctness: {
      exactOutputText: result.text,
      tokenCount: result.outputTokens,
      characterCount: result.text.length,
      outputByteCount: outputBytes,
      reachedRequestedTokenLength: result.outputTokens >= testCase.targetOutputTokens,
      matchedExpectedPrefix: result.text.trimStart().startsWith(testCase.expectedPrefix),
      invalidOutput,
      repeatedOutput,
      earlyTerminated,
      error: result.error ?? null,
    },
    equalWorkEligible: exclusionReasons.length === 0,
    exclusionReasons,
  };
}

function hasPathologicalRepetition(text: string): boolean {
  const tokens = text.trim().split(/\s+/);
  let repeated = 1;
  for (let index = 1; index < tokens.length; index += 1) {
    repeated = tokens[index] === tokens[index - 1] ? repeated + 1 : 1;
    if (repeated >= 8) return true;
  }
  return false;
}