export interface GemmaGreedyGoldenCase {
  id: string;
  prompt: string;
  maxNewTokens: number;
  promptTokenIds: readonly number[];
  generatedTokenIds: readonly number[];
  text: string;
  stoppedOnEndToken: boolean;
}

export const GEMMA_GREEDY_GOLDEN_CASES: readonly GemmaGreedyGoldenCase[] = [
  {
    id: "short-greeting",
    prompt: "Say hi in one short sentence.",
    maxNewTokens: 8,
    promptTokenIds: [
      2, 105, 2364, 107, 37889, 5631, 528, 886, 2822, 13315, 236761, 106, 107,
      105, 4368, 107,
    ],
    generatedTokenIds: [10979, 236888],
    text: "Hi!",
    stoppedOnEndToken: true,
  },
  {
    id: "arithmetic",
    prompt: "Reply with only the number: 7 + 5.",
    maxNewTokens: 8,
    promptTokenIds: [
      2, 105, 2364, 107, 40654, 607, 1186, 506, 1548, 236787, 236743, 236832,
      900, 236743, 236810, 236761, 106, 107, 105, 4368, 107,
    ],
    generatedTokenIds: [236770, 236778],
    text: "12",
    stoppedOnEndToken: true,
  },
  {
    id: "arabic",
    prompt: "قل مرحباً بجملة قصيرة.",
    maxNewTokens: 12,
    promptTokenIds: [
      2, 105, 2364, 107, 18903, 47399, 99673, 39570, 7130, 237049, 96365,
      28521, 236761, 106, 107, 105, 4368, 107,
    ],
    generatedTokenIds: [237143, 236910, 70436, 60774, 236888],
    text: "أهلاً بك!",
    stoppedOnEndToken: true,
  },
  {
    id: "longer-instruction",
    prompt: "Name the three primary colors in one short sentence.",
    maxNewTokens: 16,
    promptTokenIds: [
      2, 105, 2364, 107, 1567, 506, 1806, 5905, 7913, 528, 886, 2822, 13315,
      236761, 106, 107, 105, 4368, 107,
    ],
    generatedTokenIds: [
      818, 5905, 7913, 659, 2604, 236764, 7070, 236764, 532, 3730, 236761,
    ],
    text: "The primary colors are red, yellow, and blue.",
    stoppedOnEndToken: true,
  },
  {
    id: "prefill-32-boundary",
    prompt: "Reply with only OK. word word word word word word word word word word word word word word word word word word",
    maxNewTokens: 8,
    promptTokenIds: [
      2, 105, 2364, 107, 40654, 607, 1186, 16119, 236761, 3658, 3658, 3658,
      3658, 3658, 3658, 3658, 3658, 3658, 3658, 3658, 3658, 3658, 3658, 3658,
      3658, 3658, 3658, 106, 107, 105, 4368, 107,
    ],
    generatedTokenIds: [7676],
    text: "OK",
    stoppedOnEndToken: true,
  },
];