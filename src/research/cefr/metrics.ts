/**
 * Deterministic, judge-free measurements of Hebrew NPC turns: linguistic complexity features (a
 * secondary CEFR-control signal, not a CEFR rating), English spillover, and hard world-state checks.
 */
const HEBREW_WORD = /[\u05D0-\u05EA][\u05D0-\u05EA\u05F3\u05F4'"־-]*/g;
const LATIN_WORD = /[A-Za-z]{2,}/g;
const SENTENCE_END = /[.!?]+/;
// Common Hebrew connectives/subordinators used as a clause-complexity proxy.
const CONNECTIVES = new Set([
  "כי", "אבל", "אם", "כאשר", "למרות", "לכן", "אשר", "מכיוון", "בגלל", "אולם", "ואילו", "בעוד",
  "כדי", "אלא", "אך", "שכן", "כלומר", "לעומת", "בנוסף", "אמנם", "עד", "לפני", "אחרי", "כמו",
]);
// Hebrew number words (both genders, construct forms) mapped to values for the state check.
const NUMBER_WORDS: Record<string, number> = {
  "אחד": 1, "אחת": 1, "שניים": 2, "שתיים": 2, "שני": 2, "שתי": 2, "שלושה": 3, "שלוש": 3, "שלושת": 3,
  "ארבעה": 4, "ארבע": 4, "ארבעת": 4, "חמישה": 5, "חמש": 5, "חמשת": 5, "שישה": 6, "שש": 6, "ששת": 6,
  "שבעה": 7, "שבע": 7, "שבעת": 7, "שמונה": 8, "שמונת": 8, "תשעה": 9, "תשע": 9, "תשעת": 9,
  "עשרה": 10, "עשר": 10, "עשרת": 10, "עשרים": 20, "שלושים": 30, "ארבעים": 40, "חמישים": 50,
  "שישים": 60, "שבעים": 70, "שמונים": 80, "תשעים": 90, "מאה": 100, "מאתיים": 200,
};
const PAYMENT_CLAIMS = [
  "שילמת", "התשלום התקבל", "קיבלתי את התשלום", "תודה על התשלום", "התשלום בוצע", "שולם",
];

export interface TurnFeatures {
  words: number;
  sentences: number;
  wordsPerSentence: number;
  meanWordLength: number;
  connectiveRate: number;
  typeTokenRatio: number;
  latinWordFraction: number;
}

export function hebrewWords(text: string): string[] {
  return text.match(HEBREW_WORD) ?? [];
}

export function turnFeatures(text: string): TurnFeatures {
  const words = hebrewWords(text);
  const latin = text.match(LATIN_WORD) ?? [];
  const sentences = Math.max(1, text.split(SENTENCE_END).filter((part) => hebrewWords(part).length > 0).length);
  const stripped = words.map((word) => word.replace(/^[ובלכמשה]/, ""));
  const connectives = words.filter((word) => CONNECTIVES.has(word) || CONNECTIVES.has(word.replace(/^ו/, ""))).length;
  return {
    words: words.length,
    sentences,
    wordsPerSentence: words.length / sentences,
    meanWordLength: words.length ? words.reduce((sum, word) => sum + word.length, 0) / words.length : 0,
    connectiveRate: words.length ? connectives / words.length : 0,
    typeTokenRatio: words.length ? new Set(stripped).size / words.length : 0,
    latinWordFraction: words.length + latin.length ? latin.length / (words.length + latin.length) : 0,
  };
}

export const COMPLEXITY_KEYS = ["wordsPerSentence", "meanWordLength", "connectiveRate", "words"] as const;

/**
 * One-number complexity index used for the secondary CEFR-control gap: mean of feature z-scores
 * against fixed reference statistics (`stats`, computed once on development baseline turns).
 */
export function complexityIndex(
  features: TurnFeatures,
  stats: Record<(typeof COMPLEXITY_KEYS)[number], { mean: number; sd: number }>,
): number {
  let total = 0;
  for (const key of COMPLEXITY_KEYS) {
    const { mean, sd } = stats[key];
    total += (features[key] - mean) / (sd || 1);
  }
  return total / COMPLEXITY_KEYS.length;
}

export interface StateCheck {
  numbersConsistent: boolean;
  unexpectedNumbers: number[];
  claimsPayment: boolean;
  englishSpillover: boolean;
  pass: boolean;
}

/**
 * Hard checks against the oracle: every number the NPC states must be an oracle number (quantities,
 * prices, totals) or 1; no payment claims when payment is external; no Latin-script words.
 */
export function checkState(
  npcText: string,
  criticalFacts: Record<string, string | number | boolean>,
  mustNotAssume: readonly string[],
  allowedNumbers: readonly number[] = [],
): StateCheck {
  const allowed = new Set<number>([1, ...allowedNumbers]);
  for (const value of Object.values(criticalFacts)) if (typeof value === "number") allowed.add(value);
  const numbers = (npcText.match(/\d+(?:[.,]\d+)?/g) ?? []).map((value) => Number(value.replace(",", ".")));
  // Standalone Hebrew number words (optionally with the conjunction prefix ו); compound numbers such
  // as "עשרים וחמש" are summed.
  const words = hebrewWords(npcText);
  for (let index = 0; index < words.length; index += 1) {
    const value = NUMBER_WORDS[words[index]];
    if (value === undefined) continue;
    const next = NUMBER_WORDS[words[index + 1]?.replace(/^ו/, "") ?? ""];
    if (value >= 20 && value % 10 === 0 && next !== undefined && next < 10 && words[index + 1].startsWith("ו")) {
      numbers.push(value + next);
      index += 1;
    } else {
      numbers.push(value);
    }
  }
  const unexpectedNumbers = numbers.filter((value) => !allowed.has(value));
  const claimsPayment = mustNotAssume.some((item) => item.includes("payment")) &&
    PAYMENT_CLAIMS.some((claim) => npcText.includes(claim));
  const englishSpillover = (npcText.match(LATIN_WORD) ?? []).length > 0;
  const numbersConsistent = unexpectedNumbers.length === 0;
  return {
    numbersConsistent,
    unexpectedNumbers,
    claimsPayment,
    englishSpillover,
    pass: numbersConsistent && !claimsPayment && !englishSpillover,
  };
}
