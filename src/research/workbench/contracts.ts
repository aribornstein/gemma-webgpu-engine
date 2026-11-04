import type { GemmaChatMessage } from "../../runtime/gemma-tokenizer";

export const HIDDEN_SIZE = 1536;
export const MAX_INSPECTION_BYTES = 16 * 1024 * 1024;
export const CRITERIA = [
  "grammar", "naturalness", "levelSuitability", "vocabulary", "grounding", "goal", "coherence",
] as const;
export type Criterion = typeof CRITERIA[number];
export type Verdict = "pass" | "fail" | "unknown";

export interface WorldContract {
  currency: "ILS";
  quantity: number;
  unitPrice: number;
  totalPrice: number;
  paid: boolean;
  item: string;
  topic: string;
  npcGoal: string;
  vocabularyPolicy: "unrestricted-diagnostic" | "explicit-forms";
  allowedForms?: string[];
  permittedNames?: string[];
}

export function validateWorld(world: WorldContract): void {
  for (const key of ["quantity", "unitPrice", "totalPrice"] as const) {
    if (!Number.isFinite(world[key]) || world[key] < 0) throw new Error(`Invalid world ${key}`);
  }
  if (!Number.isInteger(world.quantity) || world.quantity < 1) throw new Error("Quantity must be a positive integer");
  if (Math.abs(world.quantity * world.unitPrice - world.totalPrice) > 1e-6) {
    throw new Error("Ambiguous price contract: quantity * unitPrice must equal totalPrice");
  }
  if (world.currency !== "ILS" || typeof world.paid !== "boolean") throw new Error("Invalid currency or payment state");
  if (![world.item, world.topic, world.npcGoal].every((v) => typeof v === "string" && v.length > 0)) {
    throw new Error("World requires item, topic and NPC goal");
  }
  if (!["unrestricted-diagnostic", "explicit-forms"].includes(world.vocabularyPolicy)) {
    throw new Error("Unknown vocabulary policy");
  }
  if (world.vocabularyPolicy === "explicit-forms" && !world.allowedForms?.length) {
    throw new Error("Explicit vocabulary requires an enumerated, inflection-aware allowedForms list");
  }
}

export interface GroundingClaim {
  field: "quantity" | "unitPrice" | "totalPrice" | "currency" | "paid" | "item";
  value: string | number | boolean;
  quote: string;
}

export interface GroundingResult {
  status: Verdict;
  violations: string[];
  coverage: "adjudicated-claims" | "unadjudicated";
}

/** Semantic claim extraction is supplied by a separate assessor, never inferred from a number bag. */
export function checkGrounding(
  world: WorldContract,
  transcript: string,
  claims?: readonly GroundingClaim[],
): GroundingResult {
  validateWorld(world);
  if (!claims?.length) return { status: "unknown", violations: [], coverage: "unadjudicated" };
  const violations: string[] = [];
  for (const claim of claims) {
    if (!["quantity", "unitPrice", "totalPrice", "currency", "paid", "item"].includes(claim.field)) {
      throw new Error("Unsupported fact claim field");
    }
    if (!claim.quote.trim() || !transcript.includes(claim.quote)) throw new Error("Fact evidence is not in the transcript");
    if (claim.value !== world[claim.field]) violations.push(`${claim.field}: expected ${world[claim.field]}, got ${claim.value}`);
  }
  return { status: violations.length ? "fail" : "pass", violations, coverage: "adjudicated-claims" };
}

export function checkVocabulary(world: WorldContract, text: string): { status: Verdict; unmatched: string[] } {
  validateWorld(world);
  if (world.vocabularyPolicy === "unrestricted-diagnostic") return { status: "unknown", unmatched: [] };
  const allowed = new Set([...(world.allowedForms ?? []), ...(world.permittedNames ?? [])]);
  // Do not strip arbitrary prefixes: these can change a Hebrew word's meaning.
  const words = text.normalize("NFC").match(/[\p{L}\p{M}]+(?:['"\u05f3\u05f4][\p{L}\p{M}]+)*/gu) ?? [];
  const unmatched = [...new Set(words.filter((word) => !allowed.has(word)))];
  return { status: unmatched.length ? "fail" : "pass", unmatched };
}

export interface Rating {
  criterion: Criterion;
  score: number | null;
  evidence: string[];
}

export interface JudgeAssessment {
  episodeId: string;
  judgeId: string;
  role: "language" | "task" | "level-conversation";
  rubricVersion: "mechanistic-quality-1";
  abstain: boolean;
  ratings: Rating[];
  criticalViolations: string[];
}

const ROLE_CRITERIA: Record<JudgeAssessment["role"], readonly Criterion[]> = {
  language: ["grammar", "naturalness"],
  task: ["vocabulary", "grounding", "goal"],
  "level-conversation": ["levelSuitability", "coherence"],
};

export function validateAssessment(assessment: JudgeAssessment, episodeId: string, transcript: string): void {
  if (assessment.episodeId !== episodeId || !assessment.judgeId?.trim() ||
      assessment.rubricVersion !== "mechanistic-quality-1" || typeof assessment.abstain !== "boolean") {
    throw new Error("Invalid judge identity, episode or rubric");
  }
  const expected = ROLE_CRITERIA[assessment.role];
  if (!expected || !Array.isArray(assessment.ratings) || !Array.isArray(assessment.criticalViolations) ||
      assessment.criticalViolations.some((v) => typeof v !== "string")) throw new Error("Invalid judge packet");
  const present = assessment.ratings.map((r) => r.criterion);
  if (present.length !== expected.length || new Set(present).size !== present.length ||
      expected.some((criterion) => !present.includes(criterion))) throw new Error("Missing, duplicate or unexpected criteria");
  for (const rating of assessment.ratings) {
    if (rating.score === null) {
      if (!assessment.abstain) throw new Error("Null score requires explicit abstention");
    } else if (!Number.isFinite(rating.score) || rating.score < 0 || rating.score > 10) {
      throw new Error("Score must be null or a finite number in 0..10");
    }
    if (!Array.isArray(rating.evidence) || (!assessment.abstain && rating.evidence.length === 0) ||
        rating.evidence.some((quote) => typeof quote !== "string" || !quote.trim() || !transcript.includes(quote))) {
      throw new Error("Judge evidence must cite exact transcript spans");
    }
  }
}

export function promotionVerdict(
  assessments: readonly JudgeAssessment[],
  episodeId: string,
  transcript: string,
  hardChecks: readonly Verdict[],
): Verdict {
  if (!hardChecks.length || hardChecks.some((check) => !["pass", "fail", "unknown"].includes(check))) {
    throw new Error("Promotion requires explicit hard checks");
  }
  const roles = new Set<JudgeAssessment["role"]>();
  const judges = new Set<string>();
  for (const assessment of assessments) {
    validateAssessment(assessment, episodeId, transcript);
    if (roles.has(assessment.role) || judges.has(assessment.judgeId)) throw new Error("Duplicate panel role or judge");
    roles.add(assessment.role);
    judges.add(assessment.judgeId);
  }
  if (hardChecks.includes("fail") || assessments.some((a) =>
    a.criticalViolations.length > 0 || a.ratings.some((r) => r.score !== null && r.score < 9))) return "fail";
  if (roles.size !== 3 || hardChecks.includes("unknown") || assessments.some((a) => a.abstain)) return "unknown";
  return "pass";
}

export interface FailureCase {
  id: string;
  family: string;
  split: "discovery" | "development";
  source: string;
  humanValidated: false;
  targetLevel: "A1" | "B2";
  playerProfile: string;
  metadata: WorldContract;
  history: GemmaChatMessage[];
  cleanUser: string;
  errorUser: string;
  assistantPrefix: string;
  correctContinuation: string;
  incorrectContinuation: string;
  freeTask: string;
  phenomenon: string;
  notes: string;
}

export function validateFailureCase(value: FailureCase): void {
  if (!value.id || !value.family || !["discovery", "development"].includes(value.split) ||
      value.humanValidated !== false || !["A1", "B2"].includes(value.targetLevel)) throw new Error("Invalid case identity");
  validateWorld(value.metadata);
  if (!value.cleanUser || !value.errorUser || value.cleanUser === value.errorUser) throw new Error("Case requires a clean/error pair");
  if (!value.correctContinuation || !value.incorrectContinuation ||
      value.correctContinuation === value.incorrectContinuation) throw new Error("Case requires contrastive continuations");
  if (!Array.isArray(value.history) || !value.history.length ||
      value.history.some((m) => !["system", "user", "assistant"].includes(m.role) || typeof m.content !== "string")) {
    throw new Error("Case history must be a text conversation");
  }
}
