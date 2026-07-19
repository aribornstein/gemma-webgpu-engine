import { expect, test } from "@playwright/test";
import {
  aggregateCandidateAuditScores,
  createUniqueJudgeSchedule,
  scoreComparativeHebrewJudge,
  scoreHebrewAudit,
  selectHighestScoredCandidate,
  shuffledCandidateIndices,
} from "../src/runtime/best-of-audit";

function audit(overrides: Record<string, string> = {}): string {
  return JSON.stringify({
    task_fidelity: "fulfilled",
    actionability: "actionable",
    grammar_integrity: "clean",
    idiomaticity: "natural",
    level_fit: "appropriate",
    ...overrides,
  });
}

test("penalizes semantic contradictions more than stylistic weakness", () => {
  expect(scoreHebrewAudit(audit())).toBe(16);
  expect(scoreHebrewAudit(audit({
    task_fidelity: "contradicted",
    actionability: "unusable",
  }))).toBe(-5);
  expect(scoreHebrewAudit(audit({ idiomaticity: "acceptable" }))).toBe(14);
});

test("selects the highest independent score and abstains on ties", () => {
  expect(selectHighestScoredCandidate([
    { candidateIndex: 0, score: 14 },
    { candidateIndex: 1, score: 12 },
  ])).toBe(0);
  expect(selectHighestScoredCandidate([
    { candidateIndex: 0, score: 12 },
    { candidateIndex: 1, score: 12 },
  ])).toBeNull();
  expect(selectHighestScoredCandidate([
    { candidateIndex: 0, score: 14 },
    { candidateIndex: 1, score: null },
  ])).toBeNull();
});

test("rejects incomplete or unknown audit categories", () => {
  expect(scoreHebrewAudit("{}")).toBeNull();
  expect(scoreHebrewAudit(audit({ task_fidelity: "excellent" }))).toBeNull();
});

test("shuffles audit execution away from natural candidate order", () => {
  expect(shuffledCandidateIndices(2, () => 0.99)).toEqual([1, 0]);
  const order = shuffledCandidateIndices(4, () => 0.25);
  expect(order).toHaveLength(4);
  expect([...order].sort()).toEqual([0, 1, 2, 3]);
  expect(order).not.toEqual([0, 1, 2, 3]);
});

test("gives every comparative judge all candidates in one order", () => {
  const schedule = createUniqueJudgeSchedule(3, 3, () => 0.25);
  expect(schedule).toHaveLength(3);
  for (const judge of schedule) expect([...judge.candidateOrder].sort()).toEqual([0, 1, 2]);
});

test("keeps every judge ordering unique", () => {
  const schedule = createUniqueJudgeSchedule(3, 4, () => 0.25);
  const orders = schedule.map(({ candidateOrder }) => candidateOrder.join(","));
  expect(new Set(orders).size).toBe(4);
  expect(() => createUniqueJudgeSchedule(2, 3, () => 0.25))
    .toThrow(/At most 2 unique orders/);
});

test("samples large candidate orders without enumerating permutations", () => {
  const schedule = createUniqueJudgeSchedule(1_000, 4, () => 0.25);
  expect(schedule).toHaveLength(4);
  expect(schedule.every(({ candidateOrder }) => candidateOrder.length === 1_000)).toBe(true);
  expect(new Set(schedule.map(({ candidateOrder }) => candidateOrder.join(","))).size).toBe(4);
});

test("aggregates repeated judges before selecting a candidate", () => {
  const aggregates = aggregateCandidateAuditScores(2, 2, [
    { candidateIndex: 1, score: 10 },
    { candidateIndex: 0, score: 16 },
    { candidateIndex: 0, score: 12 },
    { candidateIndex: 1, score: 14 },
  ]);
  expect(aggregates).toEqual([
    { candidateIndex: 0, score: 14 },
    { candidateIndex: 1, score: 12 },
  ]);
  expect(selectHighestScoredCandidate(aggregates)).toBe(0);
  expect(aggregateCandidateAuditScores(2, 2, [
    { candidateIndex: 0, score: 16 },
    { candidateIndex: 0, score: null },
    { candidateIndex: 1, score: 12 },
    { candidateIndex: 1, score: 14 },
  ])[0].score).toBeNull();
});

test("maps comparative display slots back to stable candidate IDs", () => {
  const judgment = JSON.stringify({
    candidate_1: JSON.parse(audit()),
    candidate_2: JSON.parse(audit({ task_fidelity: "incomplete" })),
  });
  expect(scoreComparativeHebrewJudge(judgment, [1, 0])).toEqual([
    { candidateIndex: 1, score: 16 },
    { candidateIndex: 0, score: 12 },
  ]);
});
