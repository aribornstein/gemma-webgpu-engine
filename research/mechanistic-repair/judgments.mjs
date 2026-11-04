import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { checkGrounding, checkVocabulary, promotionVerdict, validateAssessment } from "../../src/research/workbench/contracts.ts";

const directory = resolve(process.argv[2] ?? "research-results/mechanistic-repair/discovery");
const read = (name) => JSON.parse(readFileSync(resolve(directory, name), "utf8"));
const packet = read("quality-packet.json");
const key = read("quality-key.json");
const roles = ["language", "task", "level"];
const judgments = roles.flatMap((role) => read(`judge-${role}.json`));
const seen = new Set();
for (const judgment of judgments) {
  const item = packet.find((p) => p.episodeId === judgment.episodeId);
  if (!item) throw new Error("Unrecognized judge item");
  const pair = `${judgment.episodeId}:${judgment.role}`;
  if (seen.has(pair)) throw new Error("Duplicate judgment");
  seen.add(pair);
  validateAssessment(judgment, item.episodeId, item.npcReply);
}
const result = packet.map((item) => {
  const panel = judgments.filter((j) => j.episodeId === item.episodeId);
  const task = panel.find((j) => j.role === "task");
  const grounding = checkGrounding(item.world, item.npcReply, task?.claims);
  if (grounding.status === "pass" && task.extractionComplete !== true) grounding.status = "unknown";
  const vocabulary = checkVocabulary(item.world, item.npcReply);
  return {
    artifactId: key[item.episodeId], scope: item.scope, npcReply: item.npcReply,
    ratings: Object.fromEntries(panel.flatMap((j) => j.ratings).map((r) => [r.criterion, r.score])),
    grounding, vocabulary,
    verdict: promotionVerdict(panel, item.episodeId, item.npcReply, [grounding.status, vocabulary.status]),
  };
});
const summary = {
  scope: "AI-judged discovery single turns, not a held-out full-episode quality gate",
  judgments: judgments.length, cases: result.length,
  verdicts: Object.fromEntries(["pass", "fail", "unknown"].map((v) => [v, result.filter((r) => r.verdict === v).length])),
  missingVocabularyPolicy: result.filter((r) => r.vocabulary.status === "unknown").length,
  criteria: Object.fromEntries(Object.keys(result[0].ratings).map((criterion) => {
    const scores = result.map((r) => r.ratings[criterion]).filter((v) => v !== null && v !== undefined);
    return [criterion, {
      scored: scores.length, unassessed: result.length - scores.length,
      minimum: scores.length ? Math.min(...scores) : null,
      mean: scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : null,
      atLeastNine: scores.filter((v) => v >= 9).length,
    }];
  })),
};
writeFileSync(resolve(directory, "quality-summary.json"), JSON.stringify({ summary, items: result }, null, 2));
console.log(JSON.stringify(summary, null, 2));
