// Converts episode JSONL files into blind judge items (transcript + metadata only).
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
const ROOT = resolve(import.meta.dirname, "../..");
const readJsonl = (path) => readFileSync(path, "utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line));
const [out, ...names] = process.argv.slice(2);
const scenarios = Object.fromEntries([
  ...readJsonl(resolve(ROOT, "research/cefr-agentic/scenarios.dev.jsonl")),
  ...readJsonl(resolve(ROOT, "research/cefr-agentic/scenarios.validation.jsonl")),
  ...(process.env.SEALED_PATH ? process.env.SEALED_PATH.split(",").flatMap((path) => readJsonl(path)) : []),
].map((row) => [row.scenario_id, row]));
const items = names.flatMap((name) => readJsonl(resolve(ROOT, process.env.RESEARCH_OUT ?? "research-results/cefr-agentic", `${name}.jsonl`))).map((row) => ({
  item_id: row.jobKey,
  metadata_en: { ...scenarios[row.scenarioId].metadata_en, cefr_level: row.level },
  transcript: row.playerTurns.flatMap((text, index) => [{ speaker: "learner", text }, { speaker: "npc", text: row.npcTurns[index] }]),
}));
writeFileSync(out, items.map((item) => JSON.stringify(item)).join("\n") + "\n");
console.log(`${items.length} items -> ${out}`);
