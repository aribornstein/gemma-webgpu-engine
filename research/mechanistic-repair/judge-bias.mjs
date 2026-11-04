import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { counterbalancedPreference, judgeAgreement } from "../../src/research/workbench/statistics.ts";

const root = resolve(import.meta.dirname, "../..");
const dir = resolve(root, "research-results/component-localization/judge-audit");
mkdirSync(dir, { recursive: true });
const stage = process.argv[2];
const read = (file) => JSON.parse(readFileSync(file, "utf8"));
const write = (name, value) => writeFileSync(resolve(dir, name), JSON.stringify(value, null, 2));
if (stage === "prepare") {
  const examples = read(resolve(root, "public/research/comparisons.json")).examples.filter((e) => e.experiment === "cefr" && e.level === "B2" && e.seed === 23);
  const items = [], key = {};
  for (const [index, example] of examples.entries()) {
    const base = example.arms.find((a) => a.id === "raw_base");
    const edited = example.arms.find((a) => a.id === "agent_p6");
    for (const order of ["forward", "reverse", ...(index < 2 ? ["duplicate"] : [])]) {
      const id = createHash("sha256").update(`judge-audit-v1:${example.id}:${order}`).digest("hex").slice(0, 12);
      key[id] = { exampleId: example.id, order, A: order === "reverse" ? edited.id : base.id, B: order === "reverse" ? base.id : edited.id };
      items.push({
        id, task: example.metadata,
        A: (order === "reverse" ? edited : base).turns.map((t) => ({ player: t.player, npc: t.text })),
        B: (order === "reverse" ? base : edited).turns.map((t) => ({ player: t.player, npc: t.text })),
      });
    }
  }
  items.sort((a, b) => a.id.localeCompare(b.id));
  write("items.json", items); write("key.json", key);
  console.log(`${items.length} randomized paired packets (6 reversed pairs plus 2 hidden duplicates)`);
} else if (stage === "analyze") {
  const items = read(resolve(dir, "items.json"));
  const key = read(resolve(dir, "key.json"));
  const ratings = read(resolve(dir, "ratings.json"));
  if (ratings.length !== items.length || new Set(ratings.map((r) => r.id)).size !== items.length) throw new Error("Missing or duplicated judge packets");
  for (const row of ratings) {
    const item = items.find((i) => i.id === row.id);
    if (!item || !["A", "B", "tie", null].includes(row.winner)) throw new Error("Invalid judge winner");
    for (const arm of ["A", "B"]) {
      for (const criterion of ["grammar", "naturalness"]) {
        const score = row[arm]?.[criterion];
        if (score !== null && (!Number.isFinite(score) || score < 0 || score > 10)) throw new Error("Invalid rubric value");
      }
      if (!Array.isArray(row[arm]?.evidence) || !row[arm].evidence.length ||
        row[arm].evidence.some((span) => !span || !item[arm].some((t) => t.npc.includes(span)))) throw new Error("Ungrounded judge evidence");
    }
  }
  const examples = [...new Set(Object.values(key).map((k) => k.exampleId))];
  const get = (example, order) => ratings.find((r) => key[r.id]?.exampleId === example && key[r.id].order === order);
  const preference = counterbalancedPreference(examples.map((id) => ({
    id, originalWinner: get(id, "forward").winner, reversedWinner: get(id, "reverse").winner,
  })));
  const duplicates = examples.filter((id) => get(id, "duplicate")).flatMap((id) =>
    ["A", "B"].flatMap((arm) => ["grammar", "naturalness"].map((criterion) => ({
      id: `${id}:${arm}:${criterion}`, first: get(id, "forward")[arm][criterion], second: get(id, "duplicate")[arm][criterion],
    }))));
  const report = { scope: "One AI judge invocation: descriptive position/repeatability audit on historical data, not independent linguistic validation",
    preference, hiddenDuplicateRatings: judgeAgreement(duplicates), itemCount: items.length,
    warning: "Correlated outputs and ordinal scores; do not treat items/criteria as independent replications.",
  };
  write("summary.json", report);
  writeFileSync(resolve(root, "public/research/judge-audit.json"), JSON.stringify(report));
  console.log(JSON.stringify(report, null, 2));
} else throw new Error("prepare | analyze");
