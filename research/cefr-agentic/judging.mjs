// Blind judging harness. Prepares anonymized, order-shuffled judge batches from episodes or
// calibration items, and collects judge outputs (JSONL following judgment.schema.json, plus
// "estimated_level" for the level estimator). Judges never see arm, edit, seed or hypothesis.
//
//   node research/cefr-agentic/judging.mjs prepare <role> <items.jsonl> <batch-name>
//   node research/cefr-agentic/judging.mjs collect <batch-name>
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const ROOT = resolve(import.meta.dirname, "../..");
const DIR = resolve(ROOT, process.env.RESEARCH_OUT ?? "research-results/cefr-agentic", "judging");
mkdirSync(DIR, { recursive: true });
const readJsonl = (path) => readFileSync(path, "utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line));

export const ROLES = {
  language: {
    criteria: ["morphosyntactic_correctness", "natural_hebrew"],
    showTargetLevel: false,
    instructions:
      "Rate only the NPC's Hebrew. morphosyntactic_correctness: grammatical formation and agreement (gender, number, numerals, verb agreement, prepositions, construct state). natural_hebrew: idiomatic, natural modern Israeli Hebrew for this character and situation. Ignore whether the level is appropriate.",
  },
  cefr: {
    criteria: ["cefr_suitability", "pedagogical_appropriateness"],
    showTargetLevel: true,
    instructions:
      "The learner's target CEFR level is given. cefr_suitability: how well the NPC's language fits that level using communicative descriptors (A1: very simple concrete phrases and basic needs; B2: flexible, precise connected dialogue with reasons and alternatives), not length alone. A reply that is far simpler or far more complex than the target is unsuitable. pedagogical_appropriateness: useful and comprehensible for that learner without misleading feedback.",
  },
  level: {
    criteria: [],
    showTargetLevel: false,
    instructions:
      "Estimate the CEFR level (A1, A2, B1, B2, C1 or C2) of the NPC's Hebrew language: vocabulary range, sentence length and structure, subordinate clauses, connectives, abstraction. Judge the language actually produced, not the topic. Use null only if there is too little Hebrew to tell.",
  },
  level2: {
    criteria: [],
    showTargetLevel: false,
    instructions:
      "Reserved second level rater. Read only the NPC turns and decide which CEFR band (A1, A2, B1, B2, C1, C2) a learner would need to understand them comfortably. Consider word frequency, morphology, sentence length, clause embedding and discourse connectives. Ignore content correctness. Use null only if there is almost no Hebrew.",
  },
  conversation: {
    criteria: ["character_register_fidelity", "multi_turn_coherence"],
    showTargetLevel: false,
    instructions:
      "character_register_fidelity: the NPC keeps the role, personality, register and goal from the metadata. multi_turn_coherence: replies follow from the learner's turns, refer back correctly, repair misunderstandings and progress toward the NPC goal across turns.",
  },
};

const ANCHORS =
  "Scale 0-4: 4 = fully meets the criterion with no issues; 3 = meets it with minor issues (pass); 2 = noticeable problems; 1 = serious problems; 0 = fails entirely. Set insufficient_evidence true and score null only when the text gives no basis. Cite short exact spans from the NPC text as evidence.";

function shuffle(items, seed) {
  const out = [...items];
  let state = seed >>> 0 || 1;
  for (let i = out.length - 1; i > 0; i -= 1) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const j = state % (i + 1);
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

const [command, ...args] = process.argv.slice(2);
if (command === "prepare") {
  const [role, itemsPath, name, seedText = "11"] = args;
  const spec = ROLES[role];
  if (!spec) throw new Error(`unknown role ${role}`);
  const items = readJsonl(itemsPath);
  const hidden = {};
  const shown = shuffle(items, Number(seedText)).map((item, index) => {
    const blindId = `item-${createHash("sha256").update(`${name}:${item.item_id}`).digest("hex").slice(0, 10)}`;
    hidden[blindId] = item.item_id;
    const metadata = { ...item.metadata_en };
    if (!spec.showTargetLevel) delete metadata.cefr_level;
    return { order: index + 1, item_id: blindId, metadata_en: metadata, transcript: item.transcript };
  });
  const prompt = [
    `You are an independent blind judge (role: ${role}) for Hebrew NPC dialogue in a language-learning RPG.`,
    "You do not know how any transcript was produced; judge each one on its own.",
    spec.instructions,
    spec.criteria.length ? ANCHORS : "",
    `Rubric version: hebrew-cefr-1.0. Criteria to rate: ${spec.criteria.join(", ") || "(none; estimate level only)"}.`,
    "Output exactly one JSON object per item, one per line (JSONL), in the same order, each with keys:",
    `episode_id (the item_id), judge_model_id (your model name if known, else "unknown"), rubric_version "hebrew-cefr-1.0", ratings (array of {criterion, score (0-4 integer or null), insufficient_evidence (bool), hard_violation (bool), evidence: [{span, finding}] up to 3}), estimated_level (A1/A2/B1/B2/C1/C2 or null)${role === "level" || role === "level2" ? " (required)" : " (null)"}, critical_violation_codes (array of short codes), overall_abstain (bool), short_summary (one sentence).`,
  ].filter(Boolean).join("\n");
  writeFileSync(resolve(DIR, `${name}.prompt.txt`), prompt);
  writeFileSync(resolve(DIR, `${name}.items.json`), JSON.stringify({ role, criteria: spec.criteria, items: shown }, null, 1));
  writeFileSync(resolve(DIR, `${name}.key.json`), JSON.stringify(hidden));
  console.log(`${name}: ${shown.length} items for role ${role}`);
} else if (command === "collect") {
  const [name] = args;
  const key = JSON.parse(readFileSync(resolve(DIR, `${name}.key.json`), "utf8"));
  const outputPath = resolve(DIR, `${name}.output.jsonl`);
  if (!existsSync(outputPath)) throw new Error(`missing ${outputPath}`);
  const rows = readFileSync(outputPath, "utf8").split("\n").filter((line) => line.trim().startsWith("{")).map((line) => JSON.parse(line));
  const collected = rows.filter((row) => key[row.episode_id]).map((row) => ({ ...row, item_id: key[row.episode_id], batch: name }));
  writeFileSync(resolve(DIR, `${name}.collected.jsonl`), collected.map((row) => JSON.stringify(row)).join("\n") + "\n");
  console.log(`${name}: collected ${collected.length}/${Object.keys(key).length}`);
} else {
  console.error("usage: judging.mjs prepare <role> <items.jsonl> <name> [seed] | collect <name>");
  process.exit(2);
}
