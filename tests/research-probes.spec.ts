import { expect, test } from "@playwright/test";
import { checkState, turnFeatures } from "../src/research/cefr/metrics";
import { atLevel, episodeMessages, systemMessage, type CefrScenario } from "../src/research/cefr/prompts";
import {
  auc,
  fitLogistic,
  fitMeanDifference,
  groupBootstrapAuc,
  probeScore,
  shuffledLabels,
} from "../src/research/probes";

function dataset(n: number, signal: number, seed: number) {
  let state = seed;
  const next = () => ((state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 4294967296) - 0.5;
  const x: Float32Array[] = [];
  const y: number[] = [];
  const group: string[] = [];
  for (let i = 0; i < n; i += 1) {
    const label = i % 2;
    const row = Float32Array.from({ length: 16 }, () => next());
    row[3] += label ? signal : -signal;
    x.push(row);
    y.push(label);
    group.push(`g${i % 6}`);
  }
  return { x, y, group };
}

test("linear probes separate a planted signal and shuffled labels do not", () => {
  const train = dataset(120, 0.6, 1);
  const testSet = dataset(120, 0.6, 2);
  for (const probe of [fitMeanDifference(train), fitLogistic(train)]) {
    const scores = testSet.x.map((row) => probeScore(probe, row));
    expect(auc(scores, testSet.y)).toBeGreaterThan(0.9);
  }
  const shuffled = fitLogistic({ ...train, y: shuffledLabels(train.y, 5) });
  const control = auc(testSet.x.map((row) => probeScore(shuffled, row)), testSet.y);
  expect(Math.abs(control - 0.5)).toBeLessThan(0.2);
  const [low, high] = groupBootstrapAuc(testSet.x.map((row) => probeScore(fitMeanDifference(train), row)), testSet.y, testSet.group, 200);
  expect(low).toBeGreaterThan(0.8);
  expect(high).toBeLessThanOrEqual(1);
});

test("AUC handles ties and missing classes", () => {
  expect(auc([1, 1, 0, 0], [1, 0, 1, 0])).toBe(0.5);
  expect(auc([1, 2], [1, 1])).toBeNaN();
  expect(shuffledLabels([1, 0, 1, 0], 3).sort()).toEqual([0, 0, 1, 1]);
});

test("Hebrew complexity features and hard state checks", () => {
  const simple = turnFeatures("שלום. התפוחים עולים עשרה שקלים.");
  const complex = turnFeatures("שלום, אם תרצה לקנות שני תפוחים, כדאי לדעת שהמחיר הכולל הוא עשרה שקלים, כי זה מחיר קבוע.");
  expect(complex.wordsPerSentence).toBeGreaterThan(simple.wordsPerSentence);
  expect(complex.connectiveRate).toBeGreaterThan(simple.connectiveRate);
  const facts = { quantity: 2, total_price_ils: 10 };
  expect(checkState("שני תפוחים עולים עשרה שקלים.", facts, ["payment_completed"]).pass).toBe(true);
  expect(checkState("שני תפוחים עולים עשרים שקלים.", facts, ["payment_completed"]).unexpectedNumbers).toEqual([20]);
  expect(checkState("זה עולה 25 שקלים", facts, []).unexpectedNumbers).toEqual([25]);
  expect(checkState("עשרים וחמש", facts, []).unexpectedNumbers).toEqual([25]);
  expect(checkState("תודה על התשלום!", facts, ["payment_completed"]).claimsPayment).toBe(true);
  expect(checkState("It costs ten", facts, []).englishSpillover).toBe(true);
});

test("prompts keep world state identical across levels and policies", () => {
  const scenario: CefrScenario = {
    scenario_id: "s",
    scenario_family: "f",
    split: "development",
    human_validated: false,
    metadata_en: { language: "Hebrew", cefr_level: "A1", scenario: "x" },
    player_policy: { type: "scripted_hebrew", player_turns_he: ["א", "ב"] },
    oracle_en: { critical_facts: {}, must_not_assume: [], payment_update_mode: "external_oracle_only" },
  };
  const a1 = systemMessage(atLevel(scenario, "A1"), "canonical");
  const b2 = systemMessage(atLevel(scenario, "B2"), "canonical");
  expect(a1.replace('"A1"', '"B2"')).toBe(b2);
  expect(systemMessage(scenario, "level_specific_strong")).toContain("CEFR A1");
  const messages = episodeMessages(scenario, "canonical", ["נ1"], 1);
  expect(messages.map((message) => message.role)).toEqual(["system", "user", "assistant", "user"]);
});
