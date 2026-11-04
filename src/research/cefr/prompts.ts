import type { GemmaChatMessage } from "../../runtime/gemma-tokenizer";

export type CefrLevel = "A1" | "A2" | "B1" | "B2";
export type PromptPolicy = "canonical" | "level_specific_strong";

export interface CefrScenario {
  scenario_id: string;
  scenario_family: string;
  split: string;
  human_validated: boolean;
  metadata_en: Record<string, unknown> & { cefr_level: CefrLevel };
  player_policy: { type: string; player_turns_he: string[] };
  oracle_en: {
    critical_facts: Record<string, string | number | boolean>;
    must_not_assume: string[];
    payment_update_mode: string;
  };
  notes?: string;
}

const LEVEL_GUIDANCE: Record<CefrLevel, string> = {
  A1: "The learner is CEFR A1. Use very short, simple sentences in the present tense with the most common everyday words. One idea per sentence. Avoid subordinate clauses, idioms and rare words.",
  A2: "The learner is CEFR A2. Use short connected sentences about routine transactions with common words and simple connectors.",
  B1: "The learner is CEFR B1. Use connected everyday explanations, simple subordinate clauses and repairs, with moderately varied vocabulary.",
  B2: "The learner is CEFR B2. Use flexible, precise, natural Hebrew: longer connected sentences, subordinate clauses, reasons and alternatives, and richer vocabulary, without artificial complexity.",
};

/** Returns the scenario with its requested level replaced; world state is otherwise unchanged. */
export function atLevel(scenario: CefrScenario, level: CefrLevel): CefrScenario {
  return { ...scenario, metadata_en: { ...scenario.metadata_en, cefr_level: level } };
}

/** System message: fixed English instructions plus the English metadata as compact JSON. */
export function systemMessage(scenario: CefrScenario, policy: PromptPolicy): string {
  const lines = [
    "You are a non-player character (NPC) in a role-playing game for learners of Hebrew.",
    "The scenario metadata below is in English. Stay in character, keep every fact, quantity and price exactly as given, never say that payment or the quest is complete unless the metadata says so, and reply only in Hebrew with one short NPC turn.",
  ];
  if (policy === "level_specific_strong") {
    lines.push(LEVEL_GUIDANCE[scenario.metadata_en.cefr_level]);
    lines.push("Match the learner's requested CEFR level in every turn; the level matters more than being thorough.");
  }
  lines.push(`Scenario metadata: ${JSON.stringify(scenario.metadata_en)}`);
  return lines.join("\n");
}

/** Messages for the NPC's turn after `playerTurnIndex` player turns, with prior NPC turns. */
export function episodeMessages(
  scenario: CefrScenario,
  policy: PromptPolicy,
  npcTurns: readonly string[],
  playerTurnIndex: number,
): GemmaChatMessage[] {
  const messages: GemmaChatMessage[] = [{ role: "system", content: systemMessage(scenario, policy) }];
  for (let turn = 0; turn <= playerTurnIndex; turn += 1) {
    messages.push({ role: "user", content: scenario.player_policy.player_turns_he[turn] });
    if (turn < playerTurnIndex) messages.push({ role: "assistant", content: npcTurns[turn] });
  }
  return messages;
}
