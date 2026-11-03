/**
 * Browser-side experiment primitives for the Hebrew CEFR RPG task plugin. They run on the owned
 * WebGPU session; all numbers come from the engine (no other model is involved).
 */
import type { GemmaGenerationSession } from "../../runtime/gemma-session";
import type { GemmaResearchCaptureSite } from "../engine-types";
import { atLevel, episodeMessages, systemMessage, type CefrLevel, type CefrScenario, type PromptPolicy } from "./prompts";

const END_OF_TURN = 106;

export const EPISODE_DECODING = {
  temperature: 1,
  topK: 64,
  topP: 0.95,
  maxNewTokens: 96,
} as const;

export interface EpisodeResult {
  scenarioId: string;
  scenarioFamily: string;
  level: CefrLevel;
  policy: PromptPolicy;
  seed: number;
  playerTurns: string[];
  npcTurns: string[];
  npcTokenCounts: number[];
  stopReasons: string[];
  wallMs: number;
}

/** Scripted paired replay: identical player turns and per-turn seeds across arms. */
export async function runEpisode(
  session: GemmaGenerationSession,
  scenario: CefrScenario,
  level: CefrLevel,
  policy: PromptPolicy,
  seed: number,
  maxTurns?: number,
): Promise<EpisodeResult> {
  const leveled = atLevel(scenario, level);
  const npcTurns: string[] = [];
  const npcTokenCounts: number[] = [];
  const stopReasons: string[] = [];
  const started = performance.now();
  const turns = Math.min(maxTurns ?? Infinity, leveled.player_policy.player_turns_he.length);
  for (let turn = 0; turn < turns; turn += 1) {
    const result = await session.generate(episodeMessages(leveled, policy, npcTurns, turn), {
      ...EPISODE_DECODING,
      seed: seed * 100 + turn,
    });
    npcTurns.push(result.text.trim());
    npcTokenCounts.push(result.generatedTokenIds.length);
    stopReasons.push(result.stopReason);
  }
  return {
    scenarioId: scenario.scenario_id,
    scenarioFamily: scenario.scenario_family,
    level,
    policy,
    seed,
    playerTurns: leveled.player_policy.player_turns_he.slice(0, turns),
    npcTurns,
    npcTokenCounts,
    stopReasons,
    wallMs: performance.now() - started,
  };
}

interface SessionTokenizer {
  encodeInput(input: unknown): number[];
  encodeText(text: string): number[];
}

function tokenizerOf(session: GemmaGenerationSession): SessionTokenizer {
  return (session as unknown as { tokenizer: SessionTokenizer }).tokenizer;
}

/** Position of the last metadata token: the token before the system turn's end-of-turn marker. */
export function endOfMetadataPosition(tokenIds: readonly number[]): number {
  const index = tokenIds.indexOf(END_OF_TURN);
  if (index < 1) throw new Error("System turn end marker not found");
  return index - 1;
}

export interface CaptureRow {
  id: string;
  scenarioFamily: string;
  split: string;
  level: CefrLevel;
  variant: string;
  position: string;
  site: GemmaResearchCaptureSite;
  values: number[];
}

/** Level-probe captures at the end of metadata and at the last prompt token before the NPC turn. */
export async function captureLevelStates(
  session: GemmaGenerationSession,
  scenario: CefrScenario,
  level: CefrLevel,
  playerTurnIndex: number,
  sites: readonly GemmaResearchCaptureSite[],
): Promise<CaptureRow[]> {
  const leveled = atLevel(scenario, level);
  const messages = [
    { role: "system" as const, content: systemMessage(leveled, "canonical") },
    { role: "user" as const, content: leveled.player_policy.player_turns_he[playerTurnIndex] },
  ];
  const ids = tokenizerOf(session).encodeInput(messages);
  const positions = { end_of_metadata: endOfMetadataPosition(ids), last_prompt_token: ids.length - 1 };
  const captures = await session.researchCaptureActivations(ids, Object.values(positions), sites);
  return captures.map((capture) => ({
    id: `${scenario.scenario_id}:${level}:turn${playerTurnIndex}`,
    scenarioFamily: scenario.scenario_family,
    split: scenario.split,
    level,
    variant: `turn${playerTurnIndex}`,
    position: capture.tokenPosition === positions.end_of_metadata ? "end_of_metadata" : "last_prompt_token",
    site: capture.site,
    values: Array.from(capture.values),
  }));
}

export interface CrossedRow extends CaptureRow {
  style: "A1" | "B2";
  matched: boolean;
  meanLogProb: number;
  sumLogProb: number;
  tokenCount: number;
}

/**
 * Crossed suitability: under requested level `level`, read a reference NPC reply written in `style`;
 * capture at the candidate boundary (last reply token) and record its teacher-forced likelihood.
 */
export async function captureCrossed(
  session: GemmaGenerationSession,
  scenario: CefrScenario,
  level: CefrLevel,
  style: "A1" | "B2",
  reply: string,
  sites: readonly GemmaResearchCaptureSite[],
): Promise<CrossedRow[]> {
  const tokenizer = tokenizerOf(session);
  const leveled = atLevel(scenario, level);
  const prompt = [
    { role: "system" as const, content: systemMessage(leveled, "canonical") },
    { role: "user" as const, content: leveled.player_policy.player_turns_he[0] },
  ];
  const promptIds = tokenizer.encodeInput(prompt);
  const replyIds = tokenizer.encodeText(reply);
  const score = await session.researchScoreContinuation(promptIds, replyIds);
  const ids = [...promptIds, ...replyIds];
  const captures = await session.researchCaptureActivations(ids, [ids.length - 1], sites);
  return captures.map((capture) => ({
    id: `${scenario.scenario_id}:${level}:${style}`,
    scenarioFamily: scenario.scenario_family,
    split: scenario.split,
    level,
    variant: style,
    position: "fixed_candidate_boundary",
    site: capture.site,
    values: Array.from(capture.values),
    style,
    matched: style === level,
    meanLogProb: score.meanLogProb,
    sumLogProb: score.sumLogProb,
    tokenCount: score.tokenCount,
  }));
}

export interface ContrastRow {
  id: string;
  phenomenon: string;
  split: string;
  lexicalGroup: string;
  correctSum: number;
  incorrectSum: number;
  correctMean: number;
  incorrectMean: number;
  correctTokens: number;
  incorrectTokens: number;
  firstDivergenceCorrect: number;
  firstDivergenceIncorrect: number;
  prefersCorrect: boolean;
}

/** Minimal-pair likelihood: same prompt, compare the correct and incorrect Hebrew phrase. */
export async function scoreContrast(
  session: GemmaGenerationSession,
  pair: { id: string; phenomenon: string; split: string; lexical_group: string; context_en: string; prefix_he?: string; correct_he: string; incorrect_he: string },
): Promise<ContrastRow> {
  const tokenizer = tokenizerOf(session);
  const promptIds = tokenizer.encodeInput([
    { role: "user", content: `Say this in natural Hebrew, in a few words: ${pair.context_en}` },
  ]);
  const prefix = pair.prefix_he ? `${pair.prefix_he} ` : "";
  const correct = tokenizer.encodeText(prefix + pair.correct_he);
  const incorrect = tokenizer.encodeText(prefix + pair.incorrect_he);
  let shared = 0;
  while (shared < correct.length && shared < incorrect.length && correct[shared] === incorrect[shared]) shared += 1;
  const a = await session.researchScoreContinuation(promptIds, correct);
  const b = await session.researchScoreContinuation(promptIds, incorrect);
  return {
    id: pair.id,
    phenomenon: pair.phenomenon,
    split: pair.split,
    lexicalGroup: pair.lexical_group,
    correctSum: a.sumLogProb,
    incorrectSum: b.sumLogProb,
    correctMean: a.meanLogProb,
    incorrectMean: b.meanLogProb,
    correctTokens: a.tokenCount,
    incorrectTokens: b.tokenCount,
    firstDivergenceCorrect: a.tokenLogProbs[shared] ?? Number.NaN,
    firstDivergenceIncorrect: b.tokenLogProbs[shared] ?? Number.NaN,
    prefersCorrect: a.sumLogProb > b.sumLogProb,
  };
}

/**
 * Mean activation per site over the tokens of an NPC reply read under a fixed prompt
 * (teacher-forced), used to build contrastive edit directions (A1-style minus B2-style replies).
 */
export async function meanReplyStates(
  session: GemmaGenerationSession,
  scenario: CefrScenario,
  level: CefrLevel,
  reply: string,
  sites: readonly GemmaResearchCaptureSite[],
): Promise<Record<string, number[]>> {
  const tokenizer = tokenizerOf(session);
  const leveled = atLevel(scenario, level);
  const promptIds = tokenizer.encodeInput([
    { role: "system", content: systemMessage(leveled, "canonical") },
    { role: "user", content: leveled.player_policy.player_turns_he[0] },
  ]);
  const replyIds = tokenizer.encodeText(reply);
  const ids = [...promptIds, ...replyIds];
  // States before each reply token (they predict the reply).
  const positions = Array.from({ length: replyIds.length }, (_, index) => promptIds.length - 1 + index);
  const captures = await session.researchCaptureActivations(ids, positions, sites, 1 << 27);
  const out: Record<string, number[]> = {};
  for (const site of sites) {
    const rows = captures.filter((capture) => capture.site === site);
    const key = typeof site === "object" ? `residual${site.residual}` : String(site);
    const sum = new Float64Array(rows[0].values.length);
    for (const row of rows) for (let i = 0; i < sum.length; i += 1) sum[i] += row.values[i];
    out[key] = Array.from(sum, (value) => value / rows.length);
  }
  return out;
}

export interface TrajectoryReread {
  replyTokens: number;
  /** Per site: mean state over reply positions under each prompt policy, and the paired mean difference. */
  canonical: Record<string, number[]>;
  strong: Record<string, number[]>;
}

/**
 * Re-reads one NPC reply (from an episode's turn `turn`, with the episode's earlier turns as history)
 * under the canonical and the level-specific strong system prompt, capturing the residual states that
 * predict each reply token. The reply tokens are identical in both reads, so states are paired.
 */
export async function rereadReply(
  session: GemmaGenerationSession,
  scenario: CefrScenario,
  level: CefrLevel,
  npcTurns: readonly string[],
  turn: number,
  sites: readonly GemmaResearchCaptureSite[],
): Promise<TrajectoryReread> {
  const tokenizer = tokenizerOf(session);
  const leveled = atLevel(scenario, level);
  const replyIds = tokenizer.encodeText(npcTurns[turn]);
  const out: TrajectoryReread = { replyTokens: replyIds.length, canonical: {}, strong: {} };
  for (const policy of ["canonical", "level_specific_strong"] as const) {
    const promptIds = tokenizer.encodeInput(episodeMessages(leveled, policy, npcTurns, turn));
    const ids = [...promptIds, ...replyIds];
    const positions = Array.from({ length: replyIds.length }, (_, index) => promptIds.length - 1 + index);
    const captures = await session.researchCaptureActivations(ids, positions, sites, 1 << 27);
    const target = policy === "canonical" ? out.canonical : out.strong;
    for (const site of sites) {
      const key = typeof site === "object" ? `residual${site.residual}` : String(site);
      const rows = captures.filter((capture) => capture.site === site);
      const sum = new Float64Array(rows[0].values.length);
      for (const row of rows) for (let i = 0; i < sum.length; i += 1) sum[i] += row.values[i];
      target[key] = Array.from(sum, (value) => value / rows.length);
    }
  }
  return out;
}
