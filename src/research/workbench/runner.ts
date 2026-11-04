import type { GemmaGenerationSession } from "../../runtime/gemma-session";
import type { GemmaChatMessage } from "../../runtime/gemma-tokenizer";
import type { GemmaResearchEvaluationRequest, GemmaResearchStage, GemmaResearchOverlay } from "../engine-types";
import { validateWeightSliceRequest, type WeightSliceRequest } from "./weights";
import { checkGrounding, checkVocabulary, promotionVerdict, type GroundingClaim, type JudgeAssessment, type WorldContract } from "./contracts";

type Backend = Pick<GemmaGenerationSession, "researchTokenize" | "researchEvaluate" | "researchInspectWeights" | "destroy">;
interface ConversationRequest {
  id: string;
  messages: GemmaChatMessage[];
  assistantPrefix?: string;
}
export interface ReplayArm {
  id: string;
  label: string;
  overlays?: { layer: number; a: number[]; b: number[]; coefficient: number }[];
}
export interface PairedReplayRequest extends ConversationRequest {
  op: "pairedReplay";
  protocol: "fixed-prefix" | "conversation";
  playerTurns: string[];
  arms: ReplayArm[];
  maxNewTokens: number;
  temperature?: number;
  seed?: number;
}
export type ExperimentRequest =
  | PairedReplayRequest
  | (ConversationRequest & { op: "replay"; maxNewTokens: number; temperature?: number; seed?: number })
  | (ConversationRequest & { op: "score"; continuation: string })
  | (ConversationRequest & { op: "capture"; layers: number[]; positions?: number[]; stages?: GemmaResearchStage[] })
  | (ConversationRequest & {
    op: "patch"; layer: number; tokenPosition?: number; mode: "replace" | "add"; amount: number;
    vector: number[]; sourceArtifactIds: string[]; continuation?: string; maxNewTokens?: number;
    temperature?: number; seed?: number;
    stage?: Exclude<GemmaResearchStage, "layer-input">;
  })
  | { op: "inspectWeights"; id: string; slice: WeightSliceRequest }
  | { op: "compare"; id: string; artifactIds: string[] }
  | {
    op: "assess"; id: string; episodeArtifactId: string; world: WorldContract;
    claims?: GroundingClaim[]; extractionComplete?: boolean; assessments: JudgeAssessment[];
  };

export interface ResearchProvenance {
  revision: string;
  checkpointRevision: string;
  checkpointSha256: string;
  environment: string;
  runId: string;
}
export interface Usage {
  evaluatedPositions: number;
  generatedTokens: number;
  diagnostics: number;
  episodes: number;
}
export interface Artifact {
  schemaVersion: 1;
  id: string;
  request: ExperimentRequest;
  requestHash: string;
  provenance: ResearchProvenance;
  startedAt: string;
  finishedAt?: string;
  status: "running" | "completed" | "failed" | "cancelled";
  usage: Usage;
  result?: unknown;
  resultHash?: string;
  error?: string;
}
export interface ArtifactStore {
  get(id: string): Promise<Artifact | undefined>;
  put(artifact: Artifact): Promise<void>;
  list(): Promise<Artifact[]>;
}
export interface WorkbenchProgress {
  id: string;
  phase: string;
  completed: number;
  total: number | null;
}
export const DEFAULT_LIMITS = {
  evaluatedPositions: 1_000_000, generatedTokens: 180_000, diagnostics: 2_000, episodes: 640,
};
const emptyUsage = (): Usage => ({ evaluatedPositions: 0, generatedTokens: 0, diagnostics: 0, episodes: 0 });

export async function hashJson(value: unknown): Promise<string> {
  const text = JSON.stringify(value);
  if (text === undefined) throw new Error("Cannot hash an undefined artifact");
  const bytes = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.from(new Uint8Array(bytes), (v) => v.toString(16).padStart(2, "0")).join("");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
function numbers(value: unknown): value is number[] {
  return Array.isArray(value) && value.every((v) => typeof v === "number" && Number.isFinite(v));
}
function evaluationResult(artifact: Artifact): {
  text: string; tokenIds: number[]; tokenLogProbs: number[]; promptTokenIds: number[]; stopReason: string;
} {
  const value = artifact.result;
  if (artifact.status !== "completed" || !isRecord(value) || typeof value.text !== "string" ||
      !numbers(value.tokenIds) || !numbers(value.tokenLogProbs) || !numbers(value.promptTokenIds) ||
      typeof value.stopReason !== "string") {
    throw new Error(`Artifact ${artifact.id} is not a completed model evaluation`);
  }
  return {
    text: value.text, tokenIds: value.tokenIds, tokenLogProbs: value.tokenLogProbs,
    promptTokenIds: value.promptTokenIds, stopReason: value.stopReason,
  };
}

function validateRequest(request: ExperimentRequest): void {
  if (!request || typeof request.id !== "string" || !/^[a-zA-Z0-9._:-]{1,120}$/.test(request.id)) {
    throw new Error("Experiment id must be 1..120 safe identifier characters");
  }
  if ("messages" in request) {
    if (!Array.isArray(request.messages) || !request.messages.length ||
        request.messages.some((m) => !["system", "user", "assistant"].includes(m.role) ||
          typeof m.content !== "string" || m.content.length > 20000)) throw new Error("Expected bounded text messages");
    if (request.assistantPrefix !== undefined && typeof request.assistantPrefix !== "string") throw new Error("Invalid assistant prefix");
  }
  switch (request.op) {
    case "pairedReplay":
      if (!["fixed-prefix", "conversation"].includes(request.protocol) || request.assistantPrefix ||
          !Array.isArray(request.playerTurns) || !request.playerTurns.length || request.playerTurns.length > 6 ||
          request.playerTurns.some((t) => typeof t !== "string" || !t.trim() || t.length > 4000) ||
          (request.protocol === "fixed-prefix" && request.playerTurns.length !== 1)) throw new Error("Invalid paired replay protocol or script");
      if (!Array.isArray(request.arms) || request.arms.length < 2 || request.arms.length > 3 ||
          new Set(request.arms.map((a) => a.id)).size !== request.arms.length ||
          request.arms.some((a) => !a.id || typeof a.label !== "string")) throw new Error("Paired replay needs 2..3 uniquely identified arms");
      for (const arm of request.arms) {
        if ((arm.overlays?.length ?? 0) > 4) throw new Error("At most four residual steering sites");
        for (const overlay of arm.overlays ?? []) {
          if (!Number.isInteger(overlay.layer) || overlay.layer < 0 || overlay.layer >= 35 ||
              !numbers(overlay.a) || !numbers(overlay.b) || overlay.a.length !== 1536 || overlay.b.length !== 1536 ||
              !Number.isFinite(overlay.coefficient) || Math.abs(overlay.coefficient) > 1) throw new Error("Invalid paired residual steering overlay");
        }
      }
      if (request.arms[0].overlays?.length) throw new Error("First arm must be an unmodified matched baseline");
      if (!Number.isInteger(request.maxNewTokens) || request.maxNewTokens < 1 || request.maxNewTokens > 128) throw new Error("Invalid paired output budget");
      break;
    case "replay":
      if (!Number.isInteger(request.maxNewTokens) || request.maxNewTokens < 1 || request.maxNewTokens > 128) {
        throw new Error("Workbench reply budget must be 1..128 tokens");
      }
      break;
    case "score":
      if (typeof request.continuation !== "string" || !request.continuation.length) throw new Error("Scoring requires continuation text");
      break;
    case "capture":
      if (!Array.isArray(request.layers) || !request.layers.length || request.layers.length > 8) throw new Error("Capture requires 1..8 layers");
      break;
    case "patch":
      if (!Array.isArray(request.vector) || request.vector.length !== 1536 || !request.vector.every(Number.isFinite) ||
          !Array.isArray(request.sourceArtifactIds) || !request.sourceArtifactIds.length ||
          !request.sourceArtifactIds.every((id) => typeof id === "string")) throw new Error("Patch needs a finite donor vector and provenance artifact IDs");
      if ((request.continuation === undefined) === (request.maxNewTokens === undefined)) {
        throw new Error("Patch must request either forced continuation or free generation");
      }
      if (request.maxNewTokens !== undefined && (!Number.isInteger(request.maxNewTokens) || request.maxNewTokens < 1 || request.maxNewTokens > 128)) {
        throw new Error("Invalid patch generation budget");
      }
      if (request.continuation !== undefined && (typeof request.continuation !== "string" || !request.continuation)) throw new Error("Invalid patch continuation");
      break;
    case "inspectWeights": validateWeightSliceRequest(request.slice); break;
    case "compare":
      if (!Array.isArray(request.artifactIds) || request.artifactIds.length < 2 || request.artifactIds.length > 32) throw new Error("Compare needs 2..32 artifacts");
      break;
    case "assess": break;
    default: throw new Error("Unknown experiment operation");
  }
}

export class ResearchWorkbench {
  private readonly backend: Backend;
  private readonly store: ArtifactStore;
  private readonly provenance: ResearchProvenance;
  private readonly limits: Usage;
  private readonly onProgress: (progress: WorkbenchProgress) => void;
  private active: AbortController | null = null;
  private tail: Promise<void> = Promise.resolve();
  private paused = false;
  private closed = false;
  private resumeWaiters: (() => void)[] = [];
  private used: Usage = emptyUsage();

  constructor(
    backend: Backend,
    store: ArtifactStore,
    provenance: ResearchProvenance,
    limits: Usage = DEFAULT_LIMITS,
    onProgress: (progress: WorkbenchProgress) => void = () => {},
  ) {
    if (!provenance.revision || !provenance.runId || !/^[0-9a-f]{64}$/.test(provenance.checkpointSha256)) throw new Error("Missing research provenance");
    for (const value of Object.values(limits)) if (!Number.isSafeInteger(value) || value < 1) throw new Error("Invalid research budget");
    this.backend = backend;
    this.store = store;
    this.provenance = { ...provenance };
    this.limits = { ...limits };
    this.onProgress = onProgress;
  }

  async restoreUsage(): Promise<void> {
    if (this.active) throw new Error("Cannot restore budgets during a job");
    const records = (await this.store.list()).filter((a) => a.provenance.runId === this.provenance.runId);
    this.used = emptyUsage();
    for (const artifact of records) {
      if (artifact.status === "running") {
        artifact.status = "failed";
        artifact.error = "Interrupted: incomplete output cannot be resumed as a successful job";
        artifact.finishedAt = new Date().toISOString();
        await this.store.put(artifact);
      }
      for (const key of Object.keys(this.used) as (keyof Usage)[]) this.used[key] += artifact.usage[key];
    }
  }

  get usage(): Usage { return { ...this.used }; }
  get isPaused(): boolean { return this.paused; }
  pause(): void { this.paused = true; }
  resume(): void {
    this.paused = false;
    for (const resolve of this.resumeWaiters.splice(0)) resolve();
  }
  cancel(): void {
    this.active?.abort(new DOMException("Research job cancelled", "AbortError"));
  }
  async close(): Promise<void> {
    this.closed = true;
    this.cancel();
    // Closing must release paused waiters so they reject as closed, not execute.
    this.resume();
    await this.tail;
    this.backend.destroy();
  }
  async artifacts(): Promise<Artifact[]> { return this.store.list(); }
  async artifact(id: string): Promise<Artifact> {
    const artifact = await this.store.get(id);
    if (!artifact) throw new Error(`Unknown artifact ${id}`);
    return artifact;
  }

  run(input: ExperimentRequest): Promise<Artifact> {
    // Snapshot queued requests so callers cannot change an experiment while another one runs.
    const request = structuredClone(input);
    const result = this.tail.then(async () => {
      if (this.paused) await new Promise<void>((resolve) => this.resumeWaiters.push(resolve));
      if (this.closed) throw new Error("Research workbench is closed");
      return this.execute(request);
    });
    // Keep the queue usable after a failed job; the returned promise still rejects.
    this.tail = result.then(() => {}, () => {});
    return result;
  }

  private reserve(usage: Usage): void {
    for (const key of Object.keys(usage) as (keyof Usage)[]) {
      if (this.used[key] + usage[key] > this.limits[key]) throw new Error(`Research budget exhausted: ${key}`);
    }
  }

  private async execute(request: ExperimentRequest): Promise<Artifact> {
    validateRequest(request);
    const requestHash = await hashJson({ request, provenance: this.provenance });
    const id = `${this.provenance.runId}:${request.id}`;
    const existing = await this.store.get(id);
    if (existing) {
      if (existing.requestHash !== requestHash) throw new Error("Job id reused with different request or source revision");
      if (existing.status !== "completed") throw new Error(`Prior job ${id} is ${existing.status}; use an explicit new attempt id`);
      return existing;
    }
    const usage = emptyUsage();
    const artifact: Artifact = {
      schemaVersion: 1, id, request, requestHash, provenance: { ...this.provenance },
      startedAt: new Date().toISOString(), status: "running", usage,
    };
    const controller = new AbortController();
    this.reserve({ ...emptyUsage(), diagnostics: 1 });
    this.active = controller;
    this.used.diagnostics += 1;
    usage.diagnostics = 1;
    try {
      await this.store.put(artifact);
      let result: unknown;
      if (request.op === "pairedReplay") {
        const histories = request.arms.map(() => structuredClone(request.messages));
        const arms = request.arms.map((arm) => ({
          id: arm.id, label: arm.label, intervention: arm.overlays?.length ? "historical-residual-steering" : "none",
          turns: [] as { player: string; text: string; tokenIds: number[]; tokenLogProbs: number[]; stopReason: string; promptHash: string; milliseconds: number }[],
        }));
        for (let turn = 0; turn < request.playerTurns.length; turn += 1) {
          for (let index = 0; index < request.arms.length; index += 1) {
            if (controller.signal.aborted) throw controller.signal.reason;
            const history = histories[index];
            history.push({ role: "user", content: request.playerTurns[turn] });
            const { promptTokenIds } = this.backend.researchTokenize(history);
            const upperPositions = promptTokenIds.length + request.maxNewTokens - 1;
            this.reserve({ evaluatedPositions: upperPositions, generatedTokens: request.maxNewTokens, episodes: 1, diagnostics: 0 });
            this.used.episodes += 1;
            usage.episodes += 1;
            const previous = { ...usage };
            await this.store.put({ ...artifact, usage: {
              ...usage, evaluatedPositions: previous.evaluatedPositions + upperPositions,
              generatedTokens: previous.generatedTokens + request.maxNewTokens,
            } });
            const overlays: GemmaResearchOverlay[] = (request.arms[index].overlays ?? []).map((o) => ({
              site: { layer: o.layer }, a: Float32Array.from(o.a), b: Float32Array.from(o.b), coefficient: o.coefficient,
            }));
            const evaluated = await this.backend.researchEvaluate({
              promptTokenIds, overlays,
              generate: { maxNewTokens: request.maxNewTokens, temperature: request.temperature ?? 0, seed: (request.seed ?? 23) + turn },
              signal: controller.signal,
              onProgress: (progress) => {
                const nextPositions = previous.evaluatedPositions + progress.evaluatedPositions;
                const nextTokens = previous.generatedTokens + progress.generatedTokens;
                this.used.evaluatedPositions += nextPositions - usage.evaluatedPositions;
                this.used.generatedTokens += nextTokens - usage.generatedTokens;
                usage.evaluatedPositions = nextPositions;
                usage.generatedTokens = nextTokens;
                this.onProgress({ id, phase: `arm ${index + 1}/${arms.length}, turn ${turn + 1}/${request.playerTurns.length}`, completed: progress.evaluatedPositions, total: upperPositions });
              },
            });
            arms[index].turns.push({
              player: request.playerTurns[turn], text: evaluated.text, tokenIds: evaluated.tokenIds,
              tokenLogProbs: evaluated.tokenLogProbs, stopReason: evaluated.stopReason,
              promptHash: await hashJson(promptTokenIds), milliseconds: evaluated.milliseconds,
            });
            history.push({ role: "assistant", content: evaluated.text });
            // Partial paired results survive cancellation and are explicitly not a completed comparison.
            artifact.result = { kind: "paired-replay", protocol: request.protocol, arms, complete: false };
            await this.store.put(artifact);
          }
        }
        result = {
          kind: "paired-replay", protocol: request.protocol, arms, complete: true,
          caveat: request.protocol === "conversation"
            ? "Same player script and per-turn seeds; each arm keeps its own NPC history. Later prompts intentionally diverge."
            : "Same frozen prefix, player turn, decoding, seed and full-logit path. Residual steering is not a native weight repair.",
          automatedQualityJudgment: "not performed",
        };
      } else if (request.op === "inspectWeights") {
        result = await this.backend.researchInspectWeights(request.slice, controller.signal);
      } else if (request.op === "compare") {
        const artifacts = await Promise.all(request.artifactIds.map((key) => this.artifact(key)));
        const evaluations = artifacts.map(evaluationResult);
        result = {
          sameTokenPrefix: evaluations.every((e) => JSON.stringify(e.promptTokenIds) === JSON.stringify(evaluations[0].promptTokenIds)),
          rows: evaluations.map((e, i) => {
            const request = artifacts[i].request;
            const paired = JSON.stringify(e.promptTokenIds) === JSON.stringify(evaluations[0].promptTokenIds) &&
              e.tokenIds[0] === evaluations[0].tokenIds[0] && e.tokenLogProbs.length > 0;
            return {
              artifactId: artifacts[i].id, text: e.text, tokenIds: e.tokenIds,
              tokenCount: e.tokenLogProbs.length,
              sumLogProbability: e.tokenLogProbs.reduce((a, b) => a + b, 0),
              firstTokenLogProbability: e.tokenLogProbs[0] ?? null,
              deltaFirstTokenLogProbability: paired ? e.tokenLogProbs[0] - evaluations[0].tokenLogProbs[0] : null,
              intervention: request.op === "patch" ? {
                layer: request.layer, position: request.tokenPosition ?? e.promptTokenIds.length - 1, mode: request.mode,
              } : null,
              stopReason: e.stopReason,
            };
          }),
          caveat: "Different prefixes are observational contrasts, not identical-context patch effects; sequence totals also depend on tokenization and length.",
        };
      } else if (request.op === "assess") {
        const episode = evaluationResult(await this.artifact(request.episodeArtifactId));
        const grounding = checkGrounding(request.world, episode.text, request.claims);
        const vocabulary = checkVocabulary(request.world, episode.text);
        if (grounding.status === "pass" && request.extractionComplete !== true) grounding.status = "unknown";
        result = {
          grounding, vocabulary,
          verdict: promotionVerdict(request.assessments, request.episodeArtifactId, episode.text, [grounding.status, vocabulary.status]),
          assessments: request.assessments,
        };
      } else {
        const continuation = request.op === "score" || request.op === "patch" ? request.continuation ?? "" : "";
        const tokens = this.backend.researchTokenize(request.messages, request.assistantPrefix ?? "", continuation);
        const maxNew = request.op === "replay" || request.op === "patch" ? request.maxNewTokens ?? 0 : 0;
        const maxPositions = tokens.promptTokenIds.length + Math.max(0, (maxNew || tokens.continuationTokenIds.length) - 1);
        this.reserve({ evaluatedPositions: maxPositions, generatedTokens: maxNew, episodes: maxNew ? 1 : 0, diagnostics: 0 });
        if (maxNew) { usage.episodes = 1; this.used.episodes += 1; }
        if (request.op === "patch") {
          for (const sourceId of request.sourceArtifactIds) {
            if ((await this.artifact(sourceId)).status !== "completed") throw new Error("Patch source is not completed");
          }
        }
        this.onProgress({ id, phase: "evaluating", completed: 0, total: maxPositions });
        const evaluation: GemmaResearchEvaluationRequest = {
          promptTokenIds: tokens.promptTokenIds,
          continuationTokenIds: tokens.continuationTokenIds,
          signal: controller.signal,
          onProgress: (progress) => {
            this.used.evaluatedPositions += progress.evaluatedPositions - usage.evaluatedPositions;
            this.used.generatedTokens += progress.generatedTokens - usage.generatedTokens;
            usage.evaluatedPositions = progress.evaluatedPositions;
            usage.generatedTokens = progress.generatedTokens;
            this.onProgress({ id, phase: "evaluating", completed: progress.evaluatedPositions, total: maxPositions });
          },
        };
        if (maxNew && (request.op === "replay" || request.op === "patch")) {
          evaluation.generate = { maxNewTokens: maxNew, temperature: request.temperature ?? 0, seed: request.seed ?? 23 };
        }
        if (request.op === "capture") evaluation.capture = {
          layers: request.layers, positions: request.positions ?? [tokens.promptTokenIds.length - 1],
          stages: request.stages,
        };
        if (request.op === "patch") evaluation.patch = {
          layer: request.layer, tokenPosition: request.tokenPosition ?? tokens.promptTokenIds.length - 1,
          vector: Float32Array.from(request.vector), mode: request.mode, amount: request.amount,
          stage: request.stage,
        };
        // Reserve worst-case work durably. A crashed tab conservatively consumes the reservation.
        await this.store.put({ ...artifact, usage: { ...usage, evaluatedPositions: maxPositions, generatedTokens: maxNew } });
        const measured = await this.backend.researchEvaluate(evaluation);
        result = {
          ...measured,
          assistantPrefix: request.assistantPrefix ?? "",
          fullAssistantText: (request.assistantPrefix ?? "") + measured.text,
          promptPieces: tokens.pieces.slice(0, tokens.promptTokenIds.length),
          continuationPieces: tokens.pieces.slice(tokens.promptTokenIds.length),
          captures: measured.captures.map((capture) => ({
            ...capture, dtype: "f32", shape: [1536],
            semantics: `residual at ${capture.stage ?? "layer-output"}, after any patch at that boundary; not an isolated component output`,
            values: Array.from(capture.values),
          })),
        };
      }
      artifact.status = "completed";
      artifact.result = result;
      artifact.resultHash = await hashJson(result);
      artifact.finishedAt = new Date().toISOString();
      await this.store.put(artifact);
      this.onProgress({ id, phase: "completed", completed: 1, total: 1 });
      return artifact;
    } catch (error) {
      artifact.status = controller.signal.aborted ? "cancelled" : "failed";
      artifact.error = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
      artifact.finishedAt = new Date().toISOString();
      await this.store.put(artifact);
      this.onProgress({ id, phase: artifact.status, completed: usage.evaluatedPositions, total: null });
      throw error;
    } finally {
      this.active = null;
    }
  }
}

export class IndexedArtifactStore implements ArtifactStore {
  private readonly database: Promise<IDBDatabase>;
  constructor(name = "gemma-mechanistic-workbench-v1") {
    this.database = new Promise((resolve, reject) => {
      const request = indexedDB.open(name, 1);
      request.onupgradeneeded = () => request.result.createObjectStore("artifacts", { keyPath: "id" });
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
      request.onblocked = () => reject(new Error("Research artifact database upgrade blocked by another tab"));
    });
  }
  private async transact<T>(mode: IDBTransactionMode, operation: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
    const db = await this.database;
    return new Promise((resolve, reject) => {
      const tx = db.transaction("artifacts", mode);
      const request = operation(tx.objectStore("artifacts"));
      tx.oncomplete = () => resolve(request.result);
      tx.onerror = () => reject(tx.error ?? request.error);
      tx.onabort = () => reject(tx.error ?? new Error("Research artifact transaction aborted"));
    });
  }
  get(id: string): Promise<Artifact | undefined> { return this.transact("readonly", (store) => store.get(id)); }
  async put(artifact: Artifact): Promise<void> { await this.transact("readwrite", (store) => store.put(artifact)); }
  list(): Promise<Artifact[]> { return this.transact("readonly", (store) => store.getAll()); }
}

/** Acknowledge external persistence before a job can reach model evaluation. */
export class DurableArtifactStore implements ArtifactStore {
  private readonly local: ArtifactStore;
  private readonly persist: (artifact: Artifact) => Promise<void>;
  constructor(local: ArtifactStore, persist: (artifact: Artifact) => Promise<void>) {
    this.local = local;
    this.persist = persist;
  }
  get(id: string): Promise<Artifact | undefined> { return this.local.get(id); }
  list(): Promise<Artifact[]> { return this.local.list(); }
  async put(artifact: Artifact): Promise<void> {
    const snapshot = structuredClone(artifact);
    await this.persist(snapshot);
    await this.local.put(snapshot);
  }
}
