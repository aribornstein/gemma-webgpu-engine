import "./style.css";
import { loadGemmaGenerationSession } from "../../runtime/gemma-session";
import { DurableArtifactStore, hashJson, IndexedArtifactStore, ResearchWorkbench, type Artifact, type ExperimentRequest, type ResearchProvenance } from "./runner";

const root = document.querySelector<HTMLDivElement>("#workbench");
if (!root) throw new Error("Missing workbench root");
root.innerHTML = `
<header><div><p class="eyebrow">GEMMA / RESEARCH</p><h1>Mechanism-first workbench</h1>
<p>Inspect a failure. Test a cause. Preserve what works.</p></div><a href="./index.html">Generation console</a></header>
<section class="notice"><strong>Research only.</strong> Activation patches are diagnostics, not weight repairs.
No candidate is promoted without every required criterion at 9/10, preservation tests, and fresh evaluation.</section>
<section class="toolbar" aria-label="Session controls">
<label>Weights<select id="source"><option value="local">Local pinned checkpoint</option><option value="cache">Existing browser cache</option></select></label>
<label>Run ID<input id="run-id" value="interactive" pattern="[a-zA-Z0-9._-]+"></label>
<button id="load">Load research session</button><button id="pause" disabled>Pause queue</button>
<button id="resume" disabled>Resume</button><button id="cancel" disabled>Cancel active job</button>
<button id="export">Export evidence</button><label class="file-label">Open saved evidence<input id="import" type="file" accept=".json,.jsonl"></label>
</section>
<p id="status" role="status" aria-live="polite">No model loaded. Saved evidence can be viewed without a GPU.</p>
<progress id="progress" max="1" value="0" aria-label="Experiment progress"></progress>
<main><section class="request-panel"><h2>Experiment</h2>
<label>Operation<select id="operation">
<option value="replay">Replay conversation</option><option value="score">Score continuation</option>
<option value="capture">Capture decision states</option><option value="inspectWeights">Inspect selected weights</option>
<option value="patch">Token-local activation patch</option><option value="compare">Compare artifacts</option>
</select></label>
<p class="hint">The UI and agents submit the same bounded JSON requests. A patch must cite its donor artifacts. No checkpoint bytes are modified.</p>
<label for="request">Request JSON</label><textarea id="request" spellcheck="false" rows="20"></textarea>
<button id="run" disabled>Run experiment</button>
<h2>Run ledger</h2><p id="usage">No model evaluations.</p><ol id="ledger"></ol></section>
<section class="inspector"><h2>Evidence inspector</h2><p id="artifact-title">Select an artifact from the ledger.</p>
<div id="answer" dir="auto"></div><div id="decision"></div><div id="captures"></div>
<details><summary>Request and provenance</summary><pre id="metadata"></pre></details>
<details><summary>Bounded result preview</summary><pre id="details"></pre></details>
</section></main>`;

function element<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw new Error(`Missing workbench element ${id}`);
  return found as T;
}
const status = element<HTMLParagraphElement>("status");
const progress = element<HTMLProgressElement>("progress");
const editor = element<HTMLTextAreaElement>("request");
const runButton = element<HTMLButtonElement>("run");
const loadButton = element<HTMLButtonElement>("load");
let controller: ResearchWorkbench | null = null;
let records: Artifact[] = [];
let busy = false;
let lastEvent = "idle";
let activeStarted = 0;
let loadEpoch = 0;
let pendingLoad: Promise<void> | null = null;
let closing = false;
const store = new IndexedArtifactStore();
const messages = [
  { role: "system", content: "You are a Hebrew NPC. Two apples cost 10 ILS in total (5 ILS each). Payment has not occurred. Reply naturally in Hebrew." },
  { role: "user", content: "אני רוצה שתי תפוחים. כמה הם עולים?" },
];
let nextId = 1;
function example(op: string): unknown {
  const id = `interactive-${nextId++}`;
  switch (op) {
    case "replay": return { id, op, messages, maxNewTokens: 48, temperature: 0, seed: 23 };
    case "score": return { id, op, messages, continuation: "שני תפוחים עולים עשרה שקלים." };
    case "capture": return { id, op, messages, layers: [12, 20, 28, 34] };
    case "inspectWeights": return { id, op, slice: { layer: 20, projection: "down", rowStart: 0, rowCount: 4, columnStart: 0, columnCount: 32 } };
    case "patch": return { id, op, messages, layer: 20, amount: 1, mode: "replace", vector: [], sourceArtifactIds: [], continuation: "שני תפוחים עולים עשרה שקלים." };
    case "compare": return { id, op, artifactIds: records.filter((r) => r.status === "completed").slice(-2).map((r) => r.id) };
    default: throw new Error("Unsupported example operation");
  }
}
editor.value = JSON.stringify(example("replay"), null, 2);
element<HTMLSelectElement>("operation").addEventListener("change", (event) => {
  editor.value = JSON.stringify(example((event.target as HTMLSelectElement).value), null, 2);
});
function fail(error: unknown): void {
  status.dataset.error = "true";
  status.textContent = error instanceof Error ? error.message : String(error);
}
function refreshLedger(): void {
  const ledger = element<HTMLOListElement>("ledger");
  ledger.replaceChildren();
  for (const artifact of records.toReversed()) {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.textContent = `${artifact.status} · ${artifact.request.op} · ${artifact.request.id}`;
    button.addEventListener("click", () => inspect(artifact));
    item.append(button);
    ledger.append(item);
  }
  if (controller) element("usage").textContent = JSON.stringify(controller.usage);
}
function object(value: unknown): value is Record<string, unknown> { return typeof value === "object" && value !== null; }
function table(container: HTMLElement, headers: string[], rows: string[][]): void {
  container.replaceChildren();
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const top = document.createElement("tr");
  for (const header of headers) { const th = document.createElement("th"); th.textContent = header; top.append(th); }
  thead.append(top);
  table.append(thead);
  const body = document.createElement("tbody");
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const text of row) { const td = document.createElement("td"); td.textContent = text; tr.append(td); }
    body.append(tr);
  }
  table.append(body);
  container.append(table);
}
function inspect(artifact: Artifact): void {
  element("artifact-title").textContent = `${artifact.id} · ${artifact.status}`;
  element("metadata").textContent = JSON.stringify({ request: artifact.request, provenance: artifact.provenance, usage: artifact.usage, requestHash: artifact.requestHash }, null, 2);
  const result = artifact.result;
  element("answer").textContent = object(result) && typeof result.fullAssistantText === "string"
    ? result.fullAssistantText : object(result) && typeof result.text === "string" ? result.text : artifact.error ?? "";
  element("decision").replaceChildren();
  element("captures").replaceChildren();
  if (object(result) && Array.isArray(result.firstDecision)) {
    table(element("decision"), ["Next token", "Log probability", "Raw logit"], result.firstDecision.filter(object).map((row) =>
      [String(row.text), Number(row.logProbability).toFixed(4), Number(row.rawLogit).toFixed(4)]));
  }
  if (object(result) && Array.isArray(result.captures)) {
    table(element("captures"), ["Site", "Position", "Values", "L2 norm"], result.captures.filter(object).map((row) => {
      const values = Array.isArray(row.values) ? row.values : [];
      return [JSON.stringify(row.site), String(row.tokenPosition), String(values.length), Math.sqrt(values.reduce((sum, v) => sum + Number(v) ** 2, 0)).toFixed(4)];
    }));
    for (const capture of result.captures.filter(object)) {
      if (!object(capture.site) || typeof capture.site.residual !== "number" || !Array.isArray(capture.values)) continue;
      const layer = capture.site.residual;
      const button = document.createElement("button");
      button.textContent = `Use layer ${capture.site.residual} as diagnostic donor`;
      button.addEventListener("click", () => {
        editor.value = JSON.stringify({
          id: `patch-${nextId++}`, op: "patch", messages, layer,
          mode: "replace", amount: 1, vector: capture.values, sourceArtifactIds: [artifact.id],
          continuation: "שני תפוחים עולים עשרה שקלים.",
        }, null, 2);
      });
      element("captures").append(button);
    }
  }
  if (object(result) && Array.isArray(result.rows)) {
    table(element("decision"), ["Artifact", "Text", "First token log p", "Total log p"], result.rows.filter(object).map((row) =>
      [String(row.artifactId), String(row.text), String(row.firstTokenLogProbability), String(row.sumLogProbability)]));
    const map = document.createElement("div");
    map.className = "causal-map";
    for (const row of result.rows.filter(object)) {
      if (!object(row.intervention) || typeof row.deltaFirstTokenLogProbability !== "number") continue;
      const cell = document.createElement("div");
      const delta = row.deltaFirstTokenLogProbability;
      cell.textContent = `L${row.intervention.layer} @ token ${row.intervention.position}: ${delta >= 0 ? "+" : ""}${delta.toFixed(3)} log p`;
      cell.title = String(row.artifactId);
      cell.style.borderColor = delta > 0.01 ? "#77d7c6" : delta < -0.01 ? "#ffb6a4" : "#56717f";
      map.append(cell);
    }
    const note = document.createElement("p");
    note.className = "hint";
    note.textContent = "Patch map: change from the first artifact, only for identical prefixes and first target tokens. Likelihood changes are not proof of useful or regression-free repair.";
    element("captures").append(note, map);
  }
  if (object(result) && Array.isArray(result.values) && typeof result.tensor === "string") {
    table(element("captures"), ["Tensor", "Shape", "Bits", "Source bytes"], [[result.tensor, JSON.stringify(result.shape), String(result.packedBits), String(result.bytesRead)]]);
  }
  element("details").textContent = JSON.stringify(result, null, 2)?.slice(0, 18000) ?? "";
}

function load(provenance?: ResearchProvenance, restored: Artifact[] = [], durable = false): Promise<void> {
  if (controller || pendingLoad || busy || closing) return Promise.reject(new Error("Research session already loaded, loading or closing"));
  const epoch = ++loadEpoch;
  const task = loadSession(epoch, provenance, restored, durable).finally(() => {
    if (pendingLoad === task) pendingLoad = null;
  });
  pendingLoad = task;
  return task;
}

async function loadSession(epoch: number, provenance?: ResearchProvenance, restored: Artifact[] = [], durable = false): Promise<void> {
  const checkCancelled = () => {
    if (epoch !== loadEpoch) throw new DOMException("Research session loading was cancelled", "AbortError");
  };
  loadButton.disabled = true;
  busy = true;
  activeStarted = Date.now();
  let loadingSession: Awaited<ReturnType<typeof loadGemmaGenerationSession>> | null = null;
  status.dataset.error = "false";
  status.textContent = "Loading pinned research session...";
  progress.removeAttribute("value");
  try {
    const persist = window.persistResearchArtifact;
    if (durable && !persist) throw new Error("Durable CLI store binding is unavailable");
    const artifactStore = durable && persist ? new DurableArtifactStore(store, persist) : store;
    for (const artifact of restored) {
      if (!provenance || artifact.provenance.runId !== provenance.runId ||
          artifact.provenance.revision !== provenance.revision ||
          artifact.requestHash !== await hashJson({ request: artifact.request, provenance: artifact.provenance }) ||
          (artifact.status === "completed" && artifact.resultHash !== await hashJson(artifact.result))) {
        throw new Error("Cannot resume altered artifacts or a different source revision");
      }
      checkCancelled();
      await artifactStore.put(artifact);
    }
    checkCancelled();
    const session = await loadGemmaGenerationSession({
      sourceUrl: element<HTMLSelectElement>("source").value === "local" ? new URL("models/gemma-4-e2b/model.safetensors", document.baseURI).href : undefined,
      cacheCapacity: 2048, prefillStrategy: "sequential",
    });
    loadingSession = session;
    checkCancelled();
    controller = new ResearchWorkbench(session, artifactStore, provenance ?? {
      revision: import.meta.env.VITE_RESEARCH_REVISION ?? "interactive-unverified-source",
      checkpointRevision: "9fcec64df66cb1e4d972fc5cdc142afb25b2362c",
      checkpointSha256: "efab429012b97ab986c4d4838a46ff3ad95d618b42ce514771ca40fadc76a9a4",
      environment: navigator.userAgent,
      runId: element<HTMLInputElement>("run-id").value,
    }, undefined, (event) => {
      lastEvent = `${event.phase}: ${event.id}`;
      progress.max = event.total ?? 1;
      if (event.total === null) progress.removeAttribute("value"); else progress.value = event.completed;
      status.textContent = `${lastEvent} ${event.completed}/${event.total ?? "?"}`;
      window.dispatchEvent(new CustomEvent("research-progress", { detail: event }));
    });
    await controller.restoreUsage();
    checkCancelled();
    records = await controller.artifacts();
    checkCancelled();
    refreshLedger();
    status.textContent = "Research session ready · sequential prefill · matched full-logit selection";
    runButton.disabled = false;
    for (const id of ["pause", "resume", "cancel"]) element<HTMLButtonElement>(id).disabled = false;
  } catch (error) {
    if (controller) await controller.close(); else loadingSession?.destroy();
    controller = null;
    throw error;
  } finally { busy = false; }
}
async function run(request: ExperimentRequest): Promise<Artifact> {
  if (!controller || pendingLoad || closing) throw new Error("Load a research session first");
  const activeController = controller;
  busy = true;
  activeStarted = Date.now();
  runButton.disabled = true;
  status.dataset.error = "false";
  try {
    const artifact = await activeController.run(request);
    inspect(artifact);
    return artifact;
  } catch (error) {
    fail(error);
    throw error;
  } finally {
    busy = false;
    runButton.disabled = closing || controller !== activeController;
    records = await activeController.artifacts();
    refreshLedger();
  }
}
loadButton.addEventListener("click", () => { void load().catch((error) => { loadButton.disabled = false; fail(error); }); });
runButton.addEventListener("click", () => {
  try { void run(JSON.parse(editor.value)).catch(fail); } catch (error) { fail(error); }
});
element("pause").addEventListener("click", () => { controller?.pause(); status.textContent = "Queue paused after the active job"; });
element("resume").addEventListener("click", () => { controller?.resume(); status.textContent = "Queue resumed"; });
element("cancel").addEventListener("click", () => controller?.cancel());
element("export").addEventListener("click", () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(records)], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url; link.download = "research-artifacts.json"; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
element<HTMLInputElement>("import").addEventListener("change", async (event) => {
  try {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (!file) return;
    const text = await file.text();
    const parsed: unknown = text.trim().startsWith("[") ? JSON.parse(text) : text.trim().split("\n").map((line) => JSON.parse(line));
    if (!Array.isArray(parsed) || parsed.some((v) => !object(v) || v.schemaVersion !== 1 || typeof v.id !== "string" || !object(v.request))) {
      throw new Error("Expected exported research artifacts");
    }
    // Imported evidence is display-only, never trusted as a live intervention source.
    records = parsed as Artifact[];
    refreshLedger();
    status.textContent = `Opened ${records.length} saved artifacts (display-only)`;
  } catch (error) { fail(error); }
});
setInterval(() => {
  if (busy) console.info(`HEARTBEAT research ${Math.floor((Date.now() - activeStarted) / 1000)}s ${lastEvent}`);
}, 15000);

declare global {
  interface Window {
    persistResearchArtifact?: (artifact: Artifact) => Promise<void>;
    researchWorkbench: {
      load: typeof load; run: typeof run; pause(): void; resume(): void; cancel(): void;
      artifacts(): Promise<Artifact[]>; close(): Promise<void>;
    };
  }
}
window.researchWorkbench = {
  load, run,
  pause: () => controller?.pause(), resume: () => controller?.resume(), cancel: () => controller?.cancel(),
  artifacts: () => controller ? controller.artifacts() : Promise.resolve(records),
  close: async () => {
    closing = true;
    loadEpoch += 1;
    try {
      if (pendingLoad) {
        try { await pendingLoad; } catch (error) {
          if (!(error instanceof DOMException) || error.name !== "AbortError") throw error;
        }
      } else {
        await controller?.close();
      }
    } finally {
      controller = null;
      closing = false;
      runButton.disabled = true;
      loadButton.disabled = false;
      for (const id of ["pause", "resume", "cancel"]) element<HTMLButtonElement>(id).disabled = true;
      status.textContent = "Research session closed";
    }
  },
};
