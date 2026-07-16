import "./style.css";
import {
  GEMMA_4_E2B_CACHE_SPEC,
  ReadonlySafetensorsCache,
} from "./model/cached-safetensors";
import {
  compileGenerationConstraint,
  type GenerationConstraint,
  type JsonWhitespace,
} from "./runtime/constraints";
import {
  DEFAULT_GENERATION_CONFIG,
  resolveGemmaGenerationConfig,
  type GemmaGenerationOptions,
} from "./runtime/generation-config";
import { calculateGemmaGenerationThroughput } from "./runtime/generation-throughput";
import {
  availableGemmaOutputTokens,
  GEMMA_MODEL_CONTEXT_CAPACITY,
  GEMMA_VALIDATED_CONTEXT_CAPACITY,
} from "./runtime/gemma-context";
import type {
  GemmaGenerationSession,
  GemmaGenerationTiming,
} from "./runtime/gemma-session";
import type { GemmaGenerationInput } from "./runtime/gemma-tokenizer";
import type { GemmaDurableBenchmarkArtifact } from "./runtime/durable-benchmark";

interface GemmaCacheInitializationProgress {
  status?: string;
  kind?: string;
  fraction?: number;
  loaded?: number;
  total?: number;
  fromCache?: boolean;
  message?: string;
}

type GenerationExampleControl =
  | "temperature"
  | "maxNewTokens"
  | "topK"
  | "topP"
  | "minP"
  | "typicalP"
  | "repetitionPenalty"
  | "repetitionWindow"
  | "frequencyPenalty"
  | "presencePenalty"
  | "seed";

interface GenerationExample {
  id: string;
  label: string;
  prompt: string;
  controls: Partial<Record<GenerationExampleControl, number>>;
  image?: {
    url: string;
    filename: string;
  };
  constraint?: GenerationConstraint;
  expandProbabilityControls?: boolean;
}

const GENERATION_EXAMPLES: readonly GenerationExample[] = [
  {
    id: "vision-dolphin-caption",
    label: "Dolphin caption · Vision",
    prompt: "Transcribe the printed caption in this image as accurately as possible, including the credit line. Then briefly describe whether the photograph matches the caption. Mark any unreadable words as [unclear] and do not use outside knowledge.",
    controls: { temperature: 0, maxNewTokens: 256 },
    image: {
      url: "/examples/dolphin_capt_image.png",
      filename: "dolphin_capt_image.png",
    },
  },
  {
    id: "vision-gottingen",
    label: "Göttingen mathematicians · Vision",
    prompt: "Describe this historical group photograph. Read the printed caption in the image and use only text you can actually discern there to identify people by row. Quote uncertain spellings as uncertain, and do not infer identities from faces or outside knowledge.",
    controls: { temperature: 0, maxNewTokens: 256 },
    image: {
      url: "/examples/the-mathematics-club-of-gottingen-1902.jpg",
      filename: "the-mathematics-club-of-gottingen-1902.jpg",
    },
  },
  {
    id: "greedy-colors",
    label: "Primary colors · Greedy",
    prompt: "Name the three primary colors in one short sentence.",
    controls: { temperature: 0, maxNewTokens: 32 },
  },
  {
    id: "sampling-tagline",
    label: "Pocket observatory · Sampling",
    prompt: "Write one vivid, playful sentence describing a pocket-sized observatory.",
    controls: {
      temperature: 0.8,
      maxNewTokens: 48,
      topK: 40,
      topP: 0.9,
      minP: 0.05,
      typicalP: 0.95,
      repetitionPenalty: 1.08,
      seed: 7,
    },
    expandProbabilityControls: true,
  },
  {
    id: "regex-sky",
    label: "Sky color · Regex",
    prompt: "Return exactly one lowercase word for the color of a clear daytime sky.",
    controls: { temperature: 0, maxNewTokens: 4 },
    constraint: { type: "regex", pattern: "(?:blue|gray)" },
  },
  {
    id: "json-city",
    label: "Paris facts · JSON",
    prompt: "Return one compact JSON object about Paris with keys city, country, and landmarks, where landmarks is an array of exactly two strings. Return JSON only.",
    controls: { temperature: 0, maxNewTokens: 64 },
    constraint: { type: "json", maxDepth: 3, whitespace: "compact" },
  },
  {
    id: "schema-triage",
    label: "Support triage · Schema",
    prompt: "Classify this support request: The app crashes after sign-in. Return JSON with exactly these keys in this order: category, urgent, summary.",
    controls: { temperature: 0, maxNewTokens: 64 },
    constraint: {
      type: "json-schema",
      maxDepth: 3,
      whitespace: "compact",
      schema: {
        type: "object",
        properties: {
          category: { enum: ["bug", "billing", "account"] },
          urgent: { type: "boolean" },
          summary: { type: "string" },
        },
        required: ["category", "urgent", "summary"],
        additionalProperties: false,
      },
    },
  },
];

declare global {
  interface Window {
    __gemmaEngineCacheInitializer?: (
      onProgress: (progress: GemmaCacheInitializationProgress) => void,
    ) => Promise<void>;
  }
}

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("Missing #app");

app.innerHTML = `
  <header class="topbar">
    <div class="brand-lockup">
      <span class="brand-mark" aria-hidden="true">G4</span>
      <div>
        <p class="eyebrow">GEMMA 4 E2B / OWNED WEBGPU</p>
        <h1>Generation console</h1>
      </div>
    </div>
    <div class="runtime-badges" aria-label="Runtime status">
      <span id="gpu-status" class="badge" data-state="pending">WebGPU</span>
      <span id="cache-status" class="badge" data-state="pending">Cache</span>
      <span id="model-status" class="badge" data-state="idle">Model offline</span>
    </div>
  </header>

  <main class="console-grid">
    <section class="generation-panel" aria-labelledby="generation-heading">
      <div class="section-heading">
        <div>
          <p class="eyebrow">SESSION</p>
          <h2 id="generation-heading">Prompt and response</h2>
        </div>
        <button id="load-model" class="secondary-command" type="button" disabled>Load model</button>
      </div>

      <div id="model-progress" class="model-progress" data-state="pending" hidden>
        <div class="progress-heading">
          <span id="progress-phase">Preparing model</span>
          <strong id="progress-percent">-</strong>
        </div>
        <progress id="progress-bar" max="1" aria-label="Model download progress"></progress>
        <span id="progress-detail" class="progress-detail">Waiting for download information</span>
      </div>

      <form id="generation-form">
        <label class="field example-picker">Example
          <select id="generation-example" name="generationExample">
            <option value="">Custom</option>
            ${GENERATION_EXAMPLES.map(({ id, label }) => `<option value="${id}">${label}</option>`).join("")}
          </select>
        </label>

        <label class="field prompt-field" for="prompt">
          <span>Prompt</span>
          <textarea id="prompt" name="prompt" rows="6" spellcheck="true">Name the three primary colors in one short sentence.</textarea>
        </label>
        <div class="image-input-row">
          <label class="image-picker" for="image-input">
            <span>Add image</span>
            <input id="image-input" name="image" type="file" accept="image/*">
          </label>
          <div id="image-preview" class="image-preview" hidden>
            <img id="image-thumbnail" alt="Selected image preview">
            <span id="image-name"></span>
            <button id="remove-image" class="quiet-command" type="button">Remove</button>
          </div>
        </div>
        <p id="prompt-budget" class="prompt-budget" data-valid="true" aria-live="polite">Load the model to calculate context usage</p>

        <div class="command-row">
          <button id="generate" class="primary-command" type="submit" disabled>Generate</button>
          <button id="cancel" class="danger-command" type="button" disabled>Stop</button>
          <span id="request-status" class="request-status" role="status">Waiting for model</span>
        </div>
      </form>

      <div class="output-tool" aria-labelledby="output-heading">
        <div class="tool-heading">
          <h3 id="output-heading">Output</h3>
          <span id="output-token-count">0 tokens</span>
        </div>
        <div id="generation-output" class="generation-output" role="log" aria-live="polite">No output yet.</div>
      </div>

      <div class="telemetry" aria-label="Generation telemetry">
        <div><span>TTFT</span><strong id="metric-ttft">-</strong></div>
        <div title="Average wall-clock time between emitted tokens"><span>TPOT</span><strong id="metric-tpot">-</strong></div>
        <div title="Median and p95 wall-clock latency between emitted tokens"><span>ITL</span><strong id="metric-itl">-</strong></div>
        <div title="Model evaluation only, excluding callback and CPU selection"><span>Decode</span><strong id="metric-decode">-</strong></div>
        <div title="Generated tokens after the first token, divided by measured decode time"><span>Decode tok/s</span><strong id="metric-decode-rate">-</strong></div>
        <div title="All emitted tokens divided by total request time, including prefill"><span>Overall tok/s</span><strong id="metric-overall-rate">-</strong></div>
        <div><span>Total</span><strong id="metric-total">-</strong></div>
        <div title="Browser preprocessing plus vision tower and projection"><span>Vision</span><strong id="metric-vision">-</strong></div>
        <div><span>Prefill</span><strong id="metric-prefill">-</strong></div>
        <div><span>GPU buffers</span><strong id="metric-memory">-</strong></div>
        <div><span>Stop</span><strong id="metric-stop">-</strong></div>
      </div>
    </section>

    <aside class="controls-panel" aria-labelledby="controls-heading">
      <div class="section-heading compact-heading">
        <div>
          <p class="eyebrow">DECODER</p>
          <h2 id="controls-heading">Controls</h2>
        </div>
        <button id="reset-controls" class="quiet-command" type="button">Reset</button>
      </div>

      <form id="controls-form">
        <div class="control-grid core-controls">
          <label class="field">Temperature
            <input name="temperature" type="number" min="0" step="0.05" value="${DEFAULT_GENERATION_CONFIG.temperature}">
          </label>
          <label class="field">Max tokens
            <input name="maxNewTokens" type="number" min="1" max="${GEMMA_VALIDATED_CONTEXT_CAPACITY}" step="1" value="${DEFAULT_GENERATION_CONFIG.maxNewTokens}">
          </label>
          <label class="field">Top K
            <input name="topK" type="number" min="0" step="1" value="${DEFAULT_GENERATION_CONFIG.topK}">
          </label>
          <label class="field">Top P
            <input name="topP" type="number" min="0" max="1" step="0.01" value="${DEFAULT_GENERATION_CONFIG.topP}">
          </label>
          <label class="field">Seed
            <input name="seed" type="number" step="1" value="${DEFAULT_GENERATION_CONFIG.seed}">
          </label>
          <label class="field">Stop token IDs
            <input name="stopTokenIds" type="text" inputmode="numeric" placeholder="1, 50, 106">
          </label>
        </div>

        <details class="control-section">
          <summary>Probability and penalties</summary>
          <div class="control-grid detail-controls">
            <label class="field">Min P
              <input name="minP" type="number" min="0" max="1" step="0.01" value="${DEFAULT_GENERATION_CONFIG.minP}">
            </label>
            <label class="field">Typical P
              <input name="typicalP" type="number" min="0" max="1" step="0.01" value="${DEFAULT_GENERATION_CONFIG.typicalP}">
            </label>
            <label class="field">Repetition
              <input name="repetitionPenalty" type="number" min="0.01" step="0.01" value="${DEFAULT_GENERATION_CONFIG.repetitionPenalty}">
            </label>
            <label class="field">History window
              <input name="repetitionWindow" type="number" min="0" step="1" value="${DEFAULT_GENERATION_CONFIG.repetitionWindow}">
            </label>
            <label class="field">Frequency
              <input name="frequencyPenalty" type="number" min="0" step="0.05" value="${DEFAULT_GENERATION_CONFIG.frequencyPenalty}">
            </label>
            <label class="field">Presence
              <input name="presencePenalty" type="number" min="0" step="0.05" value="${DEFAULT_GENERATION_CONFIG.presencePenalty}">
            </label>
          </div>
        </details>

        <fieldset class="constraint-section">
          <legend>Constraint</legend>
          <div class="segmented-control" role="radiogroup" aria-label="Output constraint">
            <label><input type="radio" name="constraintMode" value="none" checked><span>None</span></label>
            <label><input type="radio" name="constraintMode" value="regex"><span>Regex</span></label>
            <label><input type="radio" name="constraintMode" value="json"><span>JSON</span></label>
            <label><input type="radio" name="constraintMode" value="json-schema"><span>Schema</span></label>
          </div>

          <div id="constraint-regex" class="constraint-fields" hidden>
            <label class="field">Pattern
              <textarea name="regexPattern" rows="3" spellcheck="false">.+</textarea>
            </label>
          </div>

          <div id="constraint-json" class="constraint-fields" hidden>
            <div class="control-grid">
              <label class="field">Maximum depth
                <input name="jsonMaxDepth" type="number" min="1" max="8" step="1" value="4">
              </label>
              <label class="field">Whitespace
                <select name="jsonWhitespace">
                  <option value="none">None</option>
                  <option value="compact" selected>Compact</option>
                  <option value="any">Any</option>
                </select>
              </label>
            </div>
          </div>

          <div id="constraint-json-schema" class="constraint-fields" hidden>
            <label class="field">JSON Schema
              <textarea name="jsonSchema" rows="10" spellcheck="false">{
  "type": "object",
  "properties": {
    "answer": { "type": "string" }
  },
  "required": ["answer"],
  "additionalProperties": false
}</textarea>
            </label>
          </div>
        </fieldset>

        <p id="config-status" class="config-status" data-valid="true" role="status">Exact greedy configuration</p>
      </form>

      <section class="benchmark-tool" aria-labelledby="benchmark-heading">
        <div class="benchmark-heading">
          <div>
            <p class="eyebrow">CERTIFICATION</p>
            <h3 id="benchmark-heading">Long-context boundary</h3>
          </div>
          <button id="benchmark-download" class="quiet-command" type="button" disabled>Download JSON</button>
        </div>
        <label class="field">Capacity
          <select id="benchmark-capacity">
            <option value="8192">8,192 positions</option>
            <option value="32768" selected>32,768 positions</option>
            <option value="131072">131,072 positions</option>
          </select>
        </label>
        <div class="benchmark-command-row">
          <button id="benchmark-run" class="primary-command" type="button" disabled>Run exact-fit</button>
          <button id="benchmark-stop" class="danger-command" type="button" disabled>Stop</button>
        </div>
        <p id="benchmark-status" class="benchmark-status" role="status">No retained benchmark</p>
        <dl id="benchmark-details" class="benchmark-details" hidden>
          <div><dt>Phase</dt><dd id="benchmark-phase">-</dd></div>
          <div><dt>Progress</dt><dd id="benchmark-progress">-</dd></div>
          <div><dt>Memory</dt><dd id="benchmark-memory">-</dd></div>
          <div><dt>Total</dt><dd id="benchmark-total">-</dd></div>
        </dl>
      </section>
    </aside>
  </main>

  <footer class="runtime-footer">
    <span id="origin-label"></span>
    <span>google/gemma-4-E2B-it-qat-mobile-transformers</span>
    <span>${GEMMA_VALIDATED_CONTEXT_CAPACITY.toLocaleString()} validated / ${GEMMA_MODEL_CONTEXT_CAPACITY.toLocaleString()} model positions</span>
  </footer>
`;

const gpuStatus = element<HTMLSpanElement>("gpu-status");
const cacheStatus = element<HTMLSpanElement>("cache-status");
const modelStatus = element<HTMLSpanElement>("model-status");
const loadButton = element<HTMLButtonElement>("load-model");
const generateButton = element<HTMLButtonElement>("generate");
const cancelButton = element<HTMLButtonElement>("cancel");
const resetButton = element<HTMLButtonElement>("reset-controls");
const generationForm = element<HTMLFormElement>("generation-form");
const controlsForm = element<HTMLFormElement>("controls-form");
const exampleSelect = element<HTMLSelectElement>("generation-example");
const promptInput = element<HTMLTextAreaElement>("prompt");
const imageInput = element<HTMLInputElement>("image-input");
const imagePreview = element<HTMLDivElement>("image-preview");
const imageThumbnail = element<HTMLImageElement>("image-thumbnail");
const imageName = element<HTMLSpanElement>("image-name");
const removeImageButton = element<HTMLButtonElement>("remove-image");
const output = element<HTMLDivElement>("generation-output");
const outputTokenCount = element<HTMLSpanElement>("output-token-count");
const requestStatus = element<HTMLSpanElement>("request-status");
const configStatus = element<HTMLParagraphElement>("config-status");
const promptBudget = element<HTMLParagraphElement>("prompt-budget");
const modelProgress = element<HTMLDivElement>("model-progress");
const progressPhase = element<HTMLSpanElement>("progress-phase");
const progressPercent = element<HTMLElement>("progress-percent");
const progressBar = element<HTMLProgressElement>("progress-bar");
const progressDetail = element<HTMLSpanElement>("progress-detail");
const benchmarkCapacity = element<HTMLSelectElement>("benchmark-capacity");
const benchmarkRunButton = element<HTMLButtonElement>("benchmark-run");
const benchmarkStopButton = element<HTMLButtonElement>("benchmark-stop");
const benchmarkDownloadButton = element<HTMLButtonElement>("benchmark-download");
const benchmarkStatus = element<HTMLParagraphElement>("benchmark-status");
const benchmarkDetails = element<HTMLDListElement>("benchmark-details");

element<HTMLSpanElement>("origin-label").textContent = location.origin;

let session: GemmaGenerationSession | null = null;
let generationController: AbortController | null = null;
let cacheAvailable = false;
let controlsValidationTimer: number | null = null;
let progressPhaseKey = "";
let progressPhaseFraction = 0;
let selectedImage: File | null = null;
let imagePreviewUrl: string | null = null;
let benchmarkController: AbortController | null = null;
let benchmarkArtifact: GemmaDurableBenchmarkArtifact | null = null;

void initializeCapabilities();
void restoreBenchmarkArtifact();
renderConstraintFields();
validateControls();

loadButton.addEventListener("click", () => void loadModel());
exampleSelect.addEventListener("change", () => {
  const example = GENERATION_EXAMPLES.find(({ id }) => id === exampleSelect.value);
  if (example) void selectGenerationExample(example);
});
generationForm.addEventListener("submit", (event) => {
  event.preventDefault();
  void generate();
});
cancelButton.addEventListener("click", () => {
  generationController?.abort(new DOMException("Generation cancelled", "AbortError"));
});
resetButton.addEventListener("click", () => {
  controlsForm.reset();
  exampleSelect.value = "";
  renderConstraintFields();
  validateControls();
});
controlsForm.addEventListener("input", () => {
  exampleSelect.value = "";
  renderConstraintFields();
  if (controlsValidationTimer !== null) window.clearTimeout(controlsValidationTimer);
  controlsValidationTimer = window.setTimeout(validateControls, 120);
});
promptInput.addEventListener("input", () => {
  exampleSelect.value = "";
  validateControls();
});
imageInput.addEventListener("change", () => {
  const file = imageInput.files?.[0] ?? null;
  if (!file) {
    clearSelectedImage();
    return;
  }
  selectImage(file, true);
});
removeImageButton.addEventListener("click", clearSelectedImage);
benchmarkRunButton.addEventListener("click", () => void runBoundaryBenchmark());
benchmarkStopButton.addEventListener("click", () => {
  benchmarkController?.abort(new DOMException("Benchmark cancelled", "AbortError"));
});
benchmarkDownloadButton.addEventListener("click", () => void downloadBenchmarkArtifact());
window.addEventListener("beforeunload", () => {
  if (imagePreviewUrl) URL.revokeObjectURL(imagePreviewUrl);
  session?.destroy();
});

async function initializeCapabilities(): Promise<void> {
  const hasWebGpu = Boolean(navigator.gpu);
  setBadge(gpuStatus, hasWebGpu ? "WebGPU ready" : "WebGPU unavailable", hasWebGpu ? "ready" : "error");
  try {
    cacheAvailable = await inventoryModelCache();
    setBadge(
      cacheStatus,
      cacheAvailable ? "Cache ready" : "Cache absent",
      cacheAvailable ? "ready" : "error",
    );
  } catch {
    cacheAvailable = false;
    setBadge(cacheStatus, "Cache unknown", "error");
  }
  const canInitializeCache = typeof window.__gemmaEngineCacheInitializer === "function";
  loadButton.disabled = !hasWebGpu || (!cacheAvailable && !canInitializeCache);
  benchmarkRunButton.disabled = !hasWebGpu || !cacheAvailable;
  if (!cacheAvailable && canInitializeCache) {
    loadButton.textContent = "Initialize cache";
    requestStatus.textContent = "Model cache is not initialized in this browser";
  } else if (!cacheAvailable) {
    requestStatus.textContent = `Open this console on the origin containing ${GEMMA_4_E2B_CACHE_SPEC.databaseName}`;
  } else if (hasWebGpu) {
    requestStatus.textContent = "Model ready to load";
  }
}

async function restoreBenchmarkArtifact(): Promise<void> {
  try {
    const { recoverInterruptedGemmaBenchmark } = await import(
      "./runtime/gemma-long-context-benchmark"
    );
    benchmarkArtifact = recoverInterruptedGemmaBenchmark();
    renderBenchmarkArtifact(benchmarkArtifact);
  } catch (error) {
    benchmarkStatus.textContent = `Stored benchmark unavailable: ${errorMessage(error)}`;
    benchmarkStatus.dataset.state = "failed";
  }
}

async function runBoundaryBenchmark(): Promise<void> {
  if (benchmarkController || generationController || !cacheAvailable || !navigator.gpu) return;
  const capacity = Number(benchmarkCapacity.value);
  benchmarkController = new AbortController();
  session?.destroy();
  session = null;
  setBadge(modelStatus, "Benchmark session", "pending");
  loadButton.textContent = "Load model";
  setBenchmarkRunning(true);
  try {
    const { runGemmaLongContextBenchmark } = await import(
      "./runtime/gemma-long-context-benchmark"
    );
    benchmarkArtifact = await runGemmaLongContextBenchmark({
      cacheCapacity: capacity,
      targetPromptTokens: capacity,
      maxNewTokens: 1,
      prefillStrategy: "auto",
      signal: benchmarkController.signal,
      onArtifact: (artifact) => {
        benchmarkArtifact = artifact;
        renderBenchmarkArtifact(artifact);
      },
    });
    setBadge(
      modelStatus,
      benchmarkArtifact.phase === "completed" ? "Benchmark complete" : "Benchmark stopped",
      benchmarkArtifact.phase === "completed" ? "ready" : "error",
    );
  } catch (error) {
    setBadge(modelStatus, "Benchmark failed", "error");
    benchmarkStatus.textContent = errorMessage(error);
    benchmarkStatus.dataset.state = "failed";
  } finally {
    benchmarkController = null;
    setBenchmarkRunning(false);
  }
}

async function downloadBenchmarkArtifact(): Promise<void> {
  if (!benchmarkArtifact) return;
  const { serializeGemmaBenchmarkArtifact } = await import(
    "./runtime/gemma-long-context-benchmark"
  );
  const blob = new Blob([serializeGemmaBenchmarkArtifact(benchmarkArtifact)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `gemma-long-context-${benchmarkArtifact.configuration.cacheCapacity}-${benchmarkArtifact.runId}.json`;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

function renderBenchmarkArtifact(artifact: GemmaDurableBenchmarkArtifact | null): void {
  benchmarkDownloadButton.disabled = benchmarkController !== null || !artifact;
  benchmarkDetails.hidden = !artifact;
  if (!artifact) {
    benchmarkStatus.textContent = "No retained benchmark";
    benchmarkStatus.dataset.state = "idle";
    return;
  }
  benchmarkCapacity.value = String(artifact.configuration.cacheCapacity);
  benchmarkStatus.textContent = artifact.error ?? artifact.progress.message;
  benchmarkStatus.dataset.state = artifact.phase;
  element<HTMLElement>("benchmark-phase").textContent = artifact.phase;
  element<HTMLElement>("benchmark-progress").textContent = artifact.progress.promptTokens === null
    ? artifact.progress.message
    : `${artifact.progress.promptTokens.toLocaleString()} prompt · ${artifact.progress.generatedTokens.toLocaleString()} output`;
  element<HTMLElement>("benchmark-memory").textContent = artifact.memory
    ? formatBytes(artifact.memory.retainedGpuBufferBytes)
    : "-";
  const totalMs = artifact.timing?.totalMs;
  element<HTMLElement>("benchmark-total").textContent = typeof totalMs === "number"
    ? formatMilliseconds(totalMs)
    : "-";
}

function setBenchmarkRunning(running: boolean): void {
  benchmarkRunButton.disabled = running || !cacheAvailable || !navigator.gpu;
  benchmarkStopButton.disabled = !running;
  benchmarkCapacity.disabled = running;
  benchmarkDownloadButton.disabled = running || !benchmarkArtifact;
  loadButton.disabled = running || !cacheAvailable || !navigator.gpu;
  generateButton.disabled = running || !session;
  exampleSelect.disabled = running;
  promptInput.disabled = running;
  imageInput.disabled = running;
  removeImageButton.disabled = running;
  for (const control of controlsForm.elements) {
    if (control instanceof HTMLInputElement || control instanceof HTMLSelectElement ||
        control instanceof HTMLTextAreaElement || control instanceof HTMLButtonElement) {
      control.disabled = running;
    }
  }
}

async function loadModel(): Promise<void> {
  if (!navigator.gpu || generationController) return;
  const cacheInitializer = window.__gemmaEngineCacheInitializer;
  if (!cacheAvailable && !cacheInitializer) return;
  loadButton.disabled = true;
  const startedAt = performance.now();
  let cacheInitializationAttempted = false;
  try {
    if (!cacheAvailable && cacheInitializer) {
      cacheInitializationAttempted = true;
      await initializeModelCache(cacheInitializer);
    }
    loadButton.textContent = "Loading...";
    requestStatus.textContent = "Loading checkpoint and compiling pipelines";
    setBadge(modelStatus, "Model loading", "pending");
    setModelProgress(
      "Loading WebGPU engine",
      null,
      cacheAvailable ? "Reading cached weights and compiling pipelines" : "Preparing runtime",
    );
    session?.destroy();
    const { loadGemmaGenerationSession } = await import("./runtime/gemma-session");
    const loadOwnedSession = () => loadGemmaGenerationSession({
      cacheCapacity: GEMMA_VALIDATED_CONTEXT_CAPACITY,
        prefillStrategy: "auto",
      });
    try {
      session = await loadOwnedSession();
    } catch (error) {
      if (!cacheInitializer || cacheInitializationAttempted || !isMissingCachedTensor(error)) {
        throw error;
      }
      cacheInitializationAttempted = true;
      await initializeModelCache(cacheInitializer);
      loadButton.textContent = "Loading...";
      requestStatus.textContent = "Loading repaired cache and compiling pipelines";
      setBadge(modelStatus, "Model loading", "pending");
      setModelProgress(
        "Loading WebGPU engine",
        null,
        "Reading repaired weights and compiling pipelines",
      );
      session = await loadOwnedSession();
    }
    const loadSeconds = (performance.now() - startedAt) / 1000;
    const memory = session.estimateRetainedGpuMemory();
    setBadge(modelStatus, "Model ready", "ready");
    requestStatus.textContent = `Loaded in ${formatSeconds(loadSeconds)}`;
    setModelProgress("Model ready", 1, `Loaded in ${formatSeconds(loadSeconds)}`, "ready");
    element<HTMLElement>("metric-memory").textContent = formatBytes(memory.gpuBufferBytes);
    generateButton.disabled = !validateControls();
    loadButton.textContent = "Reload model";
  } catch (error) {
    session = null;
    setBadge(modelStatus, "Load failed", "error");
    requestStatus.textContent = errorMessage(error);
    setModelProgress("Model load failed", null, errorMessage(error), "error");
    loadButton.textContent = cacheAvailable ? "Retry load" : "Retry initialization";
  } finally {
    loadButton.disabled = !navigator.gpu ||
      (!cacheAvailable && typeof window.__gemmaEngineCacheInitializer !== "function");
  }
}

async function initializeModelCache(
  cacheInitializer: NonNullable<Window["__gemmaEngineCacheInitializer"]>,
): Promise<void> {
  progressPhaseKey = "";
  progressPhaseFraction = 0;
  loadButton.textContent = "Initializing...";
  setBadge(cacheStatus, "Cache initializing", "pending");
  setBadge(modelStatus, "Model downloading", "pending");
  requestStatus.textContent = "Preparing model cache";
  showModelProgress("Preparing download", "Contacting model host");
  await cacheInitializer(renderCacheProgress);
  cacheAvailable = await inventoryModelCache();
  if (!cacheAvailable) {
    throw new Error(`Cache initialization did not create ${GEMMA_4_E2B_CACHE_SPEC.databaseName}`);
  }
  setBadge(cacheStatus, "Cache ready", "ready");
  setModelProgress("Finalizing cache", 1, "Model weights cached in this browser");
}

async function generate(): Promise<void> {
  if (!session || generationController) return;
  const prompt = promptInput.value.trim();
  if (!prompt) {
    requestStatus.textContent = "Prompt is required";
    promptInput.focus();
    return;
  }

  let options: GemmaGenerationOptions;
  try {
    options = readGenerationOptions();
    renderPromptBudget(options.maxNewTokens ?? DEFAULT_GENERATION_CONFIG.maxNewTokens);
    renderValidConfiguration(options);
  } catch (error) {
    configStatus.textContent = errorMessage(error);
    configStatus.dataset.valid = "false";
    return;
  }

  generationController = new AbortController();
  if (controlsValidationTimer !== null) {
    window.clearTimeout(controlsValidationTimer);
    controlsValidationTimer = null;
  }
  options.signal = generationController.signal;
  options.onVisionProgress = (progress) => {
    const image = `image ${progress.imageIndex + 1}/${progress.imageCount}`;
    requestStatus.textContent = progress.phase === "preprocessing"
      ? `Preparing ${image}`
      : `Encoding ${image} · layer ${progress.completedLayers}/${progress.totalLayers}`;
  };
  options.onToken = (update) => {
    output.textContent = update.text;
    outputTokenCount.textContent = `${update.generatedTokenIds.length} ${pluralize(update.generatedTokenIds.length, "token")}`;
  };
  setGenerating(true);
  output.textContent = "";
  outputTokenCount.textContent = "0 tokens";
  clearTelemetry();
  requestStatus.textContent = "Generating";

  try {
    const measured = await session.generateMeasured(generationInput(prompt), options);
    output.textContent = measured.result.text || "No text output.";
    outputTokenCount.textContent = `${measured.result.generatedTokenIds.length} ${pluralize(measured.result.generatedTokenIds.length, "token")}`;
    requestStatus.textContent = measured.result.stopReason === "length"
      ? "Token limit reached"
      : "Generation complete";
    renderTelemetry(
      measured.timing,
      measured.result.stopReason,
      measured.result.generatedTokenIds.length,
    );
  } catch (error) {
    if (generationController.signal.aborted) {
      requestStatus.textContent = "Generation stopped";
      element<HTMLElement>("metric-stop").textContent = "cancelled";
    } else {
      requestStatus.textContent = errorMessage(error);
      element<HTMLElement>("metric-stop").textContent = "error";
      if (isDeviceFailure(error)) {
        session.destroy();
        session = null;
        setBadge(modelStatus, "Reload required", "error");
      }
    }
  } finally {
    generationController = null;
    setGenerating(false);
  }
}

function readGenerationOptions(): GemmaGenerationOptions {
  const data = new FormData(controlsForm);
  const stopTokenIds = String(data.get("stopTokenIds") ?? "")
    .split(/[\s,]+/)
    .filter(Boolean)
    .map(Number);
  const config = resolveGemmaGenerationConfig({
    temperature: numberValue(data, "temperature"),
    topK: numberValue(data, "topK"),
    topP: numberValue(data, "topP"),
    minP: numberValue(data, "minP"),
    typicalP: numberValue(data, "typicalP"),
    repetitionPenalty: numberValue(data, "repetitionPenalty"),
    repetitionWindow: numberValue(data, "repetitionWindow"),
    frequencyPenalty: numberValue(data, "frequencyPenalty"),
    presencePenalty: numberValue(data, "presencePenalty"),
    maxNewTokens: numberValue(data, "maxNewTokens"),
    seed: numberValue(data, "seed"),
    stopTokenIds,
  });
  const constraint = readConstraint(data);
  if (constraint) compileGenerationConstraint(constraint);
  return { ...config, constraint };
}

function applyGenerationExample(example: GenerationExample): void {
  controlsForm.reset();
  promptInput.value = example.prompt;
  for (const [name, value] of Object.entries(example.controls)) {
    setControlValue(name, value);
  }
  const mode = example.constraint?.type ?? "none";
  const modeInput = controlsForm.querySelector<HTMLInputElement>(
    `input[name="constraintMode"][value="${mode}"]`,
  );
  if (!modeInput) throw new Error(`Missing constraint mode ${mode}`);
  modeInput.checked = true;
  if (example.constraint?.type === "regex") {
    setControlValue("regexPattern", example.constraint.pattern);
  } else if (example.constraint?.type === "json" ||
      example.constraint?.type === "json-schema") {
    setControlValue("jsonMaxDepth", example.constraint.maxDepth ?? 4);
    setControlValue("jsonWhitespace", example.constraint.whitespace ?? "compact");
    if (example.constraint.type === "json-schema") {
      setControlValue("jsonSchema", JSON.stringify(example.constraint.schema, null, 2));
    }
  }
  const probabilityControls = controlsForm.querySelector<HTMLDetailsElement>(".control-section");
  if (probabilityControls) probabilityControls.open = example.expandProbabilityControls ?? false;
  renderConstraintFields();
  validateControls();
}

async function selectGenerationExample(example: GenerationExample): Promise<void> {
  clearSelectedImage();
  applyGenerationExample(example);
  if (!example.image) return;
  try {
    const response = await fetch(example.image.url);
    if (!response.ok) {
      throw new Error(`Image request failed with HTTP ${response.status}`);
    }
    const contentType = response.headers.get("content-type") ?? "";
    if (!contentType.startsWith("image/")) {
      throw new Error(`Image request returned ${contentType || "an unknown content type"}`);
    }
    const blob = await response.blob();
    if (exampleSelect.value !== example.id) return;
    selectImage(new File([blob], example.image.filename, { type: blob.type }), false);
  } catch (error) {
    if (exampleSelect.value !== example.id) return;
    exampleSelect.value = "";
    clearSelectedImage();
    configStatus.textContent = `Could not load example image: ${errorMessage(error)}`;
    configStatus.dataset.valid = "false";
    generateButton.disabled = true;
  }
}

function setControlValue(name: string, value: string | number): void {
  const control = controlsForm.elements.namedItem(name);
  if (!(control instanceof HTMLInputElement || control instanceof HTMLSelectElement ||
      control instanceof HTMLTextAreaElement)) {
    throw new Error(`Missing generation control ${name}`);
  }
  control.value = String(value);
}

function readConstraint(data: FormData): GenerationConstraint | undefined {
  const mode = String(data.get("constraintMode"));
  if (mode === "none") return undefined;
  if (mode === "regex") {
    const pattern = String(data.get("regexPattern") ?? "");
    if (!pattern) throw new Error("Regex pattern is required");
    return { type: "regex", pattern };
  }
  const maxDepth = numberValue(data, "jsonMaxDepth");
  const whitespace = String(data.get("jsonWhitespace")) as JsonWhitespace;
  if (mode === "json") return { type: "json", maxDepth, whitespace };
  const schemaText = String(data.get("jsonSchema") ?? "");
  const schema: unknown = JSON.parse(schemaText);
  if (!schema || typeof schema !== "object" || Array.isArray(schema)) {
    throw new Error("JSON Schema must be an object");
  }
  return { type: "json-schema", schema, maxDepth, whitespace };
}

function validateControls(): boolean {
  if (generationController) return configStatus.dataset.valid === "true";
  try {
    const options = readGenerationOptions();
    renderPromptBudget(options.maxNewTokens ?? DEFAULT_GENERATION_CONFIG.maxNewTokens);
    renderValidConfiguration(options);
    if (session && !generationController) generateButton.disabled = false;
    return true;
  } catch (error) {
    configStatus.textContent = errorMessage(error);
    configStatus.dataset.valid = "false";
    generateButton.disabled = true;
    return false;
  }
}

function renderPromptBudget(maxNewTokens: number): void {
  const prompt = promptInput.value.trim();
  if (!session || !prompt) {
    promptBudget.textContent = session
      ? "Enter a prompt to calculate context usage"
      : "Load the model to calculate context usage";
    promptBudget.dataset.valid = "true";
    return;
  }
  const promptTokens = session.promptTokenCount(generationInput(prompt));
  const availableOutput = availableGemmaOutputTokens(promptTokens, session.cacheCapacity);
  const requestedPositions = promptTokens + maxNewTokens - 1;
  promptBudget.textContent = `${promptTokens.toLocaleString()} prompt + ${maxNewTokens.toLocaleString()} output / ${requestedPositions.toLocaleString()} of ${session.cacheCapacity.toLocaleString()} positions`;
  promptBudget.dataset.valid = String(maxNewTokens <= availableOutput);
  if (maxNewTokens > availableOutput) {
    throw new Error(
      `Output exceeds context capacity; this prompt allows at most ${availableOutput.toLocaleString()} ${availableOutput === 1 ? "token" : "tokens"}`,
    );
  }
}

function renderValidConfiguration(options: GemmaGenerationOptions): void {
  const greedy = options.temperature === 0 &&
    options.repetitionPenalty === 1 &&
    options.frequencyPenalty === 0 &&
    options.presencePenalty === 0;
  configStatus.textContent = options.constraint
    ? `${options.constraint.type} constraint valid`
    : greedy ? "Exact greedy configuration" : "Sampling configuration valid";
  configStatus.dataset.valid = "true";
}

function renderConstraintFields(): void {
  const data = new FormData(controlsForm);
  const mode = String(data.get("constraintMode"));
  element<HTMLElement>("constraint-regex").hidden = mode !== "regex";
  element<HTMLElement>("constraint-json").hidden = mode !== "json" && mode !== "json-schema";
  element<HTMLElement>("constraint-json-schema").hidden = mode !== "json-schema";
}

function renderTelemetry(
  timing: GemmaGenerationTiming,
  stopReason: string,
  generatedTokenCount: number,
): void {
  const throughput = calculateGemmaGenerationThroughput([timing], generatedTokenCount);
  element<HTMLElement>("metric-ttft").textContent = formatMilliseconds(timing.timeToFirstTokenMs);
  element<HTMLElement>("metric-tpot").textContent = timing.timePerOutputTokenMs === null
    ? "-"
    : formatMilliseconds(timing.timePerOutputTokenMs);
  const sortedItl = [...timing.interTokenLatencyMs].sort((left, right) => left - right);
  element<HTMLElement>("metric-itl").textContent = sortedItl.length > 0
    ? `${formatMilliseconds(median(sortedItl))} / ${formatMilliseconds(percentile(sortedItl, 0.95))}`
    : "-";
  element<HTMLElement>("metric-decode").textContent = timing.decodeTokenMs.length > 0
    ? `${formatMilliseconds(median(timing.decodeTokenMs))} med.`
    : "-";
  element<HTMLElement>("metric-decode-rate").textContent = formatTokensPerSecond(
    throughput.warmDecodeTokensPerSecond,
  );
  element<HTMLElement>("metric-overall-rate").textContent = formatTokensPerSecond(
    throughput.endToEndTokensPerSecond,
  );
  element<HTMLElement>("metric-total").textContent = formatMilliseconds(timing.totalMs);
  element<HTMLElement>("metric-vision").textContent = timing.visionEncodeMs > 0 ||
      timing.visionPreprocessMs > 0
    ? formatMilliseconds(timing.visionPreprocessMs + timing.visionEncodeMs)
    : "-";
  element<HTMLElement>("metric-vision").title = timing.visionEncodeMs > 0 ||
      timing.visionPreprocessMs > 0
    ? `Preprocess ${formatMilliseconds(timing.visionPreprocessMs)} · encode ${formatMilliseconds(timing.visionEncodeMs)}`
    : "";
  element<HTMLElement>("metric-prefill").textContent = timing.prefillMode;
  element<HTMLElement>("metric-stop").textContent = stopReason;
}

function clearTelemetry(): void {
  for (const id of [
    "metric-ttft",
    "metric-tpot",
    "metric-itl",
    "metric-decode",
    "metric-decode-rate",
    "metric-overall-rate",
    "metric-total",
    "metric-vision",
    "metric-prefill",
    "metric-stop",
  ]) {
    element<HTMLElement>(id).textContent = "-";
  }
}

function setGenerating(generating: boolean): void {
  generateButton.disabled = generating || !session;
  cancelButton.disabled = !generating;
  loadButton.disabled = generating || !cacheAvailable || !navigator.gpu;
  exampleSelect.disabled = generating;
  promptInput.disabled = generating;
  imageInput.disabled = generating;
  removeImageButton.disabled = generating;
  for (const control of controlsForm.elements) {
    if (control instanceof HTMLInputElement || control instanceof HTMLSelectElement ||
        control instanceof HTMLTextAreaElement || control instanceof HTMLButtonElement) {
      control.disabled = generating;
    }
  }
}

function generationInput(prompt: string): GemmaGenerationInput {
  if (!selectedImage) return prompt;
  return {
    messages: [{
      role: "user",
      content: [
        { type: "image" },
        { type: "text", text: prompt },
      ],
    }],
    images: [selectedImage],
  };
}

function selectImage(file: File, clearExample: boolean): void {
  selectedImage = file;
  if (imagePreviewUrl) URL.revokeObjectURL(imagePreviewUrl);
  imagePreviewUrl = URL.createObjectURL(file);
  imageThumbnail.src = imagePreviewUrl;
  imageName.textContent = file.name;
  imagePreview.hidden = false;
  if (clearExample) exampleSelect.value = "";
  validateControls();
}

function clearSelectedImage(): void {
  selectedImage = null;
  imageInput.value = "";
  imagePreview.hidden = true;
  imageThumbnail.removeAttribute("src");
  imageName.textContent = "";
  if (imagePreviewUrl) URL.revokeObjectURL(imagePreviewUrl);
  imagePreviewUrl = null;
  validateControls();
}

function setBadge(target: HTMLElement, text: string, state: string): void {
  target.textContent = text;
  target.dataset.state = state;
}

async function inventoryModelCache(): Promise<boolean> {
  try {
    const cache = await ReadonlySafetensorsCache.open();
    cache.close();
    return true;
  } catch {
    return false;
  }
}

function isMissingCachedTensor(error: unknown): boolean {
  return /Tensor .+ is not present in the readonly cache/.test(errorMessage(error));
}

function renderCacheProgress(progress: GemmaCacheInitializationProgress): void {
  const phaseKey = `${progress.status ?? "unknown"}:${progress.kind ?? "unknown"}:${progress.fromCache ? "cache" : "network"}`;
  const rawFraction = normalizedProgressFraction(progress);
  if (phaseKey !== progressPhaseKey) {
    progressPhaseKey = phaseKey;
    progressPhaseFraction = 0;
  }
  if (rawFraction !== null) progressPhaseFraction = Math.max(progressPhaseFraction, rawFraction);
  const fraction = rawFraction === null ? null : progressPhaseFraction;
  const phase = progress.status === "ready"
    ? "Finalizing cache"
    : progress.kind === "tensors"
      ? "Preparing model tensors"
      : progress.fromCache
        ? "Reading cached weights"
        : "Downloading model weights";
  const total = progress.total ?? (
    progress.kind === "bytes" ? GEMMA_4_E2B_CACHE_SPEC.fileSize : undefined
  );
  const detail = progress.loaded !== undefined && total !== undefined
    ? progress.kind === "bytes"
      ? `${formatBytes(progress.loaded)} of ${formatBytes(total)}`
      : `${progress.loaded.toLocaleString()} of ${total.toLocaleString()} tensors`
    : progress.message || (fraction === null
      ? "Download in progress"
      : `${formatProgressPercent(fraction)} complete`);
  setModelProgress(phase, fraction, detail);
  requestStatus.textContent = fraction === null
    ? phase
    : `${phase} · ${formatProgressPercent(fraction)}`;
  loadButton.textContent = fraction === null
    ? phase.replace(" model weights", "...")
    : `${phase.replace(" model weights", "")} ${formatProgressPercent(fraction)}`;
}

function normalizedProgressFraction(progress: GemmaCacheInitializationProgress): number | null {
  const fraction = progress.fraction ?? (
    progress.loaded !== undefined && (progress.total || progress.kind === "bytes")
      ? progress.loaded / (progress.total || GEMMA_4_E2B_CACHE_SPEC.fileSize)
      : undefined
  );
  if (fraction === undefined || !Number.isFinite(fraction)) return null;
  return Math.max(0, Math.min(1, fraction > 1 ? fraction / 100 : fraction));
}

function formatProgressPercent(fraction: number): string {
  const percent = Math.max(0, Math.min(100, fraction * 100));
  return percent === 0 || percent === 100
    ? `${percent.toFixed(0)}%`
    : `${percent.toFixed(1)}%`;
}

function showModelProgress(phase: string, detail: string): void {
  modelProgress.hidden = false;
  setModelProgress(phase, null, detail);
}

function setModelProgress(
  phase: string,
  fraction: number | null,
  detail: string,
  state: "pending" | "ready" | "error" = "pending",
): void {
  modelProgress.hidden = false;
  modelProgress.dataset.state = state;
  progressPhase.textContent = phase;
  progressDetail.textContent = detail;
  if (fraction === null) {
    progressBar.removeAttribute("value");
    progressPercent.textContent = "-";
  } else {
    const normalized = Math.max(0, Math.min(1, fraction));
    progressBar.value = normalized;
    progressPercent.textContent = formatProgressPercent(normalized);
  }
}

function element<T extends HTMLElement>(id: string): T {
  const target = document.getElementById(id);
  if (!target) throw new Error(`Missing #${id}`);
  return target as T;
}

function numberValue(data: FormData, name: string): number {
  return Number(data.get(name));
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)];
}

function percentile(values: readonly number[], quantile: number): number {
  const index = Math.min(
    values.length - 1,
    Math.max(0, Math.ceil(values.length * quantile) - 1),
  );
  return values[index];
}

function formatMilliseconds(value: number): string {
  return value >= 1000 ? formatSeconds(value / 1000) : `${value.toFixed(1)} ms`;
}

function formatSeconds(value: number): string {
  return `${value.toFixed(1)} s`;
}

function formatTokensPerSecond(value: number | null): string {
  if (value === null) return "-";
  return `${value < 10 ? value.toFixed(2) : value.toFixed(1)} tok/s`;
}

function formatBytes(value: number): string {
  return `${(value / (1024 ** 2)).toFixed(1)} MiB`;
}

function pluralize(count: number, noun: string): string {
  return count === 1 ? noun : `${noun}s`;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function isDeviceFailure(error: unknown): boolean {
  return /device|GPU|lost/i.test(errorMessage(error));
}
