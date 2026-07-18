import "./style.css";
import {
  GEMMA_4_E2B_CACHE_SPEC,
  ReadonlySafetensorsCache,
} from "./model/cached-safetensors";
import {
  initializeGemmaSafetensorsCache,
  type SafetensorsCacheInitializationProgress,
} from "./model/safetensors-cache-initializer";
import { GEMMA_LOCAL_SAFETENSORS_URL } from "./model/pinned-safetensors";
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
import {
  commitGemmaConversationTurn,
  createGemmaConversation,
  prepareGemmaConversationEdit,
  prepareGemmaConversationTurn,
  type GemmaConversation,
  type PreparedGemmaConversationTurn,
} from "./runtime/gemma-conversation";
import { parseGemmaResponse } from "./runtime/gemma-response";
import type {
  GemmaGenerationSession,
  GemmaGenerationTiming,
} from "./runtime/gemma-session";
import type {
  GemmaChatMessage,
  GemmaFunctionTool,
} from "./runtime/gemma-tokenizer";
import {
  validateGemmaVisionTokenBudget,
  type GemmaVisionImageSource,
  type GemmaVisionTokenBudget,
} from "./runtime/gemma-vision-input";
import type { GemmaAudioSource } from "./runtime/gemma-audio-input";
import {
  startGemmaMicrophoneCapture,
  type GemmaMicrophoneCapture,
} from "./runtime/gemma-microphone";
import type { GemmaDurableBenchmarkArtifact } from "./runtime/durable-benchmark";

type GemmaCacheInitializationProgress = SafetensorsCacheInitializationProgress;

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

type ConsoleWorkspace = "custom" | "chat" | "example";
type ConstraintEditor = "all" | "none" | "regex" | "json" | "json-schema";

interface ExampleUiProfile {
  vision: boolean;
  audio?: boolean;
  sampling: boolean;
  constraint: Exclude<ConstraintEditor, "all">;
}

interface ConsoleUiProfile {
  workspace: ConsoleWorkspace;
  vision: boolean;
  audio: boolean;
  sampling: boolean;
  constraint: ConstraintEditor;
}

interface GenerationExample {
  id: string;
  label: string;
  prompt: string;
  enableThinking?: boolean;
  requireReasoning?: boolean;
  controls: Partial<Record<GenerationExampleControl, number>>;
  history?: readonly GemmaChatMessage[];
  tools?: readonly GemmaFunctionTool[];
  image?: {
    url: string;
    filename: string;
  };
  audio?: {
    url: string;
    filename: string;
  };
  visionTokenBudget?: GemmaVisionTokenBudget;
  constraint?: GenerationConstraint;
  expandProbabilityControls?: boolean;
  ui: ExampleUiProfile;
}

const GENERATION_EXAMPLES: readonly GenerationExample[] = [
  {
    id: "audio-transcription",
    label: "Browser audio · Transcription",
    prompt: "What is said in this audio? Return only the spoken words.",
    controls: { temperature: 0, maxNewTokens: 48 },
    audio: {
      url: "/examples/gemma-audio-demo.wav",
      filename: "gemma-audio-demo.wav",
    },
    ui: { vision: false, audio: true, sampling: false, constraint: "none" },
  },
  {
    id: "tool-weather",
    label: "Boston weather · Tool call",
    prompt: "Use get_current_weather to look up the current weather in Boston.",
    controls: {
      temperature: 1,
      maxNewTokens: 48,
      topK: 64,
      topP: 0.95,
      seed: 42,
    },
    expandProbabilityControls: true,
    history: [{
      role: "system",
      content: "When a listed tool can answer the request, call that tool instead of answering in prose.",
    }],
    tools: [{
      type: "function",
      function: {
        name: "get_current_weather",
        description: "Get the current weather for a city.",
        parameters: {
          type: "object",
          properties: {
            location: {
              type: "string",
              description: "City and region, for example Boston, MA.",
            },
          },
          required: ["location"],
        },
      },
    }],
    ui: { vision: false, sampling: true, constraint: "none" },
  },
  {
    id: "vision-dolphin-caption",
    label: "Dolphin caption · Vision",
    prompt: "Transcribe the printed caption in this image as accurately as possible, including the credit line. Then briefly describe whether the photograph matches the caption. Mark any unreadable words as [unclear] and do not use outside knowledge.",
    controls: { temperature: 0, maxNewTokens: 256 },
    image: {
      url: "/examples/dolphin_capt_image.png",
      filename: "dolphin_capt_image.png",
    },
    visionTokenBudget: 280,
    ui: { vision: true, sampling: false, constraint: "none" },
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
    visionTokenBudget: 280,
    ui: { vision: true, sampling: false, constraint: "none" },
  },
  {
    id: "greedy-colors",
    label: "Primary colors · Greedy",
    prompt: "Name the three primary colors in one short sentence.",
    controls: { temperature: 0, maxNewTokens: 32 },
    ui: { vision: false, sampling: false, constraint: "none" },
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
    ui: { vision: false, sampling: true, constraint: "none" },
  },
  {
    id: "regex-sky",
    label: "Sky color · Regex",
    prompt: "Return exactly one lowercase word for the color of a clear daytime sky.",
    controls: { temperature: 0, maxNewTokens: 4 },
    constraint: { type: "regex", pattern: "(?:blue|gray)" },
    ui: { vision: false, sampling: false, constraint: "regex" },
  },
  {
    id: "json-city",
    label: "Paris facts · JSON",
    prompt: "Return one compact JSON object about Paris with keys city, country, and landmarks, where landmarks is an array of exactly two strings. Return JSON only.",
    controls: { temperature: 0, maxNewTokens: 64 },
    constraint: { type: "json", maxDepth: 3, whitespace: "compact" },
    ui: { vision: false, sampling: false, constraint: "json" },
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
    ui: { vision: false, sampling: false, constraint: "json-schema" },
  },
  {
    id: "reasoning-logic",
    label: "River crossing · Reasoning",
    prompt: "A farmer must take a fox, a chicken, and grain across a river. The boat carries only the farmer and one item. The fox cannot be left with the chicken, and the chicken cannot be left with the grain. Reason through the safe sequence, then give the answer clearly.",
    enableThinking: true,
    requireReasoning: true,
    controls: { temperature: 0, maxNewTokens: 384 },
    ui: { vision: false, sampling: false, constraint: "none" },
  },
  {
    id: "schema-reasoning",
    label: "Cold-chain decision · Schema + reasoning",
    prompt: "A medication shipment was above 8°C for 18 minutes. Policy says a cold-chain breach occurs only when it is above 8°C for more than 15 minutes, and medication breaches require inspection. Reason through the policy, then return JSON with exactly these keys in this order: breach, action, rationale.",
    enableThinking: true,
    requireReasoning: true,
    controls: { temperature: 0, maxNewTokens: 256 },
    constraint: {
      type: "json-schema",
      maxDepth: 2,
      whitespace: "compact",
      schema: {
        type: "object",
        properties: {
          breach: { type: "boolean" },
          action: { enum: ["release", "inspect"] },
          rationale: { type: "string" },
        },
        required: ["breach", "action", "rationale"],
        additionalProperties: false,
      },
    },
    ui: { vision: false, sampling: false, constraint: "json-schema" },
  },
];

const MODEL_CACHE_SENTINELS = [
  "lm_head.weight",
  "model.language_model.layers.34.mlp.down_proj.weight",
  "model.language_model.layers.34.self_attn.q_proj.weight",
] as const;

declare global {
  interface Window {
    __gemmaEngineCacheInitializer?: (
      onProgress: (progress: GemmaCacheInitializationProgress) => void,
    ) => Promise<void>;
    __gemmaEngineForceDeviceLoss?: () => boolean;
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
    <section id="generation-panel" class="generation-panel" data-workspace="custom" aria-labelledby="generation-heading">
      <div class="section-heading">
        <div>
          <p class="eyebrow">SESSION</p>
          <h2 id="generation-heading">Conversation</h2>
        </div>
        <div class="heading-commands">
          <button id="clear-transcript" class="quiet-command" type="button">Clear transcript</button>
          <button id="new-chat" class="quiet-command" type="button" disabled>New chat</button>
          <button id="load-model" class="secondary-command" type="button" title="Rebuild the WebGPU engine from existing local or cached weights" disabled>Load model</button>
        </div>
      </div>

      <div id="model-progress" class="model-progress" data-state="pending" hidden>
        <div class="progress-heading">
          <span id="progress-phase">Preparing model</span>
          <strong id="progress-percent">-</strong>
        </div>
        <progress id="progress-bar" max="1" aria-label="Model download progress"></progress>
        <span id="progress-detail" class="progress-detail">Waiting for download information</span>
      </div>

      <label class="field example-picker" id="workspace-picker">Workspace
        <select id="generation-example" name="generationExample">
          <optgroup label="Workspaces">
            <option value="custom">Custom</option>
            <option value="chat">Chat</option>
          </optgroup>
          <optgroup label="Examples">
            ${GENERATION_EXAMPLES.map(({ id, label }) => `<option value="${id}">${label}</option>`).join("")}
          </optgroup>
        </select>
      </label>

      <form id="generation-form">
        <label class="field prompt-field" for="prompt">
          <span>Message</span>
          <textarea id="prompt" name="prompt" rows="6" spellcheck="true">Name the three primary colors in one short sentence.</textarea>
        </label>
        <div id="image-controls" class="image-input-row">
          <label id="image-picker" class="image-picker" for="image-input">
            <span>Add image</span>
            <input id="image-input" name="image" type="file" accept="image/*">
          </label>
          <div id="image-preview" class="image-preview" hidden>
            <img id="image-thumbnail" alt="Selected image preview">
            <span id="image-name"></span>
            <button id="remove-image" class="quiet-command" type="button">Remove</button>
          </div>
          <label id="audio-picker" class="image-picker" for="audio-input">
            <span>Add audio</span>
            <input id="audio-input" name="audio" type="file" accept="audio/*">
          </label>
          <button id="record-audio" class="image-picker" type="button">Record mic</button>
          <span id="recording-status" class="recording-status" role="status" hidden></span>
          <div id="audio-preview" class="image-preview" hidden>
            <audio id="audio-player" controls preload="metadata"></audio>
            <span id="audio-name"></span>
            <button id="remove-audio" class="quiet-command" type="button">Remove</button>
          </div>
          <label id="image-budget" class="field image-budget">Visual tokens
            <select id="vision-token-budget" name="visionTokenBudget">
              <option value="70">Fast · 70</option>
              <option value="140" selected>Balanced · 140</option>
              <option value="280">Quality · 280</option>
            </select>
          </label>
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
          <h3 id="output-heading">Transcript</h3>
          <span id="output-token-count">0 tokens</span>
        </div>
        <div id="generation-output" class="generation-output" role="log" aria-live="polite">No conversation yet.</div>
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
        <div title="Session-retained materialized vision weights on the CPU"><span>Vision CPU</span><strong id="metric-vision-memory">-</strong></div>
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
          <label class="field" data-control="sampling">Temperature
            <input name="temperature" type="number" min="0" step="0.05" value="${DEFAULT_GENERATION_CONFIG.temperature}">
          </label>
          <label class="field">Max tokens
            <input name="maxNewTokens" type="number" min="1" max="${GEMMA_VALIDATED_CONTEXT_CAPACITY}" step="1" value="${DEFAULT_GENERATION_CONFIG.maxNewTokens}">
          </label>
          <label class="field" data-control="sampling">Top K
            <input name="topK" type="number" min="0" step="1" value="${DEFAULT_GENERATION_CONFIG.topK}">
          </label>
          <label class="field" data-control="sampling">Top P
            <input name="topP" type="number" min="0" max="1" step="0.01" value="${DEFAULT_GENERATION_CONFIG.topP}">
          </label>
          <label class="field" data-control="sampling">Seed
            <input name="seed" type="number" step="1" value="${DEFAULT_GENERATION_CONFIG.seed}">
          </label>
          <label class="field" data-control="stop-tokens">Stop token IDs
            <input name="stopTokenIds" type="text" inputmode="numeric" placeholder="1, 50, 106">
          </label>
        </div>

        <label id="thinking-control" class="toggle-control">
          <span>
            <strong>Thinking</strong>
            <small>Separate reasoning from the final answer</small>
          </span>
          <input id="enable-thinking" name="enableThinking" type="checkbox">
          <span class="toggle-track" aria-hidden="true"></span>
        </label>

        <details id="probability-controls" class="control-section">
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

        <fieldset id="constraint-section" class="constraint-section">
          <legend>Constraint</legend>
          <div id="constraint-mode-selector" class="segmented-control" role="radiogroup" aria-label="Output constraint">
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

      <details class="diagnostics-section">
        <summary>Diagnostics</summary>
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
      </details>
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
const clearTranscriptButton = element<HTMLButtonElement>("clear-transcript");
const newChatButton = element<HTMLButtonElement>("new-chat");
const generationPanel = element<HTMLElement>("generation-panel");
const generateButton = element<HTMLButtonElement>("generate");
const cancelButton = element<HTMLButtonElement>("cancel");
const resetButton = element<HTMLButtonElement>("reset-controls");
const generationForm = element<HTMLFormElement>("generation-form");
const controlsForm = element<HTMLFormElement>("controls-form");
const exampleSelect = element<HTMLSelectElement>("generation-example");
const promptInput = element<HTMLTextAreaElement>("prompt");
const imageInput = element<HTMLInputElement>("image-input");
const audioInput = element<HTMLInputElement>("audio-input");
const imagePicker = element<HTMLLabelElement>("image-picker");
const audioPicker = element<HTMLLabelElement>("audio-picker");
const recordAudioButton = element<HTMLButtonElement>("record-audio");
const recordingStatus = element<HTMLSpanElement>("recording-status");
const visionTokenBudgetInput = element<HTMLSelectElement>("vision-token-budget");
const thinkingControl = element<HTMLElement>("thinking-control");
const enableThinkingInput = element<HTMLInputElement>("enable-thinking");
const imageControls = element<HTMLDivElement>("image-controls");
const imageBudget = element<HTMLLabelElement>("image-budget");
const imagePreview = element<HTMLDivElement>("image-preview");
const imageThumbnail = element<HTMLImageElement>("image-thumbnail");
const imageName = element<HTMLSpanElement>("image-name");
const removeImageButton = element<HTMLButtonElement>("remove-image");
const audioPreview = element<HTMLDivElement>("audio-preview");
const audioPlayer = element<HTMLAudioElement>("audio-player");
const audioName = element<HTMLSpanElement>("audio-name");
const removeAudioButton = element<HTMLButtonElement>("remove-audio");
const outputTool = element<HTMLDivElement>("output-heading").closest<HTMLDivElement>(".output-tool");
if (!outputTool) throw new Error("Missing output tool");
const ownedOutputTool = outputTool;
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
const telemetry = generationPanel.querySelector<HTMLDivElement>(".telemetry");
if (!telemetry) throw new Error("Missing generation telemetry");
const ownedTelemetry = telemetry;
const probabilityControls = element<HTMLDetailsElement>("probability-controls");
const constraintSection = element<HTMLFieldSetElement>("constraint-section");
const constraintModeSelector = element<HTMLDivElement>("constraint-mode-selector");
const samplingControls = Array.from(
  controlsForm.querySelectorAll<HTMLElement>("[data-control='sampling']"),
);
const stopTokenControl = controlsForm.querySelector<HTMLElement>("[data-control='stop-tokens']");
if (!stopTokenControl) throw new Error("Missing stop-token control");
const ownedStopTokenControl = stopTokenControl;

element<HTMLSpanElement>("origin-label").textContent = location.origin;

let session: GemmaGenerationSession | null = null;
let generationController: AbortController | null = null;
let modelLifecycleBusy = false;
let deviceRecoveryPending = false;
let lostSession: GemmaGenerationSession | null = null;
let cacheAvailable = false;
let localModelAvailable = false;
let controlsValidationTimer: number | null = null;
let progressPhaseKey = "";
let progressPhaseFraction = 0;
let selectedImage: GemmaVisionImageSource | null = null;
let selectedAudio: GemmaAudioSource | null = null;
let imagePreviewUrl: string | null = null;
let audioPreviewUrl: string | null = null;
let audioRecorder: GemmaMicrophoneCapture | null = null;
let audioRecordingTimer: number | null = null;
let audioRecordingStartedAt = 0;
let activeWorkspace: ConsoleWorkspace = "custom";
let activeExample: GenerationExample | null = null;
let forceFreshGeneration = false;
let editBaseConversation: GemmaConversation | null = null;
let benchmarkController: AbortController | null = null;
let benchmarkArtifact: GemmaDurableBenchmarkArtifact | null = null;
let conversation: GemmaConversation = createGemmaConversation();

if (import.meta.env.DEV) {
  window.__gemmaEngineForceDeviceLoss = () => {
    if (!session) return false;
    session.simulateDeviceLoss();
    return true;
  };
}

void initializeCapabilities();
void restoreBenchmarkArtifact();
renderConstraintFields();
renderConsoleProfile();
validateControls();

loadButton.addEventListener("click", () => {
  void loadModel(session ? "manual-reload" : "initial");
});
newChatButton.addEventListener("click", clearConversation);
clearTranscriptButton.addEventListener("click", clearTranscript);
exampleSelect.addEventListener("change", () => {
  if (exampleSelect.value === "custom") {
    activateWorkspace("custom");
    return;
  }
  if (exampleSelect.value === "chat") {
    activateWorkspace("chat");
    return;
  }
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
  visionTokenBudgetInput.value = "140";
  activateWorkspace("custom");
  renderConstraintFields();
  validateControls();
});
controlsForm.addEventListener("input", () => {
  markExampleCustomized();
  renderConstraintFields();
  if (controlsValidationTimer !== null) window.clearTimeout(controlsValidationTimer);
  controlsValidationTimer = window.setTimeout(validateControls, 120);
});
promptInput.addEventListener("input", () => {
  markExampleCustomized();
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
audioInput.addEventListener("change", () => {
  const file = audioInput.files?.[0] ?? null;
  if (!file) {
    clearSelectedAudio();
    return;
  }
  selectAudio(file);
});
recordAudioButton.addEventListener("click", () => void toggleAudioRecording());
visionTokenBudgetInput.addEventListener("change", () => {
  markExampleCustomized();
  validateControls();
});
removeImageButton.addEventListener("click", () => {
  markExampleCustomized();
  clearSelectedImage();
});
removeAudioButton.addEventListener("click", () => {
  markExampleCustomized();
  clearSelectedAudio();
});
benchmarkRunButton.addEventListener("click", () => void runBoundaryBenchmark());
benchmarkStopButton.addEventListener("click", () => {
  benchmarkController?.abort(new DOMException("Benchmark cancelled", "AbortError"));
});
benchmarkDownloadButton.addEventListener("click", () => void downloadBenchmarkArtifact());
window.addEventListener("beforeunload", () => {
  discardAudioRecording();
  if (imagePreviewUrl) URL.revokeObjectURL(imagePreviewUrl);
  if (audioPreviewUrl) URL.revokeObjectURL(audioPreviewUrl);
  session?.destroy();
});

async function initializeCapabilities(): Promise<void> {
  const hasWebGpu = Boolean(navigator.gpu);
  setBadge(gpuStatus, hasWebGpu ? "WebGPU ready" : "WebGPU unavailable", hasWebGpu ? "ready" : "error");
  let cacheExists = false;
  try {
    localModelAvailable = await localModelExists();
    cacheExists = await modelCacheDatabaseExists();
    cacheAvailable = localModelAvailable ? false : await inventoryModelCache();
    setBadge(
      cacheStatus,
      localModelAvailable
        ? "Local weights"
        : cacheAvailable ? "Cache ready" : cacheExists ? "Cache partial" : "Cache absent",
      localModelAvailable || cacheAvailable ? "ready" : "error",
    );
  } catch {
    cacheAvailable = false;
    localModelAvailable = false;
    setBadge(cacheStatus, "Cache unknown", "error");
  }
  const canInitializeCache = typeof indexedDB !== "undefined";
  loadButton.disabled = !hasWebGpu ||
    (!localModelAvailable && !cacheAvailable && !canInitializeCache);
  benchmarkRunButton.disabled = !hasWebGpu || !cacheAvailable;
  if (localModelAvailable) {
    loadButton.textContent = "Load model";
    requestStatus.textContent = "Local model ready to load";
  } else if (!cacheAvailable && canInitializeCache) {
    loadButton.textContent = cacheExists ? "Resume download" : "Download model";
    requestStatus.textContent = cacheExists
      ? "Resume the model download and load it on this origin"
      : "Download and load the model on this origin";
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
  loadButton.disabled = running || (!localModelAvailable && !cacheAvailable) || !navigator.gpu;
  newChatButton.disabled = running || !hasConversationState();
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

type ModelLoadReason = "initial" | "manual-reload" | "device-recovery";

async function loadModel(reason: ModelLoadReason = "initial"): Promise<void> {
  if (!navigator.gpu || generationController || benchmarkController || modelLifecycleBusy) return;
  const cachedOnly = reason !== "initial";
  const cacheInitializer = cachedOnly
    ? undefined
    : window.__gemmaEngineCacheInitializer ?? initializeGemmaSafetensorsCache;
  if (!localModelAvailable && !cacheAvailable && !cacheInitializer) return;
  modelLifecycleBusy = true;
  loadButton.disabled = true;
  generateButton.disabled = true;
  const startedAt = performance.now();
  let cacheInitializationAttempted = false;
  try {
    const previousSession = session;
    session = null;
    previousSession?.destroy();
    if (reason === "manual-reload") {
      loadButton.textContent = "Reloading...";
      requestStatus.textContent = "Recreating WebGPU device and engine from cached weights";
      setBadge(modelStatus, "Engine reloading", "pending");
      const { resetWebGpuDevice } = await import("./webgpu/device");
      await resetWebGpuDevice();
    }
    if (!localModelAvailable && !cacheAvailable && cacheInitializer) {
      cacheInitializationAttempted = true;
      await initializeModelCache(cacheInitializer);
    }
    loadButton.textContent = "Loading...";
    requestStatus.textContent = reason === "device-recovery"
      ? "Recovering WebGPU engine from cached weights"
      : "Loading checkpoint and compiling pipelines";
    setBadge(modelStatus, reason === "device-recovery" ? "Engine recovering" : "Model loading", "pending");
    setModelProgress(
      reason === "device-recovery" ? "Recovering WebGPU engine" : "Loading WebGPU engine",
      null,
      localModelAvailable
        ? "Reading local weights and compiling pipelines"
        : cacheAvailable ? "Reading cached weights and compiling pipelines" : "Preparing runtime",
    );
    const { loadGemmaGenerationSession } = await import("./runtime/gemma-session");
    const loadOwnedSession = () => loadGemmaGenerationSession({
      cacheCapacity: GEMMA_VALIDATED_CONTEXT_CAPACITY,
      sourceUrl: localModelAvailable ? GEMMA_LOCAL_SAFETENSORS_URL : undefined,
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
    watchSessionDeviceLoss(session);
    const loadSeconds = (performance.now() - startedAt) / 1000;
    const memory = session.estimateRetainedGpuMemory();
    setBadge(modelStatus, "Model ready", "ready");
    requestStatus.textContent = reason === "device-recovery"
      ? `Recovered in ${formatSeconds(loadSeconds)}`
      : `Loaded in ${formatSeconds(loadSeconds)}`;
    setModelProgress(
      "Model ready",
      1,
      reason === "device-recovery"
        ? `Recovered in ${formatSeconds(loadSeconds)}`
        : `Loaded in ${formatSeconds(loadSeconds)}`,
      "ready",
    );
    renderMemory(memory);
    generateButton.disabled = !validateControls();
    loadButton.textContent = "Reload engine";
  } catch (error) {
    session = null;
    setBadge(modelStatus, reason === "device-recovery" ? "Recovery failed" : "Load failed", "error");
    requestStatus.textContent = errorMessage(error);
    setModelProgress(
      reason === "device-recovery" ? "Engine recovery failed" : "Model load failed",
      null,
      errorMessage(error),
      "error",
    );
    loadButton.textContent = localModelAvailable || cacheAvailable
      ? reason === "device-recovery" ? "Retry recovery" : "Retry load"
      : "Retry initialization";
  } finally {
    modelLifecycleBusy = false;
    loadButton.disabled = !navigator.gpu ||
      (!localModelAvailable && !cacheAvailable && typeof indexedDB === "undefined");
    generateButton.disabled = !session || !validateControls();
  }
}

function watchSessionDeviceLoss(target: GemmaGenerationSession): void {
  void target.deviceLost.then((info) => {
    if (session !== target) return;
    queueDeviceRecovery(target, info.message || "WebGPU device lost");
  });
}

function queueDeviceRecovery(target: GemmaGenerationSession, detail: string): void {
  if (session !== target && lostSession !== target) return;
  session = null;
  lostSession = target;
  deviceRecoveryPending = true;
  setBadge(gpuStatus, "WebGPU recovering", "pending");
  setBadge(modelStatus, "Engine recovery queued", "pending");
  requestStatus.textContent = detail;
  generateButton.disabled = true;
  loadButton.disabled = true;
  generationController?.abort(new DOMException("WebGPU device lost", "AbortError"));
  void recoverDeviceSessionWhenIdle();
}

async function recoverDeviceSessionWhenIdle(): Promise<void> {
  if (!deviceRecoveryPending || generationController || benchmarkController || modelLifecycleBusy) return;
  deviceRecoveryPending = false;
  lostSession?.destroy();
  lostSession = null;
  await loadModel("device-recovery");
  if (session) setBadge(gpuStatus, "WebGPU ready", "ready");
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
  await verifyModelCache();
  cacheAvailable = true;
  setBadge(cacheStatus, "Cache ready", "ready");
  setModelProgress("Finalizing cache", 1, "Model weights cached in this browser");
}

async function generate(): Promise<void> {
  if (!session || generationController) return;
  const activeSession = session;
  const prompt = promptInput.value.trim();
  if (!prompt) {
    requestStatus.textContent = "Prompt is required";
    promptInput.focus();
    return;
  }

  let options: GemmaGenerationOptions;
  let turn: PreparedGemmaConversationTurn;
  const requestConversation = editBaseConversation ?? conversation;
  const editingRequest = editBaseConversation !== null;
  try {
    options = readGenerationOptions();
    if (activeExample?.requireReasoning) options.requireReasoning = true;
    if (forceFreshGeneration) options.reusePromptCache = false;
    turn = prepareGemmaConversationTurn(
      requestConversation,
      prompt,
      selectedImage ?? undefined,
      readVisionTokenBudget(),
      enableThinkingInput.checked,
      selectedAudio ?? undefined,
    );
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
  options.onAudioProgress = (progress) => {
    const audio = `audio ${progress.audioIndex + 1}/${progress.audioCount}`;
    requestStatus.textContent = progress.phase === "preprocessing"
      ? `Preparing ${audio}`
      : `Encoding ${audio} · layer ${progress.completedLayers}/${progress.totalLayers}`;
  };
  let draftText = "";
  let draftReasoning = "";
  options.onToken = (update) => {
    const draft = visibleGemmaDraft(update.rawText, update.text, enableThinkingInput.checked);
    draftText = draft.text;
    draftReasoning = draft.reasoning;
    renderConversation(turn.userMessage, draftText, "streaming", draftReasoning);
    outputTokenCount.textContent = `${update.generatedTokenIds.length} ${pluralize(update.generatedTokenIds.length, "token")}`;
  };
  setGenerating(true);
  renderConversation(turn.userMessage, "");
  outputTokenCount.textContent = "0 tokens";
  clearTelemetry();
  requestStatus.textContent = "Generating";

  try {
    const measured = await activeSession.generateMeasured(turn.input, options);
    const hasToolCalls = measured.result.toolCalls.length > 0;
    draftText = hasToolCalls
      ? measured.result.toolCalls.map((call) =>
          `${call.function.name}\n${JSON.stringify(call.function.arguments, null, 2)}`)
        .join("\n\n")
      : measured.result.text;
    if (requestConversation.tools.length > 0) {
      renderConversation(
        turn.userMessage,
        draftText || "No tool call output.",
        hasToolCalls ? "tool-call" : "stopped",
        measured.result.reasoning,
      );
    } else if (draftText.trim()) {
      conversation = commitGemmaConversationTurn(
        requestConversation,
        turn,
        measured.result.text,
        measured.result.reasoning,
      );
      forceFreshGeneration = false;
      editBaseConversation = null;
      if (activeWorkspace === "chat") promptInput.value = "";
      clearSelectedImage();
      clearSelectedAudio();
      renderConversation();
    } else {
      renderConversation(turn.userMessage, "No text output.", "failed");
    }
    outputTokenCount.textContent = measured.result.reasoningTokenCount > 0
      ? `${measured.result.generatedTokenIds.length} tokens · ${measured.result.reasoningTokenCount} reasoning`
      : `${measured.result.generatedTokenIds.length} ${pluralize(measured.result.generatedTokenIds.length, "token")}`;
    requestStatus.textContent = hasToolCalls
      ? "Tool call ready"
      : measured.result.stopReason === "length" ? "Token limit reached" : "Generation complete";
    renderTelemetry(
      measured.timing,
      measured.result.stopReason,
      measured.result.generatedTokenIds.length,
    );
    if (session === activeSession) renderMemory(activeSession.estimateRetainedGpuMemory());
  } catch (error) {
    if (generationController.signal.aborted) {
      requestStatus.textContent = deviceRecoveryPending
        ? "WebGPU device lost · recovery queued"
        : "Generation stopped";
      element<HTMLElement>("metric-stop").textContent = deviceRecoveryPending
        ? "device-lost"
        : "cancelled";
      renderConversation(
        turn.userMessage,
        draftText || (deviceRecoveryPending ? "WebGPU device lost" : ""),
        deviceRecoveryPending ? "failed" : "stopped",
        draftReasoning,
      );
    } else {
      const message = errorMessage(error);
      const incompleteConstraint = options.constraint &&
        /constraint|constrained|max tokens/i.test(message);
      requestStatus.textContent = incompleteConstraint
        ? "No valid constrained output · increase Max tokens"
        : message;
      element<HTMLElement>("metric-stop").textContent = "error";
      renderConversation(
        turn.userMessage,
        incompleteConstraint
          ? "No valid constrained output. Increase Max tokens and run again."
          : draftText || message,
        "failed",
        draftReasoning,
      );
      if (isDeviceFailure(error)) {
        queueDeviceRecovery(activeSession, errorMessage(error));
      }
    }
    cancelConversationEdit();
    if (editingRequest) renderConversation();
  } finally {
    generationController = null;
    setGenerating(false);
    validateControls();
    void recoverDeviceSessionWhenIdle();
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

function visibleGemmaDraft(
  rawText: string | undefined,
  decodedText: string,
  thinking: boolean,
): { text: string; reasoning: string } {
  if (!thinking || !rawText?.includes("<|channel>thought")) {
    return { text: decodedText, reasoning: "" };
  }
  const reasoningStart = rawText.indexOf("<|channel>thought") + "<|channel>thought".length;
  const reasoningEnd = rawText.indexOf("<channel|>", reasoningStart);
  if (reasoningEnd < 0) {
    return { text: "", reasoning: rawText.slice(reasoningStart).trim() };
  }
  return parseGemmaResponse(rawText, decodedText);
}

function applyGenerationExample(example: GenerationExample): void {
  controlsForm.reset();
  enableThinkingInput.checked = example.enableThinking ?? false;
  conversation = createGemmaConversation(example.history, [], example.tools);
  renderConversation();
  newChatButton.disabled = !hasConversationState();
  promptInput.value = example.prompt;
  visionTokenBudgetInput.value = String(example.visionTokenBudget ?? 140);
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
  activeWorkspace = "example";
  activeExample = example;
  exampleSelect.value = example.id;
  renderConsoleProfile();
  clearSelectedImage();
  clearSelectedAudio();
  applyGenerationExample(example);
  try {
    if (example.image) {
      const image = await loadExampleImage(example.image.url);
      if (exampleSelect.value !== example.id) return;
      selectImage(image, false, example.image.url, example.image.filename);
    }
    if (example.audio) {
      const audio = await loadExampleAudio(example.audio.url, example.audio.filename);
      if (exampleSelect.value !== example.id) return;
      selectAudio(audio, false);
    }
  } catch (error) {
    if (exampleSelect.value !== example.id) return;
    activateWorkspace("custom");
    clearSelectedImage();
    configStatus.textContent = `Could not load example media: ${errorMessage(error)}`;
    configStatus.dataset.valid = "false";
    generateButton.disabled = true;
  }
}

async function loadExampleAudio(url: string, filename: string): Promise<File> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  const audio = await response.blob();
  return new File([audio], filename, { type: audio.type || "audio/wav" });
}

async function loadExampleImage(url: string): Promise<ImageData> {
  const image = new Image();
  image.src = url;
  await image.decode();
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Could not decode example image");
  context.drawImage(image, 0, 0);
  return context.getImageData(0, 0, canvas.width, canvas.height);
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
    if (session && !generationController) {
      generateButton.disabled = promptInput.value.trim().length === 0 || audioRecorder !== null;
    }
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
      ? "Enter a message to calculate context usage"
      : "Load the model to calculate context usage";
    promptBudget.dataset.valid = "true";
    return;
  }
  const turn = prepareGemmaConversationTurn(
    editBaseConversation ?? conversation,
    prompt,
    selectedImage ?? undefined,
    readVisionTokenBudget(),
    false,
    selectedAudio ?? undefined,
  );
  const promptTokens = session.promptTokenCount(turn.input);
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

function readVisionTokenBudget(): GemmaVisionTokenBudget {
  const tokenBudget = Number(visionTokenBudgetInput.value);
  validateGemmaVisionTokenBudget(tokenBudget);
  return tokenBudget;
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
  const editor = activeConsoleProfile().constraint;
  constraintSection.hidden = editor === "none";
  constraintModeSelector.hidden = editor !== "all";
  const mode = editor === "all" ? String(data.get("constraintMode")) : editor;
  element<HTMLElement>("constraint-regex").hidden = mode !== "regex";
  element<HTMLElement>("constraint-json").hidden = mode !== "json" && mode !== "json-schema";
  element<HTMLElement>("constraint-json-schema").hidden = mode !== "json-schema";
}

function activeConsoleProfile(): ConsoleUiProfile {
  if (activeWorkspace === "custom") {
    return { workspace: "custom", vision: true, audio: true, sampling: true, constraint: "all" };
  }
  if (activeWorkspace === "chat") {
    return { workspace: "chat", vision: true, audio: true, sampling: true, constraint: "none" };
  }
  if (!activeExample) throw new Error("Example workspace is missing its UI profile");
  return { workspace: "example", audio: false, ...activeExample.ui };
}

function renderConsoleProfile(): void {
  const profile = activeConsoleProfile();
  generationPanel.dataset.workspace = profile.workspace;
  if (profile.workspace === "chat") generationPanel.insertBefore(ownedOutputTool, generationForm);
  else generationPanel.insertBefore(ownedOutputTool, ownedTelemetry);
  element<HTMLElement>("generation-heading").textContent = profile.workspace === "chat"
    ? "Chat"
    : "Conversation";
  element<HTMLElement>("output-heading").textContent = profile.workspace === "chat"
    ? "Messages"
    : "Transcript";
  generateButton.textContent = profile.workspace === "chat" ? "Send" : "Generate";
  promptInput.rows = profile.workspace === "chat" ? 3 : 6;
  imageControls.hidden = !profile.vision && !profile.audio;
  imagePicker.hidden = !profile.vision;
  imagePreview.hidden = !profile.vision || !selectedImage;
  audioPicker.hidden = !profile.audio;
  recordAudioButton.hidden = !profile.audio;
  audioPreview.hidden = !profile.audio || !selectedAudio;
  imageBudget.hidden = !profile.vision || (profile.workspace === "chat" && !selectedImage);
  thinkingControl.hidden = profile.workspace === "example" && !activeExample?.enableThinking;
  for (const control of samplingControls) control.hidden = !profile.sampling;
  probabilityControls.hidden = !profile.sampling;
  ownedStopTokenControl.hidden = profile.workspace === "example";
  renderConstraintFields();
}

function activateWorkspace(workspace: Exclude<ConsoleWorkspace, "example">): void {
  const enteringChat = workspace === "chat" && activeWorkspace !== "chat";
  activeWorkspace = workspace;
  activeExample = null;
  exampleSelect.value = workspace;
  if (enteringChat) {
    conversation = createGemmaConversation();
    editBaseConversation = null;
    promptInput.value = "";
    clearSelectedImage();
    clearSelectedAudio();
    renderConversation();
  }
  renderConsoleProfile();
}

function markExampleCustomized(): void {
  if (activeWorkspace !== "example") return;
  activateWorkspace("custom");
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
    ? [
        `Preprocess ${formatMilliseconds(timing.visionPreprocessMs)}`,
        `weights ${formatMilliseconds(timing.visionWeightLoadMs)}`,
        `patch ${formatMilliseconds(timing.visionPatchEmbedMs)}`,
        `layer setup ${formatMilliseconds(timing.visionLayerSetupMs)}`,
        `layer execute ${formatMilliseconds(timing.visionLayerExecutionMs)}`,
        `postprocess ${formatMilliseconds(timing.visionPostprocessMs)}`,
      ].join(" · ")
    : "";
  element<HTMLElement>("metric-prefill").textContent = timing.prefillMode;
  element<HTMLElement>("metric-prefill").title = timing.promptTokensReused > 0
    ? `${timing.promptTokensReused.toLocaleString()} prompt tokens reused`
    : "No prompt tokens reused";
  element<HTMLElement>("metric-stop").textContent = stopReason;
}

function renderMemory(memory: ReturnType<GemmaGenerationSession["estimateRetainedGpuMemory"]>): void {
  element<HTMLElement>("metric-memory").textContent = formatBytes(memory.gpuBufferBytes);
  const visionMemory = element<HTMLElement>("metric-vision-memory");
  visionMemory.textContent = formatBytes(memory.visionWeightMaterializedBytes);
  visionMemory.title = memory.visionWeightEntryCount > 0
    ? `${memory.visionWeightEntryCount} cached entries · ${formatBytes(memory.visionWeightSourceBytes)} source weights`
    : "No materialized vision weights cached";
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
  loadButton.disabled = generating || modelLifecycleBusy || deviceRecoveryPending ||
    (!localModelAvailable && !cacheAvailable) || !navigator.gpu;
  newChatButton.disabled = generating || !hasConversationState();
  clearTranscriptButton.disabled = generating;
  exampleSelect.disabled = generating;
  promptInput.disabled = generating;
  imageInput.disabled = generating;
  audioInput.disabled = generating;
  recordAudioButton.disabled = generating;
  removeImageButton.disabled = generating;
  removeAudioButton.disabled = generating;
  for (const control of controlsForm.elements) {
    if (control instanceof HTMLInputElement || control instanceof HTMLSelectElement ||
        control instanceof HTMLTextAreaElement || control instanceof HTMLButtonElement) {
      if (control instanceof HTMLTextAreaElement && control.closest("#constraint-section")) {
        control.readOnly = generating;
        continue;
      }
      control.disabled = generating;
    }
  }
}

function clearTranscript(): void {
  conversation = createGemmaConversation(
    activeExample?.history ?? [],
    [],
    activeExample?.tools ?? [],
  );
  editBaseConversation = null;
  forceFreshGeneration = true;
  renderConversation();
  outputTokenCount.textContent = "0 tokens";
  clearTelemetry();
  newChatButton.disabled = !hasConversationState();
  requestStatus.textContent = session ? "Transcript cleared · next run starts fresh" : "Waiting for model";
  validateControls();
}

function clearConversation(): void {
  conversation = createGemmaConversation();
  editBaseConversation = null;
  promptInput.value = "";
  clearSelectedImage();
  clearSelectedAudio();
  renderConversation();
  outputTokenCount.textContent = "0 tokens";
  newChatButton.disabled = true;
  requestStatus.textContent = session ? "New conversation" : "Waiting for model";
  validateControls();
}

function renderConversation(
  pendingUser?: GemmaChatMessage,
  assistantDraft?: string,
  draftState: "streaming" | "stopped" | "failed" | "tool-call" = "streaming",
  assistantReasoning?: string,
): void {
  output.replaceChildren();
  const displayedConversation = editBaseConversation ?? conversation;
  for (const tool of displayedConversation.tools) output.append(renderTool(tool));
  const messages = pendingUser
    ? [...displayedConversation.messages, pendingUser]
    : [...displayedConversation.messages];
  let imageIndex = 0;
  for (const [messageIndex, message] of messages.entries()) {
    const imageCount = messageImageCount(message);
    const images = displayedConversation.images.slice(imageIndex, imageIndex + imageCount);
    imageIndex += imageCount;
    output.append(renderMessage(message, undefined, messageIndex, images));
  }
  if (pendingUser) {
    output.append(renderMessage({
      role: "assistant",
      content: assistantDraft || "Generating...",
      ...(assistantReasoning ? { reasoning: assistantReasoning } : {}),
    }, draftState));
  }
  if (messages.length === 0 && displayedConversation.tools.length === 0) {
    output.textContent = activeWorkspace === "chat" ? "Start a new conversation." : "No conversation yet.";
  }
  output.scrollTop = output.scrollHeight;
}

function renderTool(tool: GemmaFunctionTool): HTMLElement {
  const article = document.createElement("article");
  article.className = "conversation-message";
  article.dataset.role = "tool";
  const role = document.createElement("span");
  role.className = "conversation-role";
  role.textContent = "Tool";
  const content = document.createElement("div");
  content.className = "conversation-content";
  content.textContent = `${tool.function.name} - ${tool.function.description}`;
  article.append(role, content);
  return article;
}

function renderMessage(
  message: GemmaChatMessage,
  state?: "streaming" | "stopped" | "failed" | "tool-call",
  messageIndex?: number,
  images: readonly GemmaVisionImageSource[] = [],
): HTMLElement {
  const article = document.createElement("article");
  article.className = "conversation-message";
  article.dataset.role = message.role;
  if (state) article.dataset.state = state;
  const role = document.createElement("span");
  role.className = "conversation-role";
  role.textContent = message.role === "assistant" ? "Model" : message.role;
  const content = document.createElement("div");
  content.className = "conversation-content";
  content.textContent = messageContentText(message);
  const body = document.createElement("div");
  body.className = "conversation-body";
  body.append(content);
  if (message.role === "assistant" && message.reasoning) {
    const reasoning = document.createElement("details");
    reasoning.className = "message-reasoning";
    reasoning.open = true;
    const summary = document.createElement("summary");
    summary.textContent = "Reasoning";
    const reasoningContent = document.createElement("div");
    reasoningContent.className = "message-reasoning-content";
    reasoningContent.textContent = message.reasoning;
    reasoning.append(summary, reasoningContent);
    body.append(reasoning);
  }
  if (images.length > 0) {
    const attachments = document.createElement("div");
    attachments.className = "conversation-attachments";
    for (const image of images) attachments.append(renderConversationImage(image));
    body.prepend(attachments);
  }
  if (activeWorkspace === "chat" && message.role === "user" && state === undefined &&
      messageIndex !== undefined) {
    const editButton = document.createElement("button");
    editButton.type = "button";
    editButton.className = "message-edit quiet-command";
    editButton.textContent = "Edit";
    editButton.addEventListener("click", () => beginConversationEdit(messageIndex));
    body.append(editButton);
  }
  article.append(role, body);
  return article;
}

function renderConversationImage(image: GemmaVisionImageSource): HTMLElement {
  const frame = document.createElement("div");
  frame.className = "conversation-image";
  if (image instanceof Blob) {
    const thumbnail = document.createElement("img");
    const url = URL.createObjectURL(image);
    thumbnail.alt = image instanceof File ? image.name : "Attached image";
    thumbnail.src = url;
    thumbnail.addEventListener("load", () => URL.revokeObjectURL(url), { once: true });
    frame.append(thumbnail);
    return frame;
  }
  const canvas = document.createElement("canvas");
  canvas.width = image.width;
  canvas.height = image.height;
  const context = canvas.getContext("2d");
  if (context) {
    if (image instanceof ImageData) context.putImageData(image, 0, 0);
    else context.drawImage(image, 0, 0);
  }
  frame.append(canvas);
  return frame;
}

function beginConversationEdit(messageIndex: number): void {
  const message = conversation.messages[messageIndex];
  if (!message || message.role !== "user") return;
  const edit = prepareGemmaConversationEdit(
    conversation,
    messageIndex,
    messageText(message),
    undefined,
    readVisionTokenBudget(),
  );
  editBaseConversation = edit.conversation;
  promptInput.value = messageText(message);
  if (edit.turn.image) selectImage(edit.turn.image, false);
  else clearSelectedImage();
  requestStatus.textContent = "Editing message";
  renderConversation();
  promptInput.focus();
}

function cancelConversationEdit(): void {
  editBaseConversation = null;
}

function messageImageCount(message: GemmaChatMessage): number {
  return typeof message.content === "string"
    ? 0
    : message.content.filter((part) => part.type === "image").length;
}

function messageText(message: GemmaChatMessage): string {
  return typeof message.content === "string"
    ? message.content
    : message.content.filter((part) => part.type === "text").map((part) => part.text).join("\n");
}

function messageContentText(message: GemmaChatMessage): string {
  if (typeof message.content === "string") return message.content;
  return message.content.map((part) => part.type === "image"
    ? "[Image]"
    : part.type === "audio" ? "[Audio]" : part.text).join("\n");
}

function hasConversationState(): boolean {
  return conversation.messages.length > 0 || conversation.images.length > 0 ||
    conversation.audios.length > 0 || conversation.tools.length > 0;
}

function selectImage(
  image: GemmaVisionImageSource,
  clearExample: boolean,
  previewUrl?: string,
  filename?: string,
): void {
  selectedImage = image;
  if (imagePreviewUrl) URL.revokeObjectURL(imagePreviewUrl);
  imagePreviewUrl = previewUrl || !(image instanceof Blob) ? null : URL.createObjectURL(image);
  imageThumbnail.src = previewUrl ?? imagePreviewUrl!;
  imageName.textContent = filename ?? (image instanceof File ? image.name : "Image");
  imagePreview.hidden = false;
  if (clearExample) markExampleCustomized();
  renderConsoleProfile();
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
  renderConsoleProfile();
  validateControls();
}

function selectAudio(audio: Blob, clearExample = true): void {
  selectedAudio = audio;
  if (audioPreviewUrl) URL.revokeObjectURL(audioPreviewUrl);
  audioPreviewUrl = URL.createObjectURL(audio);
  audioPlayer.src = audioPreviewUrl;
  audioName.textContent = audio instanceof File ? audio.name : "Audio";
  audioPreview.hidden = false;
  if (clearExample) markExampleCustomized();
  validateControls();
}

function clearSelectedAudio(): void {
  selectedAudio = null;
  audioInput.value = "";
  audioPreview.hidden = true;
  audioPlayer.pause();
  audioPlayer.removeAttribute("src");
  audioPlayer.load();
  audioName.textContent = "";
  if (audioPreviewUrl) URL.revokeObjectURL(audioPreviewUrl);
  audioPreviewUrl = null;
  validateControls();
}

async function toggleAudioRecording(): Promise<void> {
  if (audioRecorder) {
    await finishAudioRecording(audioRecorder);
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia || typeof AudioContext === "undefined") {
    showRecordingStatus("Microphone recording is not supported in this browser", "error");
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    audioRecorder = await startGemmaMicrophoneCapture(stream);
    audioRecordingStartedAt = performance.now();
    recordAudioButton.textContent = "Stop recording";
    recordAudioButton.dataset.recording = "true";
    showRecordingStatus("Recording 0:00", "recording");
    audioRecordingTimer = window.setInterval(renderRecordingDuration, 250);
    validateControls();
  } catch (error) {
    showRecordingStatus(`Microphone unavailable: ${errorMessage(error)}`, "error");
    releaseAudioRecorder();
  }
}

async function finishAudioRecording(capture: GemmaMicrophoneCapture): Promise<void> {
  try {
    const wav = await capture.stop();
    const recording = new File([wav], "microphone-recording.wav", { type: wav.type });
    releaseAudioRecorder();
    recordingStatus.hidden = true;
    selectAudio(recording);
  } catch (error) {
    releaseAudioRecorder();
    showRecordingStatus(errorMessage(error), "error");
  }
}

function renderRecordingDuration(): void {
  const elapsedSeconds = Math.floor((performance.now() - audioRecordingStartedAt) / 1_000);
  const minutes = Math.floor(elapsedSeconds / 60);
  const seconds = String(elapsedSeconds % 60).padStart(2, "0");
  showRecordingStatus(`Recording ${minutes}:${seconds}`, "recording");
}

function showRecordingStatus(message: string, state: "recording" | "error"): void {
  recordingStatus.textContent = message;
  recordingStatus.dataset.state = state;
  recordingStatus.hidden = false;
}

function releaseAudioRecorder(): void {
  if (audioRecordingTimer !== null) window.clearInterval(audioRecordingTimer);
  audioRecordingTimer = null;
  audioRecorder = null;
  recordAudioButton.textContent = "Record mic";
  delete recordAudioButton.dataset.recording;
  validateControls();
}

function discardAudioRecording(): void {
  void audioRecorder?.discard();
  releaseAudioRecorder();
}

function setBadge(target: HTMLElement, text: string, state: string): void {
  target.textContent = text;
  target.dataset.state = state;
}

async function inventoryModelCache(): Promise<boolean> {
  try {
    await verifyModelCache();
    return true;
  } catch {
    return false;
  }
}

async function verifyModelCache(): Promise<void> {
  let cache: ReadonlySafetensorsCache | null = null;
  try {
    cache = await ReadonlySafetensorsCache.open();
    for (const name of MODEL_CACHE_SENTINELS) {
      try {
        await cache.readTensorSlice(name, 0, 1);
      } catch (error) {
        throw new Error(`Cache verification failed for ${name}: ${errorMessage(error)}`);
      }
    }
  } finally {
    cache?.close();
  }
}

async function modelCacheDatabaseExists(): Promise<boolean> {
  return (await indexedDB.databases()).some(
    (candidate) => candidate.name === GEMMA_4_E2B_CACHE_SPEC.databaseName,
  );
}

async function localModelExists(): Promise<boolean> {
  try {
    const response = await fetch(GEMMA_LOCAL_SAFETENSORS_URL, { method: "HEAD" });
    return response.ok &&
      response.headers.get("content-length") === String(GEMMA_4_E2B_CACHE_SPEC.fileSize);
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
