import { expect, test, type Page } from "@playwright/test";
import { execFileSync } from "node:child_process";
import { appendFile, mkdir, rename, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import type {
  ReliabilitySmokeInitialization,
  ReliabilitySmokeScenario,
  ReliabilitySmokeScenarioResult,
} from "../src/reliability/smoke-harness";

const ENABLED = process.env.RELIABILITY_SMOKE === "1";
const DURATION_MINUTES = parseDurationMinutes(process.env.RELIABILITY_SMOKE_MINUTES);
const TARGET_DURATION_MS = DURATION_MINUTES * 60_000;
const SOURCE_URL = "/models/gemma-4-e2b/model.safetensors";
const MANDATORY_SCENARIOS: readonly ReliabilitySmokeScenario[] = [
  "golden",
  "sampling",
  "regex-constraint",
  "json-constraint",
  "prefix-reuse",
  "cancellation-recovery",
  "vision",
  "audio",
  "lifecycle-reload",
  "device-loss-recovery",
];
const STEADY_SCENARIOS: readonly ReliabilitySmokeScenario[] = [
  "golden",
  "sampling",
  "regex-constraint",
  "golden",
  "json-constraint",
  "prefix-reuse",
  "golden",
  "cancellation-recovery",
];

test.skip(!ENABLED, "Set RELIABILITY_SMOKE=1 to run the cached-model reliability smoke");
test.setTimeout(Math.max(15, DURATION_MINUTES + 15) * 60_000);

test("runs the durable Owned WebGPU reliability smoke", async ({ page }) => {
  const startedAtMs = Date.now();
  const outputDirectory = process.env.RELIABILITY_SMOKE_OUTPUT_DIR
    ? path.resolve(process.env.RELIABILITY_SMOKE_OUTPUT_DIR)
    : defaultOutputDirectory();
  const eventsPath = path.join(outputDirectory, "events.jsonl");
  const progressPath = path.join(outputDirectory, "progress.json");
  const summaryPath = path.join(outputDirectory, "summary.json");
  await mkdir(outputDirectory, { recursive: true });
  await writeFile(eventsPath, "");
  console.log(`[reliability] output: ${outputDirectory}`);

  let currentScenario: ReliabilitySmokeScenario | "initializing" | null = "initializing";
  const browserErrors: string[] = [];
  const expectedBrowserEvents: string[] = [];
  page.on("pageerror", (error) => browserErrors.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() !== "error") return;
    const event = `console: ${message.text()}`;
    if (currentScenario === "device-loss-recovery" &&
        message.text().startsWith("WebGPU device lost GPUDeviceLostInfo")) {
      expectedBrowserEvents.push(event);
      return;
    }
    browserErrors.push(event);
  });

  const environment = await captureEnvironment(page);
  const events: ReliabilityEvent[] = [];
  let initialization: ReliabilitySmokeInitialization | null = null;
  let failure: Error | null = null;
  await writeProgress(progressPath, createProgress({
    startedAtMs,
    targetDurationMs: TARGET_DURATION_MS,
    events,
    currentScenario,
    status: "running",
    browserErrors,
  }));

  try {
    await page.goto("/");
    initialization = await initialize(page);
    await appendEvent(eventsPath, {
      schemaVersion: 1,
      sequence: -1,
      scenario: "initialize",
      startedAt: new Date(startedAtMs).toISOString(),
      completedAt: new Date().toISOString(),
      durationMs: initialization.loadMs,
      passed: true,
      details: initialization,
    });

    let sequence = 0;
    for (const scenario of MANDATORY_SCENARIOS) {
      currentScenario = scenario;
      await writeProgress(progressPath, createProgress({
        startedAtMs,
        targetDurationMs: TARGET_DURATION_MS,
        events,
        currentScenario,
        status: "running",
        browserErrors,
      }));
      events.push(await runScenario(page, eventsPath, scenario, sequence));
      sequence += 1;
      printProgress(startedAtMs, TARGET_DURATION_MS, events.at(-1)!);
    }

    while (Date.now() - startedAtMs < TARGET_DURATION_MS) {
      const scenario = selectSteadyScenario(sequence);
      currentScenario = scenario;
      await writeProgress(progressPath, createProgress({
        startedAtMs,
        targetDurationMs: TARGET_DURATION_MS,
        events,
        currentScenario,
        status: "running",
        browserErrors,
      }));
      events.push(await runScenario(page, eventsPath, scenario, sequence));
      sequence += 1;
      printProgress(startedAtMs, TARGET_DURATION_MS, events.at(-1)!);
    }
  } catch (error) {
    failure = error instanceof Error ? error : new Error(String(error));
  } finally {
    currentScenario = null;
    try {
      await dispose(page);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      browserErrors.push(`dispose: ${message}`);
      failure ??= new Error(`Reliability smoke disposal failed: ${message}`);
    }
  }

  const summary = createSummary({
    startedAtMs,
    targetDurationMs: TARGET_DURATION_MS,
    initialization,
    events,
    browserErrors,
    expectedBrowserEvents,
    environment,
    failure,
  });
  await atomicWriteJson(summaryPath, summary);
  await writeProgress(progressPath, createProgress({
    startedAtMs,
    targetDurationMs: TARGET_DURATION_MS,
    events,
    currentScenario,
    status: summary.passed ? "completed" : "failed",
    browserErrors,
    failure: failure?.message ?? null,
  }));

  expect(summary.browserErrors, "Browser errors were recorded").toEqual([]);
  expect(summary.failure, "Reliability scenario failed").toBeNull();
  expect(summary.missingMandatoryScenarios, "Mandatory scenarios did not complete").toEqual([]);
  expect(summary.memory.gpuBufferByteSpread, "Retained GPU memory did not plateau")
    .toBeLessThanOrEqual(1024 * 1024);
  expect(summary.passed).toBe(true);
});

interface ReliabilityEvent {
  schemaVersion: 1;
  sequence: number;
  scenario: ReliabilitySmokeScenario;
  startedAt: string;
  completedAt: string;
  durationMs: number;
  passed: boolean;
  result?: ReliabilitySmokeScenarioResult;
  error?: string;
}

interface ProgressOptions {
  startedAtMs: number;
  targetDurationMs: number;
  events: readonly ReliabilityEvent[];
  currentScenario: ReliabilitySmokeScenario | "initializing" | null;
  status: "running" | "completed" | "failed";
  browserErrors: readonly string[];
  failure?: string | null;
}

async function initialize(page: Page): Promise<ReliabilitySmokeInitialization> {
  return page.evaluate(async ({ sourceUrl }) => {
    const modulePath = "/src/reliability/smoke-harness.ts";
    const harness = await import(modulePath);
    return harness.initializeReliabilitySmoke({ sourceUrl, cacheCapacity: 1024 });
  }, { sourceUrl: SOURCE_URL });
}

async function runScenario(
  page: Page,
  eventsPath: string,
  scenario: ReliabilitySmokeScenario,
  sequence: number,
): Promise<ReliabilityEvent> {
  const startedAt = new Date().toISOString();
  try {
    const result = await page.evaluate(async ({ scenario: selectedScenario, sequence: index }) => {
      const modulePath = "/src/reliability/smoke-harness.ts";
      const harness = await import(modulePath);
      return harness.executeReliabilitySmokeScenario(selectedScenario, index);
    }, { scenario, sequence });
    const event: ReliabilityEvent = {
      schemaVersion: 1,
      sequence,
      scenario,
      startedAt,
      completedAt: new Date().toISOString(),
      durationMs: result.durationMs,
      passed: true,
      result,
    };
    await appendEvent(eventsPath, event);
    return event;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const event: ReliabilityEvent = {
      schemaVersion: 1,
      sequence,
      scenario,
      startedAt,
      completedAt: new Date().toISOString(),
      durationMs: Date.now() - Date.parse(startedAt),
      passed: false,
      error: message,
    };
    await appendEvent(eventsPath, event);
    throw new Error(`Reliability scenario ${scenario} failed: ${message}`, { cause: error });
  }
}

async function dispose(page: Page): Promise<void> {
  if (page.isClosed()) return;
  await page.evaluate(async () => {
    const modulePath = "/src/reliability/smoke-harness.ts";
    const harness = await import(modulePath);
    await harness.disposeReliabilitySmoke();
  });
}

function selectSteadyScenario(sequence: number): ReliabilitySmokeScenario {
  if (sequence % 60 === 0) return "audio";
  if (sequence % 45 === 0) return "device-loss-recovery";
  if (sequence % 40 === 0) return "vision";
  if (sequence % 30 === 0) return "lifecycle-reload";
  return STEADY_SCENARIOS[sequence % STEADY_SCENARIOS.length];
}

function createProgress(options: ProgressOptions) {
  const elapsedMs = Date.now() - options.startedAtMs;
  const completed = options.events.filter((event) => event.passed).length;
  const failures = options.events.filter((event) => !event.passed).length;
  return {
    schemaVersion: 1,
    profile: "smoke",
    status: options.status,
    phase: options.currentScenario ?? options.status,
    currentScenario: options.currentScenario,
    completed,
    failures,
    browserErrors: options.browserErrors.length,
    startedAt: new Date(options.startedAtMs).toISOString(),
    updatedAt: new Date().toISOString(),
    targetDurationMs: options.targetDurationMs,
    elapsedMs,
    remainingMs: Math.max(0, options.targetDurationMs - elapsedMs),
    percentComplete: options.targetDurationMs === 0
      ? 100
      : Math.min(100, Math.round(elapsedMs / options.targetDurationMs * 10_000) / 100),
    failure: options.failure ?? null,
    latest: options.events.at(-1) ?? null,
  };
}

function createSummary(options: {
  startedAtMs: number;
  targetDurationMs: number;
  initialization: ReliabilitySmokeInitialization | null;
  events: readonly ReliabilityEvent[];
  browserErrors: readonly string[];
  expectedBrowserEvents: readonly string[];
  environment: Awaited<ReturnType<typeof captureEnvironment>>;
  failure: Error | null;
}) {
  const completedEvents = options.events.filter((event) => event.passed && event.result);
  const completedScenarios = new Set(completedEvents.map((event) => event.scenario));
  const missingMandatoryScenarios = MANDATORY_SCENARIOS.filter((scenario) =>
    !completedScenarios.has(scenario)
  );
  const gpuBufferBytes = completedEvents.flatMap((event) =>
    event.result ? [event.result.memory.gpuBufferBytes] : []
  );
  const gpuBufferCounts = completedEvents.flatMap((event) =>
    event.result ? [event.result.memory.gpuBufferCount] : []
  );
  const scenarioCounts = Object.fromEntries(MANDATORY_SCENARIOS.map((scenario) => [
    scenario,
    completedEvents.filter((event) => event.scenario === scenario).length,
  ]));
  const gpuBufferByteMinimum = gpuBufferBytes.length > 0 ? Math.min(...gpuBufferBytes) : null;
  const gpuBufferByteMaximum = gpuBufferBytes.length > 0 ? Math.max(...gpuBufferBytes) : null;
  const gpuBufferByteSpread = gpuBufferByteMinimum === null || gpuBufferByteMaximum === null
    ? null
    : gpuBufferByteMaximum - gpuBufferByteMinimum;
  const passed = options.failure === null && options.browserErrors.length === 0 &&
    missingMandatoryScenarios.length === 0 && gpuBufferByteSpread !== null &&
    gpuBufferByteSpread <= 1024 * 1024;
  return {
    schemaVersion: 1,
    profile: "smoke",
    passed,
    startedAt: new Date(options.startedAtMs).toISOString(),
    completedAt: new Date().toISOString(),
    elapsedMs: Date.now() - options.startedAtMs,
    targetDurationMs: options.targetDurationMs,
    environment: options.environment,
    initialization: options.initialization,
    completedScenarios: completedEvents.length,
    scenarioCounts,
    missingMandatoryScenarios,
    failedEvents: options.events.filter((event) => !event.passed),
    browserErrors: options.browserErrors,
    expectedBrowserEvents: options.expectedBrowserEvents,
    failure: options.failure?.message ?? null,
    memory: {
      gpuBufferByteMinimum,
      gpuBufferByteMaximum,
      gpuBufferByteSpread,
      gpuBufferCountMinimum: gpuBufferCounts.length > 0 ? Math.min(...gpuBufferCounts) : null,
      gpuBufferCountMaximum: gpuBufferCounts.length > 0 ? Math.max(...gpuBufferCounts) : null,
    },
  };
}

async function captureEnvironment(page: Page) {
  const browser = await page.evaluate(async () => {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    return {
      userAgent: navigator.userAgent,
      language: navigator.language,
      visibilityState: document.visibilityState,
      adapterInfo: adapter?.info ? {
        vendor: adapter.info.vendor,
        architecture: adapter.info.architecture,
        device: adapter.info.device,
        description: adapter.info.description,
      } : null,
      features: adapter ? [...adapter.features].sort() : [],
    };
  });
  return {
    capturedAt: new Date().toISOString(),
    gitCommit: currentGitCommit(),
    operatingSystem: `${os.type()} ${os.release()} ${os.arch()}`,
    physicalDevice: os.hostname(),
    cpu: os.cpus()[0]?.model ?? "unknown",
    totalRamBytes: os.totalmem(),
    browser,
  };
}

async function appendEvent(filePath: string, event: unknown): Promise<void> {
  await appendFile(filePath, `${JSON.stringify(event)}\n`);
}

async function writeProgress(filePath: string, progress: unknown): Promise<void> {
  await atomicWriteJson(filePath, progress);
}

async function atomicWriteJson(filePath: string, value: unknown): Promise<void> {
  const temporaryPath = `${filePath}.tmp`;
  await writeFile(temporaryPath, `${JSON.stringify(value, null, 2)}\n`);
  await rename(temporaryPath, filePath);
}

function printProgress(
  startedAtMs: number,
  targetDurationMs: number,
  event: ReliabilityEvent,
): void {
  const elapsedMs = Date.now() - startedAtMs;
  const percent = targetDurationMs === 0
    ? 100
    : Math.min(100, Math.round(elapsedMs / targetDurationMs * 1_000) / 10);
  console.log(
    `[reliability] ${percent.toFixed(1)}% #${event.sequence} ${event.scenario} ` +
    `${(event.durationMs / 1000).toFixed(2)}s`,
  );
}

function defaultOutputDirectory(): string {
  const timestamp = new Date().toISOString().replaceAll(":", "-").replace(".", "-");
  return path.join(process.cwd(), "benchmarks", "reliability", `${timestamp}-smoke`);
}

function parseDurationMinutes(value: string | undefined): number {
  if (value === undefined) return 20;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0 || parsed > 40) {
    throw new Error("RELIABILITY_SMOKE_MINUTES must be between 0 and 40");
  }
  return parsed;
}

function currentGitCommit(): string {
  try {
    return execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim();
  } catch {
    return "unknown";
  }
}