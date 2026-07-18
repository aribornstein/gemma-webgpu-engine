import { expect, test, chromium, type BrowserContext, type Page } from "@playwright/test";
import { execFileSync } from "node:child_process";
import { appendFile, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { renderBenchmarkArtifacts } from "../src/benchmark/suite/report";
import { createBenchmarkSchedule } from "../src/benchmark/suite/schedule";
import { summarizeBenchmarkRuns } from "../src/benchmark/suite/summary";
import type {
  BenchmarkCase,
  BenchmarkEnvironment,
  BenchmarkScheduleEntry,
  BrowserMode,
  LoadOptions,
  RawBenchmarkRun,
} from "../src/benchmark/suite/types";
import { createBenchmarkWorkloads, createSmokeWorkload } from "../src/benchmark/suite/workloads";

type RuntimeId = "owned-webgpu" | "transformers-js" | "litert-lm-web";

const ENABLED = process.env.BENCHMARK_SUITE_LIVE === "1";
const PROFILE = process.env.BENCHMARK_SUITE_PROFILE === "smoke" ? "smoke" : "full";
const BROWSER_MODE: BrowserMode = process.env.BENCHMARK_BROWSER_MODE === "headed" ? "headed" : "headless";
const SEED = Number(process.env.BENCHMARK_SEED ?? 20260717);
const RUNTIMES = parseRuntimeIds(process.env.BENCHMARK_RUNTIMES);
const SKIPPED_WARM_RUNTIMES = parseOptionalRuntimeIds(process.env.BENCHMARK_SKIP_WARM_RUNTIMES);
const FULL_LIMITATIONS = [
  "Track B compares complete deployable browser stacks and does not isolate library implementation performance.",
  "Token counts are runtime-tokenizer-specific; cross-artifact tokens/second is not a unit-invariant measure.",
  "LiteRT-LM exposes chunk callbacks rather than guaranteed token callbacks, so only chunk intervals are reported.",
  "Pinned Hugging Face current-browser execution is unavailable because its pinned browser bundle is absent.",
  "The 8,192-token input row is executed only by runtimes whose configured context can also hold the requested output.",
];

test.skip(!ENABLED, "Set BENCHMARK_SUITE_LIVE=1 to run multi-GB rigorous benchmarks");
test.setTimeout(8 * 60 * 60_000);

test("runs the reproducible browser benchmark suite", async () => {
  const profile = PROFILE === "smoke"
    ? { workloads: [createSmokeWorkload()], warmups: 1, measured: 2, startup: 1, conversation: 2 }
    : { workloads: [...createBenchmarkWorkloads()], warmups: 5, measured: 30, startup: 3, conversation: 30 };
  const root = path.join(process.cwd(), "test-results", "benchmark-suite-profiles", BROWSER_MODE);
  await mkdir(root, { recursive: true });
  const outputDirectory = process.env.BENCHMARK_SUITE_RESUME_DIR
    ? path.resolve(process.env.BENCHMARK_SUITE_RESUME_DIR)
    : outputPath(BROWSER_MODE, PROFILE);
  const rawResultsPath = path.join(outputDirectory, "raw-results.jsonl");
  const progressPath = path.join(outputDirectory, "progress.json");
  await mkdir(outputDirectory, { recursive: true });
  const runs = process.env.BENCHMARK_SUITE_RESUME_DIR
    ? await readCheckpointRuns(rawResultsPath)
    : [];
  if (!process.env.BENCHMARK_SUITE_RESUME_DIR) await writeFile(rawResultsPath, "");
  const completedRunIds = new Set(runs.map((run) => run.runId));
  const completedAtStart = completedRunIds.size;
  const unavailableRuntimes = new Map<RuntimeId, string>(SKIPPED_WARM_RUNTIMES.map((runtimeId) => [
    runtimeId,
    `${runtimeId} warm runtime disabled by BENCHMARK_SKIP_WARM_RUNTIMES after external failure`,
  ]));
  const startupWorkloads = PROFILE === "full" ? [profile.workloads[0]] : profile.workloads;
  const allSchedule = createAllSchedule(profile, startupWorkloads);
  const startedAtMs = Date.now();
  await writeProgress(progressPath, progressForRun(
    allSchedule,
    completedRunIds,
    runs,
    startedAtMs,
    completedAtStart,
    null,
    unavailableRuntimes,
  ));
  const checkpoint = async (run: RawBenchmarkRun) => {
    if (completedRunIds.has(run.runId)) return;
    runs.push(run);
    completedRunIds.add(run.runId);
    await appendFile(rawResultsPath, `${JSON.stringify(run)}\n`);
    const progress = progressForRun(
      allSchedule,
      completedRunIds,
      runs,
      startedAtMs,
      completedAtStart,
      run,
      unavailableRuntimes,
    );
    await writeProgress(progressPath, progress);
    printProgress(progress);
  };
  const limitations = [...FULL_LIMITATIONS, ...unavailableRuntimes.values()];
  let capturedEnvironment: BenchmarkEnvironment | null = null;
  const reportWarmupProgress = async (progress: WarmupProgress) => {
    await writeProgress(progressPath, {
      ...progressForRun(
        allSchedule,
        completedRunIds,
        runs,
        startedAtMs,
        completedAtStart,
        null,
        unavailableRuntimes,
      ),
      phase: "warmup",
      currentRuntime: progress.runtimeId,
      currentWorkload: progress.workloadId,
      phaseProgress: {
        completed: progress.completed,
        total: progress.total,
        unit: "warmup-generations",
        attempt: progress.attempt,
        workloadIndex: progress.workloadIndex,
        workloadTotal: progress.workloadTotal,
      },
    });
  };
  const networkSchedule = scheduleFor("network-cold-startup", startupWorkloads, profile.startup, 0);
  for (const entry of networkSchedule) {
    if (!supportsEntry(entry, startupWorkloads) || completedRunIds.has(runIdFor(entry))) continue;
    await writePhaseProgress(
      progressPath,
      allSchedule,
      completedRunIds,
      runs,
      startedAtMs,
      completedAtStart,
      unavailableRuntimes,
      entry,
    );
    const profilePath = path.join(root, entry.runtimeId);
    await rm(profilePath, { recursive: true, force: true });
    const instance = await launchRuntime(profilePath);
    try {
      await clearColdStorage(instance.page, instance.context);
      const monitor = await startNetworkMonitor(instance.context, instance.page);
      const initialized = await initializeRuntime(instance.page, entry.runtimeId as RuntimeId, {
        mode: "network-cold-startup",
        cacheCapacity: contextCapacity(entry.runtimeId),
        ...(entry.runtimeId === "owned-webgpu" ? { sourceUrl: "/models/gemma-4-e2b/model.safetensors" } : {}),
      });
      const network = await monitor.finish();
      await applyNetworkObservation(instance.page, network.bytes, network.durationMs);
      capturedEnvironment ??= initialized.environment;
      limitations.push(...initialized.limitations);
      const workload = await calibrateOrRetain(instance.page, workloadFor(entry, startupWorkloads), limitations);
      await checkpoint(await executeEntry(instance.page, workload, entry));
    } finally {
      await disposeRuntime(instance.page);
      await instance.context.close();
    }
  }

  const cachedSchedule = scheduleFor(
    "cached-cold-startup",
    startupWorkloads,
    profile.startup,
    networkSchedule.length,
  );
  for (const entry of cachedSchedule) {
    if (!supportsEntry(entry, startupWorkloads) || completedRunIds.has(runIdFor(entry))) continue;
    await writePhaseProgress(
      progressPath,
      allSchedule,
      completedRunIds,
      runs,
      startedAtMs,
      completedAtStart,
      unavailableRuntimes,
      entry,
    );
    const instance = await launchRuntime(path.join(root, entry.runtimeId));
    try {
      const monitor = await startNetworkMonitor(instance.context, instance.page);
      const initialized = await initializeRuntime(instance.page, entry.runtimeId as RuntimeId, {
        mode: "cached-cold-startup",
        cacheCapacity: contextCapacity(entry.runtimeId),
        ...(entry.runtimeId === "owned-webgpu" ? { sourceUrl: "/models/gemma-4-e2b/model.safetensors" } : {}),
      });
      const network = await monitor.finish();
      await applyNetworkObservation(instance.page, network.bytes, network.durationMs);
      capturedEnvironment ??= initialized.environment;
      const workload = await calibrateOrRetain(instance.page, workloadFor(entry, startupWorkloads), limitations);
      await checkpoint(await executeEntry(instance.page, workload, entry));
    } finally {
      await disposeRuntime(instance.page);
      await instance.context.close();
    }
  }

  const warmInstances = new Map<RuntimeId, Awaited<ReturnType<typeof launchRuntime>>>();
  const calibrated = new Map<string, BenchmarkCase>();
  const markRuntimeUnavailable = async (runtimeId: RuntimeId, error: unknown) => {
    const reason = `${runtimeId} warm runtime unavailable: ${String(error)}`;
    unavailableRuntimes.set(runtimeId, reason);
    limitations.push(reason);
    const instance = warmInstances.get(runtimeId);
    if (instance) {
      await disposeRuntime(instance.page).catch(() => {});
      await instance.context.close().catch(() => {});
      warmInstances.delete(runtimeId);
    }
    const progress = progressForRun(
      allSchedule,
      completedRunIds,
      runs,
      startedAtMs,
      completedAtStart,
      null,
      unavailableRuntimes,
    );
    await writeProgress(progressPath, {
      ...progress,
      phase: "runtime-unavailable",
      currentRuntime: runtimeId,
    });
    console.warn(`[benchmark] ${reason}; continuing remaining runtimes`);
  };
  try {
    for (const runtimeId of RUNTIMES) {
      if (unavailableRuntimes.has(runtimeId)) continue;
      await writeProgress(progressPath, {
        ...progressForRun(
          allSchedule,
          completedRunIds,
          runs,
          startedAtMs,
          completedAtStart,
          null,
          unavailableRuntimes,
        ),
        phase: "warmup",
        currentRuntime: runtimeId,
      });
      try {
        const initialized = await recoverWarmRuntimeWithRetry(
          runtimeId,
          warmInstances,
          calibrated,
          root,
          profile.workloads,
          profile.warmups,
          limitations,
          reportWarmupProgress,
        );
        capturedEnvironment ??= initialized.environment;
      } catch (error) {
        await markRuntimeUnavailable(runtimeId, error);
      }
    }

    const warmSchedule = scheduleFor(
      "warm-steady-state",
      profile.workloads,
      profile.measured,
      networkSchedule.length + cachedSchedule.length,
    );
    for (const entry of warmSchedule) {
      const runtimeId = entry.runtimeId as RuntimeId;
      if (unavailableRuntimes.has(runtimeId) || !supportsEntry(entry, profile.workloads) || completedRunIds.has(runIdFor(entry))) continue;
      await writePhaseProgress(
        progressPath,
        allSchedule,
        completedRunIds,
        runs,
        startedAtMs,
        completedAtStart,
        unavailableRuntimes,
        entry,
      );
      const instance = required(warmInstances.get(runtimeId), `Missing ${runtimeId} instance`);
      const workload = required(calibrated.get(`${runtimeId}\u0000${entry.workloadId}`), "Missing calibrated workload");
      const run = await executeEntryOrTransportFailure(instance.page, workload, entry, runs);
      await checkpoint(run);
      if (run.correctness.error) {
        try {
          await recoverWarmRuntimeWithRetry(
            runtimeId,
            warmInstances,
            calibrated,
            root,
            profile.workloads,
            profile.warmups,
            limitations,
            reportWarmupProgress,
          );
        } catch (error) {
          await markRuntimeUnavailable(runtimeId, error);
        }
      }
      await cooldown();
    }

    const conversationSchedule = scheduleFor(
      "conversation-cache",
      profile.workloads,
      profile.conversation,
      networkSchedule.length + cachedSchedule.length + warmSchedule.length,
      true,
    );
    for (const entry of conversationSchedule) {
      const runtimeId = entry.runtimeId as RuntimeId;
      if (unavailableRuntimes.has(runtimeId) || !supportsEntry(entry, profile.workloads) || completedRunIds.has(runIdFor(entry))) continue;
      await writePhaseProgress(
        progressPath,
        allSchedule,
        completedRunIds,
        runs,
        startedAtMs,
        completedAtStart,
        unavailableRuntimes,
        entry,
      );
      const instance = required(warmInstances.get(runtimeId), `Missing ${runtimeId} instance`);
      const workload = required(calibrated.get(`${runtimeId}\u0000${entry.workloadId}`), "Missing calibrated workload");
      const run = await executeEntryOrTransportFailure(
        instance.page,
        workload,
        entry,
        runs,
        entry.conversationVariant === "reused",
      );
      await checkpoint(run);
      if (run.correctness.error) {
        try {
          await recoverWarmRuntimeWithRetry(
            runtimeId,
            warmInstances,
            calibrated,
            root,
            profile.workloads,
            profile.warmups,
            limitations,
            reportWarmupProgress,
          );
        } catch (error) {
          await markRuntimeUnavailable(runtimeId, error);
        }
      }
      await cooldown();
    }

    const environment = required(capturedEnvironment, "No benchmark environment was captured");
    const summary = summarizeBenchmarkRuns(runs, environment, allSchedule, limitations);
    const reports = renderBenchmarkArtifacts(runs, summary);
    await Promise.all([
      writeFile(rawResultsPath, reports.rawResultsJsonl),
      writeFile(path.join(outputDirectory, "summary.json"), reports.summaryJson),
      writeFile(path.join(outputDirectory, "report.md"), reports.reportMarkdown),
      writeFile(path.join(outputDirectory, "report.csv"), reports.reportCsv),
      writeFile(path.join(outputDirectory, "report.html"), reports.reportHtml),
      writeProgress(progressPath, {
        status: "completed",
        phase: "complete",
        currentRuntime: null,
        currentWorkload: null,
        progressUnit: "measured-runs",
        phaseProgress: null,
        completed: runs.length,
        total: allSchedule.length,
        remaining: 0,
        skipped: allSchedule.length - runs.length,
        errors: runs.filter((run) => run.correctness.error !== null).length,
        startedAt: new Date(startedAtMs).toISOString(),
        updatedAt: new Date().toISOString(),
        elapsedMs: Date.now() - startedAtMs,
        estimatedRemainingMs: 0,
        percentComplete: 100,
        unavailableRuntimes: [...unavailableRuntimes].map(([runtimeId, reason]) => ({ runtimeId, reason })),
      }),
    ]);
    expect(runs.length).toBeGreaterThan(0);
    expect(summary.environment.browserMode).toBe(BROWSER_MODE);
  } finally {
    for (const instance of warmInstances.values()) {
      await disposeRuntime(instance.page).catch(() => {});
      await instance.context.close().catch(() => {});
    }
  }
});

function scheduleFor(
  mode: BenchmarkScheduleEntry["mode"],
  workloads: readonly BenchmarkCase[],
  iterations: number,
  sequenceOffset: number,
  includeConversationVariants = false,
): BenchmarkScheduleEntry[] {
  return createBenchmarkSchedule({
    seed: SEED + sequenceOffset,
    modes: [mode],
    workloads,
    runtimeIds: RUNTIMES,
    iterations,
    includeConversationVariants,
  }).map((entry, index) => ({ ...entry, sequence: sequenceOffset + index }));
}

function createAllSchedule(
  profile: {
    workloads: readonly BenchmarkCase[];
    measured: number;
    startup: number;
    conversation: number;
  },
  startupWorkloads: readonly BenchmarkCase[],
): BenchmarkScheduleEntry[] {
  const network = scheduleFor("network-cold-startup", startupWorkloads, profile.startup, 0);
  const cached = scheduleFor("cached-cold-startup", startupWorkloads, profile.startup, network.length);
  const warm = scheduleFor(
    "warm-steady-state",
    profile.workloads,
    profile.measured,
    network.length + cached.length,
  );
  const conversation = scheduleFor(
    "conversation-cache",
    profile.workloads,
    profile.conversation,
    network.length + cached.length + warm.length,
    true,
  );
  return [...network, ...cached, ...warm, ...conversation].filter((entry) =>
    supportsEntry(entry, entry.mode === "network-cold-startup" || entry.mode === "cached-cold-startup"
      ? startupWorkloads
      : profile.workloads),
  );
}

async function launchRuntime(userDataDir: string) {
  const context = await chromium.launchPersistentContext(userDataDir, {
    channel: "chrome",
    headless: BROWSER_MODE === "headless",
    serviceWorkers: "block",
  });
  const page = context.pages()[0] ?? await context.newPage();
  await page.goto("http://127.0.0.1:5174/");
  return { context, page };
}

interface PhaseProgress {
  completed: number;
  total: number;
  unit: "warmup-generations";
  attempt: number;
  workloadIndex: number;
  workloadTotal: number;
}

interface WarmupProgress extends PhaseProgress {
  runtimeId: RuntimeId;
  workloadId: string;
}

interface ProgressSnapshot {
  status: "running" | "completed";
  phase: string;
  currentRuntime: string | null;
  currentWorkload: string | null;
  progressUnit: "measured-runs";
  phaseProgress: PhaseProgress | null;
  completed: number;
  total: number;
  remaining: number;
  skipped: number;
  errors: number;
  startedAt: string;
  updatedAt: string;
  elapsedMs: number;
  estimatedRemainingMs: number | null;
  percentComplete: number;
  unavailableRuntimes: Array<{ runtimeId: string; reason: string }>;
}

async function writePhaseProgress(
  progressPath: string,
  schedule: readonly BenchmarkScheduleEntry[],
  completedRunIds: ReadonlySet<string>,
  runs: readonly RawBenchmarkRun[],
  startedAtMs: number,
  completedAtStart: number,
  unavailableRuntimes: ReadonlyMap<RuntimeId, string>,
  entry: BenchmarkScheduleEntry,
): Promise<void> {
  await writeProgress(progressPath, {
    ...progressForRun(
      schedule,
      completedRunIds,
      runs,
      startedAtMs,
      completedAtStart,
      null,
      unavailableRuntimes,
    ),
    phase: entry.mode,
    currentRuntime: entry.runtimeId,
    currentWorkload: entry.workloadId,
  });
}

async function writeProgress(progressPath: string, progress: ProgressSnapshot): Promise<void> {
  await writeFile(progressPath, `${JSON.stringify(progress, null, 2)}\n`);
}

function progressForRun(
  schedule: readonly BenchmarkScheduleEntry[],
  completedRunIds: ReadonlySet<string>,
  runs: readonly RawBenchmarkRun[],
  startedAtMs: number,
  completedAtStart: number,
  latestRun: RawBenchmarkRun | null,
  unavailableRuntimes: ReadonlyMap<RuntimeId, string>,
): ProgressSnapshot {
  const completed = completedRunIds.size;
  const total = schedule.length;
  const skipped = schedule.filter((entry) =>
    unavailableRuntimes.has(entry.runtimeId as RuntimeId) && !completedRunIds.has(runIdFor(entry))
  ).length;
  const remaining = Math.max(0, total - completed - skipped);
  const elapsedMs = Date.now() - startedAtMs;
  const completedThisInvocation = completed - completedAtStart;
  const runsPerMs = completedThisInvocation > 0 ? completedThisInvocation / elapsedMs : 0;
  return {
    status: "running",
    phase: latestRun?.mode ?? "initializing",
    currentRuntime: latestRun?.runtime.id ?? null,
    currentWorkload: latestRun?.workload.id ?? null,
    progressUnit: "measured-runs",
    phaseProgress: null,
    completed,
    total,
    remaining,
    skipped,
    errors: runs.filter((run) => run.correctness.error !== null).length,
    startedAt: new Date(startedAtMs).toISOString(),
    updatedAt: new Date().toISOString(),
    elapsedMs,
    estimatedRemainingMs: runsPerMs > 0 ? Math.round(remaining / runsPerMs) : null,
    percentComplete: total === 0 ? 100 : Math.round((completed + skipped) / total * 10000) / 100,
    unavailableRuntimes: [...unavailableRuntimes].map(([runtimeId, reason]) => ({ runtimeId, reason })),
  };
}

function printProgress(progress: ProgressSnapshot): void {
  const eta = progress.estimatedRemainingMs === null
    ? "ETA pending"
    : `ETA ${formatDuration(progress.estimatedRemainingMs)}`;
  console.log(
    `[benchmark] ${progress.completed}/${progress.total} ` +
    `(${progress.percentComplete.toFixed(1)}%) ${progress.phase} ` +
    `${progress.currentRuntime ?? ""} ${progress.currentWorkload ?? ""}; ` +
    `${progress.errors} errors; ${eta}`,
  );
}

function formatDuration(milliseconds: number): string {
  const seconds = Math.round(milliseconds / 1000);
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainder = seconds % 60;
  return hours > 0
    ? `${hours}h ${minutes}m`
    : minutes > 0
      ? `${minutes}m ${remainder}s`
      : `${remainder}s`;
}

async function recoverWarmRuntime(
  runtimeId: RuntimeId,
  instances: Map<RuntimeId, Awaited<ReturnType<typeof launchRuntime>>>,
  calibrated: Map<string, BenchmarkCase>,
  root: string,
  workloads: readonly BenchmarkCase[],
  warmups: number,
  limitations: string[],
  onWarmupProgress: (progress: Omit<WarmupProgress, "attempt" | "runtimeId">) => Promise<void>,
) {
  const previous = instances.get(runtimeId);
  if (previous) {
    await disposeRuntime(previous.page).catch(() => {});
    await previous.context.close().catch(() => {});
  }
  const instance = await launchRuntime(path.join(root, runtimeId));
  instances.set(runtimeId, instance);
  const initialized = await initializeRuntime(instance.page, runtimeId, {
    mode: "cached-cold-startup",
    cacheCapacity: contextCapacity(runtimeId),
    ...(runtimeId === "owned-webgpu" ? { sourceUrl: "/models/gemma-4-e2b/model.safetensors" } : {}),
  });
  const supportedWorkloads = workloads.filter((workload) => supportsWorkload(runtimeId, workload));
  for (const [workloadOffset, workload] of supportedWorkloads.entries()) {
    const adjusted = await calibrateOrRetain(instance.page, workload, limitations);
    calibrated.set(`${runtimeId}\u0000${workload.id}`, adjusted);
    await warmupRuntime(instance.page, adjusted, warmups, async (completed, total) => {
      await onWarmupProgress({
        completed,
        total,
        unit: "warmup-generations",
        workloadId: workload.id,
        workloadIndex: workloadOffset + 1,
        workloadTotal: supportedWorkloads.length,
      });
    });
  }
  return initialized;
}

async function recoverWarmRuntimeWithRetry(
  runtimeId: RuntimeId,
  instances: Map<RuntimeId, Awaited<ReturnType<typeof launchRuntime>>>,
  calibrated: Map<string, BenchmarkCase>,
  root: string,
  workloads: readonly BenchmarkCase[],
  warmups: number,
  limitations: string[],
  onWarmupProgress: (progress: WarmupProgress) => Promise<void>,
) {
  const maximumAttempts = 3;
  let lastError: unknown;
  for (let attempt = 1; attempt <= maximumAttempts; attempt += 1) {
    try {
      return await recoverWarmRuntime(
        runtimeId,
        instances,
        calibrated,
        root,
        workloads,
        warmups,
        limitations,
        async (progress) => onWarmupProgress({ ...progress, attempt, runtimeId }),
      );
    } catch (error) {
      lastError = error;
      console.warn(`[benchmark] ${runtimeId} warm setup failed (${attempt}/${maximumAttempts}): ${String(error)}`);
    }
  }
  throw new Error(`${runtimeId} warm setup failed after ${maximumAttempts} attempts`, { cause: lastError });
}

async function initializeRuntime(page: Page, runtimeId: RuntimeId, options: LoadOptions) {
  return page.evaluate(async ({ runtimeId: id, options: loadOptions, environmentInput }) => {
    const modulePath = "/src/benchmark/suite/browser-harness.ts";
    const harness = await import(modulePath);
    return harness.initializeBenchmarkRuntime(id, loadOptions, environmentInput);
  }, { runtimeId, options, environmentInput: nodeEnvironmentInput() });
}

async function executeEntry(page: Page, workload: BenchmarkCase, schedule: BenchmarkScheduleEntry) {
  return page.evaluate(async ({ workload: selectedWorkload, schedule: selectedSchedule }) => {
    const modulePath = "/src/benchmark/suite/browser-harness.ts";
    const harness = await import(modulePath);
    return harness.executeBenchmarkScheduleEntry(selectedWorkload, selectedSchedule);
  }, { workload, schedule });
}

async function executeEntryOrTransportFailure(
  page: Page,
  workload: BenchmarkCase,
  schedule: BenchmarkScheduleEntry,
  runs: readonly RawBenchmarkRun[],
  prepareConversation = false,
): Promise<RawBenchmarkRun> {
  const startedAtMs = performance.now();
  try {
    if (prepareConversation) await prepareReusedConversation(page, workload);
    return await executeEntry(page, workload, schedule);
  } catch (error) {
    const completionMs = performance.now();
    const message = error instanceof Error ? error.message : String(error);
    const runtimeTemplate = [...runs].reverse().find((run) => run.runtime.id === schedule.runtimeId);
    if (!runtimeTemplate) throw error;
    return {
      schemaVersion: 2,
      runId: runIdFor(schedule),
      capturedAt: new Date().toISOString(),
      seed: SEED,
      browserMode: BROWSER_MODE,
      mode: schedule.mode,
      trackEligibility: runtimeTemplate.trackEligibility,
      runtime: runtimeTemplate.runtime,
      workload,
      schedule,
      actualInputTokens: 0,
      actualOutputTokens: 0,
      generatedCharacters: 0,
      outputBytes: 0,
      external: {
        requestStartMs: startedAtMs,
        firstVisibleOutputMs: null,
        completionMs,
        ttftMs: null,
        totalMs: completionMs - startedAtMs,
        aggregateDecodeTokensPerSecond: null,
        charactersPerSecond: 0,
        streamChunkCount: 0,
        streamChunkIntervalMs: [],
      },
      nativeMetrics: [],
      correctness: {
        exactOutputText: "",
        tokenCount: 0,
        characterCount: 0,
        outputByteCount: 0,
        reachedRequestedTokenLength: false,
        matchedExpectedPrefix: false,
        invalidOutput: true,
        repeatedOutput: false,
        earlyTerminated: true,
        error: `Browser transport failure: ${message}`,
      },
      equalWorkEligible: false,
      exclusionReasons: ["runtime-error", "invalid-output"],
      memoryBytes: null,
    };
  }
}

async function calibrateOrRetain(page: Page, workload: BenchmarkCase, limitations: string[]) {
  try {
    return await page.evaluate(async (selectedWorkload) => {
      const modulePath = "/src/benchmark/suite/browser-harness.ts";
      const harness = await import(modulePath);
      return harness.calibrateBenchmarkWorkload(selectedWorkload);
    }, workload);
  } catch (error) {
    limitations.push(`Runtime prompt calibration unavailable for ${workload.id}: ${String(error)}`);
    return workload;
  }
}

async function warmupRuntime(
  page: Page,
  workload: BenchmarkCase,
  iterations: number,
  onProgress: (completed: number, total: number) => Promise<void>,
) {
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    await onProgress(iteration, iterations);
    await page.evaluate(async (selectedWorkload) => {
      const modulePath = "/src/benchmark/suite/browser-harness.ts";
      const harness = await import(modulePath);
      await harness.warmupBenchmarkRuntime(selectedWorkload, 1);
    }, workload);
  }
  await onProgress(iterations, iterations);
}

async function prepareReusedConversation(page: Page, workload: BenchmarkCase) {
  await page.evaluate(async (selectedWorkload) => {
    const modulePath = "/src/benchmark/suite/browser-harness.ts";
    const harness = await import(modulePath);
    await harness.prepareReusedConversation(selectedWorkload);
  }, workload);
}

async function disposeRuntime(page: Page) {
  if (page.isClosed()) return;
  await page.evaluate(async () => {
    const modulePath = "/src/benchmark/suite/browser-harness.ts";
    const harness = await import(modulePath);
    await harness.disposeBenchmarkRuntime();
  });
}

async function applyNetworkObservation(page: Page, bytes: number, durationMs: number) {
  await page.evaluate(({ bytesTransferred, observedDurationMs }) => {
    const modulePath = "/src/benchmark/suite/browser-harness.ts";
    return import(modulePath).then((harness) =>
      harness.applyStartupNetworkObservation(bytesTransferred, observedDurationMs)
    );
  }, { bytesTransferred: bytes, observedDurationMs: durationMs });
}

async function clearColdStorage(page: Page, context: BrowserContext) {
  const cdp = await context.newCDPSession(page);
  await cdp.send("Network.enable");
  await cdp.send("Network.clearBrowserCache");
  await page.evaluate(async () => {
    localStorage.clear();
    sessionStorage.clear();
    for (const registration of await navigator.serviceWorker?.getRegistrations?.() ?? []) {
      await registration.unregister();
    }
    for (const cacheName of await caches.keys()) await caches.delete(cacheName);
    const databases = await indexedDB.databases?.() ?? [];
    await Promise.all(databases.flatMap((database) => database.name
      ? [new Promise<void>((resolve) => {
          const request = indexedDB.deleteDatabase(database.name!);
          request.onsuccess = request.onerror = request.onblocked = () => resolve();
        })]
      : []));
    const storage = navigator.storage as StorageManager & {
      getDirectory?: () => Promise<FileSystemDirectoryHandle>;
    };
    const directory = await storage.getDirectory?.();
    if (directory) {
      for await (const name of directory.keys()) await directory.removeEntry(name, { recursive: true });
    }
  });
  await cdp.detach();
}

async function startNetworkMonitor(context: BrowserContext, page: Page) {
  const cdp = await context.newCDPSession(page);
  await cdp.send("Network.enable");
  let bytes = 0;
  let firstRequestSeconds: number | null = null;
  let lastFinishSeconds: number | null = null;
  cdp.on("Network.requestWillBeSent", (event) => {
    firstRequestSeconds ??= event.timestamp;
  });
  cdp.on("Network.loadingFinished", (event) => {
    bytes += event.encodedDataLength;
    lastFinishSeconds = event.timestamp;
  });
  return {
    async finish() {
      await cdp.detach();
      return {
        bytes: Math.round(bytes),
        durationMs: firstRequestSeconds === null || lastFinishSeconds === null
          ? 0
          : Math.max(0, (lastFinishSeconds - firstRequestSeconds) * 1000),
      };
    },
  };
}

function nodeEnvironmentInput() {
  return {
    browserMode: BROWSER_MODE,
    gitCommit: gitCommit(),
    benchmarkSeed: SEED,
    operatingSystem: `${os.type()} ${os.release()} ${os.arch()}`,
    physicalDevice: os.hostname(),
    cpu: os.cpus()[0]?.model ?? "unknown",
    totalRamBytes: os.totalmem(),
    browserFlags: [] as string[],
  };
}

function supportsEntry(entry: BenchmarkScheduleEntry, workloads: readonly BenchmarkCase[]): boolean {
  return supportsWorkload(entry.runtimeId, workloadFor(entry, workloads));
}

function supportsWorkload(runtimeId: string, workload: BenchmarkCase): boolean {
  if (runtimeId === "litert-lm-web" && workload.targetInputTokens >= 4096) return false;
  return workload.supportsLongContext || runtimeId === "transformers-js";
}

function workloadFor(entry: BenchmarkScheduleEntry, workloads: readonly BenchmarkCase[]): BenchmarkCase {
  return required(workloads.find((workload) => workload.id === entry.workloadId), `Unknown workload ${entry.workloadId}`);
}

function contextCapacity(runtimeId: string): number {
  return runtimeId === "transformers-js" ? 16384 : 8192;
}

function parseRuntimeIds(value: string | undefined): RuntimeId[] {
  const allowed = new Set<RuntimeId>(["owned-webgpu", "transformers-js", "litert-lm-web"]);
  const selected = value?.split(",").map((item) => item.trim()).filter(Boolean) ?? [...allowed];
  for (const runtimeId of selected) {
    if (!allowed.has(runtimeId as RuntimeId)) throw new Error(`Unknown benchmark runtime: ${runtimeId}`);
  }
  return selected as RuntimeId[];
}

function parseOptionalRuntimeIds(value: string | undefined): RuntimeId[] {
  if (!value?.trim()) return [];
  return parseRuntimeIds(value);
}

function outputPath(browserMode: BrowserMode, profile: string): string {
  const stamp = new Date().toISOString().replaceAll(/[:.]/g, "-");
  return path.join(process.cwd(), "benchmarks", "suite", browserMode, `${stamp}-${profile}`);
}

function runIdFor(entry: BenchmarkScheduleEntry): string {
  return `${BROWSER_MODE}-${entry.sequence}-${entry.runtimeId}-${entry.workloadId}`;
}

async function readCheckpointRuns(rawResultsPath: string): Promise<RawBenchmarkRun[]> {
  try {
    const content = await readFile(rawResultsPath, "utf8");
    return content.split("\n").filter(Boolean).map((line) => JSON.parse(line) as RawBenchmarkRun);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw error;
  }
}

function gitCommit(): string {
  try {
    return execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim();
  } catch {
    return "unknown";
  }
}

function required<T>(value: T | null | undefined, message: string): T {
  if (value === null || value === undefined) throw new Error(message);
  return value;
}

async function cooldown(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, PROFILE === "smoke" ? 0 : 250));
}