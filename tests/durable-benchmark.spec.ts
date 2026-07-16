import { expect, test } from "@playwright/test";

test("persists and restores a benchmark checkpoint across reload", async ({ page }) => {
  await page.goto("/");
  const saved = await page.evaluate(async () => {
    const modulePath = "/src/runtime/durable-benchmark.ts";
    const {
      createGemmaDurableBenchmarkArtifact,
      saveGemmaDurableBenchmarkArtifact,
      updateGemmaDurableBenchmarkArtifact,
    } = await import(modulePath);
    let artifact = createGemmaDurableBenchmarkArtifact({
      cacheCapacity: 32_768,
      targetPromptTokens: 32_768,
      maxNewTokens: 1,
      prefillStrategy: "auto",
    }, "run-32k", new Date("2026-07-16T12:00:00.000Z"));
    saveGemmaDurableBenchmarkArtifact(localStorage, artifact);
    artifact = updateGemmaDurableBenchmarkArtifact(localStorage, artifact, {
      phase: "running",
      progress: {
        promptTokens: 32_768,
        generatedTokens: 0,
        message: "Generating exact-fit boundary token",
      },
    }, new Date("2026-07-16T12:01:00.000Z"));
    return artifact;
  });

  expect(saved.phase).toBe("running");
  await page.reload();
  await expect(page.getByRole("status").filter({
    hasText: "The benchmark page or browser process reloaded before completion",
  })).toBeVisible();
  const restored = await page.evaluate(async () => {
    const modulePath = "/src/runtime/durable-benchmark.ts";
    const { loadGemmaDurableBenchmarkArtifact } = await import(modulePath);
    return loadGemmaDurableBenchmarkArtifact(localStorage);
  });

  expect(restored).toMatchObject({
    schemaVersion: 1,
    runId: "run-32k",
    phase: "interrupted",
    configuration: {
      cacheCapacity: 32_768,
      targetPromptTokens: 32_768,
      maxNewTokens: 1,
    },
    progress: {
      promptTokens: 32_768,
      generatedTokens: 0,
      message: "Benchmark interrupted by page or process reload",
    },
  });
});

test("rejects a benchmark request beyond cache capacity", async ({ page }) => {
  await page.goto("/");
  const message = await page.evaluate(async () => {
    const modulePath = "/src/runtime/durable-benchmark.ts";
    const { createGemmaDurableBenchmarkArtifact } = await import(modulePath);
    try {
      createGemmaDurableBenchmarkArtifact({
        cacheCapacity: 32_768,
        targetPromptTokens: 32_768,
        maxNewTokens: 2,
        prefillStrategy: "auto",
      });
      return null;
    } catch (error) {
      return error instanceof Error ? error.message : String(error);
    }
  });

  expect(message).toMatch(/exceed cache capacity/);
});

test("constructs an exact synthetic prompt and recovers interrupted work", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const durablePath = "/src/runtime/durable-benchmark.ts";
    const runnerPath = "/src/runtime/gemma-long-context-benchmark.ts";
    const { createGemmaDurableBenchmarkArtifact, saveGemmaDurableBenchmarkArtifact } =
      await import(durablePath);
    const {
      createExactGemmaPrompt,
      recoverInterruptedGemmaBenchmark,
      serializeGemmaBenchmarkArtifact,
    } = await import(runnerPath);
    const counter = {
      promptTokenCount(prompt: string) {
        if (prompt.length === 0) throw new Error("empty prompts are invalid");
        return 9 + prompt.length / 2;
      },
    };
    const prompt = createExactGemmaPrompt(counter, 32_768);
    const running = createGemmaDurableBenchmarkArtifact({
      cacheCapacity: 32_768,
      targetPromptTokens: 32_768,
      maxNewTokens: 1,
      prefillStrategy: "auto",
    }, "interrupted-32k");
    running.phase = "running";
    saveGemmaDurableBenchmarkArtifact(localStorage, running);
    const recovered = recoverInterruptedGemmaBenchmark(localStorage)!;
    return {
      promptLength: prompt.length,
      promptTokens: counter.promptTokenCount(prompt),
      phase: recovered.phase,
      message: recovered.progress.message,
      serialized: serializeGemmaBenchmarkArtifact(recovered),
    };
  });

  expect(result.promptTokens).toBe(32_768);
  expect(result.promptLength).toBe((32_768 - 9) * 2);
  expect(result.phase).toBe("interrupted");
  expect(result.message).toMatch(/interrupted by page or process reload/);
  expect(JSON.parse(result.serialized)).toMatchObject({
    runId: "interrupted-32k",
    phase: "interrupted",
  });
});