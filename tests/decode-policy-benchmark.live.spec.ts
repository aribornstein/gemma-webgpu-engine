import { chromium, expect, test, type Page } from "@playwright/test";
import { existsSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

// Requires either a browser profile whose origin already holds the model cache
// (DECODE_POLICY_PROFILE, served from that origin via DECODE_POLICY_URL) or the checkpoint at
// public/models/gemma-4-e2b/model.safetensors.
const LIVE = process.env.DECODE_POLICY_BENCHMARK === "1";
const PROFILE = process.env.DECODE_POLICY_PROFILE;
const TARGET_URL = process.env.DECODE_POLICY_URL;
const LABEL = process.env.DECODE_POLICY_LABEL ?? "chrome";
const ITERATIONS = Number(process.env.DECODE_POLICY_ITERATIONS ?? 5);
const LOCAL_CHECKPOINT = resolve("public/models/gemma-4-e2b/model.safetensors");

test("writes the owned decoding-policy benchmark artifact", async ({ page, baseURL }) => {
  test.skip(!LIVE, "Set DECODE_POLICY_BENCHMARK=1 to run the live decoding-policy benchmark");
  test.skip(
    !PROFILE && !existsSync(LOCAL_CHECKPOINT),
    "Set DECODE_POLICY_PROFILE or provide the local checkpoint",
  );
  test.setTimeout(1_800_000);
  const context = PROFILE
    ? await chromium.launchPersistentContext(PROFILE, { channel: "chrome", headless: true })
    : null;
  try {
    const target: Page = context ? context.pages()[0] ?? await context.newPage() : page;
    await target.goto(TARGET_URL ?? baseURL ?? "/");
    const artifact = await target.evaluate(async ({ iterations, useLocal }) => {
      const modulePath = "/src/benchmark/decode-policy-benchmark.ts";
      const { runDecodePolicyBenchmark } = await import(modulePath);
      return runDecodePolicyBenchmark({
        iterations,
        ...(useLocal ? { sourceUrl: "/models/gemma-4-e2b/model.safetensors" } : {}),
      });
    }, { iterations: ITERATIONS, useLocal: !PROFILE });
    expect(artifact.workloads.map((workload: { id: string }) => workload.id)).toEqual([
      "greedy-warm",
      "sampled-warm",
      "json-schema-warm",
      "greedy-fresh-session",
    ]);
    const output = resolve("benchmarks", `decode-policy.${LABEL}.json`);
    await mkdir(resolve("benchmarks"), { recursive: true });
    await writeFile(output, `${JSON.stringify(artifact, null, 2)}\n`);
  } finally {
    await context?.close();
  }
});
