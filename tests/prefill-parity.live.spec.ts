import { chromium, expect, test, type Page } from "@playwright/test";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

// Batched (32-row) prefill must produce the same next-token distribution as sequential
// prefill. Requires a browser profile whose origin holds the model cache (PREFILL_PARITY_PROFILE
// with PREFILL_PARITY_URL) or the checkpoint at public/models/gemma-4-e2b/model.safetensors.
const LIVE = process.env.PREFILL_PARITY === "1";
const PROFILE = process.env.PREFILL_PARITY_PROFILE;
const TARGET_URL = process.env.PREFILL_PARITY_URL;
const LOCAL_CHECKPOINT = resolve("public/models/gemma-4-e2b/model.safetensors");

test("batched prefill matches sequential prefill on the next token", async ({ page, baseURL }) => {
  test.skip(!LIVE, "Set PREFILL_PARITY=1 to run the live prefill parity check");
  test.skip(
    !PROFILE && !existsSync(LOCAL_CHECKPOINT),
    "Set PREFILL_PARITY_PROFILE or provide the local checkpoint",
  );
  test.setTimeout(600_000);
  const context = PROFILE
    ? await chromium.launchPersistentContext(PROFILE, { channel: "chrome", headless: true })
    : null;
  try {
    const target: Page = context ? context.pages()[0] ?? await context.newPage() : page;
    await target.goto(TARGET_URL ?? baseURL ?? "/");
    const results = await target.evaluate(async (useLocal) => {
      const modulePath = "/src/runtime/gemma-session.ts";
      const { loadGemmaGenerationSession } = await import(modulePath);
      const sourceUrl = useLocal ? "/models/gemma-4-e2b/model.safetensors" : undefined;
      const words = Array.from({ length: 60 }, (_, index) => `item${index}`).join(" ");
      const prompts = [
        "Say hi in one short sentence.",
        "What is 17 * 23? Answer with the number.",
        `Repeat the first three items exactly: ${words}`,
      ];
      const run = async (prefillStrategy: "sequential" | "fixed-32") => {
        const session = await loadGemmaGenerationSession({
          cacheCapacity: 4096,
          prefillStrategy,
          ...(sourceUrl ? { sourceUrl } : {}),
        });
        try {
          const outputs = [];
          for (const prompt of prompts) {
            const result = await session.generate(prompt, {
              maxNewTokens: 1,
              captureTokenLogProbabilities: true,
              reusePromptCache: false,
            });
            outputs.push({
              token: result.generatedTokenIds[0],
              logProbability: result.tokenLogProbabilities?.[0] ?? Number.NaN,
            });
          }
          return outputs;
        } finally {
          session.destroy();
        }
      };
      const sequential = await run("sequential");
      const batched = await run("fixed-32");
      return sequential.map((expected, index) => ({ expected, actual: batched[index] }));
    }, !PROFILE);
    for (const { expected, actual } of results) {
      expect(actual.token).toBe(expected.token);
      expect(Math.abs(actual.logProbability - expected.logProbability)).toBeLessThan(0.25);
    }
  } finally {
    await context?.close();
  }
});
