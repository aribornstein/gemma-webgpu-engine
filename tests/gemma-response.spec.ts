import { expect, test } from "@playwright/test";

test("parses canonical Gemma function call arguments", async ({ page }) => {
  await page.goto("/");
  const calls = await page.evaluate(async () => {
    const modulePath = "/src/runtime/gemma-response.ts";
    const { parseGemmaToolCalls } = await import(modulePath);
    return parseGemmaToolCalls(
      'prefix<|tool_call>call:search{query:<|"|>Gemma 4<|"|>,' +
      'limit:3,filters:{fresh:true,tags:[<|"|>webgpu<|"|>,<|"|>gemma<|"|>]}}' +
      "<tool_call|>",
    );
  });

  expect(calls).toEqual([{
    type: "function",
    function: {
      name: "search",
      arguments: {
        query: "Gemma 4",
        limit: 3,
        filters: { fresh: true, tags: ["webgpu", "gemma"] },
      },
    },
  }]);
});

test("rejects malformed Gemma tool call boundaries", async ({ page }) => {
  await page.goto("/");
  const message = await page.evaluate(async () => {
    const modulePath = "/src/runtime/gemma-response.ts";
    const { parseGemmaToolCalls } = await import(modulePath);
    try {
      parseGemmaToolCalls("<|tool_call>call:weather{city:<|\"|>Boston<|\"|>}");
      return "no error";
    } catch (error) {
      return error instanceof Error ? error.message : String(error);
    }
  });

  expect(message).toBe("Gemma emitted a malformed tool call");
});

test("separates canonical thought channels from the final answer", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const modulePath = "/src/runtime/gemma-response.ts";
    const { countGemmaReasoningTokens, parseGemmaResponse } = await import(modulePath);
    return {
      parsed: parseGemmaResponse(
        "<|channel>thought\nCheck the premise.\n<channel|>The answer is <four>.<turn|>",
        "Check the premise. The answer is 4.",
      ),
      tokenCount: countGemmaReasoningTokens([
        "<|channel>thought", "\n", "Check", " premise", ".", "<channel|>", "Answer",
      ]),
    };
  });

  expect(result).toEqual({
    parsed: { reasoning: "Check the premise.", text: "The answer is <four>." },
    tokenCount: 4,
  });
});

test("rejects an unterminated canonical thought channel", async ({ page }) => {
  await page.goto("/");
  const message = await page.evaluate(async () => {
    const modulePath = "/src/runtime/gemma-response.ts";
    const { parseGemmaResponse } = await import(modulePath);
    try {
      parseGemmaResponse("<|channel>thought\nunfinished", "unfinished");
      return "no error";
    } catch (error) {
      return error instanceof Error ? error.message : String(error);
    }
  });
  expect(message).toBe("Gemma emitted an unterminated thought channel");
});

test("rejects a malformed later thought channel", async ({ page }) => {
  await page.goto("/");
  const message = await page.evaluate(async () => {
    const modulePath = "/src/runtime/gemma-response.ts";
    const { parseGemmaResponse } = await import(modulePath);
    try {
      parseGemmaResponse(
        "<|channel>thought\nfirst<channel|>answer<|channel>thought\nunfinished",
        "first answer unfinished",
      );
      return "no error";
    } catch (error) {
      return error instanceof Error ? error.message : String(error);
    }
  });
  expect(message).toBe("Gemma emitted an unterminated thought channel");
});