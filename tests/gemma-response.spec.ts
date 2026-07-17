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