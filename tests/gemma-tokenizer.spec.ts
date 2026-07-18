import { expect, test } from "@playwright/test";
import {
  GEMMA_GREEDY_GOLDEN_CASES,
  type GemmaGreedyGoldenCase,
} from "../src/runtime/gemma-golden";

test("loads the pinned local tokenizer and applies the Gemma chat template", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const modulePath = "/src/runtime/gemma-tokenizer.ts";
    const { loadGemmaTokenizer } = await import(modulePath);
    const tokenizer = await loadGemmaTokenizer();
    const tokenIds = tokenizer.encodePrompt("Hello");
    return {
      tokenIds,
      decoded: tokenizer.decodeTokens(tokenIds),
      ends: [1, 50, 106].map((tokenId) => tokenizer.isEndToken(tokenId)),
      ordinaryEnds: tokenizer.isEndToken(2),
    };
  });

  expect(result.tokenIds[0]).toBe(2);
  expect(result.tokenIds.every(Number.isInteger)).toBe(true);
  expect(result.decoded).toContain("Hello");
  expect(result.ends).toEqual([true, true, true]);
  expect(result.ordinaryEnds).toBe(false);
});

test("emits the pinned audio marker for structured audio parts", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const modulePath = "/src/runtime/gemma-tokenizer.ts";
    const { loadGemmaTokenizer } = await import(modulePath);
    const tokenizer = await loadGemmaTokenizer();
    const tokenIds = tokenizer.encodeMessages([{
      role: "user",
      content: [{ type: "audio" }, { type: "text", text: "Transcribe this." }],
    }]);
    return {
      audioTokenId: tokenizer.audioTokenId,
      beginAudioTokenId: tokenizer.beginAudioTokenId,
      endAudioTokenId: tokenizer.endAudioTokenId,
      markerCount: tokenIds.filter((tokenId: number) => tokenId === tokenizer.audioTokenId).length,
      decoded: tokenizer.decodeRawTokens(tokenIds),
    };
  });

  expect(result.audioTokenId).toBe(258881);
  expect(result.beginAudioTokenId).toBe(256000);
  expect(result.endAudioTokenId).toBe(258883);
  expect(result.markerCount).toBe(1);
  expect(result.decoded).toContain("<|audio|>");
});

test("emits the pinned video marker for structured video parts", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const modulePath = "/src/runtime/gemma-tokenizer.ts";
    const { loadGemmaTokenizer } = await import(modulePath);
    const tokenizer = await loadGemmaTokenizer();
    const tokenIds = tokenizer.encodeMessages([{
      role: "user",
      content: [{ type: "video" }, { type: "text", text: "Describe this video." }],
    }]);
    return {
      videoTokenId: tokenizer.videoTokenId,
      markerCount: tokenIds.filter((tokenId: number) => tokenId === tokenizer.videoTokenId).length,
      decoded: tokenizer.decodeRawTokens(tokenIds),
    };
  });

  expect(result.videoTokenId).toBe(258884);
  expect(result.markerCount).toBe(1);
  expect(result.decoded).toContain("<|video|>");
});

test("wraps audio soft-token slots in canonical boundary tokens", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const modulePath = "/src/runtime/gemma-tokenizer.ts";
    const { expandGemmaAudioTokenIds } = await import(modulePath);
    return expandGemmaAudioTokenIds(3);
  });

  expect(result).toEqual([256000, 258881, 258881, 258881, 258883]);
});

test("preserves the greedy golden prompt and output token vectors", async ({ page }) => {
  await page.goto("/");
  const results = await page.evaluate(async () => {
    const tokenizerModulePath = "/src/runtime/gemma-tokenizer.ts";
    const goldenModulePath = "/src/runtime/gemma-golden.ts";
    const [{ loadGemmaTokenizer }, golden] = await Promise.all([
      import(tokenizerModulePath),
      import(goldenModulePath),
    ]);
    const tokenizer = await loadGemmaTokenizer();
    return (golden.GEMMA_GREEDY_GOLDEN_CASES as readonly GemmaGreedyGoldenCase[])
      .map((testCase) => ({
      id: testCase.id,
      promptTokenIds: tokenizer.encodePrompt(testCase.prompt),
      text: tokenizer.decodeTokens(testCase.generatedTokenIds),
      }));
  });

  expect(results).toEqual(GEMMA_GREEDY_GOLDEN_CASES.map((testCase) => ({
    id: testCase.id,
    promptTokenIds: testCase.promptTokenIds,
    text: testCase.text,
  })));
});

test("tokenizes structured single and multi-turn conversations", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const tokenizerModulePath = "/src/runtime/gemma-tokenizer.ts";
    const goldenModulePath = "/src/runtime/gemma-golden.ts";
    const [{ loadGemmaTokenizer }, golden] = await Promise.all([
      import(tokenizerModulePath),
      import(goldenModulePath),
    ]);
    const tokenizer = await loadGemmaTokenizer();
    const prompt = golden.GEMMA_GREEDY_GOLDEN_CASES[0].prompt;
    const single = tokenizer.encodeMessages([{ role: "user", content: prompt }]);
    const conversation = [
      { role: "system", content: "Answer briefly." },
      { role: "user", content: "What is two plus two?" },
      { role: "assistant", content: "Four." },
      { role: "user", content: "Write it as a digit." },
    ];
    return {
      single,
      prompt: tokenizer.encodePrompt(prompt),
      conversation: tokenizer.encodeMessages(conversation),
      repeated: tokenizer.encodeMessages(conversation),
      decoded: tokenizer.decodeTokens(tokenizer.encodeMessages(conversation)),
    };
  });

  expect(result.single).toEqual(result.prompt);
  expect(result.conversation).toEqual(result.repeated);
  expect(result.conversation).toEqual([
    2, 105, 9731, 107, 7925, 21485, 236761, 106, 107, 105, 2364, 107, 3689,
    563, 1156, 2915, 1156, 236881, 106, 107, 105, 4368, 107, 26391, 236761,
    106, 107, 105, 2364, 107, 6974, 625, 618, 496, 15958, 236761, 106, 107,
    105, 4368, 107,
  ]);
  expect(result.decoded).toBe(
    "system\nAnswer briefly.\nuser\nWhat is two plus two?\n" +
    "model\nFour.\nuser\nWrite it as a digit.\nmodel\n",
  );
});

test("applies the canonical boolean thinking template switch", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const modulePath = "/src/runtime/gemma-tokenizer.ts";
    const tokenizerModule = await import(modulePath);
    const { loadGemmaTokenizer } = tokenizerModule;
    const tokenizer = await loadGemmaTokenizer();
    const messages = [{ role: "user", content: "What is two plus two?" }] as const;
    const disabled = tokenizer.encodeInput({ messages });
    const enabled = tokenizer.encodeInput({ messages, enableThinking: true });
    return {
      disabled: tokenizer.decodeRawTokens(disabled),
      enabled: tokenizer.decodeRawTokens(enabled),
      enabledTokenIds: enabled,
      startChannel: tokenizer.decodeRawTokens([tokenizerModule.GEMMA_START_CHANNEL_TOKEN_ID]),
      endChannel: tokenizer.decodeRawTokens([tokenizerModule.GEMMA_END_CHANNEL_TOKEN_ID]),
      repeatedEnabled: tokenizer.encodeInput({ messages, enableThinking: true }),
    };
  });

  expect(result.disabled).not.toContain("<|think|>");
  expect(result.enabled).toContain("<|turn>system\n<|think|>\n<turn|>");
  expect(result.enabled).toContain("<|turn>user\nWhat is two plus two?<turn|>");
  expect(result.startChannel).toBe("<|channel>");
  expect(result.endChannel).toBe("<channel|>");
  expect(result.repeatedEnabled).toEqual(result.enabledTokenIds);
});

test("preserves canonical reasoning through a tool-response continuation", async ({ page }) => {
  await page.goto("/");
  const rawPrompt = await page.evaluate(async () => {
    const modulePath = "/src/runtime/gemma-tokenizer.ts";
    const { loadGemmaTokenizer } = await import(modulePath);
    const tokenizer = await loadGemmaTokenizer();
    const tokenIds = tokenizer.encodeInput({
      enableThinking: true,
      preserveThinking: true,
      messages: [
        { role: "user", content: "What is the weather in Boston?" },
        {
          role: "assistant",
          content: "",
          reasoning: "I need the weather tool.",
          toolCalls: [{
            id: "weather-1",
            type: "function",
            function: { name: "get_weather", arguments: { city: "Boston" } },
          }],
        },
        {
          role: "tool",
          content: "72 F and clear",
          toolCallId: "weather-1",
        },
      ],
    });
    return tokenizer.decodeRawTokens(tokenIds);
  });

  expect(rawPrompt).toContain("<|channel>thought\nI need the weather tool.\n<channel|>");
  expect(rawPrompt).toContain("<|tool_call>call:get_weather{city:<|\"|>Boston<|\"|>}<tool_call|>");
  expect(rawPrompt).toContain("<|tool_response>response:get_weather{value:<|\"|>72 F and clear<|\"|>}<tool_response|>");
  expect(rawPrompt.endsWith("<|channel>thought\n")).toBe(true);
});

test("commits completed turns without mutating in-flight conversation history", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const modulePath = "/src/runtime/gemma-conversation.ts";
    const {
      commitGemmaConversationTurn,
      createGemmaConversation,
      prepareGemmaConversationTurn,
    } = await import(modulePath);
    const initial = createGemmaConversation([
      { role: "system", content: "Answer briefly." },
      { role: "user", content: "What is two plus two?" },
      { role: "assistant", content: "Four." },
    ]);
    const pending = prepareGemmaConversationTurn(initial, "Write it as a digit.");
    const committed = commitGemmaConversationTurn(initial, pending, "4");
    return {
      initialMessages: initial.messages,
      pendingInput: pending.input,
      committedMessages: committed.messages,
    };
  });

  expect(result.initialMessages).toHaveLength(3);
  expect(result.pendingInput).toEqual([
    ...result.initialMessages,
    { role: "user", content: "Write it as a digit." },
  ]);
  expect(result.committedMessages).toEqual([
    ...result.pendingInput,
    { role: "assistant", content: "4" },
  ]);
});

test("prepares edited user turns without retaining dependent history", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const modulePath = "/src/runtime/gemma-conversation.ts";
    const {
      commitGemmaConversationTurn,
      createGemmaConversation,
      prepareGemmaConversationEdit,
    } = await import(modulePath);
    const initial = createGemmaConversation([
      { role: "user", content: "What is two plus two?" },
      { role: "assistant", content: "Four." },
      { role: "user", content: "Spell it." },
      { role: "assistant", content: "F-O-U-R." },
    ]);
    const edit = prepareGemmaConversationEdit(initial, 0, "What is three plus three?");
    const committed = commitGemmaConversationTurn(edit.conversation, edit.turn, "Six.");
    return {
      initial: initial.messages,
      pending: edit.turn.input,
      committed: committed.messages,
    };
  });

  expect(result.initial).toHaveLength(4);
  expect(result.pending).toEqual([{ role: "user", content: "What is three plus three?" }]);
  expect(result.committed).toEqual([
    { role: "user", content: "What is three plus three?" },
    { role: "assistant", content: "Six." },
  ]);
});

test("preserves image ownership when preparing an edited multimodal turn", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const modulePath = "/src/runtime/gemma-conversation.ts";
    const { createGemmaConversation, prepareGemmaConversationEdit } = await import(modulePath);
    const firstImage = new ImageData(48, 48);
    const editedImage = new ImageData(96, 48);
    const conversation = createGemmaConversation([
      { role: "user", content: [{ type: "image" }, { type: "text", text: "First image" }] },
      { role: "assistant", content: "First answer" },
      { role: "user", content: [{ type: "image" }, { type: "text", text: "Second image" }] },
      { role: "assistant", content: "Second answer" },
    ], [firstImage, editedImage]);
    const edit = prepareGemmaConversationEdit(conversation, 2, "Describe it differently.");
    const input = edit.turn.input;
    if (Array.isArray(input) || typeof input === "string") throw new Error("Expected image input");
    return {
      retainedMessages: edit.conversation.messages,
      retainedImages: edit.conversation.images.length,
      pendingMessages: input.messages,
      pendingImages: input.images?.map((image: unknown) =>
        image === firstImage ? "first" : "edited"),
    };
  });

  expect(result.retainedMessages).toHaveLength(2);
  expect(result.retainedImages).toBe(1);
  expect(result.pendingMessages).toHaveLength(3);
  expect(result.pendingImages).toEqual(["first", "edited"]);
});

test("retains the selected visual-token budget in multimodal turns", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const modulePath = "/src/runtime/gemma-conversation.ts";
    const { createGemmaConversation, prepareGemmaConversationTurn } = await import(modulePath);
    const image = new ImageData(48, 48);
    return prepareGemmaConversationTurn(
      createGemmaConversation(),
      "Describe this image.",
      image,
      70,
    ).input;
  });

  expect(result).toMatchObject({
    visionTokenBudget: 70,
    images: [{}],
  });
});

test("retains video ownership in multimodal turns", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const modulePath = "/src/runtime/gemma-conversation.ts";
    const { commitGemmaConversationTurn, createGemmaConversation, prepareGemmaConversationTurn } =
      await import(modulePath);
    const video = new Blob([new Uint8Array([1, 2, 3])], { type: "video/webm" });
    const initial = createGemmaConversation();
    const turn = prepareGemmaConversationTurn(
      initial,
      "Describe this video.",
      undefined,
      70,
      false,
      undefined,
      video,
    );
    const input = turn.input;
    if (Array.isArray(input) || typeof input === "string") throw new Error("Expected video input");
    const committed = commitGemmaConversationTurn(initial, turn, "A short recording.");
    return {
      content: input.messages.at(-1)?.content,
      videos: input.videos?.length,
      committedVideos: committed.videos.length,
    };
  });

  expect(result).toEqual({
    content: [{ type: "video" }, { type: "text", text: "Describe this video." }],
    videos: 1,
    committedVideos: 1,
  });
});

test("preserves video ownership when editing a multimodal turn", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const modulePath = "/src/runtime/gemma-conversation.ts";
    const { createGemmaConversation, prepareGemmaConversationEdit } = await import(modulePath);
    const firstVideo = new Blob([new Uint8Array([1])], { type: "video/webm" });
    const editedVideo = new Blob([new Uint8Array([2])], { type: "video/webm" });
    const conversation = createGemmaConversation([
      { role: "user", content: [{ type: "video" }, { type: "text", text: "First video" }] },
      { role: "assistant", content: "First answer" },
      { role: "user", content: [{ type: "video" }, { type: "text", text: "Second video" }] },
      { role: "assistant", content: "Second answer" },
    ], [], [], [], [firstVideo, editedVideo]);
    const edit = prepareGemmaConversationEdit(conversation, 2, "Describe it differently.");
    const input = edit.turn.input;
    if (Array.isArray(input) || typeof input === "string") throw new Error("Expected video input");
    return {
      retainedVideos: edit.conversation.videos.length,
      pendingVideos: input.videos?.map((video: Blob) => video === firstVideo ? "first" : "edited"),
    };
  });

  expect(result).toEqual({ retainedVideos: 1, pendingVideos: ["first", "edited"] });
});

test("renders function declarations through the canonical tool system block", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const modulePath = "/src/runtime/gemma-tokenizer.ts";
    const { loadGemmaTokenizer } = await import(modulePath);
    const tokenizer = await loadGemmaTokenizer();
    const tokenIds = tokenizer.encodeInput({
      messages: [{ role: "user", content: "What is the weather in Boston?" }],
      tools: [{
        type: "function",
        function: {
          name: "get_current_weather",
          description: "Get the current weather for a city.",
          parameters: {
            type: "object",
            properties: {
              location: { type: "string", description: "City and region." },
            },
            required: ["location"],
          },
        },
      }],
    });
    return {
      tokenIds,
      decoded: tokenizer.decodeTokens(tokenIds),
    };
  });

  expect(result.tokenIds[0]).toBe(2);
  expect(result.tokenIds).toContain(46);
  expect(result.tokenIds).toContain(47);
  expect(result.decoded).toBe(
    "system\ndeclaration:get_current_weather{description:Get the current weather for a city.," +
    "parameters:{properties:{location:{description:City and region.,type:STRING}}," +
    "required:[location],type:OBJECT}}\nuser\nWhat is the weather in Boston?\nmodel\n",
  );
});

test("emits one pinned image marker for each structured image part", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const modulePath = "/src/runtime/gemma-tokenizer.ts";
    const { GEMMA_IMAGE_TOKEN_ID, loadGemmaTokenizer } = await import(modulePath);
    const tokenizer = await loadGemmaTokenizer();
    const tokenIds = tokenizer.encodeMessages([{
      role: "user",
      content: [
        { type: "image" },
        { type: "text", text: "What is shown?" },
      ],
    }]);
    return {
      exportedTokenId: GEMMA_IMAGE_TOKEN_ID,
      tokenizerTokenId: tokenizer.imageTokenId,
      markerCount: tokenIds.filter((tokenId: number) =>
        tokenId === GEMMA_IMAGE_TOKEN_ID).length,
      markerIndex: tokenIds.indexOf(GEMMA_IMAGE_TOKEN_ID),
      tokenCount: tokenIds.length,
    };
  });

  expect(result.exportedTokenId).toBe(258880);
  expect(result.tokenizerTokenId).toBe(258880);
  expect(result.markerCount).toBe(1);
  expect(result.markerIndex).toBeGreaterThan(0);
  expect(result.markerIndex).toBeLessThan(result.tokenCount - 1);
});

test("maps vocabulary tokens to the pinned decoder's exact output bytes", async ({ page }) => {
  await page.goto("/");
  const results = await page.evaluate(async () => {
    const tokenizerModulePath = "/src/runtime/gemma-tokenizer.ts";
    const goldenModulePath = "/src/runtime/gemma-golden.ts";
    const [{ loadGemmaTokenizer }, golden] = await Promise.all([
      import(tokenizerModulePath),
      import(goldenModulePath),
    ]);
    const tokenizer = await loadGemmaTokenizer();
    const decoder = new TextDecoder("utf-8", { fatal: true });
    return {
      vocabularySize: tokenizer.vocabularySize,
      specialBytes: [0, 1, 2].map((tokenId) => tokenizer.tokenBytes(tokenId)),
      byteFallback: Array.from(tokenizer.tokenBytes(248) ?? []),
      outputs: golden.GEMMA_GREEDY_GOLDEN_CASES.map((testCase: GemmaGreedyGoldenCase) => {
        const parts = testCase.generatedTokenIds.map(
          (tokenId: number) => tokenizer.tokenBytes(tokenId),
        );
        const size = parts.reduce(
          (sum: number, part: Uint8Array | null) => sum + (part?.length ?? 0),
          0,
        );
        const bytes = new Uint8Array(size);
        let offset = 0;
        for (const part of parts) {
          if (!part) throw new Error("Golden output contains a special token");
          bytes.set(part, offset);
          offset += part.length;
        }
        return decoder.decode(bytes);
      }),
    };
  });

  expect(results.vocabularySize).toBe(262144);
  expect(results.specialBytes).toEqual([null, null, null]);
  expect(results.byteFallback).toEqual([10]);
  expect(results.outputs).toEqual(GEMMA_GREEDY_GOLDEN_CASES.map((testCase) => testCase.text));
});