import { expect, test } from "@playwright/test";

test("offers a direct model download when the cache is absent", async ({ page }) => {
  await hideLocalCheckpoint(page);
  await page.goto("/");

  await expect(page).toHaveTitle("Gemma WebGPU Generation Console");
  await expect(page.getByRole("heading", { name: "Generation console" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Conversation" })).toBeVisible();
  await expect(page.getByRole("button", { name: "New chat" })).toBeDisabled();
  await expect(page.getByRole("heading", { name: "Transcript" })).toBeVisible();
  await expect(page.getByText("WebGPU ready", { exact: true })).toBeVisible();
  await expect(page.getByText("Cache absent", { exact: true })).toBeVisible();
  const modelButton = page.getByRole("button", { name: "Download model" });
  await expect(modelButton).toBeEnabled();
  await expect(modelButton).toHaveAttribute(
    "title",
    "Rebuild the WebGPU engine from existing local or cached weights",
  );
  await expect(page.getByRole("button", { name: "Generate" })).toBeDisabled();
  await expect(page.getByText("Download and load the model on this origin")).toBeVisible();
  await expect(page.getByText("Decode tok/s", { exact: true })).toBeVisible();
  await expect(page.getByText("Overall tok/s", { exact: true })).toBeVisible();
  await expect(page.getByText("Vision", { exact: true })).toBeVisible();
  await expect(page.getByText("Vision CPU", { exact: true })).toBeVisible();
  await expect(page.getByText("Diagnostics", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Long-context boundary" })).toBeHidden();
  await page.getByText("Diagnostics", { exact: true }).click();
  await expect(page.getByRole("heading", { name: "Long-context boundary" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Run exact-fit" })).toBeDisabled();
  await expect(page.getByLabel("Capacity")).toHaveValue("32768");
  await expect(page.getByRole("spinbutton", { name: "Max tokens" })).toHaveAttribute("max", "32768");
  await expect(page.getByLabel("Visual tokens")).toHaveValue("140");
  await expect(page.getByText("32,768 validated / 131,072 model positions", { exact: true }))
    .toBeVisible();
});

test("offers to resume an incomplete model cache", async ({ page }) => {
  await hideLocalCheckpoint(page);
  await page.goto("/");
  await page.evaluate(async () => {
    await new Promise<void>((resolve, reject) => {
      const request = indexedDB.open("safetensors-cache-v1", 2);
      request.onupgradeneeded = () => {
        request.result.createObjectStore("chunks");
        request.result.createObjectStore("meta");
      };
      request.onsuccess = () => {
        request.result.close();
        resolve();
      };
      request.onerror = () => reject(request.error);
    });
  });
  await page.reload();

  await expect(page.getByText("Cache partial", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Resume download" })).toBeEnabled();
  await expect(page.getByText("Resume the model download and load it on this origin"))
    .toBeVisible();
  await page.evaluate(async () => {
    await new Promise<void>((resolve, reject) => {
      const request = indexedDB.deleteDatabase("safetensors-cache-v1");
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  });
});

test("enables a host-provided client-side cache initializer", async ({ page }) => {
  await hideLocalCheckpoint(page);
  await page.addInitScript(() => {
    const host = window as typeof window & {
      __gemmaEngineCacheInitializer?: (
        onProgress: (progress: { fraction?: number }) => void,
      ) => Promise<void>;
    };
    host.__gemmaEngineCacheInitializer = async (onProgress) => {
      onProgress({
        status: "weights",
        kind: "bytes",
        loaded: 1_250_000_000,
        total: 2_458_111_846,
      });
      await new Promise((resolve) => setTimeout(resolve, 500));
      throw new Error("fixture initialization stopped");
    };
  });
  await page.goto("/");

  const initialize = page.getByRole("button", { name: "Download model" });
  await expect(initialize).toBeEnabled();
  await initialize.click();
  await expect(page.getByRole("button", { name: "Downloading 50.9%" })).toBeVisible();
  await expect(page.getByText("Downloading model weights · 50.9%", { exact: true })).toBeVisible();
  await expect(page.getByText("1192.1 MiB of 2344.2 MiB", { exact: true })).toBeVisible();
  await expect(page.getByRole("progressbar", { name: "Model download progress" }))
    .toHaveAttribute("value", /0\.5085/);
  await expect(page.locator("#request-status")).toHaveText("fixture initialization stopped");
  await expect(page.getByRole("button", { name: "Retry initialization" })).toBeEnabled();
});

test("validates controls and switches constraint editors", async ({ page }) => {
  await page.goto("/");

  await page.getByRole("radio", { name: "Regex" }).check();
  await expect(page.getByRole("textbox", { name: "Pattern" })).toBeVisible();
  await page.getByRole("textbox", { name: "Pattern" }).fill("a(?=b)");
  await expect(page.getByRole("status").filter({ hasText: /assertions are not supported/ }))
    .toBeVisible();

  await page.getByRole("radio", { name: "Schema" }).check();
  await expect(page.getByRole("textbox", { name: "JSON Schema" })).toBeVisible();
  await expect(page.getByRole("spinbutton", { name: "Maximum depth" })).toBeVisible();
  await page.getByRole("textbox", { name: "JSON Schema" }).fill("{");
  await expect(page.getByRole("status").filter({ hasText: /Expected property name|JSON/ }))
    .toBeVisible();
  await page.getByRole("textbox", { name: "JSON Schema" }).fill(`{
    "type": "object",
    "properties": { "ok": { "const": true } },
    "required": ["ok"],
    "additionalProperties": false
  }`);
  await expect(page.getByText("json-schema constraint valid", { exact: true })).toBeVisible();
  await page.getByRole("checkbox", { name: "Thinking" }).check();
  await expect(page.getByText("json-schema constraint valid", { exact: true })).toBeVisible();

  await page.getByRole("radio", { name: "Object" }).check();
  await expect(page.getByRole("spinbutton", { name: "Maximum depth" })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "JSON Schema" })).toBeHidden();
  await expect(page.getByText("json-object constraint valid", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: "Reset" }).click();
  await expect(page.getByRole("radio", { name: "None" })).toBeChecked();
  await expect(page.getByText("Exact greedy configuration", { exact: true })).toBeVisible();
  await expect(page.getByRole("spinbutton", { name: "Temperature" })).toHaveValue("0");
});

test("applies editable generation examples", async ({ page }) => {
  await page.goto("/");

  const examples = page.getByLabel("Workspace");
  await examples.selectOption("webcam-video");
  await expect(page.getByRole("textbox", { name: "Message" })).toHaveValue(/chronological order/);
  await expect(page.getByRole("button", { name: "Add video" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Record webcam" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Add image" })).toBeHidden();
  await expect(page.getByRole("button", { name: "Add audio" })).toBeHidden();
  await expect(page.getByLabel("Visual tokens")).toHaveValue("70");

  await examples.selectOption("tool-weather");
  await expect(page.getByRole("log")).toContainText("get_current_weather");
  await expect(page.getByRole("textbox", { name: "Message" }))
    .toHaveValue(/current weather in Boston/);

  await examples.selectOption("greedy-colors");
  await expect(page.getByRole("textbox", { name: "Message" }))
    .toHaveValue("Name the three primary colors in one short sentence.");
  await expect(page.getByText("Exact greedy configuration", { exact: true })).toBeVisible();

  await examples.selectOption("sampling-tagline");
  await expect(page.getByRole("textbox", { name: "Message" }))
    .toHaveValue("Write one vivid, playful sentence describing a pocket-sized observatory.");
  await expect(page.getByRole("spinbutton", { name: "Temperature" })).toHaveValue("0.8");
  await expect(page.getByRole("spinbutton", { name: "Top K" })).toHaveValue("40");
  await expect(page.getByText("Sampling configuration valid", { exact: true })).toBeVisible();

  await examples.selectOption("regex-sky");
  await expect(page.getByRole("textbox", { name: "Pattern" })).toHaveValue("(?:blue|gray)");
  await expect(page.getByText("regex constraint valid", { exact: true })).toBeVisible();

  await examples.selectOption("json-city");
  await expect(page.getByRole("spinbutton", { name: "Maximum depth" })).toHaveValue("3");
  await expect(page.getByRole("textbox", { name: "JSON Schema" })).toBeHidden();
  await expect(page.getByText("json-object constraint valid", { exact: true })).toBeVisible();

  await examples.selectOption("schema-triage");
  await expect(page.getByRole("textbox", { name: "JSON Schema" })).toHaveValue(/"urgent"/);
  await expect(page.getByText("json-schema constraint valid", { exact: true })).toBeVisible();

  await expect(examples.getByRole("option", { name: "Best of N · Comparative judge" })).toBeAttached();
  await expect(examples.getByRole("option", { name: "Best of N · Log likelihood" })).toBeAttached();
  await examples.selectOption("best-of-n-judge-hebrew");
  const languageLevel = page.getByLabel("Language level");
  await expect(languageLevel).toHaveValue("A1");
  await expect(page.getByText("Reasoning-free drafts · Comparative Thinking judges", { exact: true })).toBeVisible();
  const candidateCount = page.getByRole("spinbutton", { name: "Candidates" });
  await expect(candidateCount).toHaveValue("2");
  await expect(candidateCount).not.toHaveAttribute("max");
  const judgeCount = page.getByRole("spinbutton", { name: /^Judges\b/ });
  await expect(judgeCount).toHaveValue("2");
  await expect(judgeCount).toHaveAttribute("max", "2");
  await expect(page.getByRole("button", { name: "Generate 2 + 2 judges" })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "Generation prompt" })).toBeVisible();
  const judgePrompt = page.getByRole("textbox", { name: "Example judge prompt" });
  await expect(judgePrompt).toHaveValue(/Compare all anonymous Hebrew candidates/);
  await expect(page.getByRole("checkbox", { name: "Thinking" })).toBeHidden();
  await expect(page.getByRole("spinbutton", { name: "Max tokens" })).toHaveValue("320");
  const generationPrompt = page.getByRole("textbox", { name: "Generation prompt" });
  await expect(generationPrompt).not.toHaveValue(/[\u0590-\u05FF\u0600-\u06FF]/);
  await expect(generationPrompt).toHaveValue(/generate the dialogue yourself/);
  await expect(generationPrompt).toHaveValue(/level A1/);
  await expect(generationPrompt).toHaveValue(/modern Israeli Hebrew/);
  await expect(page.getByRole("textbox", { name: "JSON Schema" }))
    .toHaveValue(/"level": \{\s+"const": "A1"[\s\S]*"visitor_request"[\s\S]*"local_direction"/);
  await expect(page.getByRole("textbox", { name: "JSON Schema" })).not.toHaveValue(/jerusalem_arabic/);
  await expect(page.getByRole("textbox", { name: "JSON Schema" })).not.toHaveValue(/route_plan/);
  await candidateCount.fill("3");
  await expect(judgeCount).toHaveAttribute("max", "4");
  await judgeCount.fill("3");
  await judgePrompt.fill("Custom independent Hebrew audit instructions.");
  await expect(page.getByRole("button", { name: "Generate 3 + 3 judges" })).toBeVisible();
  await languageLevel.selectOption("C3");
  await expect(examples).toHaveValue("best-of-n-judge-hebrew");
  await expect(candidateCount).toHaveValue("3");
  await expect(judgeCount).toHaveValue("3");
  await expect(judgePrompt).toHaveValue("Custom independent Hebrew audit instructions.");
  await candidateCount.fill("20");
  await expect(candidateCount).toHaveValue("20");
  await expect(page.getByRole("button", { name: "Generate 20 + 3 judges" })).toBeVisible();
  await expect(generationPrompt)
    .toHaveValue(/level C3[\s\S]*experimental beyond-CEFR/);
  await expect(page.getByRole("textbox", { name: "JSON Schema" }))
    .toHaveValue(/"level": \{\s+"const": "C3"[\s\S]*"visitor_contextual_request"[\s\S]*"local_graceful_close"/);
  await expect(page.getByRole("spinbutton", { name: "Max tokens" })).toHaveValue("768");
  await expect(page.getByText("json-schema constraint valid", { exact: true })).toBeVisible();

  await examples.selectOption("best-of-n-likelihood-hebrew");
  await expect(page.getByText("Thinking candidates · Joint generated-token likelihood", { exact: true })).toBeVisible();
  await expect(page.getByRole("spinbutton", { name: /^Judges\b/ })).toBeHidden();
  await expect(page.getByRole("textbox", { name: "Example judge prompt" })).toBeHidden();
  await expect(page.getByRole("button", { name: "Generate 2 + rank" })).toBeVisible();
  await expect(page.getByRole("checkbox", { name: "Thinking" })).toBeVisible();
  await expect(page.getByRole("checkbox", { name: "Thinking" })).toBeChecked();

  await examples.selectOption("reasoning-logic");
  await expect(page.getByRole("textbox", { name: "Message" })).toHaveValue(/farmer.*fox.*chicken/s);
  await expect(page.getByRole("checkbox", { name: "Thinking" })).toBeVisible();
  await expect(page.getByRole("checkbox", { name: "Thinking" })).toBeChecked();

  await examples.selectOption("schema-reasoning");
  await expect(page.getByRole("textbox", { name: "Message" })).toHaveValue(/medication shipment/);
  await expect(page.getByRole("textbox", { name: "JSON Schema" })).toHaveValue(/"inspect"/);
  await expect(page.getByRole("checkbox", { name: "Thinking" })).toBeVisible();
  await expect(page.getByRole("checkbox", { name: "Thinking" })).toBeChecked();
  await expect(page.getByText("json-schema constraint valid", { exact: true })).toBeVisible();

  await page.getByRole("spinbutton", { name: "Max tokens" }).fill("32");
  await expect(examples).toHaveValue("schema-reasoning");
  await expect(page.getByRole("textbox", { name: "JSON Schema" })).toBeVisible();
});

test("records webcam output into the video attachment preview", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: {
        getUserMedia: async () => {
          const canvas = document.createElement("canvas");
          canvas.width = 64;
          canvas.height = 48;
          const context = canvas.getContext("2d")!;
          let frame = 0;
          window.setInterval(() => {
            context.fillStyle = frame++ % 2 === 0 ? "#e65332" : "#16857a";
            context.fillRect(0, 0, canvas.width, canvas.height);
          }, 50);
          return canvas.captureStream(10);
        },
      },
    });
  });
  await page.goto("/");
  await page.getByLabel("Workspace").selectOption("webcam-video");

  await page.getByRole("button", { name: "Record webcam" }).click();
  await expect(page.getByRole("button", { name: "Stop recording" })).toBeVisible();
  await expect(page.getByText("Recording 0:00 / 0:10", { exact: true })).toBeVisible();
  await page.waitForTimeout(1_100);
  await page.getByRole("button", { name: "Stop recording" }).click();

  await expect(page.getByText(/webcam-recording\.(?:webm|mp4)/)).toBeVisible();
  const preview = page.locator("#video-player");
  await expect(preview).toBeVisible();
  await expect(preview).toHaveAttribute("src", /^blob:/);
  await expect.poll(() => preview.evaluate((element) =>
    Number.isFinite((element as HTMLVideoElement).duration))).toBe(true);
  await preview.evaluate((element) => (element as HTMLVideoElement).play());
  await expect.poll(() => preview.evaluate((element) =>
    (element as HTMLVideoElement).currentTime)).toBeGreaterThan(0);
});

test("keeps Generate prompts while Chat clears its sent composer", async ({ page }) => {
  await page.goto("/");
  const workspace = page.getByLabel("Workspace");
  const prompt = page.getByRole("textbox", { name: "Message" });

  await workspace.selectOption("schema-reasoning");
  const examplePrompt = await prompt.inputValue();
  await page.evaluate(() => document.querySelector<HTMLFormElement>("#generation-form")?.dispatchEvent(
    new Event("submit", { bubbles: true, cancelable: true }),
  ));
  await expect(prompt).toHaveValue(examplePrompt);

  await workspace.selectOption("chat");
  await prompt.fill("Keep this only until it is sent.");
  await expect(prompt).toHaveValue("Keep this only until it is sent.");
});

test("clears the transcript without clearing the configured example prompt", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Workspace").selectOption("schema-reasoning");
  const prompt = page.getByRole("textbox", { name: "Message" });
  const examplePrompt = await prompt.inputValue();
  const clearTranscript = page.getByRole("button", { name: "Clear transcript" });

  await expect(clearTranscript.locator("xpath=parent::*")).toHaveClass(/tool-heading-actions/);
  await clearTranscript.click();

  await expect(page.getByRole("log")).toHaveText("No conversation yet.");
  await expect(prompt).toHaveValue(examplePrompt);
  await expect(page.locator("#request-status"))
    .toHaveText(/Transcript cleared|Waiting for model/);
});

test("loads the synthesized speech example through the audio attachment path", async ({ page }) => {
  await page.goto("/");

  await page.getByLabel("Workspace").selectOption("audio-transcription");
  await expect(page.getByRole("textbox", { name: "Message" }))
    .toHaveValue("What is said in this audio? Return only the spoken words.");
  await expect(page.getByRole("button", { name: "Add image" })).toBeHidden();
  await expect(page.getByRole("button", { name: "Add audio" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Record mic" })).toBeVisible();
  await expect(page.getByText("gemma-audio-demo.wav", { exact: true })).toBeVisible();
  await expect(page.locator("#audio-player")).toBeVisible();
});

test("records microphone output into the audio attachment preview", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: {
        getUserMedia: async () => {
          const context = new AudioContext();
          const destination = context.createMediaStreamDestination();
          const oscillator = context.createOscillator();
          oscillator.connect(destination);
          oscillator.start();
          await context.resume();
          return destination.stream;
        },
      },
    });
  });
  await page.goto("/");

  await page.getByRole("button", { name: "Record mic" }).click();
  await expect(page.getByRole("button", { name: "Stop recording" })).toBeVisible();
  await expect(page.getByText("Recording 0:00", { exact: true })).toBeVisible();
  await page.waitForTimeout(250);
  await page.getByRole("button", { name: "Stop recording" }).click();
  await expect(page.getByText("microphone-recording.wav", { exact: true })).toBeVisible();
  const player = page.locator("#audio-player");
  await expect(player).toBeVisible();
  await expect.poll(() => player.evaluate((audio) =>
    (audio as HTMLAudioElement).duration)).toBeGreaterThan(0);
});

test("keeps the vision example grounded in the selected image", async ({ page }) => {
  await page.goto("/");

  await page.getByLabel("Workspace").selectOption("vision-dolphin-caption");
  const captionPrompt = await page.getByRole("textbox", { name: "Message" }).inputValue();
  expect(captionPrompt).toContain("Transcribe the printed caption in this image");
  expect(captionPrompt).not.toMatch(/spinner|stenella|kona|hawaii|brian skerry/i);
  await expect(page.getByRole("img", { name: "Selected image preview" })).toBeVisible();
  await expect(page.getByText("dolphin_capt_image.png", { exact: true })).toBeVisible();
  await expect(page.getByLabel("Visual tokens")).toHaveValue("280");

  await page.getByLabel("Workspace").selectOption("vision-gottingen");
  const prompt = await page.getByRole("textbox", { name: "Message" }).inputValue();
  expect(prompt).toContain("Read the printed caption in the image");
  expect(prompt).not.toMatch(
    /Abraham|Schilling|Hilbert|Klein|Schwarzschild|Young|Diestel|Zermelo|Fanla|Hansen|Müller|Dawncy|Schmidt|Yoshiye|Epsteen|Fleisher|Bernstein|Blumenthal|Hamel/,
  );
  await expect(page.getByRole("img", { name: "Selected image preview" })).toBeVisible();
  await expect(page.getByText("the-mathematics-club-of-gottingen-1902.jpg", { exact: true }))
    .toBeVisible();
});

test("shows only controls relevant to each example while Custom exposes all", async ({ page }) => {
  await page.goto("/");
  const workspace = page.getByLabel("Workspace");
  const addImage = page.getByRole("button", { name: "Add image" });
  const addAudio = page.getByRole("button", { name: "Add audio" });
  const recordMic = page.getByRole("button", { name: "Record mic" });
  const visualTokens = page.getByLabel("Visual tokens");
  const temperature = page.getByRole("spinbutton", { name: "Temperature" });
  const probability = page.getByText("Probability and penalties", { exact: true });
  const constraintModes = page.getByRole("radiogroup", { name: "Output constraint" });
  const thinking = page.getByRole("checkbox", { name: "Thinking" });
  const languageLevel = page.getByLabel("Language level");

  await expect(workspace.locator("optgroup").first()).toHaveAttribute("label", "Workspaces");
  await expect(workspace.locator("optgroup").last()).toHaveAttribute("label", "Examples");
  await expect(addImage).toBeVisible();
  await expect(addAudio).toBeVisible();
  await expect(recordMic).toBeVisible();
  await expect(visualTokens).toBeVisible();
  await expect(temperature).toBeVisible();
  await expect(probability).toBeVisible();
  await expect(constraintModes).toBeVisible();
  await expect(thinking).toBeVisible();
  await expect(thinking).not.toBeChecked();
  await expect(languageLevel).toBeHidden();
  await expect(page.getByRole("textbox", { name: "Stop token IDs" })).toBeVisible();

  await workspace.selectOption("greedy-colors");
  await expect(addImage).toBeHidden();
  await expect(addAudio).toBeHidden();
  await expect(recordMic).toBeHidden();
  await expect(visualTokens).toBeHidden();
  await expect(temperature).toBeHidden();
  await expect(probability).toBeHidden();
  await expect(constraintModes).toBeHidden();
  await expect(thinking).toBeHidden();

  await workspace.selectOption("audio-transcription");
  await expect(addImage).toBeHidden();
  await expect(addAudio).toBeVisible();
  await expect(recordMic).toBeVisible();
  await expect(visualTokens).toBeHidden();

  await workspace.selectOption("sampling-tagline");
  await expect(temperature).toBeVisible();
  await expect(probability).toBeVisible();
  await expect(constraintModes).toBeHidden();

  await workspace.selectOption("regex-sky");
  await expect(constraintModes).toBeHidden();
  await expect(page.getByRole("textbox", { name: "Pattern" })).toBeVisible();
  await expect(page.getByRole("spinbutton", { name: "Maximum depth" })).toBeHidden();

  await workspace.selectOption("json-city");
  await expect(page.getByRole("textbox", { name: "Pattern" })).toBeHidden();
  await expect(page.getByRole("spinbutton", { name: "Maximum depth" })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "JSON Schema" })).toBeHidden();

  await workspace.selectOption("schema-triage");
  await expect(page.getByRole("textbox", { name: "JSON Schema" })).toBeVisible();

  await workspace.selectOption("best-of-n-judge-hebrew");
  await expect(languageLevel).toBeVisible();
  await expect(languageLevel).toHaveValue("A1");
  await expect(page.getByRole("textbox", { name: "JSON Schema" })).toBeVisible();

  await workspace.selectOption("schema-reasoning");
  await expect(languageLevel).toBeHidden();
  await expect(page.getByRole("textbox", { name: "JSON Schema" })).toBeVisible();
  await expect(thinking).toBeVisible();
  await expect(thinking).toBeChecked();

  await workspace.selectOption("reasoning-logic");
  await expect(page.getByRole("textbox", { name: "JSON Schema" })).toBeHidden();
  await expect(thinking).toBeVisible();
  await expect(thinking).toBeChecked();
});

test("keeps Chat selected through composing and image attachment", async ({ page }) => {
  await page.goto("/");
  const workspace = page.getByLabel("Workspace");
  await workspace.selectOption("chat");

  await expect(page.getByRole("heading", { name: "Chat" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Messages" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Send" })).toBeVisible();
  await expect(page.getByRole("checkbox", { name: "Thinking" })).toBeVisible();
  await expect(page.getByLabel("Visual tokens")).toBeHidden();
  await page.getByRole("textbox", { name: "Message" }).fill("What is in this image?");
  await expect(workspace).toHaveValue("chat");

  await page.getByRole("button", { name: "Add image" }).setInputFiles(
    "public/examples/dolphin_capt_image.png",
  );
  await expect(page.getByRole("img", { name: "Selected image preview" })).toBeVisible();
  await expect(page.getByLabel("Visual tokens")).toBeVisible();
  await expect(workspace).toHaveValue("chat");

  const transcriptBeforeComposer = await page.evaluate(() => {
    const transcript = document.querySelector(".output-tool");
    const composer = document.querySelector("#generation-form");
    return Boolean(transcript && composer &&
      (transcript.compareDocumentPosition(composer) & Node.DOCUMENT_POSITION_FOLLOWING));
  });
  expect(transcriptBeforeComposer).toBe(true);
});

for (const viewport of [
  { name: "desktop", width: 1440, height: 900 },
  { name: "mobile", width: 390, height: 844 },
] as const) {
  test(`keeps Chat controls coherent on ${viewport.name}`, async ({ page }) => {
    await page.setViewportSize(viewport);
    await page.goto("/");
    await page.getByLabel("Workspace").selectOption("chat");
    await page.getByRole("textbox", { name: "Message" }).fill("Responsive layout check");

    const layout = await page.evaluate(() => {
      const documentWidth = document.documentElement.scrollWidth;
      const viewportWidth = document.documentElement.clientWidth;
      const transcript = document.querySelector<HTMLElement>(".output-tool")?.getBoundingClientRect();
      const composer = document.querySelector<HTMLElement>("#generation-form")?.getBoundingClientRect();
      const send = document.querySelector<HTMLElement>("#generate")?.getBoundingClientRect();
      if (!transcript || !composer || !send) throw new Error("Missing responsive UI surface");
      return {
        horizontalOverflow: documentWidth > viewportWidth,
        transcriptBeforeComposer: transcript.bottom <= composer.top,
        sendInsideViewport: send.left >= 0 && send.right <= viewportWidth,
      };
    });

    expect(layout).toEqual({
      horizontalOverflow: false,
      transcriptBeforeComposer: true,
      sendInsideViewport: true,
    });
  });
}

async function hideLocalCheckpoint(page: import("@playwright/test").Page): Promise<void> {
  await page.route("**/models/gemma-4-e2b/model.safetensors", async (route) => {
    if (route.request().method() === "HEAD") {
      await route.fulfill({ status: 404 });
      return;
    }
    await route.continue();
  });
}
