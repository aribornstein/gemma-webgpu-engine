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

  await page.getByRole("button", { name: "Reset" }).click();
  await expect(page.getByRole("radio", { name: "None" })).toBeChecked();
  await expect(page.getByText("Exact greedy configuration", { exact: true })).toBeVisible();
  await expect(page.getByRole("spinbutton", { name: "Temperature" })).toHaveValue("0");
});

test("applies editable generation examples", async ({ page }) => {
  await page.goto("/");

  const examples = page.getByLabel("Workspace");
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
  await expect(page.getByText("json constraint valid", { exact: true })).toBeVisible();

  await examples.selectOption("schema-triage");
  await expect(page.getByRole("textbox", { name: "JSON Schema" })).toHaveValue(/"urgent"/);
  await expect(page.getByText("json-schema constraint valid", { exact: true })).toBeVisible();

  await page.getByRole("spinbutton", { name: "Max tokens" }).fill("32");
  await expect(examples).toHaveValue("custom");
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
  const visualTokens = page.getByLabel("Visual tokens");
  const temperature = page.getByRole("spinbutton", { name: "Temperature" });
  const probability = page.getByText("Probability and penalties", { exact: true });
  const constraintModes = page.getByRole("radiogroup", { name: "Output constraint" });
  const thinking = page.getByRole("checkbox", { name: "Thinking" });

  await expect(workspace.locator("optgroup").first()).toHaveAttribute("label", "Workspaces");
  await expect(workspace.locator("optgroup").last()).toHaveAttribute("label", "Examples");
  await expect(addImage).toBeVisible();
  await expect(visualTokens).toBeVisible();
  await expect(temperature).toBeVisible();
  await expect(probability).toBeVisible();
  await expect(constraintModes).toBeVisible();
  await expect(thinking).toBeVisible();
  await expect(thinking).not.toBeChecked();
  await expect(page.getByRole("textbox", { name: "Stop token IDs" })).toBeVisible();

  await workspace.selectOption("greedy-colors");
  await expect(addImage).toBeHidden();
  await expect(visualTokens).toBeHidden();
  await expect(temperature).toBeHidden();
  await expect(probability).toBeHidden();
  await expect(constraintModes).toBeHidden();
  await expect(thinking).toBeHidden();

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
