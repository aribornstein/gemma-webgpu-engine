import { expect, test } from "@playwright/test";

test("gates model loading to the cache-owning origin", async ({ page }) => {
  await page.goto("/");

  await expect(page).toHaveTitle("Gemma WebGPU Generation Console");
  await expect(page.getByRole("heading", { name: "Generation console" })).toBeVisible();
  await expect(page.getByText("WebGPU ready", { exact: true })).toBeVisible();
  await expect(page.getByText("Cache absent", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Load model" })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Generate" })).toBeDisabled();
  await expect(page.getByText(/origin containing safetensors-cache-v1/)).toBeVisible();
  await expect(page.getByText("Decode tok/s", { exact: true })).toBeVisible();
  await expect(page.getByText("Overall tok/s", { exact: true })).toBeVisible();
  await expect(page.getByText("Vision", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Long-context boundary" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Run exact-fit" })).toBeDisabled();
  await expect(page.getByLabel("Capacity")).toHaveValue("32768");
  await expect(page.getByRole("spinbutton", { name: "Max tokens" })).toHaveAttribute("max", "32768");
  await expect(page.getByText("32,768 validated / 131,072 model positions", { exact: true }))
    .toBeVisible();
});

test("enables a host-provided client-side cache initializer", async ({ page }) => {
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

  const initialize = page.getByRole("button", { name: "Initialize cache" });
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

  const examples = page.getByLabel("Example");
  await examples.selectOption("greedy-colors");
  await expect(page.getByRole("textbox", { name: "Prompt" }))
    .toHaveValue("Name the three primary colors in one short sentence.");
  await expect(page.getByText("Exact greedy configuration", { exact: true })).toBeVisible();

  await examples.selectOption("sampling-tagline");
  await expect(page.getByRole("textbox", { name: "Prompt" }))
    .toHaveValue("Write one vivid, playful sentence describing a pocket-sized observatory.");
  await expect(page.getByRole("spinbutton", { name: "Temperature" })).toHaveValue("0.8");
  await expect(page.getByRole("spinbutton", { name: "Top K" })).toHaveValue("40");
  await expect(page.getByText("Sampling configuration valid", { exact: true })).toBeVisible();

  await examples.selectOption("regex-sky");
  await expect(page.getByRole("radio", { name: "Regex" })).toBeChecked();
  await expect(page.getByRole("textbox", { name: "Pattern" })).toHaveValue("(?:blue|gray)");
  await expect(page.getByText("regex constraint valid", { exact: true })).toBeVisible();

  await examples.selectOption("json-city");
  await expect(page.getByRole("radio", { name: "JSON", exact: true })).toBeChecked();
  await expect(page.getByRole("spinbutton", { name: "Maximum depth" })).toHaveValue("3");
  await expect(page.getByText("json constraint valid", { exact: true })).toBeVisible();

  await examples.selectOption("schema-triage");
  await expect(page.getByRole("radio", { name: "Schema" })).toBeChecked();
  await expect(page.getByRole("textbox", { name: "JSON Schema" })).toHaveValue(/"urgent"/);
  await expect(page.getByText("json-schema constraint valid", { exact: true })).toBeVisible();

  await page.getByRole("spinbutton", { name: "Max tokens" }).fill("32");
  await expect(examples).toHaveValue("");
});

test("keeps the vision example grounded in the selected image", async ({ page }) => {
  await page.goto("/");

  await page.getByLabel("Example").selectOption("vision-dolphin-caption");
  const captionPrompt = await page.getByRole("textbox", { name: "Prompt" }).inputValue();
  expect(captionPrompt).toContain("Transcribe the printed caption in this image");
  expect(captionPrompt).not.toMatch(/spinner|stenella|kona|hawaii|brian skerry/i);
  await expect(page.getByRole("img", { name: "Selected image preview" })).toBeVisible();
  await expect(page.getByText("dolphin_capt_image.png", { exact: true })).toBeVisible();

  await page.getByLabel("Example").selectOption("vision-gottingen");
  const prompt = await page.getByRole("textbox", { name: "Prompt" }).inputValue();
  expect(prompt).toContain("Read the printed caption in the image");
  expect(prompt).not.toMatch(
    /Abraham|Schilling|Hilbert|Klein|Schwarzschild|Young|Diestel|Zermelo|Fanla|Hansen|Müller|Dawncy|Schmidt|Yoshiye|Epsteen|Fleisher|Bernstein|Blumenthal|Hamel/,
  );
  await expect(page.getByRole("img", { name: "Selected image preview" })).toBeVisible();
  await expect(page.getByText("the-mathematics-club-of-gottingen-1902.jpg", { exact: true }))
    .toBeVisible();
});
