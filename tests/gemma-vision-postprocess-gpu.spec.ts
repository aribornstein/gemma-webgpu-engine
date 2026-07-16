import { expect, test } from "@playwright/test";

test("pools and projects real Gemma vision soft tokens", async ({ page }) => {
  test.setTimeout(60_000);
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const sourcePath = "/src/model/pinned-safetensors.ts";
    const weightsPath = "/src/model/gemma-vision-weights.ts";
    const devicePath = "/src/webgpu/device.ts";
    const postprocessPath = "/src/webgpu/vision-postprocess.ts";
    const [{ PinnedSafetensorsSource }, { loadGemmaVisionProjectorWeights },
      { getWebGpuDevice }, postprocess] = await Promise.all([
      import(sourcePath),
      import(weightsPath),
      import(devicePath),
      import(postprocessPath),
    ]);
    const source = await PinnedSafetensorsSource.open();
    const weights = await loadGemmaVisionProjectorWeights(source);
    const device = await getWebGpuDevice();
    const width = 768;
    const input = Float32Array.from({ length: 9 * width }, (_, index) =>
      Math.fround(Math.sin(index / 41) * 0.5 + (Math.floor(index / width) - 4) * 0.125));
    const inputBuffer = device.createBuffer({
      size: input.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const readback = device.createBuffer({
      size: 1536 * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    device.queue.writeBuffer(inputBuffer, 0, input);
    const resources = await postprocess.createGemmaVisionPostprocessResources(
      device,
      inputBuffer,
      3,
      3,
      weights,
    );
    device.pushErrorScope("validation");
    device.pushErrorScope("internal");
    try {
      const encoder = device.createCommandEncoder();
      postprocess.encodeGemmaVisionPostprocess(encoder, resources);
      encoder.copyBufferToBuffer(resources.output, 0, readback, 0, 1536 * 4);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const actual = new Float32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      const pooled = new Float32Array(width);
      let squareSum = 0;
      for (let column = 0; column < width; column += 1) {
        let sum = 0;
        for (let row = 0; row < 9; row += 1) sum += input[row * width + column];
        pooled[column] = sum * Math.sqrt(width) / 9;
        squareSum += pooled[column] * pooled[column];
      }
      const inverseRms = 1 / Math.sqrt(squareSum / width + 1e-6);
      const outputColumns = [0, 1, 127, 1024, 1535];
      const expected = outputColumns.map((outputColumn) => {
        let sum = 0;
        const weightBase = outputColumn * width;
        for (let column = 0; column < width; column += 1) {
          sum += pooled[column] * inverseRms * weights.projection[weightBase + column];
        }
        return sum;
      });
      const internalError = await device.popErrorScope();
      const validationError = await device.popErrorScope();
      return {
        actual: outputColumns.map((column) => actual[column]),
        expected,
        outputRows: resources.outputRows,
        sourceBytes: weights.sourceBytes,
        gpuError: internalError?.message ?? validationError?.message ?? null,
      };
    } finally {
      postprocess.destroyGemmaVisionPostprocessResources(resources);
      inputBuffer.destroy();
      readback.destroy();
    }
  });

  expect(result.gpuError).toBeNull();
  expect(result.outputRows).toBe(1);
  expect(result.sourceBytes).toBe(4_718_592);
  for (let index = 0; index < result.actual.length; index += 1) {
    expect(Math.abs(result.actual[index] - result.expected[index])).toBeLessThan(3e-4);
  }
});