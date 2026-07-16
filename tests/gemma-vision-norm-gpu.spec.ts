import { expect, test } from "@playwright/test";

test("normalizes real Gemma vision layer-0 Q heads with checkpoint RMS weights", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const sourcePath = "/src/model/pinned-safetensors.ts";
    const weightsPath = "/src/model/gemma-vision-weights.ts";
    const devicePath = "/src/webgpu/device.ts";
    const rmsPath = "/src/webgpu/prefill-rms.ts";
    const [{ PinnedSafetensorsSource }, weightsModule, { getWebGpuDevice }, rms] =
      await Promise.all([
        import(sourcePath),
        import(weightsPath),
        import(devicePath),
        import(rmsPath),
      ]);
    const source = await PinnedSafetensorsSource.open();
    const weights = await weightsModule.loadGemmaVisionLayerNormWeights(source, 0);
    const device = await getWebGpuDevice();
    const heads = 12;
    const dimension = 64;
    const input = Float32Array.from({ length: heads * dimension }, (_, index) =>
      Math.fround(Math.sin(index / 29) * 3 + (index % 11) / 17));
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const inputBuffer = device.createBuffer({ size: input.byteLength, usage: storage });
    const weightBuffer = device.createBuffer({ size: weights.query.byteLength, usage: storage });
    const readBuffer = device.createBuffer({
      size: input.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    device.queue.writeBuffer(inputBuffer, 0, input);
    device.queue.writeBuffer(weightBuffer, 0, weights.query);
    const pipeline = await rms.getGemmaPrefillRmsPipeline(device, dimension, true);
    const resources = rms.createGemmaPrefillRmsResources(
      device,
      pipeline,
      heads,
      inputBuffer,
      weightBuffer,
    );
    try {
      const encoder = device.createCommandEncoder();
      rms.encodeGemmaPrefillRms(encoder, pipeline, resources);
      encoder.copyBufferToBuffer(resources.output, 0, readBuffer, 0, input.byteLength);
      device.queue.submit([encoder.finish()]);
      await readBuffer.mapAsync(GPUMapMode.READ);
      const actual = new Float32Array(readBuffer.getMappedRange().slice(0));
      readBuffer.unmap();
      let maximumError = 0;
      for (let head = 0; head < heads; head += 1) {
        let squareSum = 0;
        for (let column = 0; column < dimension; column += 1) {
          const value = input[head * dimension + column];
          squareSum += value * value;
        }
        const inverseRms = 1 / Math.sqrt(squareSum / dimension + 1e-6);
        for (let column = 0; column < dimension; column += 1) {
          const index = head * dimension + column;
          const expected = input[index] * inverseRms * weights.query[column];
          maximumError = Math.max(maximumError, Math.abs(actual[index] - expected));
        }
      }
      return { maximumError, sourceBytes: weights.sourceBytes };
    } finally {
      rms.destroyGemmaPrefillRmsResources(resources);
      inputBuffer.destroy();
      weightBuffer.destroy();
      readBuffer.destroy();
    }
  });

  expect(result.sourceBytes).toBe(8 * 768 + 2 * 64 * 2);
  expect(result.maximumError).toBeLessThan(2e-5);
});