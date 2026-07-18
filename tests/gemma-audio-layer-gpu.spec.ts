import { expect, test } from "@playwright/test";

test("executes real Gemma audio layer zero with bounded finite output", async ({ page }) => {
  test.setTimeout(180_000);
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const sourcePath = "/src/model/pinned-safetensors.ts";
    const weightsPath = "/src/model/gemma-audio-weights.ts";
    const devicePath = "/src/webgpu/device.ts";
    const layerPath = "/src/webgpu/audio-layer.ts";
    const [{ PinnedSafetensorsSource }, weightsModule, { getWebGpuDevice }, layerModule] =
      await Promise.all([
        import(sourcePath),
        import(weightsPath),
        import(devicePath),
        import(layerPath),
      ]);
    const rows = 4;
    const input = Float32Array.from({ length: rows * 1024 }, (_, index) =>
      Math.fround(Math.sin(index / 89) * 0.6 + Math.cos(index / 37) * 0.15));
    const source = await PinnedSafetensorsSource.open();
    const weights = await weightsModule.loadGemmaAudioLayerWeights(source, 0);
    const device = await getWebGpuDevice();
    const hidden = device.createBuffer({
      size: input.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(hidden, 0, input);
    const readback = device.createBuffer({
      size: input.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const resources = await layerModule.createGemmaAudioLayerResources(
      device,
      hidden,
      new Uint32Array(rows).fill(1),
      rows,
      weights,
    );
    try {
      device.pushErrorScope("validation");
      const encoder = device.createCommandEncoder({ label: "Gemma audio real layer 0 test" });
      layerModule.encodeGemmaAudioLayer(encoder, resources);
      encoder.copyBufferToBuffer(hidden, 0, readback, 0, input.byteLength);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      const validationError = await device.popErrorScope();
      await readback.mapAsync(GPUMapMode.READ);
      const output = new Float32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      let maximum = 0;
      let changed = 0;
      let checksum = 0;
      for (let index = 0; index < output.length; index += 1) {
        maximum = Math.max(maximum, Math.abs(output[index]));
        if (output[index] !== input[index]) changed += 1;
        checksum += output[index] * ((index % 29) - 14);
      }
      return {
        validationError: validationError?.message ?? null,
        finite: output.every(Number.isFinite),
        maximum,
        changed,
        checksum,
        sourceBytes: weights.sourceBytes,
      };
    } finally {
      layerModule.destroyGemmaAudioLayerResources(resources);
      hidden.destroy();
      readback.destroy();
    }
  });

  expect(result.validationError).toBeNull();
  expect(result.finite).toBe(true);
  expect(result.maximum).toBeGreaterThan(0.01);
  expect(result.maximum).toBeLessThan(100);
  expect(result.changed).toBe(4 * 1024);
  expect(Number.isFinite(result.checksum)).toBe(true);
  expect(result.sourceBytes).toBe(10_875_472);
});