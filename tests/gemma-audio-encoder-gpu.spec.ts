import { expect, test } from "@playwright/test";

test("encodes synthesized speech through all twelve Gemma audio layers", async ({ page }) => {
  test.setTimeout(300_000);
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const sourcePath = "/src/model/pinned-safetensors.ts";
    const inputPath = "/src/runtime/gemma-audio-input.ts";
    const devicePath = "/src/webgpu/device.ts";
    const encoderPath = "/src/webgpu/audio-input-encoder.ts";
    const [{ PinnedSafetensorsSource }, inputModule, { getWebGpuDevice }, encoderModule] =
      await Promise.all([
        import(sourcePath),
        import(inputPath),
        import(devicePath),
        import(encoderPath),
      ]);
    const response = await fetch("/examples/gemma-audio-demo.wav");
    if (!response.ok) throw new Error(`Could not load audio fixture: ${response.status}`);
    const input = await inputModule.prepareGemmaAudio(await response.blob());
    const source = await PinnedSafetensorsSource.open();
    const device = await getWebGpuDevice();
    const progress: number[] = [];
    device.pushErrorScope("validation");
    const resources = await encoderModule.encodeGemmaAudioFeatures(
      device,
      source,
      input,
      ({ completedLayers }: { completedLayers: number }) => progress.push(completedLayers),
    );
    const validationError = await device.popErrorScope();
    const outputBytes = resources.softTokenCount * 1536 * 4;
    const readback = device.createBuffer({
      size: outputBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const command = device.createCommandEncoder();
      command.copyBufferToBuffer(resources.output, 0, readback, 0, outputBytes);
      device.queue.submit([command.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const output = new Float32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      let maximum = 0;
      let checksum = 0;
      for (let index = 0; index < output.length; index += 1) {
        maximum = Math.max(maximum, Math.abs(output[index]));
        checksum += output[index] * ((index % 31) - 15);
      }
      return {
        validationError: validationError?.message ?? null,
        finite: output.every(Number.isFinite),
        maximum,
        checksum,
        softTokenCount: resources.softTokenCount,
        paddedTokenCount: resources.paddedTokenCount,
        sourceBytes: resources.sourceBytes,
        progress,
      };
    } finally {
      readback.destroy();
      encoderModule.destroyGemmaAudioEncodingResources(resources);
    }
  });

  expect(result.validationError).toBeNull();
  expect(result.finite).toBe(true);
  expect(result.maximum).toBeGreaterThan(0.01);
  expect(result.maximum).toBeLessThan(100);
  expect(Number.isFinite(result.checksum)).toBe(true);
  expect(result.softTokenCount).toBeGreaterThan(0);
  expect(result.softTokenCount).toBeLessThanOrEqual(result.paddedTokenCount);
  expect(result.sourceBytes).toBe(150_587_456);
  expect(result.progress).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
});