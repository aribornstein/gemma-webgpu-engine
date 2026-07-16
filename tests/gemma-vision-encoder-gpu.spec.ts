import { expect, test } from "@playwright/test";

test("executes all 16 real Gemma vision encoder layers", async ({ page }) => {
  test.setTimeout(180_000);
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const sourcePath = "/src/model/pinned-safetensors.ts";
    const devicePath = "/src/webgpu/device.ts";
    const encoderPath = "/src/webgpu/vision-encoder.ts";
    const [{ PinnedSafetensorsSource }, { getWebGpuDevice }, vision] =
      await Promise.all([
        import(sourcePath),
        import(devicePath),
        import(encoderPath),
      ]);
    const source = await PinnedSafetensorsSource.open();
    const device = await getWebGpuDevice();
    const input = Float32Array.from({ length: 768 }, (_, index) =>
      Math.fround(Math.sin(index / 37) * 0.5 + Math.cos(index / 53) * 0.25));
    const hidden = device.createBuffer({
      size: input.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const readback = device.createBuffer({
      size: input.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    device.queue.writeBuffer(hidden, 0, input);
    const completed: number[] = [];
    device.pushErrorScope("validation");
    device.pushErrorScope("internal");
    try {
      const encoded = await vision.runGemmaVisionEncoder(
        device,
        source,
        hidden,
        1,
        new Int32Array([4, 9]),
        (progress: { completedLayers: number }) => completed.push(progress.completedLayers),
      );
      const command = device.createCommandEncoder();
      command.copyBufferToBuffer(hidden, 0, readback, 0, input.byteLength);
      device.queue.submit([command.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const output = new Float32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      let finite = true;
      let changed = 0;
      for (let index = 0; index < output.length; index += 1) {
        finite &&= Number.isFinite(output[index]);
        if (output[index] !== input[index]) changed += 1;
      }
      const internalError = await device.popErrorScope();
      const validationError = await device.popErrorScope();
      return {
        layers: encoded.layers,
        sourceBytes: encoded.sourceBytes,
        elapsedMilliseconds: encoded.elapsedMilliseconds,
        completed,
        finite,
        changed,
        gpuError: internalError?.message ?? validationError?.message ?? null,
      };
    } finally {
      hidden.destroy();
      readback.destroy();
    }
  });

  expect(result.gpuError).toBeNull();
  expect(result.layers).toBe(16);
  expect(result.sourceBytes).toBe(151_737_216);
  expect(result.completed).toEqual(Array.from({ length: 16 }, (_, index) => index + 1));
  expect(result.finite).toBe(true);
  expect(result.changed).toBeGreaterThan(700);
  expect(result.elapsedMilliseconds).toBeGreaterThan(0);
});