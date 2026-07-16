import { expect, test } from "@playwright/test";

test("executes real Gemma vision layer 0 deterministically on WebGPU", async ({ page }) => {
  test.setTimeout(60_000);
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const sourcePath = "/src/model/pinned-safetensors.ts";
    const weightsPath = "/src/model/gemma-vision-weights.ts";
    const devicePath = "/src/webgpu/device.ts";
    const layerPath = "/src/webgpu/vision-layer.ts";
    const [{ PinnedSafetensorsSource }, { loadGemmaVisionLayerWeights },
      { getWebGpuDevice }, layer] = await Promise.all([
      import(sourcePath),
      import(weightsPath),
      import(devicePath),
      import(layerPath),
    ]);
    const device = await getWebGpuDevice();
    const source = await PinnedSafetensorsSource.open();
    const weights = await loadGemmaVisionLayerWeights(source, 0);
    const rows = 2;
    const width = 768;
    const input = Float32Array.from({ length: rows * width }, (_, index) =>
      Math.fround(Math.sin(index / 31) * 0.75 + Math.cos(index / 47) * 0.25));
    const hidden = device.createBuffer({
      size: input.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const readback = device.createBuffer({
      size: input.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const resources = await layer.createGemmaVisionLayerResources(
      device,
      hidden,
      rows,
      new Int32Array([2, 3, 5, 7]),
      weights,
    );
    const run = async () => {
      device.queue.writeBuffer(hidden, 0, input);
      const encoder = device.createCommandEncoder();
      layer.encodeGemmaVisionLayer(encoder, resources);
      encoder.copyBufferToBuffer(hidden, 0, readback, 0, input.byteLength);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const output = new Float32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      return output;
    };
    device.pushErrorScope("validation");
    device.pushErrorScope("internal");
    try {
      const first = await run();
      const second = await run();
      let bitMismatches = 0;
      let changed = 0;
      let finite = true;
      const firstBits = new Uint32Array(first.buffer);
      const secondBits = new Uint32Array(second.buffer);
      for (let index = 0; index < first.length; index += 1) {
        if (firstBits[index] !== secondBits[index]) bitMismatches += 1;
        if (first[index] !== input[index]) changed += 1;
        if (!Number.isFinite(first[index])) finite = false;
      }
      const rowDistance = first.slice(0, width).reduce(
        (sum, value, index) => sum + Math.abs(value - first[width + index]),
        0,
      );
      const internalError = await device.popErrorScope();
      const validationError = await device.popErrorScope();
      return {
        bitMismatches,
        changed,
        finite,
        rowDistance,
        sourceBytes: weights.sourceBytes,
        gpuError: internalError?.message ?? validationError?.message ?? null,
      };
    } finally {
      layer.destroyGemmaVisionLayerResources(resources);
      hidden.destroy();
      readback.destroy();
    }
  });

  expect(result.gpuError).toBeNull();
  expect(result.sourceBytes).toBe(9_483_576);
  expect(result.bitMismatches).toBe(0);
  expect(result.changed).toBeGreaterThan(1400);
  expect(result.finite).toBe(true);
  expect(result.rowDistance).toBeGreaterThan(1);
});