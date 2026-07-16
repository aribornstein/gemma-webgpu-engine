import { expect, test } from "@playwright/test";

test("projects real Gemma vision layer-0 Q weights with signed I8 SRQ", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const sourceModulePath = "/src/model/pinned-safetensors.ts";
    const weightsModulePath = "/src/model/gemma-vision-weights.ts";
    const deviceModulePath = "/src/webgpu/device.ts";
    const denseModulePath = "/src/webgpu/prefill-ple-dense.ts";
    const [{ PinnedSafetensorsSource }, { loadGemmaVisionProjectionWeights },
      { getWebGpuDevice }, dense] = await Promise.all([
      import(sourceModulePath),
      import(weightsModulePath),
      import(deviceModulePath),
      import(denseModulePath),
    ]);
    const device = await getWebGpuDevice();
    const source = await PinnedSafetensorsSource.open();
    const weights = await loadGemmaVisionProjectionWeights(
      source,
      "model.vision_tower.encoder.layers.0.self_attn.q_proj",
      768,
      768,
    );
    const input = Float32Array.from({ length: 768 }, (_, index) => Math.sin(index / 17));
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const activation = device.createBuffer({ size: input.byteLength, usage: storage });
    const codes = device.createBuffer({ size: weights.packedWeights.byteLength, usage: storage });
    const scales = device.createBuffer({ size: weights.rowScales.byteLength, usage: storage });
    const readback = device.createBuffer({
      size: 768 * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const buffers = [activation, codes, scales, readback];
    try {
      device.queue.writeBuffer(activation, 0, input);
      device.queue.writeBuffer(codes, 0, weights.packedWeights);
      device.queue.writeBuffer(scales, 0, weights.rowScales);
      const pipeline = await dense.getGemmaPrefillPleDensePipeline(device, {
        rows: 1,
        inFeatures: 768,
        outFeatures: 768,
      });
      const resources = dense.createGemmaPrefillPleDenseResources(
        device,
        pipeline,
        activation,
        {
          codes,
          rowScales: scales,
          inputScale: weights.inputScale,
          outputScale: weights.outputScale,
        },
      );
      buffers.push(...resources.ownedBuffers);
      const encoder = device.createCommandEncoder();
      dense.encodeGemmaPrefillPleDense(encoder, pipeline, resources);
      encoder.copyBufferToBuffer(resources.output, 0, readback, 0, 768 * 4);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const actual = new Float32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      const bytes = new Uint8Array(weights.packedWeights.buffer);
      const srq = (value: number, scale: number) => scale === 0
        ? value
        : Math.min(127, Math.max(-128, Math.round(value / scale))) * scale;
      const rows = [0, 1, 127, 511, 767];
      const expected = rows.map((row) => {
        let sum = 0;
        for (let column = 0; column < 768; column += 1) {
          sum += (bytes[row * 768 + column] - 128) *
            srq(input[column], weights.inputScale);
        }
        return srq(sum * weights.rowScales[row], weights.outputScale);
      });
      return {
        rows,
        actual: rows.map((row) => actual[row]),
        expected,
        inputScale: weights.inputScale,
        outputScale: weights.outputScale,
      };
    } finally {
      for (const buffer of buffers.toReversed()) buffer.destroy();
    }
  });

  expect(result.inputScale).toBeGreaterThan(0);
  expect(result.outputScale).toBeGreaterThan(0);
  for (let index = 0; index < result.actual.length; index += 1) {
    expect(Math.abs(result.actual[index] - result.expected[index])).toBeLessThan(2e-4);
  }
});