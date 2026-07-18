import { expect, test } from "@playwright/test";

test("matches blocked causal Gemma audio relative attention", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const devicePath = "/src/webgpu/device.ts";
    const attentionPath = "/src/webgpu/audio-attention.ts";
    const [{ getWebGpuDevice }, attention] = await Promise.all([
      import(devicePath),
      import(attentionPath),
    ]);
    const rows = 4;
    const values = rows * 1024;
    const query = Float32Array.from({ length: values }, (_, index) =>
      Math.fround(Math.sin(index / 97) * 0.4));
    const key = Float32Array.from({ length: values }, (_, index) =>
      Math.fround(Math.cos(index / 83) * 0.3));
    const value = Float32Array.from({ length: values }, (_, index) =>
      Math.fround(Math.sin(index / 71) * 0.7));
    const relative = Float32Array.from({ length: 13 * 1024 }, (_, index) =>
      Math.fround(Math.cos(index / 107) * 0.2));
    const perDimensionScale = Float32Array.from({ length: 128 }, (_, index) =>
      Math.fround((index - 64) / 256));
    const mask = new Uint32Array([1, 1, 0, 1]);
    const device = await getWebGpuDevice();
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const upload = (data: Float32Array | Uint32Array) => {
      const buffer = device.createBuffer({ size: data.byteLength, usage: storage });
      device.queue.writeBuffer(buffer, 0, data);
      return buffer;
    };
    const buffers = [
      upload(query),
      upload(key),
      upload(value),
      upload(relative),
      upload(perDimensionScale),
      upload(mask),
    ];
    const readback = device.createBuffer({
      size: values * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const pipeline = await attention.getGemmaAudioAttentionPipeline(device);
    const resources = attention.createGemmaAudioAttentionResources(
      device,
      pipeline,
      buffers[0],
      buffers[1],
      buffers[2],
      buffers[3],
      buffers[4],
      buffers[5],
      rows,
    );
    try {
      const encoder = device.createCommandEncoder();
      attention.encodeGemmaAudioAttention(encoder, pipeline, resources);
      encoder.copyBufferToBuffer(resources.output, 0, readback, 0, values * 4);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const actual = new Float32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      const expected = new Float32Array(values);
      const queryScale = (128 ** -0.5) / Math.log(2);
      const keyScale = Math.log(1 + Math.E) / Math.log(2);
      for (let queryRow = 0; queryRow < rows; queryRow += 1) {
        if (!mask[queryRow]) continue;
        for (let head = 0; head < 8; head += 1) {
          const logits = new Float64Array(13).fill(-1e9);
          for (let offset = 0; offset < 13; offset += 1) {
            const keyRow = queryRow + offset - 12;
            if (keyRow < 0 || keyRow >= rows || !mask[keyRow]) continue;
            let dot = 0;
            for (let dimension = 0; dimension < 128; dimension += 1) {
              const feature = head * 128 + dimension;
              const softplus = Math.log1p(Math.exp(perDimensionScale[dimension]));
              const scaledQuery = query[queryRow * 1024 + feature] * queryScale * softplus;
              const scaledKey = key[keyRow * 1024 + feature] * keyScale;
              dot += scaledQuery * (scaledKey + relative[offset * 1024 + feature]);
            }
            logits[offset] = Math.tanh(dot / 50) * 50;
          }
          const maximum = Math.max(...logits);
          const probabilities = Float64Array.from(logits, (logit) => Math.exp(logit - maximum));
          const denominator = probabilities.reduce((sum, probability) => sum + probability, 0);
          for (let dimension = 0; dimension < 128; dimension += 1) {
            let sum = 0;
            for (let offset = 0; offset < 13; offset += 1) {
              const keyRow = queryRow + offset - 12;
              if (keyRow >= 0 && keyRow < rows) {
                sum += probabilities[offset] / denominator *
                  value[keyRow * 1024 + head * 128 + dimension];
              }
            }
            expected[queryRow * 1024 + head * 128 + dimension] = sum;
          }
        }
      }
      let maximumError = 0;
      for (let index = 0; index < values; index += 1) {
        maximumError = Math.max(maximumError, Math.abs(actual[index] - expected[index]));
      }
      return { maximumError, maskedMaximum: Math.max(...actual.slice(2 * 1024, 3 * 1024)) };
    } finally {
      attention.destroyGemmaAudioAttentionResources(resources);
      for (const buffer of buffers) buffer.destroy();
      readback.destroy();
    }
  });

  expect(result.maskedMaximum).toBe(0);
  expect(result.maximumError).toBeLessThan(3e-5);
});