import { expect, test } from "@playwright/test";

test("applies bidirectional tiled attention for 64-dimensional vision heads", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const attentionModulePath = "/src/webgpu/prefill-attention.ts";
    const deviceModulePath = "/src/webgpu/device.ts";
    const [attention, { getWebGpuDevice }] = await Promise.all([
      import(attentionModulePath),
      import(deviceModulePath),
    ]);
    const device = await getWebGpuDevice();
    const rows = 3;
    const heads = 2;
    const width = 64;
    const query = Float32Array.from({ length: rows * heads * width }, (_, index) =>
      Math.sin(index / 23));
    const key = Float32Array.from({ length: query.length }, (_, index) =>
      Math.cos(index / 19));
    const value = Float32Array.from({ length: query.length }, (_, index) =>
      (index % 17) / 17);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const make = (size: number, usage = storage) => device.createBuffer({ size, usage });
    const queryBuffer = make(query.byteLength);
    const keyBuffer = make(key.byteLength);
    const valueBuffer = make(value.byteLength);
    const readBuffer = make(query.byteLength, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    const buffers = [queryBuffer, keyBuffer, valueBuffer, readBuffer];
    try {
      device.queue.writeBuffer(queryBuffer, 0, query);
      device.queue.writeBuffer(keyBuffer, 0, key);
      device.queue.writeBuffer(valueBuffer, 0, value);
      const pipeline = await attention.getGemmaPrefillAttentionPipeline(device, 64);
      const resources = attention.createGemmaPrefillAttentionResources(
        device,
        pipeline,
        queryBuffer,
        keyBuffer,
        valueBuffer,
        rows,
        rows,
        {
          sequence: rows,
          keyLength: rows,
          queryOffset: 0,
          queryHeads: heads,
          kvHeads: heads,
          window: 0,
          causal: false,
        },
      );
      buffers.push(...resources.ownedBuffers);
      const encoder = device.createCommandEncoder();
      attention.encodeGemmaPrefillAttention(encoder, pipeline, resources, rows);
      encoder.copyBufferToBuffer(resources.output, 0, readBuffer, 0, query.byteLength);
      device.queue.submit([encoder.finish()]);
      await readBuffer.mapAsync(GPUMapMode.READ);
      const actual = new Float32Array(readBuffer.getMappedRange().slice(0));
      readBuffer.unmap();
      const expected = new Float32Array(query.length);
      for (let row = 0; row < rows; row += 1) {
        for (let head = 0; head < heads; head += 1) {
          const scores = [];
          for (let keyRow = 0; keyRow < rows; keyRow += 1) {
            let score = 0;
            for (let column = 0; column < width; column += 1) {
              score += query[(row * heads + head) * width + column] *
                key[(keyRow * heads + head) * width + column];
            }
            scores.push(score);
          }
          const maximum = Math.max(...scores);
          const probabilities = scores.map((score) => Math.exp(score - maximum));
          const denominator = probabilities.reduce((sum, value) => sum + value, 0);
          for (let column = 0; column < width; column += 1) {
            let sum = 0;
            for (let keyRow = 0; keyRow < rows; keyRow += 1) {
              sum += probabilities[keyRow] *
                value[(keyRow * heads + head) * width + column];
            }
            expected[(row * heads + head) * width + column] = sum / denominator;
          }
        }
      }
      let maximumError = 0;
      for (let index = 0; index < actual.length; index += 1) {
        maximumError = Math.max(maximumError, Math.abs(actual[index] - expected[index]));
      }
      return { maximumError };
    } finally {
      for (const buffer of buffers.toReversed()) buffer.destroy();
    }
  });

  expect(result.maximumError).toBeLessThan(2e-5);
});