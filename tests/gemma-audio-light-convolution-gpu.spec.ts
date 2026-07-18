import { expect, test } from "@playwright/test";

test("matches Gemma audio GLU causal depthwise convolution", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const devicePath = "/src/webgpu/device.ts";
    const convolutionPath = "/src/webgpu/audio-light-convolution.ts";
    const [{ getWebGpuDevice }, convolution] = await Promise.all([
      import(devicePath),
      import(convolutionPath),
    ]);
    const rows = 7;
    const expanded = Float32Array.from({ length: rows * 2048 }, (_, index) =>
      Math.fround(Math.sin(index / 113) * 0.8));
    const depthwise = Float32Array.from({ length: 1024 * 5 }, (_, index) =>
      Math.fround(Math.cos(index / 41) * 0.2));
    const norm = Float32Array.from({ length: 1024 }, (_, index) =>
      Math.fround(0.75 + (index % 17) / 34));
    const device = await getWebGpuDevice();
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const upload = (data: Float32Array) => {
      const buffer = device.createBuffer({ size: data.byteLength, usage: storage });
      device.queue.writeBuffer(buffer, 0, data);
      return buffer;
    };
    const inputBuffer = upload(expanded);
    const depthwiseBuffer = upload(depthwise);
    const normBuffer = upload(norm);
    const readback = device.createBuffer({
      size: rows * 1024 * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const pipeline = await convolution.getGemmaAudioLightConvolutionPipeline(device);
    const resources = convolution.createGemmaAudioLightConvolutionResources(
      device,
      pipeline,
      inputBuffer,
      depthwiseBuffer,
      normBuffer,
      rows,
    );
    try {
      const encoder = device.createCommandEncoder();
      convolution.encodeGemmaAudioLightConvolution(encoder, pipeline, resources);
      encoder.copyBufferToBuffer(resources.output, 0, readback, 0, rows * 1024 * 4);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const actual = new Float32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      const expected = new Float32Array(rows * 1024);
      for (let row = 0; row < rows; row += 1) {
        const convolved = new Float64Array(1024);
        for (let channel = 0; channel < 1024; channel += 1) {
          for (let kernel = 0; kernel < 5; kernel += 1) {
            const sourceRow = row + kernel - 4;
            if (sourceRow < 0) continue;
            const first = expanded[sourceRow * 2048 + channel];
            const second = expanded[sourceRow * 2048 + 1024 + channel];
            convolved[channel] += first / (1 + Math.exp(-second)) *
              depthwise[channel * 5 + kernel];
          }
        }
        const meanSquare = convolved.reduce((sum, value) => sum + value * value, 0) / 1024;
        const inverseRms = 1 / Math.sqrt(meanSquare + 1e-6);
        for (let channel = 0; channel < 1024; channel += 1) {
          const normalized = convolved[channel] * inverseRms * norm[channel];
          expected[row * 1024 + channel] = normalized / (1 + Math.exp(-normalized));
        }
      }
      let maximumError = 0;
      for (let index = 0; index < actual.length; index += 1) {
        maximumError = Math.max(maximumError, Math.abs(actual[index] - expected[index]));
      }
      return { maximumError };
    } finally {
      convolution.destroyGemmaAudioLightConvolutionResources(resources);
      inputBuffer.destroy();
      depthwiseBuffer.destroy();
      normBuffer.destroy();
      readback.destroy();
    }
  });

  expect(result.maximumError).toBeLessThan(4e-5);
});