import { expect, test } from "@playwright/test";

test("matches the Gemma audio masked convolution subsampler contract", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const devicePath = "/src/webgpu/device.ts";
    const subsamplerPath = "/src/webgpu/audio-subsampler.ts";
    const [{ getWebGpuDevice }, subsampler] = await Promise.all([
      import(devicePath),
      import(subsamplerPath),
    ]);
    const frameCount = 5;
    const features = Float32Array.from({ length: frameCount * 128 }, (_, index) =>
      Math.fround(Math.sin(index / 19) + (index % 13) / 17));
    const mask = new Uint8Array([1, 1, 1, 0, 0]);
    const convolution0 = new Float32Array(128 * 3 * 3);
    for (let channel = 0; channel < 128; channel += 1) {
      convolution0[(channel * 3 + 1) * 3 + 1] = Math.fround((channel - 63.5) / 64);
    }
    const convolution1 = new Float32Array(32 * 128 * 3 * 3);
    for (let channel = 0; channel < 32; channel += 1) {
      convolution1[((channel * 128 + channel) * 3 + 1) * 3 + 1] = 1;
    }
    const projection = new Float32Array(1024 * 1024);
    for (let index = 0; index < 1024; index += 1) projection[index * 1024 + index] = 1;
    const weights = {
      convolution0,
      norm0: new Float32Array(128).fill(1),
      convolution1,
      norm1: new Float32Array(32).fill(1),
      projection,
    };
    const input = {
      features,
      mask,
      frameCount,
      validFrameCount: 3,
      softTokenCount: 1,
      sampleCount: 1,
      paddedSampleCount: 128,
    };
    const device = await getWebGpuDevice();
    const pipelines = await subsampler.getGemmaAudioSubsamplerPipelines(device);
    const resources = subsampler.createGemmaAudioSubsamplerResources(
      device,
      pipelines,
      input,
      weights,
    );
    const readback = device.createBuffer({
      size: resources.outputRows * 1024 * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = device.createCommandEncoder();
      subsampler.encodeGemmaAudioSubsampler(encoder, pipelines, resources);
      encoder.copyBufferToBuffer(
        resources.output,
        0,
        readback,
        0,
        resources.outputRows * 1024 * 4,
      );
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const actual = new Float32Array(readback.getMappedRange().slice(0));
      readback.unmap();

      const convolve = (
        values: Float32Array,
        rows: number,
        columns: number,
        inputChannels: number,
        outputChannels: number,
        valid: Uint8Array,
        kernels: Float32Array,
      ) => {
        const outputRows = Math.ceil(rows / 2);
        const outputColumns = Math.ceil(columns / 2);
        const output = new Float32Array(outputRows * outputColumns * outputChannels);
        for (let row = 0; row < outputRows; row += 1) {
          for (let column = 0; column < outputColumns; column += 1) {
            const cell = new Float64Array(outputChannels);
            for (let outputChannel = 0; outputChannel < outputChannels; outputChannel += 1) {
              for (let inputChannel = 0; inputChannel < inputChannels; inputChannel += 1) {
                for (let kernelRow = 0; kernelRow < 3; kernelRow += 1) {
                  const sourceRow = row * 2 + kernelRow - 1;
                  if (sourceRow < 0 || sourceRow >= rows || !valid[sourceRow]) continue;
                  for (let kernelColumn = 0; kernelColumn < 3; kernelColumn += 1) {
                    const sourceColumn = column * 2 + kernelColumn - 1;
                    if (sourceColumn < 0 || sourceColumn >= columns) continue;
                    const inputIndex = (sourceRow * columns + sourceColumn) *
                      inputChannels + inputChannel;
                    const weightIndex = ((outputChannel * inputChannels + inputChannel) * 3 +
                      kernelRow) * 3 + kernelColumn;
                    cell[outputChannel] += values[inputIndex] * kernels[weightIndex];
                  }
                }
              }
            }
            const mean = cell.reduce((sum, value) => sum + value, 0) / outputChannels;
            const variance = cell.reduce(
              (sum, value) => sum + (value - mean) ** 2,
              0,
            ) / outputChannels;
            const inverseStd = 1 / Math.sqrt(variance + 1e-6);
            for (let channel = 0; channel < outputChannels; channel += 1) {
              output[(row * outputColumns + column) * outputChannels + channel] =
                Math.max(0, (cell[channel] - mean) * inverseStd);
            }
          }
        }
        return output;
      };
      const mask1 = Uint8Array.from([mask[0], mask[2], mask[4]]);
      const stage0 = convolve(features, 5, 128, 1, 128, mask, convolution0);
      const stage1 = convolve(stage0, 3, 64, 128, 32, mask1, convolution1);
      let maximumError = 0;
      for (let index = 0; index < actual.length; index += 1) {
        maximumError = Math.max(maximumError, Math.abs(actual[index] - stage1[index]));
      }
      return {
        outputRows: resources.outputRows,
        outputMask: Array.from(resources.outputMask),
        maximumError,
      };
    } finally {
      readback.destroy();
      subsampler.destroyGemmaAudioSubsamplerResources(resources);
    }
  });

  expect(result.outputRows).toBe(2);
  expect(result.outputMask).toEqual([1, 0]);
  expect(result.maximumError).toBeLessThan(2e-5);
});