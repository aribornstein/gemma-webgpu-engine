import { expect, test } from "@playwright/test";

test("matches signed dense PLE arithmetic without duplicate weights", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const pleModulePath = "/src/webgpu/prefill-ple-dense.ts";
    const deviceModulePath = "/src/webgpu/device.ts";
    const {
      createGemmaPrefillPleDenseResources,
      destroyGemmaPrefillPleDenseResources,
      encodeGemmaPrefillPleDense,
      getGemmaPrefillPleDensePipeline,
    } = await import(pleModulePath);
    const { getWebGpuDevice } = await import(deviceModulePath);
    const device = await getWebGpuDevice();

    async function run(inFeatures: 256 | 1536, outFeatures: 256 | 1536) {
      const rows = 32;
      const activation = new Float32Array(rows * inFeatures);
      for (let row = 0; row < rows; row += 1) {
        activation.fill(((row % 7) - 3) * 0.6, row * inFeatures, (row + 1) * inFeatures);
      }
      const wordsPerRow = inFeatures / 4;
      const codes = new Uint32Array(outFeatures * wordsPerRow).fill(0x80808080);
      const rowScales = Float32Array.from(
        { length: outFeatures },
        (_, outputRow) => 2 ** ((outputRow % 4) - 2),
      );
      for (let outputRow = 0; outputRow < outFeatures; outputRow += 1) {
        codes[outputRow * wordsPerRow] = 0x80808081;
      }
      function upload(values: Float32Array | Uint32Array): GPUBuffer {
        const buffer = device.createBuffer({
          size: values.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(buffer, 0, values);
        return buffer;
      }
      const activationBuffer = upload(activation);
      const codeBuffer = upload(codes);
      const scaleBuffer = upload(rowScales);
      const pipeline = await getGemmaPrefillPleDensePipeline(device, {
        rows,
        inFeatures,
        outFeatures,
      });
      const resources = createGemmaPrefillPleDenseResources(
        device,
        pipeline,
        activationBuffer,
        {
          codes: codeBuffer,
          rowScales: scaleBuffer,
          inputScale: 0.5,
          outputScale: 0.125,
        },
      );
      const readback = device.createBuffer({
        size: resources.output.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      try {
        const encoder = device.createCommandEncoder();
        encodeGemmaPrefillPleDense(encoder, pipeline, resources);
        encoder.copyBufferToBuffer(resources.output, 0, readback, 0, resources.output.size);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const output = new Float32Array(readback.getMappedRange().slice(0));
        readback.unmap();
        let mismatchCount = 0;
        for (let row = 0; row < rows; row += 1) {
          const quantizedInput = Math.round(activation[row * inFeatures] / 0.5) * 0.5;
          for (let outputRow = 0; outputRow < outFeatures; outputRow += 1) {
            const linear = quantizedInput * rowScales[outputRow];
            const expected = Math.round(linear / 0.125) * 0.125;
            if (output[row * outFeatures + outputRow] !== expected) mismatchCount += 1;
          }
        }
        return mismatchCount;
      } finally {
        readback.destroy();
        destroyGemmaPrefillPleDenseResources(resources);
        activationBuffer.destroy();
        codeBuffer.destroy();
        scaleBuffer.destroy();
      }
    }

    device.pushErrorScope("validation");
    device.pushErrorScope("internal");
    const gateMismatchCount = await run(1536, 256);
    const projectionMismatchCount = await run(256, 1536);
    const internalError = await device.popErrorScope();
    const validationError = await device.popErrorScope();
    return {
      gateMismatchCount,
      projectionMismatchCount,
      gpuError: internalError?.message ?? validationError?.message ?? null,
    };
  });

  expect(result).toEqual({
    gateMismatchCount: 0,
    projectionMismatchCount: 0,
    gpuError: null,
  });
});