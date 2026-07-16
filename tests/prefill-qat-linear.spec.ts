import { expect, test } from "@playwright/test";

test("projects a 32-row QAT block exactly like individual rows", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const qatModulePath = "/src/webgpu/prefill-qat-linear.ts";
    const deviceModulePath = "/src/webgpu/device.ts";
    const {
      createGemmaPrefillQatLinearResources,
      destroyGemmaPrefillQatLinearResources,
      encodeGemmaPrefillQatLinear,
      getGemmaPrefillQatLinearPipelines,
    } = await import(qatModulePath);
    const { getWebGpuDevice } = await import(deviceModulePath);
    const device = await getWebGpuDevice();
    const rows = 32;
    const inFeatures = 64;
    const outFeatures = 1024;
    const packedWeights = Uint32Array.from(
      { length: outFeatures * inFeatures / 8 },
      (_, index) => Math.imul(index + 17, 0x9e3779b1) >>> 0,
    );
    const rowScales = Float32Array.from(
      { length: outFeatures },
      (_, index) => Math.fround(0.0005 + (index % 31) * 0.00003),
    );
    const activation = Float32Array.from(
      { length: rows * inFeatures },
      (_, index) => Math.fround(Math.sin(index * 0.17) * 3.25),
    );
    const storageUpload = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const activationBuffer = device.createBuffer({
      size: activation.byteLength,
      usage: storageUpload,
    });
    const singleActivationBuffer = device.createBuffer({
      size: inFeatures * 4,
      usage: storageUpload,
    });
    const weightBuffer = device.createBuffer({
      size: 256 + packedWeights.byteLength,
      usage: storageUpload,
    });
    const scaleBuffer = device.createBuffer({
      size: 256 + rowScales.byteLength,
      usage: storageUpload,
    });
    device.queue.writeBuffer(activationBuffer, 0, activation);
    device.queue.writeBuffer(weightBuffer, 256, packedWeights);
    device.queue.writeBuffer(scaleBuffer, 256, rowScales);
    const weights = {
      packedWeights: { buffer: weightBuffer, offset: 256, size: packedWeights.byteLength },
      rowScales: { buffer: scaleBuffer, offset: 256, size: rowScales.byteLength },
      inputScale: Math.fround(0.03125),
      outputScale: Math.fround(0.0078125),
    };

    async function run(
      input: GPUBuffer,
      rowCount: number,
    ): Promise<Uint32Array> {
      const pipelines = await getGemmaPrefillQatLinearPipelines(device, {
        rows: rowCount,
        inFeatures,
        outFeatures,
        bits: 4,
      });
      const resources = createGemmaPrefillQatLinearResources(
        device,
        pipelines,
        input,
        weights,
      );
      const outputBytes = rowCount * outFeatures * 4;
      const readback = device.createBuffer({
        size: outputBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      try {
        const encoder = device.createCommandEncoder();
        encodeGemmaPrefillQatLinear(encoder, pipelines, resources);
        encoder.copyBufferToBuffer(resources.output, 0, readback, 0, outputBytes);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const output = new Uint32Array(readback.getMappedRange().slice(0));
        readback.unmap();
        return output;
      } finally {
        readback.destroy();
        destroyGemmaPrefillQatLinearResources(resources);
      }
    }

    device.pushErrorScope("validation");
    device.pushErrorScope("internal");
    const batch = await run(activationBuffer, rows);
    let bitMismatches = 0;
    for (const row of [0, 7, 8, 31]) {
      device.queue.writeBuffer(
        singleActivationBuffer,
        0,
        activation.subarray(row * inFeatures, (row + 1) * inFeatures),
      );
      const single = await run(singleActivationBuffer, 1);
      for (let outputIndex = 0; outputIndex < outFeatures; outputIndex += 1) {
        if (batch[row * outFeatures + outputIndex] !== single[outputIndex]) {
          bitMismatches += 1;
        }
      }
    }
    const internalError = await device.popErrorScope();
    const validationError = await device.popErrorScope();
    activationBuffer.destroy();
    singleActivationBuffer.destroy();
    weightBuffer.destroy();
    scaleBuffer.destroy();
    return {
      bitMismatches,
      gpuError: internalError?.message ?? validationError?.message ?? null,
    };
  });

  expect(result).toEqual({ bitMismatches: 0, gpuError: null });
});