import { expect, test } from "@playwright/test";

test("normalizes exact RMS rows independently across a prefill block", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const rmsModulePath = "/src/webgpu/prefill-rms.ts";
    const deviceModulePath = "/src/webgpu/device.ts";
    const {
      createGemmaPrefillRmsResources,
      destroyGemmaPrefillRmsResources,
      encodeGemmaPrefillRms,
      getGemmaPrefillRmsPipeline,
    } = await import(rmsModulePath);
    const { getWebGpuDevice } = await import(deviceModulePath);
    const device = await getWebGpuDevice();

    async function compare(
      dimension: number,
      rows: number,
      weighted: boolean,
      referenceRows: readonly number[],
    ): Promise<number> {
      const input = Float32Array.from(
        { length: rows * dimension },
        (_, index) => Math.fround(Math.sin(index * 0.013) * 2.5 + (index % 7) * 0.01),
      );
      const weight = Float32Array.from(
        { length: dimension },
        (_, index) => Math.fround(0.75 + (index % 19) * 0.015625),
      );
      const storageUpload = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
      const inputBuffer = device.createBuffer({ size: input.byteLength, usage: storageUpload });
      const singleInputBuffer = device.createBuffer({ size: dimension * 4, usage: storageUpload });
      const weightBuffer = weighted
        ? device.createBuffer({ size: weight.byteLength, usage: storageUpload })
        : null;
      device.queue.writeBuffer(inputBuffer, 0, input);
      if (weightBuffer) device.queue.writeBuffer(weightBuffer, 0, weight);

      async function run(buffer: GPUBuffer, rowCount: number): Promise<Uint32Array> {
        const pipeline = await getGemmaPrefillRmsPipeline(device, dimension, weighted);
        const resources = createGemmaPrefillRmsResources(
          device,
          pipeline,
          rowCount,
          buffer,
          weightBuffer,
        );
        const outputBytes = rowCount * dimension * 4;
        const readback = device.createBuffer({
          size: outputBytes,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        try {
          const encoder = device.createCommandEncoder();
          encodeGemmaPrefillRms(encoder, pipeline, resources);
          encoder.copyBufferToBuffer(resources.output, 0, readback, 0, outputBytes);
          device.queue.submit([encoder.finish()]);
          await readback.mapAsync(GPUMapMode.READ);
          const values = new Uint32Array(readback.getMappedRange().slice(0));
          readback.unmap();
          return values;
        } finally {
          readback.destroy();
          destroyGemmaPrefillRmsResources(resources);
        }
      }

      const batch = await run(inputBuffer, rows);
      let bitMismatches = 0;
      for (const row of referenceRows) {
        device.queue.writeBuffer(
          singleInputBuffer,
          0,
          input.subarray(row * dimension, (row + 1) * dimension),
        );
        const single = await run(singleInputBuffer, 1);
        for (let index = 0; index < dimension; index += 1) {
          if (batch[row * dimension + index] !== single[index]) bitMismatches += 1;
        }
      }
      inputBuffer.destroy();
      singleInputBuffer.destroy();
      weightBuffer?.destroy();
      return bitMismatches;
    }

    device.pushErrorScope("validation");
    device.pushErrorScope("internal");
    const weightedBitMismatches = await compare(1536, 32, true, [0, 15, 31]);
    const unweightedBitMismatches = await compare(512, 4, false, [0, 3]);
    const internalError = await device.popErrorScope();
    const validationError = await device.popErrorScope();
    return {
      weightedBitMismatches,
      unweightedBitMismatches,
      gpuError: internalError?.message ?? validationError?.message ?? null,
    };
  });

  expect(result).toEqual({
    weightedBitMismatches: 0,
    unweightedBitMismatches: 0,
    gpuError: null,
  });
});