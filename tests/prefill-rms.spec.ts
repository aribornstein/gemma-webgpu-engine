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

test("fuses RMS residual and layer scale without changing output bits", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const rmsModulePath = "/src/webgpu/prefill-rms.ts";
    const elementwiseModulePath = "/src/webgpu/prefill-elementwise.ts";
    const deviceModulePath = "/src/webgpu/device.ts";
    const {
      createGemmaPrefillRmsResidualResources,
      createGemmaPrefillRmsResources,
      destroyGemmaPrefillRmsResidualResources,
      destroyGemmaPrefillRmsResources,
      encodeGemmaPrefillRms,
      encodeGemmaPrefillRmsResidual,
      getGemmaPrefillRmsPipeline,
      getGemmaPrefillRmsResidualPipeline,
    } = await import(rmsModulePath);
    const {
      createGemmaPrefillAddResources,
      createGemmaPrefillMultiplyResources,
      destroyGemmaPrefillElementwiseResources,
      encodeGemmaPrefillElementwise,
      getGemmaPrefillElementwisePipelines,
    } = await import(elementwiseModulePath);
    const { getWebGpuDevice } = await import(deviceModulePath);
    const device = await getWebGpuDevice();
    const rows = 32;
    const dimension = 64;
    const input = Float32Array.from(
      { length: rows * dimension },
      (_, index) => Math.fround(Math.sin(index * 0.013) * 2.5 + (index % 7) * 0.01),
    );
    const weight = Float32Array.from(
      { length: dimension },
      (_, index) => Math.fround(0.75 + (index % 19) * 0.015625),
    );
    const residual = Float32Array.from(
      { length: rows * dimension },
      (_, index) => Math.fround(Math.cos(index * 0.021) * 1.75),
    );
    const factors = new Float32Array([0.5, 1.25, Math.fround(-0.875)]);
    const storageUpload =
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const upload = (data: ArrayBufferView) => {
      const buffer = device.createBuffer({ size: data.byteLength, usage: storageUpload });
      device.queue.writeBuffer(buffer, 0, data);
      return buffer;
    };
    const inputBuffer = upload(input);
    const weightBuffer = upload(weight);
    const factorBuffer = upload(factors);
    const rmsPipeline = await getGemmaPrefillRmsPipeline(device, dimension, true);
    const elementwisePipelines = await getGemmaPrefillElementwisePipelines(device);
    let residualBitMismatches = 0;
    let scaledBitMismatches = 0;

    device.pushErrorScope("validation");
    device.pushErrorScope("internal");
    for (const scaled of [false, true]) {
      const currentResidual = upload(residual);
      const fusedResidual = upload(residual);
      const rms = createGemmaPrefillRmsResources(
        device,
        rmsPipeline,
        rows,
        inputBuffer,
        weightBuffer,
      );
      const add = createGemmaPrefillAddResources(
        device,
        elementwisePipelines.add,
        currentResidual,
        rms.output,
        rows * dimension,
      );
      const multiply = scaled
        ? createGemmaPrefillMultiplyResources(
            device,
            elementwisePipelines.multiply,
            currentResidual,
            factorBuffer,
            rows * dimension,
            2,
          )
        : null;
      const fusedPipeline = await getGemmaPrefillRmsResidualPipeline(
        device,
        dimension,
        scaled,
      );
      const fused = createGemmaPrefillRmsResidualResources(
        device,
        fusedPipeline,
        rows,
        inputBuffer,
        weightBuffer,
        fusedResidual,
        scaled ? factorBuffer : null,
        scaled ? 2 : 0,
      );
      const outputBytes = rows * dimension * 4;
      const readback = device.createBuffer({
        size: outputBytes * 2,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      try {
        const encoder = device.createCommandEncoder();
        encodeGemmaPrefillRms(encoder, rmsPipeline, rms);
        encodeGemmaPrefillElementwise(encoder, elementwisePipelines.add, add);
        if (multiply) {
          encodeGemmaPrefillElementwise(
            encoder,
            elementwisePipelines.multiply,
            multiply,
          );
        }
        encodeGemmaPrefillRmsResidual(encoder, fusedPipeline, fused);
        encoder.copyBufferToBuffer(currentResidual, 0, readback, 0, outputBytes);
        encoder.copyBufferToBuffer(fusedResidual, 0, readback, outputBytes, outputBytes);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const outputs = new Uint32Array(readback.getMappedRange().slice(0));
        readback.unmap();
        const outputElements = outputBytes / 4;
        for (let index = 0; index < outputElements; index += 1) {
          if (outputs[index] !== outputs[outputElements + index]) {
            if (scaled) scaledBitMismatches += 1;
            else residualBitMismatches += 1;
          }
        }
      } finally {
        readback.destroy();
        destroyGemmaPrefillRmsResources(rms);
        destroyGemmaPrefillElementwiseResources(add);
        if (multiply) destroyGemmaPrefillElementwiseResources(multiply);
        destroyGemmaPrefillRmsResidualResources(fused);
        currentResidual.destroy();
        fusedResidual.destroy();
      }
    }
    const internalError = await device.popErrorScope();
    const validationError = await device.popErrorScope();
    inputBuffer.destroy();
    weightBuffer.destroy();
    factorBuffer.destroy();
    return {
      residualBitMismatches,
      scaledBitMismatches,
      gpuError: internalError?.message ?? validationError?.message ?? null,
    };
  });

  expect(result).toEqual({
    residualBitMismatches: 0,
    scaledBitMismatches: 0,
    gpuError: null,
  });
});