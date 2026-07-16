import { expect, test } from "@playwright/test";

test("runs exact batched prefill elementwise arithmetic", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const elementwiseModulePath = "/src/webgpu/prefill-elementwise.ts";
    const deviceModulePath = "/src/webgpu/device.ts";
    const {
      createGemmaPrefillAddResources,
      createGemmaPrefillGeluMultiplyResources,
      createGemmaPrefillMultiplyResources,
      destroyGemmaPrefillElementwiseResources,
      encodeGemmaPrefillElementwise,
      getGemmaPrefillElementwisePipelines,
    } = await import(elementwiseModulePath);
    const { getWebGpuDevice } = await import(deviceModulePath);
    const device = await getWebGpuDevice();
    const pipelines = await getGemmaPrefillElementwisePipelines(device);
    const count = 300;
    const first = Float32Array.from({ length: count }, (_, index) => index);
    const second = new Float32Array(count).fill(2);
    const gate = Float32Array.from(
      { length: count },
      (_, index) => ((index % 257) - 128) * 0.25,
    );
    const lookup = Float32Array.from({ length: 256 }, (_, index) => index - 128);
    const factor = new Float32Array([99, 0.5]);
    const storageUpload = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const storageOutput = storageUpload | GPUBufferUsage.COPY_SRC;
    function upload(values: Float32Array, usage = storageUpload): GPUBuffer {
      const buffer = device.createBuffer({ size: values.byteLength, usage });
      device.queue.writeBuffer(buffer, 0, values);
      return buffer;
    }
    const firstBuffer = upload(first, storageOutput);
    const secondBuffer = upload(second);
    const gateBuffer = upload(gate);
    const factorBuffer = upload(factor);
    const lookupBuffer = upload(lookup);
    const geluOutput = device.createBuffer({
      size: first.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const addResources = createGemmaPrefillAddResources(
      device,
      pipelines.add,
      firstBuffer,
      secondBuffer,
      count,
    );
    const multiplyResources = createGemmaPrefillMultiplyResources(
      device,
      pipelines.multiply,
      firstBuffer,
      factorBuffer,
      count,
      1,
    );
    const geluResources = createGemmaPrefillGeluMultiplyResources(
      device,
      pipelines.geluMultiply,
      gateBuffer,
      secondBuffer,
      lookupBuffer,
      geluOutput,
      count,
      0.25,
    );
    const arithmeticReadback = device.createBuffer({
      size: first.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const geluReadback = device.createBuffer({
      size: first.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      device.pushErrorScope("validation");
      device.pushErrorScope("internal");
      const encoder = device.createCommandEncoder();
      encodeGemmaPrefillElementwise(encoder, pipelines.add, addResources);
      encodeGemmaPrefillElementwise(encoder, pipelines.multiply, multiplyResources);
      encodeGemmaPrefillElementwise(encoder, pipelines.geluMultiply, geluResources);
      encoder.copyBufferToBuffer(firstBuffer, 0, arithmeticReadback, 0, first.byteLength);
      encoder.copyBufferToBuffer(geluOutput, 0, geluReadback, 0, first.byteLength);
      device.queue.submit([encoder.finish()]);
      await Promise.all([
        arithmeticReadback.mapAsync(GPUMapMode.READ),
        geluReadback.mapAsync(GPUMapMode.READ),
      ]);
      const arithmetic = new Float32Array(arithmeticReadback.getMappedRange().slice(0));
      const gelu = new Float32Array(geluReadback.getMappedRange().slice(0));
      arithmeticReadback.unmap();
      geluReadback.unmap();
      let mismatchCount = 0;
      for (let index = 0; index < count; index += 1) {
        if (arithmetic[index] !== (index + 2) * 0.5) mismatchCount += 1;
        const lookupIndex = Math.min(255, index % 257);
        if (gelu[index] !== (lookupIndex - 128) * 2) mismatchCount += 1;
      }
      const internalError = await device.popErrorScope();
      const validationError = await device.popErrorScope();
      return {
        mismatchCount,
        gpuError: internalError?.message ?? validationError?.message ?? null,
      };
    } finally {
      arithmeticReadback.destroy();
      geluReadback.destroy();
      destroyGemmaPrefillElementwiseResources(addResources);
      destroyGemmaPrefillElementwiseResources(multiplyResources);
      destroyGemmaPrefillElementwiseResources(geluResources);
      firstBuffer.destroy();
      secondBuffer.destroy();
      gateBuffer.destroy();
      factorBuffer.destroy();
      lookupBuffer.destroy();
      geluOutput.destroy();
    }
  });

  expect(result).toEqual({ mismatchCount: 0, gpuError: null });
});

test("reads strided PLE multipliers without changing activation bits", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const elementwiseModulePath = "/src/webgpu/prefill-elementwise.ts";
    const copyModulePath = "/src/webgpu/prefill-strided-copy.ts";
    const deviceModulePath = "/src/webgpu/device.ts";
    const {
      createGemmaPrefillGeluMultiplyResources,
      createGemmaPrefillStridedGeluMultiplyResources,
      destroyGemmaPrefillElementwiseResources,
      encodeGemmaPrefillElementwisePass,
      getGemmaPrefillElementwisePipelines,
    } = await import(elementwiseModulePath);
    const {
      createGemmaPrefillStridedCopyResources,
      destroyGemmaPrefillStridedCopyResources,
      encodeGemmaPrefillStridedCopyPass,
      getGemmaPrefillStridedCopyPipeline,
    } = await import(copyModulePath);
    const { getWebGpuDevice } = await import(deviceModulePath);
    const device = await getWebGpuDevice();
    const rows = 32;
    const columns = 256;
    const layers = 35;
    const layerIndex = 17;
    const count = rows * columns;
    const sourceStride = layers * columns;
    const sourceStart = layerIndex * columns;
    const gate = Float32Array.from(
      { length: count },
      (_, index) => Math.fround(((index % 257) - 128) * 0.03125),
    );
    const multiplier = Float32Array.from(
      { length: rows * sourceStride },
      (_, index) => Math.fround(Math.sin(index * 0.013) * 2.5),
    );
    const lookup = Float32Array.from(
      { length: 256 },
      (_, index) => Math.fround(Math.cos((index - 128) * 0.021) * 1.75),
    );
    const uploadUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const upload = (values: Float32Array) => {
      const buffer = device.createBuffer({ size: values.byteLength, usage: uploadUsage });
      device.queue.writeBuffer(buffer, 0, values);
      return buffer;
    };
    const gateBuffer = upload(gate);
    const multiplierBuffer = upload(multiplier);
    const lookupBuffer = upload(lookup);
    const copiedMultiplier = device.createBuffer({
      size: count * 4,
      usage: GPUBufferUsage.STORAGE,
    });
    const baselineOutput = device.createBuffer({
      size: count * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const stridedOutput = device.createBuffer({
      size: count * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const pipelines = await getGemmaPrefillElementwisePipelines(device);
    const copyPipeline = await getGemmaPrefillStridedCopyPipeline(device);
    const copy = createGemmaPrefillStridedCopyResources(
      device,
      copyPipeline,
      multiplierBuffer,
      copiedMultiplier,
      {
        rows,
        sourceStride,
        sourceStart,
        destinationStride: columns,
        destinationStart: 0,
        copyColumns: columns,
      },
    );
    const baseline = createGemmaPrefillGeluMultiplyResources(
      device,
      pipelines.geluMultiply,
      gateBuffer,
      copiedMultiplier,
      lookupBuffer,
      baselineOutput,
      count,
      0.03125,
    );
    const strided = createGemmaPrefillStridedGeluMultiplyResources(
      device,
      pipelines.geluMultiplyStrided,
      gateBuffer,
      multiplierBuffer,
      lookupBuffer,
      stridedOutput,
      rows,
      columns,
      sourceStride,
      sourceStart,
      0.03125,
    );
    const readback = device.createBuffer({
      size: count * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      device.pushErrorScope("validation");
      device.pushErrorScope("internal");
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      encodeGemmaPrefillStridedCopyPass(pass, copyPipeline, copy, rows);
      encodeGemmaPrefillElementwisePass(pass, pipelines.geluMultiply, baseline);
      encodeGemmaPrefillElementwisePass(pass, pipelines.geluMultiplyStrided, strided);
      pass.end();
      encoder.copyBufferToBuffer(baselineOutput, 0, readback, 0, count * 4);
      encoder.copyBufferToBuffer(stridedOutput, 0, readback, count * 4, count * 4);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const outputs = new Uint32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      let bitMismatches = 0;
      for (let index = 0; index < count; index += 1) {
        if (outputs[index] !== outputs[count + index]) bitMismatches += 1;
      }
      const internalError = await device.popErrorScope();
      const validationError = await device.popErrorScope();
      return {
        bitMismatches,
        gpuError: internalError?.message ?? validationError?.message ?? null,
      };
    } finally {
      readback.destroy();
      destroyGemmaPrefillStridedCopyResources(copy);
      destroyGemmaPrefillElementwiseResources(baseline);
      destroyGemmaPrefillElementwiseResources(strided);
      gateBuffer.destroy();
      multiplierBuffer.destroy();
      lookupBuffer.destroy();
      copiedMultiplier.destroy();
      baselineOutput.destroy();
      stridedOutput.destroy();
    }
  });

  expect(result).toEqual({ bitMismatches: 0, gpuError: null });
});