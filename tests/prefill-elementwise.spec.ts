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