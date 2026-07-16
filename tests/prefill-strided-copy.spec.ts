import { expect, test } from "@playwright/test";

test("copies cache rows and selects a mutable last row", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const copyModulePath = "/src/webgpu/prefill-strided-copy.ts";
    const deviceModulePath = "/src/webgpu/device.ts";
    const {
      createGemmaPrefillStridedCopyResources,
      destroyGemmaPrefillStridedCopyResources,
      encodeGemmaPrefillStridedCopy,
      getGemmaPrefillStridedCopyPipeline,
      updateGemmaPrefillStridedCopy,
    } = await import(copyModulePath);
    const { getWebGpuDevice } = await import(deviceModulePath);
    const device = await getWebGpuDevice();
    const sourceValues = Float32Array.from({ length: 4 * 8 }, (_, index) => index + 0.25);
    const source = device.createBuffer({
      size: sourceValues.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const destination = device.createBuffer({
      size: 8 * 8 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(source, 0, sourceValues);
    const pipeline = await getGemmaPrefillStridedCopyPipeline(device);
    const resources = createGemmaPrefillStridedCopyResources(
      device,
      pipeline,
      source,
      destination,
      {
        rows: 4,
        sourceStride: 8,
        sourceStart: 2,
        destinationStride: 8,
        destinationStart: 3 * 8,
        copyColumns: 4,
      },
    );
    const readback = device.createBuffer({
      size: destination.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    device.pushErrorScope("validation");
    device.pushErrorScope("internal");
    try {
      let encoder = device.createCommandEncoder();
      encodeGemmaPrefillStridedCopy(encoder, pipeline, resources, 4);
      device.queue.submit([encoder.finish()]);
      updateGemmaPrefillStridedCopy(device, resources, {
        rows: 1,
        sourceStride: 8,
        sourceStart: 2 * 8,
        destinationStride: 8,
        destinationStart: 0,
        copyColumns: 8,
      });
      encoder = device.createCommandEncoder();
      encodeGemmaPrefillStridedCopy(encoder, pipeline, resources, 1);
      encoder.copyBufferToBuffer(destination, 0, readback, 0, destination.size);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const values = Array.from(new Float32Array(readback.getMappedRange().slice(0)));
      readback.unmap();
      const internalError = await device.popErrorScope();
      const validationError = await device.popErrorScope();
      return {
        selected: values.slice(0, 8),
        cached: values.slice(3 * 8, 3 * 8 + 4),
        gpuError: internalError?.message ?? validationError?.message ?? null,
      };
    } finally {
      readback.destroy();
      destroyGemmaPrefillStridedCopyResources(resources);
      source.destroy();
      destination.destroy();
    }
  });

  expect(result).toEqual({
    selected: Array.from({ length: 8 }, (_, index) => index + 16.25),
    cached: [2.25, 3.25, 4.25, 5.25],
    gpuError: null,
  });
});