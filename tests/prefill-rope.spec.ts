import { expect, test } from "@playwright/test";

test("rotates batched sliding and partial-full heads like individual rows", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const ropeModulePath = "/src/webgpu/prefill-rope.ts";
    const modelRopeModulePath = "/src/model/gemma-rope.ts";
    const deviceModulePath = "/src/webgpu/device.ts";
    const {
      createGemmaPrefillRopeResources,
      destroyGemmaPrefillRopeResources,
      encodeGemmaPrefillRope,
      getGemmaPrefillRopePipeline,
    } = await import(ropeModulePath);
    const { createGemmaRotaryBlock, createGemmaRotaryRows } = await import(modelRopeModulePath);
    const { getWebGpuDevice } = await import(deviceModulePath);
    const device = await getWebGpuDevice();

    async function compare(headDimension: 256 | 512): Promise<number> {
      const rows = 4;
      const heads = 8;
      const values = Float32Array.from(
        { length: rows * heads * headDimension },
        (_, index) => Math.fround(Math.sin(index * 0.019) * 1.75),
      );
      const storageUpload = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC;
      const batchBuffer = device.createBuffer({ size: values.byteLength, usage: storageUpload });
      const singleBuffer = device.createBuffer({
        size: heads * headDimension * 4,
        usage: storageUpload,
      });
      device.queue.writeBuffer(batchBuffer, 0, values);

      async function run(
        buffer: GPUBuffer,
        rowCount: number,
        rotary: { cosine: Float32Array; sine: Float32Array },
      ): Promise<Uint32Array> {
        const pipeline = await getGemmaPrefillRopePipeline(device, headDimension);
        const resources = createGemmaPrefillRopeResources(
          device,
          pipeline,
          buffer,
          rowCount,
          heads,
          rotary,
        );
        const byteLength = rowCount * heads * headDimension * 4;
        const readback = device.createBuffer({
          size: byteLength,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        try {
          const encoder = device.createCommandEncoder();
          encodeGemmaPrefillRope(encoder, pipeline, resources, rowCount);
          encoder.copyBufferToBuffer(buffer, 0, readback, 0, byteLength);
          device.queue.submit([encoder.finish()]);
          await readback.mapAsync(GPUMapMode.READ);
          const output = new Uint32Array(readback.getMappedRange().slice(0));
          readback.unmap();
          return output;
        } finally {
          readback.destroy();
          destroyGemmaPrefillRopeResources(resources);
        }
      }

      const block = createGemmaRotaryBlock(9, rows);
      const batch = await run(
        batchBuffer,
        rows,
        headDimension === 256 ? block.sliding : block.full,
      );
      let bitMismatches = 0;
      const rowElements = heads * headDimension;
      for (let row = 0; row < rows; row += 1) {
        device.queue.writeBuffer(
          singleBuffer,
          0,
          values.subarray(row * rowElements, (row + 1) * rowElements),
        );
        const singleRows = createGemmaRotaryRows(9 + row);
        const single = await run(
          singleBuffer,
          1,
          headDimension === 256 ? singleRows.sliding : singleRows.full,
        );
        for (let index = 0; index < rowElements; index += 1) {
          if (batch[row * rowElements + index] !== single[index]) bitMismatches += 1;
        }
      }
      batchBuffer.destroy();
      singleBuffer.destroy();
      return bitMismatches;
    }

    device.pushErrorScope("validation");
    device.pushErrorScope("internal");
    const slidingBitMismatches = await compare(256);
    const fullBitMismatches = await compare(512);
    const internalError = await device.popErrorScope();
    const validationError = await device.popErrorScope();
    return {
      slidingBitMismatches,
      fullBitMismatches,
      gpuError: internalError?.message ?? validationError?.message ?? null,
    };
  });

  expect(result).toEqual({
    slidingBitMismatches: 0,
    fullBitMismatches: 0,
    gpuError: null,
  });
});