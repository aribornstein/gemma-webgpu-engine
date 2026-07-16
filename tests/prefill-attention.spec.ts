import { expect, test } from "@playwright/test";

test("applies tiled causal sliding attention across grouped query heads", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const attentionModulePath = "/src/webgpu/prefill-attention.ts";
    const deviceModulePath = "/src/webgpu/device.ts";
    const {
      createGemmaPrefillAttentionResources,
      destroyGemmaPrefillAttentionResources,
      encodeGemmaPrefillAttention,
      getGemmaPrefillAttentionPipeline,
    } = await import(attentionModulePath);
    const { getWebGpuDevice } = await import(deviceModulePath);
    const device = await getWebGpuDevice();

    async function run(headDimension: 256 | 512): Promise<number> {
      const sequence = 32;
      const queryOffset = 4;
      const keyLength = queryOffset + sequence;
      const cacheCapacity = 40;
      const queryHeads = 8;
      const kvHeads = 1;
      const query = new Float32Array(sequence * queryHeads * headDimension);
      const key = new Float32Array(cacheCapacity * kvHeads * headDimension);
      const value = new Float32Array(cacheCapacity * kvHeads * headDimension);
      for (let position = 0; position < cacheCapacity; position += 1) {
        value.fill(position + 1, position * headDimension, (position + 1) * headDimension);
      }
      const storageUpload = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
      const queryBuffer = device.createBuffer({ size: query.byteLength, usage: storageUpload });
      const keyBuffer = device.createBuffer({ size: key.byteLength, usage: storageUpload });
      const valueBuffer = device.createBuffer({ size: value.byteLength, usage: storageUpload });
      device.queue.writeBuffer(queryBuffer, 0, query);
      device.queue.writeBuffer(keyBuffer, 0, key);
      device.queue.writeBuffer(valueBuffer, 0, value);
      const pipeline = await getGemmaPrefillAttentionPipeline(device, headDimension);
      const resources = createGemmaPrefillAttentionResources(
        device,
        pipeline,
        queryBuffer,
        keyBuffer,
        valueBuffer,
        sequence,
        cacheCapacity,
        {
          sequence,
          keyLength,
          queryOffset,
          queryHeads,
          kvHeads,
          window: 4,
        },
      );
      const readback = device.createBuffer({
        size: resources.output.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      try {
        const encoder = device.createCommandEncoder();
        encodeGemmaPrefillAttention(encoder, pipeline, resources, sequence);
        encoder.copyBufferToBuffer(resources.output, 0, readback, 0, resources.output.size);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const output = new Float32Array(readback.getMappedRange().slice(0));
        readback.unmap();
        let maximumAbsoluteError = 0;
        for (let row = 0; row < sequence; row += 1) {
          const expected = queryOffset + row - 0.5;
          for (let head = 0; head < queryHeads; head += 1) {
            const headBase = (row * queryHeads + head) * headDimension;
            for (let dimension = 0; dimension < headDimension; dimension += 1) {
              maximumAbsoluteError = Math.max(
                maximumAbsoluteError,
                Math.abs(output[headBase + dimension] - expected),
              );
            }
          }
        }
        return maximumAbsoluteError;
      } finally {
        readback.destroy();
        destroyGemmaPrefillAttentionResources(resources);
        queryBuffer.destroy();
        keyBuffer.destroy();
        valueBuffer.destroy();
      }
    }

    device.pushErrorScope("validation");
    device.pushErrorScope("internal");
    const slidingMaximumAbsoluteError = await run(256);
    const fullMaximumAbsoluteError = await run(512);
    const internalError = await device.popErrorScope();
    const validationError = await device.popErrorScope();
    return {
      slidingMaximumAbsoluteError,
      fullMaximumAbsoluteError,
      gpuError: internalError?.message ?? validationError?.message ?? null,
    };
  });

  expect(result).toEqual({
    slidingMaximumAbsoluteError: 0,
    fullMaximumAbsoluteError: 0,
    gpuError: null,
  });
});