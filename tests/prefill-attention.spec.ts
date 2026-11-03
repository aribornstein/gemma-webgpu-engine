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
// Zero queries and keys make every score equal, which hides errors in the dot-product path.
// Random inputs against an f64 reference catch reduced-precision K/V staging and score bugs.
test("matches an f64 reference with random queries, keys, and values", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const errors = await page.evaluate(async () => {
    const attentionModulePath = "/src/webgpu/prefill-attention.ts";
    const deviceModulePath = "/src/webgpu/device.ts";
    const {
      createGemmaPrefillAttentionResources,
      createGemmaPrefillAttentionShader,
      destroyGemmaPrefillAttentionResources,
      encodeGemmaPrefillAttention,
      getGemmaPrefillAttentionPipeline,
    } = await import(attentionModulePath);
    const { getWebGpuDevice } = await import(deviceModulePath);
    const device = await getWebGpuDevice();
    // Devices limited to 16 KiB of workgroup storage use 4-key tiles for head dimension 512.
    const fourKeyFullPipeline = {
      headDimension: 512 as const,
      queryTile: 32 as const,
      pipeline: await device.createComputePipelineAsync({
        layout: "auto",
        compute: {
          module: device.createShaderModule({ code: createGemmaPrefillAttentionShader(512, 4) }),
          entryPoint: "main",
        },
      }),
    };
    let state = 7;
    const random = (): number => {
      state = (Math.imul(state, 1103515245) + 12345) >>> 0;
      return state / 4294967296 - 0.5;
    };

    async function run(
      headDimension: 256 | 512,
      sequence: number,
      queryOffset: number,
      cacheCapacity: number,
      window: number,
      pipelineOverride?: typeof fourKeyFullPipeline,
    ): Promise<number> {
      const queryHeads = 8;
      const keyLength = queryOffset + sequence;
      const query = Float32Array.from({ length: sequence * queryHeads * headDimension }, random);
      const key = Float32Array.from({ length: cacheCapacity * headDimension }, random);
      const value = Float32Array.from({ length: cacheCapacity * headDimension }, random);
      const upload = (data: Float32Array): GPUBuffer => {
        const buffer = device.createBuffer({
          size: data.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(buffer, 0, data);
        return buffer;
      };
      const buffers = [upload(query), upload(key), upload(value)];
      const pipeline = pipelineOverride ??
        await getGemmaPrefillAttentionPipeline(device, headDimension);
      const resources = createGemmaPrefillAttentionResources(
        device,
        pipeline,
        buffers[0],
        buffers[1],
        buffers[2],
        sequence,
        cacheCapacity,
        { sequence, keyLength, queryOffset, queryHeads, kvHeads: 1, window },
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
          const position = queryOffset + row;
          const first = window > 0 ? Math.max(0, position + 1 - window) : 0;
          for (let head = 0; head < queryHeads; head += 1) {
            const queryBase = (row * queryHeads + head) * headDimension;
            const scores: number[] = [];
            for (let keyIndex = first; keyIndex <= position; keyIndex += 1) {
              const keyBase = (keyIndex % cacheCapacity) * headDimension;
              let score = 0;
              for (let dimension = 0; dimension < headDimension; dimension += 1) {
                score += query[queryBase + dimension] * key[keyBase + dimension];
              }
              scores.push(score);
            }
            const maximum = Math.max(...scores);
            const weights = scores.map((score) => Math.exp(score - maximum));
            const total = weights.reduce((sum, weight) => sum + weight, 0);
            for (let dimension = 0; dimension < headDimension; dimension += 1) {
              let expected = 0;
              for (let keyIndex = first; keyIndex <= position; keyIndex += 1) {
                expected += weights[keyIndex - first] *
                  value[(keyIndex % cacheCapacity) * headDimension + dimension];
              }
              maximumAbsoluteError = Math.max(
                maximumAbsoluteError,
                Math.abs(expected / total - output[queryBase + dimension]),
              );
            }
          }
        }
        return maximumAbsoluteError;
      } finally {
        readback.destroy();
        destroyGemmaPrefillAttentionResources(resources);
        for (const buffer of buffers) buffer.destroy();
      }
    }

    return {
      sliding: await run(256, 32, 0, 512, 512),
      slidingOffset: await run(256, 32, 32, 512, 512),
      full: await run(512, 32, 0, 4096, 0),
      fullPartial: await run(512, 13, 37, 4096, 0),
      fullFourKeyTiles: await run(512, 32, 5, 4096, 0, fourKeyFullPipeline),
    };
  });

  for (const error of Object.values(errors)) expect(error).toBeLessThan(1e-5);
});
