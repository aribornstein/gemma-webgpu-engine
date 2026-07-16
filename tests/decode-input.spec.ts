import { expect, test } from "@playwright/test";

test("prepares multiple input rows exactly like independent decode rows", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const inputModulePath = "/src/webgpu/decode-input.ts";
    const deviceModulePath = "/src/webgpu/device.ts";
    const {
      createGemmaDecodeInputResources,
      destroyGemmaDecodeInputResources,
      encodeGemmaDecodeInput,
      getGemmaDecodeInputPipeline,
      uploadGemmaTokenInputBatch,
    } = await import(inputModulePath);
    const { getWebGpuDevice } = await import(deviceModulePath);
    const device = await getWebGpuDevice();
    const pipelines = await getGemmaDecodeInputPipeline(device);
    const weights = {
      projectionBfloat16: new Uint32Array(8960 * 768).fill(0x3f803f80),
      projectionNorm: new Float32Array(256).fill(1),
    };
    const inputs = [
      {
        hidden: new Float32Array(1536).fill(1),
        perLayerEmbedding: new Float32Array(8960).fill(2),
      },
      {
        hidden: new Float32Array(1536).fill(-0.5),
        perLayerEmbedding: new Float32Array(8960).fill(-3),
      },
    ];

    async function run(inputRows: typeof inputs) {
      const resources = createGemmaDecodeInputResources(
        device,
        pipelines,
        weights,
        inputRows.length,
      );
      uploadGemmaTokenInputBatch(device, resources, inputRows);
      const hiddenBytes = inputRows.length * 1536 * 4;
      const perLayerBytes = inputRows.length * 8960 * 4;
      const readback = device.createBuffer({
        size: hiddenBytes + perLayerBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      try {
        const encoder = device.createCommandEncoder();
        encodeGemmaDecodeInput(encoder, pipelines, resources);
        encoder.copyBufferToBuffer(resources.hidden, 0, readback, 0, hiddenBytes);
        encoder.copyBufferToBuffer(
          resources.perLayerInputs,
          0,
          readback,
          hiddenBytes,
          perLayerBytes,
        );
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const bytes = readback.getMappedRange();
        const hidden = new Uint32Array(bytes.slice(0, hiddenBytes));
        const perLayerInputs = new Uint32Array(bytes.slice(hiddenBytes));
        readback.unmap();
        return { hidden, perLayerInputs };
      } finally {
        readback.destroy();
        destroyGemmaDecodeInputResources(resources);
      }
    }

    device.pushErrorScope("validation");
    device.pushErrorScope("internal");
    const batch = await run(inputs);
    const separate = [await run([inputs[0]]), await run([inputs[1]])];
    const internalError = await device.popErrorScope();
    const validationError = await device.popErrorScope();
    let hiddenBitMismatches = 0;
    let perLayerBitMismatches = 0;
    for (let row = 0; row < inputs.length; row += 1) {
      for (let index = 0; index < 1536; index += 1) {
        if (batch.hidden[row * 1536 + index] !== separate[row].hidden[index]) {
          hiddenBitMismatches += 1;
        }
      }
      for (let index = 0; index < 8960; index += 1) {
        if (batch.perLayerInputs[row * 8960 + index] !==
            separate[row].perLayerInputs[index]) {
          perLayerBitMismatches += 1;
        }
      }
    }
    return {
      hiddenBitMismatches,
      perLayerBitMismatches,
      gpuError: internalError?.message ?? validationError?.message ?? null,
    };
  });

  expect(result).toEqual({
    hiddenBitMismatches: 0,
    perLayerBitMismatches: 0,
    gpuError: null,
  });
});