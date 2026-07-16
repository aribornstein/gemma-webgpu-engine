import { expect, test } from "@playwright/test";

test("projects a real Gemma vision patch with learned 2D positions", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const sourceModulePath = "/src/model/pinned-safetensors.ts";
    const weightsModulePath = "/src/model/gemma-vision-weights.ts";
    const deviceModulePath = "/src/webgpu/device.ts";
    const patchModulePath = "/src/webgpu/vision-patch-embed.ts";
    const [{ PinnedSafetensorsSource }, { loadGemmaVisionPatchWeights },
      { getWebGpuDevice }, patchModule] = await Promise.all([
      import(sourceModulePath),
      import(weightsModulePath),
      import(deviceModulePath),
      import(patchModulePath),
    ]);
    const device = await getWebGpuDevice();
    const source = await PinnedSafetensorsSource.open();
    const weights = await loadGemmaVisionPatchWeights(source);
    const patches = Float32Array.from({ length: 768 }, (_, index) => (index % 256) / 255);
    const positions = new Int32Array([7, 11]);
    const make = (label: string, size: number, usage: GPUBufferUsageFlags) =>
      device.createBuffer({ label, size, usage });
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const patchBuffer = make("Vision patch fixture", patches.byteLength, storage);
    const positionBuffer = make("Vision position fixture", positions.byteLength, storage);
    const projectionBuffer = make("Vision projection fixture", weights.projection.byteLength, storage);
    const positionWeightBuffer = make("Vision position weights", weights.positions.byteLength, storage);
    const readBuffer = make(
      "Vision patch readback",
      768 * 4,
      GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    );
    const buffers = [patchBuffer, positionBuffer, projectionBuffer, positionWeightBuffer, readBuffer];
    try {
      device.queue.writeBuffer(patchBuffer, 0, patches);
      device.queue.writeBuffer(positionBuffer, 0, positions);
      device.queue.writeBuffer(projectionBuffer, 0, weights.projection);
      device.queue.writeBuffer(positionWeightBuffer, 0, weights.positions);
      const pipeline = await patchModule.getGemmaVisionPatchEmbedPipeline(device);
      const resources = patchModule.createGemmaVisionPatchEmbedResources(
        device,
        pipeline,
        patchBuffer,
        positionBuffer,
        projectionBuffer,
        positionWeightBuffer,
        1,
      );
      buffers.push(...resources.ownedBuffers);
      patchModule.updateGemmaVisionPatchEmbed(device, resources, 1);
      const encoder = device.createCommandEncoder();
      patchModule.encodeGemmaVisionPatchEmbed(encoder, pipeline, resources, 1);
      encoder.copyBufferToBuffer(resources.output, 0, readBuffer, 0, 768 * 4);
      device.queue.submit([encoder.finish()]);
      await readBuffer.mapAsync(GPUMapMode.READ);
      const actual = new Float32Array(readBuffer.getMappedRange().slice(0));
      readBuffer.unmap();

      const scratch = new DataView(new ArrayBuffer(4));
      const bf16 = (value: number) => {
        scratch.setFloat32(0, value, true);
        const bits = scratch.getUint32(0, true);
        const rounded = bits + 0x7fff + ((bits >>> 16) & 1);
        scratch.setUint32(0, rounded & 0xffff0000, true);
        return scratch.getFloat32(0, true);
      };
      const dimensions = [0, 1, 127, 511, 767];
      const expected = dimensions.map((feature) => {
        let sum = 0;
        for (let column = 0; column < 768; column += 1) {
          sum = Math.fround(sum + Math.fround(
            bf16(2 * (patches[column] - 0.5)) *
            weights.projection[feature * 768 + column],
          ));
        }
        sum = Math.fround(sum + weights.positions[(7 * 768) + feature]);
        sum = Math.fround(sum + weights.positions[((10_240 + 11) * 768) + feature]);
        return sum;
      });
      return {
        dimensions,
        actual: dimensions.map((dimension) => actual[dimension]),
        expected,
        sourceBytes: weights.sourceBytes,
      };
    } finally {
      for (const buffer of buffers.toReversed()) buffer.destroy();
    }
  });

  expect(result.sourceBytes).toBe(32_636_928);
  for (let index = 0; index < result.actual.length; index += 1) {
    expect(Math.abs(result.actual[index] - result.expected[index])).toBeLessThan(2e-5);
  }
});