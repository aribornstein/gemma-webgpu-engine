import { expect, test } from "@playwright/test";

test("hands the selected prefill row and its sum to the decode LM head", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const modulePath = "/src/webgpu/prefill-lm-head-input.ts";
    const deviceModulePath = "/src/webgpu/device.ts";
    const {
      createGemmaPrefillLmHeadInputResources,
      encodeGemmaPrefillLmHeadInputPass,
      getGemmaPrefillLmHeadInputPipeline,
      updateGemmaPrefillLmHeadInput,
    } = await import(modulePath);
    const { getWebGpuDevice } = await import(deviceModulePath);
    const device = await getWebGpuDevice();
    const rows = 32;
    const hidden = 1536;
    const source = Float32Array.from(
      { length: rows * hidden },
      (_, index) => Math.fround(Math.sin(index * 0.7) * (1 + Math.floor(index / hidden))),
    );
    const sourceBuffer = device.createBuffer({
      size: source.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(sourceBuffer, 0, source);
    const activation = device.createBuffer({
      size: hidden * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const activationSum = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readback = device.createBuffer({
      size: hidden * 4 + 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const pipeline = await getGemmaPrefillLmHeadInputPipeline(device);
    const resources = createGemmaPrefillLmHeadInputResources(
      device,
      pipeline,
      sourceBuffer,
      rows,
      activation,
      activationSum,
    );
    const checks: { row: number; copyMismatches: number; sumError: number }[] = [];
    for (const row of [0, 15, 31]) {
      updateGemmaPrefillLmHeadInput(device, resources, row);
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      encodeGemmaPrefillLmHeadInputPass(pass, pipeline, resources);
      pass.end();
      encoder.copyBufferToBuffer(activation, 0, readback, 0, hidden * 4);
      encoder.copyBufferToBuffer(activationSum, 0, readback, hidden * 4, 4);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const values = new Float32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      let copyMismatches = 0;
      let expectedSum = 0;
      for (let index = 0; index < hidden; index += 1) {
        const expected = source[row * hidden + index];
        expectedSum += expected;
        if (values[index] !== expected) copyMismatches += 1;
      }
      checks.push({ row, copyMismatches, sumError: Math.abs(values[hidden] - expectedSum) });
    }
    for (const buffer of [sourceBuffer, activation, activationSum, readback]) buffer.destroy();
    return checks;
  });

  for (const check of result) {
    expect(check.copyMismatches).toBe(0);
    expect(check.sumError).toBeLessThan(1e-2);
  }
});
