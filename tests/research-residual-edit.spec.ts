import { expect, test } from "@playwright/test";

test("research residual edit matches a CPU reference and is a no-op at coefficient zero", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const modulePath = "/src/webgpu/research-residual-edit.ts";
    const devicePath = "/src/webgpu/device.ts";
    const edit = await import(modulePath);
    const { getWebGpuDevice } = await import(devicePath);
    const device: GPUDevice = await getWebGpuDevice();
    const H = 1536;
    const wave = (scale: number, phase: number) =>
      Float32Array.from({ length: H }, (_, i) => Math.fround(scale * Math.sin(i * 0.37 + phase)));
    const hidden0 = wave(3, 0.1);
    const w12s = new Float32Array(2 * H + 1);
    w12s.set(wave(0.5, 1.3).map((v) => v + 1), 0);
    w12s.set(wave(0.5, 2.1).map((v) => v + 1), H);
    w12s[2 * H] = 1;
    const a = wave(0.02, 0.7);
    const b = wave(0.03, 1.9);
    const inScale = 0.05;
    const storage = (data: Float32Array) => {
      const buffer = device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buffer, 0, data);
      return buffer;
    };
    const read = async (buffer: GPUBuffer) => {
      const readback = device.createBuffer({ size: buffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(buffer, 0, readback, 0, buffer.size);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const values = new Float32Array(readback.getMappedRange().slice(0));
      readback.destroy();
      return values;
    };
    const pipeline = await edit.getResearchResidualEditPipeline(device);
    const run = async (coefficient: number) => {
      const hidden = storage(hidden0);
      const nextInput = storage(new Float32Array(H).fill(7));
      const nextSum = storage(new Float32Array([7]));
      const weights = storage(w12s);
      const resources = edit.createResearchResidualEditResources(
        device, pipeline, 0, { hidden, normWeightsAndScale: weights, nextInput, nextSum, inScale }, a, b, coefficient,
      );
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      edit.encodeResearchResidualEditPass(pass, resources);
      pass.end();
      device.queue.submit([encoder.finish()]);
      const out = { hidden: await read(hidden), nextInput: await read(nextInput), nextSum: (await read(nextSum))[0] };
      edit.destroyResearchResidualEditResources(resources);
      for (const buffer of [hidden, nextInput, nextSum, weights]) buffer.destroy();
      return out;
    };
    const identity = await run(0);
    const coefficient = 2.5;
    const active = await run(coefficient);
    let dot = 0;
    for (let i = 0; i < H; i += 1) dot += a[i] * hidden0[i];
    const expectedHidden = Array.from(hidden0, (h, i) => h + coefficient * dot * b[i]);
    const rms = 1 / Math.sqrt(expectedHidden.reduce((s, v) => s + v * v, 0) / H + 1e-6);
    let hiddenError = 0;
    let quantizedMismatches = 0;
    let expectedSum = 0;
    for (let i = 0; i < H; i += 1) {
      hiddenError = Math.max(hiddenError, Math.abs(active.hidden[i] - expectedHidden[i]));
      const q = Math.max(-128, Math.min(127, Math.round(expectedHidden[i] * rms * w12s[H + i] / inScale))) * inScale;
      expectedSum += q;
      if (Math.abs(active.nextInput[i] - q) > inScale * 0.5) quantizedMismatches += 1;
    }
    return {
      identityHiddenUnchanged: identity.hidden.every((v, i) => Object.is(v, hidden0[i])),
      identityNextInputUntouched: identity.nextInput.every((v) => v === 7) && identity.nextSum === 7,
      hiddenError,
      quantizedMismatches,
      sumError: Math.abs(active.nextSum - expectedSum),
      dot,
    };
  });
  expect(result.identityHiddenUnchanged).toBe(true);
  expect(result.identityNextInputUntouched).toBe(true);
  expect(Math.abs(result.dot)).toBeGreaterThan(0.01);
  expect(result.hiddenError).toBeLessThan(1e-4);
  // Rounding-boundary flips are possible in f32; none should be off by more than half a step.
  expect(result.quantizedMismatches).toBe(0);
  expect(result.sumError).toBeLessThan(0.5);
});
