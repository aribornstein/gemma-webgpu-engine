import { expect, test } from "@playwright/test";

test("beam K/V slots restore each beam's own history in linear and circular caches", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const failures = await page.evaluate(async () => {
    const cacheModulePath = "/src/webgpu/decode-kv-cache.ts";
    const slotsModulePath = "/src/webgpu/decode-kv-beam-slots.ts";
    const deviceModulePath = "/src/webgpu/device.ts";
    const { DecodeKvCache } = await import(cacheModulePath);
    const { DecodeKvBeamSlots } = await import(slotsModulePath);
    const { getWebGpuDevice } = await import(deviceModulePath);
    const device: GPUDevice = await getWebGpuDevice();
    const headDim = 4;
    const caches = [
      new DecodeKvCache(device, { capacity: 32, kvHeads: 1, headDim, mode: "linear" }),
      new DecodeKvCache(device, { capacity: 4, kvHeads: 1, headDim, mode: "circular" }),
    ];
    const row = (position: number, token: number, half: number) =>
      Float32Array.from({ length: headDim }, (_, lane) => position * 1000 + token * 10 + lane + half * 0.5);
    const prompt = [7, 8, 9];
    for (const cache of caches) {
      prompt.forEach((token, position) => cache.writeTokens(device.queue, position, row(position, token, 0), row(position, token, 1)));
    }
    const base = prompt.length;
    const beamCount = 3;
    const maxRows = 7;
    const slots = new DecodeKvBeamSlots(device, caches, beamCount, maxRows, base);
    const readCache = async (cache: InstanceType<typeof DecodeKvCache>) => {
      const readback = device.createBuffer({ size: cache.bufferBytes * 2, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(cache.keyBuffer, 0, readback, 0, cache.bufferBytes);
      encoder.copyBufferToBuffer(cache.valueBuffer, 0, readback, cache.bufferBytes, cache.bufferBytes);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const data = new Float32Array(readback.getMappedRange().slice(0));
      readback.destroy();
      return data;
    };
    const failures: string[] = [];
    let beams: number[][] = [[]];
    let seed = 12345;
    const random = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
    for (let length = 1; length <= maxRows; length += 1) {
      const parents = Array.from({ length: beamCount }, () => Math.floor(random() * beams.length));
      const tokens = parents.map(() => 1 + Math.floor(random() * 9));
      const reorder = device.createCommandEncoder();
      slots.encodeReorder(reorder, parents, length - 1);
      device.queue.submit([reorder.finish()]);
      beams = parents.map((parent, beam) => [...beams[parent], tokens[beam]]);
      for (let beam = 0; beam < beamCount; beam += 1) {
        const restore = device.createCommandEncoder();
        slots.encodeRestore(restore, beam, length - 1);
        device.queue.submit([restore.finish()]);
        const position = base + length - 1;
        for (const cache of caches) cache.writeTokens(device.queue, position, row(position, tokens[beam], 0), row(position, tokens[beam], 1));
        const history = [...prompt, ...beams[beam]];
        for (const cache of caches) {
          const data = await readCache(cache);
          const first = cache.mode === "circular" ? Math.max(0, position - cache.capacity + 1) : 0;
          for (let visible = first; visible <= position; visible += 1) {
            const physical = cache.physicalPosition(visible);
            for (const half of [0, 1]) {
              const expected = row(visible, history[visible], half);
              const offset = half * cache.capacity * headDim + physical * headDim;
              const actual = data.subarray(offset, offset + headDim);
              if (expected.some((value, lane) => value !== actual[lane])) {
                failures.push(`${cache.mode} length ${length} beam ${beam} position ${visible} half ${half}: ${Array.from(actual)}`);
              }
            }
          }
        }
        const save = device.createCommandEncoder();
        slots.encodeSave(save, beam, length - 1);
        device.queue.submit([save.finish()]);
      }
    }
    slots.destroy();
    for (const cache of caches) cache.destroy();
    return failures;
  });
  expect(failures).toEqual([]);
});
