import { expect, test } from "@playwright/test";

test("replaces an intentionally reset WebGPU device", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const modulePath = "/src/webgpu/device.ts";
    const { getWebGpuDevice, resetWebGpuDevice } = await import(modulePath);
    const first = await getWebGpuDevice();
    await resetWebGpuDevice();
    const loss = await first.lost;
    const second = await getWebGpuDevice();
    const reusedReplacement = await getWebGpuDevice() === second;
    await resetWebGpuDevice();
    return {
      replaced: first !== second,
      reusedReplacement,
      lossReason: loss.reason,
    };
  });

  expect(result).toEqual({
    replaced: true,
    reusedReplacement: true,
    lossReason: "destroyed",
  });
});

test("reacquires after an unexpected device loss", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const modulePath = "/src/webgpu/device.ts";
    const { getWebGpuDevice, resetWebGpuDevice } = await import(modulePath);
    const first = await getWebGpuDevice();
    first.destroy();
    const loss = await first.lost;
    await Promise.resolve();
    const second = await getWebGpuDevice();
    await resetWebGpuDevice();
    return {
      replaced: first !== second,
      lossReason: loss.reason,
    };
  });

  expect(result).toEqual({
    replaced: true,
    lossReason: "destroyed",
  });
});
