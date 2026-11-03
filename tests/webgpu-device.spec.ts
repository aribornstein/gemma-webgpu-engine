import { expect, test } from "@playwright/test";

test("retries WebGPU device creation after a failed adapter request", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const gpu = navigator.gpu;
    const requestAdapter = gpu.requestAdapter.bind(gpu);
    let failNext = true;
    gpu.requestAdapter = async (options?: GPURequestAdapterOptions) => {
      if (failNext) {
        failNext = false;
        return null;
      }
      return requestAdapter(options);
    };
    const modulePath = "/src/webgpu/device.ts";
    const { getWebGpuDevice, resetWebGpuDevice } = await import(modulePath);
    await resetWebGpuDevice();
    let firstError = "";
    try {
      await getWebGpuDevice();
    } catch (error) {
      firstError = error instanceof Error ? error.message : String(error);
    }
    const device = await getWebGpuDevice();
    const recovered = typeof device.createBuffer === "function";
    gpu.requestAdapter = requestAdapter;
    await resetWebGpuDevice();
    return { firstError, recovered };
  });
  expect(result.firstError).toContain("No compatible WebGPU adapter");
  expect(result.recovered).toBe(true);
});
