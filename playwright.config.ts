import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  fullyParallel: true,
  // Concurrent WebGPU Chrome workers intermittently fail to exit after their last test on
  // Apple/Metal, stalling the run until Playwright force-kills them. A single worker is reliable.
  workers: 1,
  webServer: {
    command: "npm run dev -- --host 127.0.0.1",
    port: 5174,
    reuseExistingServer: true,
  },
  use: {
    baseURL: "http://127.0.0.1:5174",
    browserName: "chromium",
    channel: "chrome",
  },
});
