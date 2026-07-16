import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  fullyParallel: true,
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
