import type { BenchmarkEnvironment, BrowserMode } from "./types";

export interface BrowserEnvironmentInput {
  browserMode: BrowserMode;
  gitCommit: string;
  benchmarkSeed: number;
  operatingSystem: string;
  physicalDevice: string;
  cpu: string;
  totalRamBytes: number;
  browserFlags?: readonly string[];
}

export async function captureBrowserEnvironment(
  input: BrowserEnvironmentInput,
): Promise<BenchmarkEnvironment> {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
  const adapterInfo = adapter?.info;
  const battery = await readBatteryState();
  const userAgentData = (navigator as Navigator & {
    userAgentData?: { brands?: readonly { brand: string; version: string }[] };
  }).userAgentData;
  const chromeBrand = userAgentData?.brands?.find((brand) => /Chrome|Chromium/.test(brand.brand));
  return {
    capturedAt: new Date().toISOString(),
    operatingSystem: input.operatingSystem,
    physicalDevice: input.physicalDevice,
    cpu: input.cpu,
    totalRamBytes: input.totalRamBytes,
    gpuAdapter: adapterInfo
      ? [adapterInfo.vendor, adapterInfo.architecture, adapterInfo.device].filter(Boolean).join(" / ")
      : "unavailable",
    webGpuAdapterInfo: adapterInfo ? serializableAdapterInfo(adapterInfo) : {},
    browserName: chromeBrand?.brand ?? "Chrome",
    browserVersion: chromeBrand?.version ?? parseChromeVersion(navigator.userAgent),
    browserMode: input.browserMode,
    browserFlags: Object.freeze([...(input.browserFlags ?? [])]),
    visibilityState: document.visibilityState,
    powerSource: battery,
    gitCommit: input.gitCommit,
    benchmarkSeed: input.benchmarkSeed,
  };
}

async function readBatteryState(): Promise<BenchmarkEnvironment["powerSource"]> {
  const batteryNavigator = navigator as Navigator & {
    getBattery?: () => Promise<{ charging: boolean }>;
  };
  if (!batteryNavigator.getBattery) return "unknown";
  try {
    return (await batteryNavigator.getBattery()).charging ? "external-power" : "battery";
  } catch {
    return "unknown";
  }
}

function serializableAdapterInfo(info: GPUAdapterInfo): Record<string, string> {
  return {
    vendor: info.vendor,
    architecture: info.architecture,
    device: info.device,
    description: info.description,
  };
}

function parseChromeVersion(userAgent: string): string {
  return /(?:Chrome|Chromium)\/([^ ]+)/.exec(userAgent)?.[1] ?? "unknown";
}