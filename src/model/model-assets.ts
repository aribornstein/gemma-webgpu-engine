const MODEL_ROOT = new URL("/models/", globalThis.location?.origin ?? "http://localhost").href;

export function modelAssetUrl(path: string): string {
  return new URL(path.replace(/^\/+/, ""), MODEL_ROOT).href;
}