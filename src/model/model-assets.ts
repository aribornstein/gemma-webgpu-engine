const MODEL_ROOT = new URL(
  "models/",
  globalThis.document?.baseURI ?? globalThis.location?.href ?? "http://localhost/",
).href;

export function modelAssetUrl(path: string): string {
  return new URL(path.replace(/^\/+/, ""), MODEL_ROOT).href;
}