const MODEL_ROOT = `${new URL(import.meta.url).origin}/models/`;

export function modelAssetUrl(path: string): string {
  return new URL(path.replace(/^\/+/, ""), MODEL_ROOT).href;
}