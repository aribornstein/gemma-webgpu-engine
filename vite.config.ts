import { cp, mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { defineConfig, type Plugin } from "vite";

const STATIC_DIRECTORIES = [
  ["public/examples", "examples"],
  ["public/models/gemma-4-e2b-tokenizer", "models/gemma-4-e2b-tokenizer"],
] as const;

const STATIC_FILES = [
  ["public/favicon.svg", "favicon.svg"],
  ["public/icons.svg", "icons.svg"],
  [
    "public/models/gemma-4-e2b/operators/decode-mlp-ple-layer0.json",
    "models/gemma-4-e2b/operators/decode-mlp-ple-layer0.json",
  ],
  [
    "public/models/gemma-4-e2b/operators/decode-mlp-ple-layer0.safetensors",
    "models/gemma-4-e2b/operators/decode-mlp-ple-layer0.safetensors",
  ],
] as const;

function staticDemoAssets(): Plugin {
  const outputRoot = resolve("build");
  return {
    name: "static-demo-assets",
    async closeBundle() {
      await writeFile(resolve(outputRoot, ".nojekyll"), "");
      for (const [source, destination] of STATIC_DIRECTORIES) {
        await cp(resolve(source), resolve(outputRoot, destination), { recursive: true });
      }
      for (const [source, destination] of STATIC_FILES) {
        const target = resolve(outputRoot, destination);
        await mkdir(resolve(target, ".."), { recursive: true });
        await cp(resolve(source), target);
      }
    },
  };
}

export default defineConfig({
  base: "./",
  publicDir: false,
  define: {
    "import.meta.env.VITE_PUBLIC_DEMO": JSON.stringify("true"),
  },
  plugins: [staticDemoAssets()],
  build: {
    outDir: "build",
    emptyOutDir: true,
    minify: "oxc",
    target: "es2022",
  },
});