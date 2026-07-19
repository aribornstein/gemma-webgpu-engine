import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
  publicDir: false,
  build: {
    outDir: "build/library",
    emptyOutDir: false,
    minify: "oxc",
    target: "es2022",
    assetsInlineLimit: 0,
    rollupOptions: {
      input: resolve("src/index.ts"),
      preserveEntrySignatures: "strict",
      output: {
        format: "es",
        entryFileNames: "gemma-webgpu-engine.min.js",
        chunkFileNames: "chunks/[name]-[hash].min.js",
        assetFileNames: "assets/[name]-[hash][extname]",
      },
    },
  },
});