# Existing weight cache reuse

## Verified facts

- The Buza bundle uses IndexedDB database `safetensors-cache-v1` with object stores `chunks` and `meta` and a 256 KiB chunk size.
- The bundle resolves `google/gemma-4-E2B-it-qat-mobile-transformers` from the floating `main` revision unless a different model root is supplied.
- Its default weight artifact is `model.safetensors`, matching the pinned upstream artifact in `public/models/gemma-4-e2b/manifest.json`.
- Buza's historical weight-cache origin is `http://localhost:8753`. Its current Playwright/dev origin is configured as `http://127.0.0.1:5173`; these are separate IndexedDB origins.
- IndexedDB is isolated by scheme, host, and port. The new engine cannot directly open Buza's existing database from a different port.
- No standalone safetensors copy exists in either workspace. The reusable 2 GB payload exists only in browser storage.

## Verified inventory

The cache survived and is artifact-compatible:

- Database version 2 with the expected `chunks` and `meta` stores.
- Source key: `https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers/resolve/main/model.safetensors`.
- Metadata reports 2,458,111,846 bytes, data start 375,400, range support, and 2,780 tensors.
- 1,371 cached tensor ranges contain 2,108,569,318 bytes.
- Tensor-level coverage is 1,371 of 2,780 tensors. The text backbone contributes 1,367 cached tensors and 230 missing tensors; those missing text tensors total only 9,486,936 bytes. All four `lm_head` tensors are cached. Audio, vision, and multimodal bridge tensors are intentionally absent.
- The first and last cached range hashes match immutable revision `9fcec64df66cb1e4d972fc5cdc142afb25b2362c`.
- All four layer-0 Q projection tensors are cached and their hashes match the immutable revision.
- The floating `main` URL resolved to the pinned revision and full-file SHA-256 during verification.

The four Q projection tensors were exported through readonly lookups into `operators/layer0-q-proj.safetensors`. The 1,581,728-byte fixture has SHA-256 `932bfa1d84087dba4ef2104a801431a9b8c9a0fd7a25f7ff65d83bea1d062be6`.

A second prefix-selected readonly export contains the 2-bit token embedding table, its scales, layer-0 input RMSNorm, and the four Q projection tensors. This seven-tensor, 103,297,096-byte local reference fixture has SHA-256 `e511c4e9e201266a9f32e06a9672a6377aacdd32d44ecb143e6e000a8cc3f03e`. It was sufficient to compute the exact first Q-projection activation without reconstructing or redownloading the full model.

## Decision

Reuse cached tensor records selectively; do not download the full weights again. A full-cache hash is not claimed because roughly 349 MB of the original file is not represented by cached tensor records.

The preferred path is to serve an inspection/export route from the Buza origin and perform a read-only inventory first: database version, meta keys, source URL/cache key, chunk count, total bytes, and a digest of selected chunks. If the source URL resolves to the same model and the cache metadata identifies the same file size/hash, either:

1. Run the owned engine from the Buza origin during migration and read the existing cache in place, or
2. Stream chunks through an explicit user-triggered export/import path into the owned cache schema.

Never delete, clear, upgrade, or overwrite `safetensors-cache-v1`. Opening the database with a higher version is prohibited because it can trigger a schema upgrade. Any inventory code must first use `indexedDB.databases()` and then open the reported existing version.

The standalone inspector and exporter live in the Buza project. They enumerate the existing database version, abort any upgrade event, use readonly transactions, and export only explicitly named tensors after a user action.