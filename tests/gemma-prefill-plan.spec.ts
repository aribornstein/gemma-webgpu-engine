import { expect, test } from "@playwright/test";
import {
  isFinalGemmaPrefillSegment,
  planGemmaPrefillSegments,
} from "../src/runtime/gemma-prefill-plan";

test("uses sequential prefill for short automatic prompts", () => {
  expect(planGemmaPrefillSegments(0, 32, 8_192, "auto", true)).toEqual([
    { mode: "sequential", start: 0, rows: 32 },
  ]);
});

test("aligns a reused prefix before chunked prefill", () => {
  const segments = planGemmaPrefillSegments(5, 636, 8_192, "auto", true);
  expect(segments[0]).toEqual({ mode: "sequential", start: 0, rows: 27 });
  expect(segments.slice(1).every(({ mode }) => mode === "fixed-32")).toBe(true);
  expect(segments.reduce((rows, segment) => rows + segment.rows, 0)).toBe(636);
  expect(segments.at(-1)).toEqual({ mode: "fixed-32", start: 635, rows: 1 });
});

test("uses a sequential tail when a padded block would exceed capacity", () => {
  expect(planGemmaPrefillSegments(8_160, 32, 8_192, "chunked-32", true)).toEqual([
    { mode: "fixed-32", start: 0, rows: 32 },
  ]);
  expect(planGemmaPrefillSegments(8_170, 22, 8_192, "chunked-32", true)).toEqual([
    { mode: "sequential", start: 0, rows: 22 },
  ]);
});

test("predicts only from the final segment", () => {
  const segments = planGemmaPrefillSegments(0, 9, 10, "chunked-32", true, 4);
  expect(segments).toEqual([
    { mode: "fixed-32", start: 0, rows: 4 },
    { mode: "fixed-32", start: 4, rows: 4 },
    { mode: "sequential", start: 8, rows: 1 },
  ]);
  expect(segments.map((segment) => isFinalGemmaPrefillSegment(segment, 9))).toEqual([
    false,
    false,
    true,
  ]);
});

test("falls back when fixed prefill resources are unavailable", () => {
  expect(planGemmaPrefillSegments(0, 641, 8_192, "auto", false)).toEqual([
    { mode: "sequential", start: 0, rows: 641 },
  ]);
});