export type GemmaPrefillMode = "fixed-32" | "chunked-32" | "sequential";
export type GemmaPrefillStrategy = "auto" | GemmaPrefillMode;

export interface GemmaPrefillSegment {
  mode: "fixed-32" | "sequential";
  start: number;
  rows: number;
}

export function planGemmaPrefillSegments(
  position: number,
  pendingRows: number,
  cacheCapacity: number,
  strategy: GemmaPrefillStrategy,
  hasFixedPrefill: boolean,
  blockRows = 32,
): readonly GemmaPrefillSegment[] {
  if (!Number.isInteger(position) || position < 0 ||
      !Number.isInteger(pendingRows) || pendingRows < 0 ||
      !Number.isInteger(cacheCapacity) || cacheCapacity < position + pendingRows ||
      !Number.isInteger(blockRows) || blockRows < 1) {
    throw new Error("Gemma prefill segment bounds are invalid");
  }
  if (pendingRows === 0) return [];
  const wantsFixed = hasFixedPrefill && strategy !== "sequential" &&
    (strategy !== "auto" || pendingRows > blockRows);
  if (!wantsFixed) return [{ mode: "sequential", start: 0, rows: pendingRows }];

  const segments: GemmaPrefillSegment[] = [];
  let start = 0;
  let nextPosition = position;
  const alignmentRows = nextPosition % blockRows === 0
    ? 0
    : Math.min(pendingRows, blockRows - nextPosition % blockRows);
  if (alignmentRows > 0) {
    segments.push({ mode: "sequential", start, rows: alignmentRows });
    start += alignmentRows;
    nextPosition += alignmentRows;
  }
  while (start < pendingRows && nextPosition + blockRows <= cacheCapacity) {
    const rows = Math.min(blockRows, pendingRows - start);
    segments.push({ mode: "fixed-32", start, rows });
    start += rows;
    nextPosition += rows;
  }
  if (start < pendingRows) {
    segments.push({ mode: "sequential", start, rows: pendingRows - start });
  }
  return segments;
}