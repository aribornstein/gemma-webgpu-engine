export interface ScoredBestOfAudit {
  candidateIndex: number;
  score: number | null;
}

export interface ScheduledBestOfJudge {
  judgeIndex: number;
  candidateOrder: number[];
}

export function scoreHebrewAudit(auditText: string): number | null {
  const parsed: unknown = JSON.parse(auditText);
  if (!parsed || typeof parsed !== "object") return null;
  const audit = parsed as Record<string, unknown>;
  const fidelity = { contradicted: -8, incomplete: 0, fulfilled: 4 }[String(audit.task_fidelity)];
  const actionability = { unusable: -5, limited: 1, actionable: 4 }[String(audit.actionability)];
  const grammar = { errors: -5, questionable: 1, clean: 3 }[String(audit.grammar_integrity)];
  const idiomaticity = { unnatural: -3, acceptable: 1, natural: 3 }[String(audit.idiomaticity)];
  const level = { below: 0, appropriate: 2, above: 0 }[String(audit.level_fit)];
  const scores = [fidelity, actionability, grammar, idiomaticity, level];
  if (!scores.every((score): score is number =>
    typeof score === "number" && Number.isFinite(score))) return null;
  return scores.reduce((sum, score) => sum + score, 0);
}

export function scoreComparativeHebrewJudge(
  judgmentText: string,
  candidateOrder: readonly number[],
): ScoredBestOfAudit[] {
  const parsed: unknown = JSON.parse(judgmentText);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    return candidateOrder.map((candidateIndex) => ({ candidateIndex, score: null }));
  }
  const judgment = parsed as Record<string, unknown>;
  return candidateOrder.map((candidateIndex, position) => {
    const audit = judgment[`candidate_${position + 1}`];
    return {
      candidateIndex,
      score: audit && typeof audit === "object" && !Array.isArray(audit)
        ? scoreHebrewAudit(JSON.stringify(audit))
        : null,
    };
  });
}

export function selectHighestScoredCandidate(audits: readonly ScoredBestOfAudit[]): number | null {
  if (audits.length === 0 || audits.some(({ score }) => score === null)) return null;
  const ranked = [...audits].sort((left, right) => (right.score ?? 0) - (left.score ?? 0));
  return ranked.length > 1 && ranked[0].score !== ranked[1].score
    ? ranked[0].candidateIndex
    : null;
}

export function shuffledCandidateIndices(candidateCount: number, random: () => number): number[] {
  if (!Number.isInteger(candidateCount) || candidateCount < 2) {
    throw new Error("Candidate count must be an integer of at least 2");
  }
  const indices = Array.from({ length: candidateCount }, (_, index) => index);
  for (let index = indices.length - 1; index > 0; index -= 1) {
    const selected = Math.floor(random() * (index + 1));
    [indices[index], indices[selected]] = [indices[selected], indices[index]];
  }
  if (indices.every((candidateIndex, index) => candidateIndex === index)) {
    indices.push(indices.shift() as number);
  }
  return indices;
}

export function createUniqueJudgeSchedule(
  candidateCount: number,
  judgeCount: number,
  random: () => number,
): ScheduledBestOfJudge[] {
  if (!Number.isInteger(judgeCount) || judgeCount < 1) {
    throw new Error("Judge count must be an integer of at least 1");
  }
  if (!Number.isSafeInteger(candidateCount) || candidateCount < 2) {
    throw new Error("Candidate count must be a safe integer of at least 2");
  }
  if (candidateCount === 2 && judgeCount > 2) {
    throw new Error("At most 2 unique orders exist for 2 candidates");
  }
  const candidates = Array.from({ length: candidateCount }, (_, index) => index);
  const orders: number[][] = [];
  const usedOrders = new Set<string>();
  while (orders.length < judgeCount) {
    let order = randomPermutation(candidates, random);
    let key = order.join(",");
    for (let attempt = 0; usedOrders.has(key) && attempt < 8; attempt += 1) {
      order = randomPermutation(candidates, random);
      key = order.join(",");
    }
    while (usedOrders.has(key)) {
      order = nextPermutation(order);
      key = order.join(",");
    }
    usedOrders.add(key);
    orders.push(order);
  }

  return orders.map((candidateOrder, judgeIndex) => ({ judgeIndex, candidateOrder }));
}

function randomPermutation(values: readonly number[], random: () => number): number[] {
  const order = [...values];
  for (let index = order.length - 1; index > 0; index -= 1) {
    const selected = Math.floor(random() * (index + 1));
    [order[index], order[selected]] = [order[selected], order[index]];
  }
  return order;
}

function nextPermutation(values: readonly number[]): number[] {
  const order = [...values];
  let pivot = order.length - 2;
  while (pivot >= 0 && order[pivot] >= order[pivot + 1]) pivot -= 1;
  if (pivot < 0) return order.reverse();
  let successor = order.length - 1;
  while (order[successor] <= order[pivot]) successor -= 1;
  [order[pivot], order[successor]] = [order[successor], order[pivot]];
  for (let left = pivot + 1, right = order.length - 1; left < right; left += 1, right -= 1) {
    [order[left], order[right]] = [order[right], order[left]];
  }
  return order;
}

export function aggregateCandidateAuditScores(
  candidateCount: number,
  judgeCount: number,
  audits: readonly ScoredBestOfAudit[],
): ScoredBestOfAudit[] {
  return Array.from({ length: candidateCount }, (_, candidateIndex) => {
    const scores = audits
      .filter((audit) => audit.candidateIndex === candidateIndex)
      .map((audit) => audit.score);
    const complete = scores.length === judgeCount && scores.every(
      (score): score is number => typeof score === "number" && Number.isFinite(score),
    );
    return {
      candidateIndex,
      score: complete
        ? scores.reduce((sum, score) => sum + score, 0) / judgeCount
        : null,
    };
  });
}
