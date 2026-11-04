import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "../..");
const [stage, outputPath] = process.argv.slice(2);
const cases = JSON.parse(readFileSync(process.env.MECHANISTIC_CASES ?? resolve(import.meta.dirname, "failure-cases.json"), "utf8"));
const read = (file) => JSON.parse(readFileSync(file, "utf8"));
const write = (file, value) => writeFileSync(file, JSON.stringify(value, null, 2));
if (stage === "prepare") {
  const requests = [];
  for (const item of cases) {
    for (const variant of ["clean", "error"]) {
      const messages = [...item.history, { role: "user", content: item[`${variant}User`] }];
      requests.push({
        id: `${item.id}.${variant}.replay`, op: "replay", messages,
        assistantPrefix: item.assistantPrefix, maxNewTokens: 64, temperature: 0, seed: 23,
      });
    }
  }
  for (const item of cases) {
    for (const variant of ["clean", "error"]) {
      const messages = [...item.history, { role: "user", content: item[`${variant}User`] }];
      for (const choice of ["correct", "incorrect"]) requests.push({
        id: `${item.id}.${variant}.${choice}`, op: "score", messages,
        assistantPrefix: item.assistantPrefix, continuation: item[`${choice}Continuation`],
      });
    }
  }
  write(outputPath, requests);
  console.log(`${requests.length} requests (${cases.length * 2} replays + ${cases.length * 4} forced alternatives) -> ${outputPath}`);
} else if (stage === "summarize") {
  const artifacts = read(resolve(outputPath, "artifacts.json"));
  const byId = new Map(artifacts.map((a) => [a.request.id, a]));
  const summaries = [];
  for (const item of cases) {
    const variants = {};
    for (const variant of ["clean", "error"]) {
      const replay = byId.get(`${item.id}.${variant}.replay`);
      const correct = byId.get(`${item.id}.${variant}.correct`);
      const incorrect = byId.get(`${item.id}.${variant}.incorrect`);
      if (![replay, correct, incorrect].every((a) => a?.status === "completed")) {
        variants[variant] = { status: "incomplete" };
        continue;
      }
      const a = correct.result, b = incorrect.result;
      let firstDifference = 0;
      while (firstDifference < Math.min(a.tokenIds.length, b.tokenIds.length) &&
          a.tokenIds[firstDifference] === b.tokenIds[firstDifference]) firstDifference += 1;
      const firstLogP = (r) => r.tokenLogProbs[firstDifference] ?? null;
      variants[variant] = {
        learner: item[`${variant}User`],
        reply: replay.result.text, replyIds: replay.result.tokenIds,
        correctLogProbability: a.tokenLogProbs.reduce((x, y) => x + y, 0),
        incorrectLogProbability: b.tokenLogProbs.reduce((x, y) => x + y, 0),
        correctTokenCount: a.tokenIds.length, incorrectTokenCount: b.tokenIds.length,
        firstDifference, correctForkToken: a.continuationPieces[firstDifference],
        incorrectForkToken: b.continuationPieces[firstDifference],
        forkLogOdds: firstLogP(a) === null || firstLogP(b) === null ? null : firstLogP(a) - firstLogP(b),
        replayArtifact: replay.id, correctArtifact: correct.id, incorrectArtifact: incorrect.id,
      };
    }
    summaries.push({
      id: item.id, family: item.family, phenomenon: item.phenomenon, targetLevel: item.targetLevel,
      metadata: item.metadata, correctContinuation: item.correctContinuation,
      incorrectContinuation: item.incorrectContinuation, variants,
    });
  }
  const summary = {
    status: "exploratory-discovery",
    caveats: ["AI-authored contrasts", "Specified continuations need not match free output", "No claims about all criteria or latent competence"],
    cases: summaries,
    usage: artifacts.reduce((sum, a) => Object.fromEntries(Object.keys(sum).map((key) => [key, sum[key] + a.usage[key]])), {
      evaluatedPositions: 0, generatedTokens: 0, diagnostics: 0, episodes: 0,
    }),
    manifestHash: createHash("sha256").update(readFileSync(resolve(outputPath, "manifest.json"))).digest("hex"),
  };
  write(resolve(outputPath, "diagnosis.json"), summary);
  console.log(JSON.stringify({ cases: summaries.length, usage: summary.usage }));
} else if (stage === "diagnosis-packet") {
  const artifacts = read(resolve(outputPath, "artifacts.json"));
  const packet = cases.map((item) => {
    const variants = {};
    for (const variant of ["clean", "error"]) {
      const replay = artifacts.find((a) => a.request.id === `${item.id}.${variant}.replay`);
      if (replay?.status !== "completed") throw new Error("Wait for all paired replays to finish");
      variants[variant] = { learner: item[`${variant}User`], reply: replay.result.text, artifactId: replay.id };
    }
    return {
      id: item.id, family: item.family, targetLevel: item.targetLevel, metadata: item.metadata,
      history: item.history, correctContinuation: item.correctContinuation,
      incorrectContinuation: item.incorrectContinuation, variants,
    };
  });
  write(resolve(outputPath, "investigator-packet.json"), packet);
  console.log("Persisted 12 paired cases for investigator review");
} else if (stage === "quality-packets") {
  const artifacts = read(resolve(outputPath, "artifacts.json"));
  const secretKey = {};
  const packets = artifacts.filter((a) => a.request.op === "replay" && a.status === "completed").map((artifact) => {
    const item = cases.find((c) => artifact.request.id.startsWith(`${c.id}.`));
    if (!item) throw new Error("Unmapped evaluation item");
    const id = createHash("sha256").update(`blind-discovery-v1:${artifact.id}`).digest("hex").slice(0, 16);
    secretKey[id] = artifact.id;
    return {
      episodeId: id, scope: "Single newly generated NPC turn; not a full-episode promotion assessment",
      targetLevel: item.targetLevel, world: item.metadata,
      conversationPrefix: artifact.request.messages,
      npcReply: artifact.result.text,
    };
  }).sort((a, b) => a.episodeId.localeCompare(b.episodeId));
  write(resolve(outputPath, "quality-key.json"), secretKey);
  write(resolve(outputPath, "quality-packet.json"), packets);
  console.log(`${packets.length} blinded single-turn quality items`);
} else if (stage === "captures") {
  // Capture before the requested decision; donor and recipient retain their own prior K/V history.
  const requests = [];
  const chosen = process.argv.slice(4);
  for (const name of chosen) {
    const item = cases.find((c) => c.id === name);
    if (!item) throw new Error(`Unknown discovery case ${name}`);
    for (const variant of ["clean", "error"]) requests.push({
      id: `${item.id}.${variant}.capture`, op: "capture",
      messages: [...item.history, { role: "user", content: item[`${variant}User`] }],
      assistantPrefix: item.assistantPrefix, layers: [0, 4, 12, 20, 28, 34],
    });
  }
  write(outputPath, requests);
} else if (stage === "patches") {
  const artifacts = read(resolve(outputPath, "artifacts.json"));
  const requests = [];
  const selected = process.argv.slice(4);
  const randomUnit = (vector, seed) => {
    let state = seed;
    const random = vector.map(() => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return state / 4294967296 - 0.5;
    });
    const norm = Math.hypot(...random);
    const target = Math.hypot(...vector);
    return random.map((x) => x * target / norm);
  };
  for (const name of selected) {
    const item = cases.find((c) => c.id === name);
    if (!item) throw new Error(`Unknown case ${name}`);
    const get = (variant) => {
      const artifact = artifacts.find((a) => a.request.id === `${name}.${variant}.capture`);
      if (!artifact || artifact.status !== "completed") throw new Error("Missing capture");
      return artifact;
    };
    const clean = get("clean"), error = get("error");
    for (const source of clean.result.captures) {
      const layer = source.site.residual;
      const original = error.result.captures.find((c) => c.site.residual === layer);
      if (!original) throw new Error("Unmatched capture layer");
      const delta = source.values.map((v, i) => v - original.values[i]);
      const controls = [
        { name: "donor", recipient: "error", mode: "replace", vector: source.values, amount: 1 },
        { name: "sham", recipient: "error", mode: "replace", vector: original.values, amount: 1 },
        { name: "random-delta", recipient: "error", mode: "add", vector: randomUnit(delta, 73 + layer), amount: 1 },
        { name: "reverse", recipient: "clean", mode: "replace", vector: original.values, amount: 1 },
      ];
      for (const control of controls) {
        for (const choice of ["correct", "incorrect", "free"]) requests.push({
          id: `${name}.L${layer}.${control.name}.${choice}`, op: "patch",
          messages: [...item.history, { role: "user", content: item[`${control.recipient}User`] }],
          assistantPrefix: item.assistantPrefix,
          layer, mode: control.mode, amount: control.amount, vector: control.vector,
          sourceArtifactIds: [clean.id, error.id],
          ...(choice === "free" ? { maxNewTokens: 64, temperature: 0, seed: 23 } : { continuation: item[`${choice}Continuation`] }),
        });
      }
    }
  }
  write(resolve(outputPath, "patch-requests.json"), requests);
  console.log(`${requests.length} bounded patch/control scores prepared`);
} else if (stage === "patch-summary") {
  const artifacts = read(resolve(outputPath, "artifacts.json"));
  const byId = new Map(artifacts.map((a) => [a.request.id, a]));
  const summary = [];
  for (const correct of artifacts.filter((a) => a.request.op === "patch" && a.request.id.endsWith(".correct"))) {
    const stem = correct.request.id.slice(0, -".correct".length);
    const incorrect = byId.get(`${stem}.incorrect`);
    const free = byId.get(`${stem}.free`);
    if (![correct, incorrect, free].every((a) => a?.status === "completed")) continue;
    let fork = 0;
    while (fork < Math.min(correct.result.tokenIds.length, incorrect.result.tokenIds.length) &&
      correct.result.tokenIds[fork] === incorrect.result.tokenIds[fork]) fork += 1;
    const [caseId, site, control] = stem.split(".");
    const baseline = byId.get(`${caseId}.${control === "reverse" ? "clean" : "error"}.replay`);
    summary.push({
      caseId, site, control, fork,
      forkLogOdds: correct.result.tokenLogProbs[fork] - incorrect.result.tokenLogProbs[fork],
      text: (free.request.assistantPrefix ?? "") + free.result.text,
      forcedAssistantPrefix: free.request.assistantPrefix ?? "",
      tokenIdenticalToRecipient: JSON.stringify(free.result.tokenIds) === JSON.stringify(baseline?.result.tokenIds),
      correctArtifact: correct.id, incorrectArtifact: incorrect.id, freeArtifact: free.id,
    });
  }
  write(resolve(outputPath, "patch-effects.json"), summary);
  console.log(JSON.stringify(summary.map(({ caseId, site, control, forkLogOdds, tokenIdenticalToRecipient }) =>
    ({ caseId, site, control, forkLogOdds: Number(forkLogOdds.toFixed(3)), tokenIdenticalToRecipient })), null, 2));
} else if (stage === "behavioral") {
  const proposal = read(resolve(outputPath, "../discovery/post-patch-review.json"));
  const originals = read(resolve(outputPath, "../discovery/artifacts.json"));
  const requests = [];
  for (const item of proposal.recognitionRequests.requests) {
    for (const label of proposal.recognitionRequests.scoreContinuations) requests.push({
      id: `${item.id}.${label}`, op: "score",
      messages: [
        { role: "system", content: proposal.recognitionRequests.system },
        { role: "user", content: item.user },
      ], continuation: label,
    });
  }
  for (const item of proposal.repairRequests.requests) requests.push({
    id: item.id, op: "replay",
    messages: [
      { role: "system", content: proposal.repairRequests.system },
      { role: "user", content: item.user },
    ], maxNewTokens: 96, temperature: 0, seed: 23,
  });
  const dependency = proposal.bookDependencyRequests;
  if (!dependency) throw new Error("Missing bounded book dependency manifest");
  for (const cell of dependency.requests) {
    const base = originals.find((a) => a.id === cell.baseTranscriptRef);
    if (!base || base.request.messages.at(-1).content !== cell.finalLearnerCheck) throw new Error("Original history cannot be resolved exactly");
    for (const [index, noun] of [" ספרים", " גילידים", " גיליונות"].entries()) requests.push({
      id: `${cell.id}.noun${index}`, op: "score", messages: base.request.messages,
      assistantPrefix: cell.assistantPrefix, continuation: noun,
    });
  }
  write(resolve(outputPath, "behavioral-requests.json"), requests);
  console.log(`${requests.length} recognition, minimal-edit and noun-dependency requests`);
} else if (stage === "behavioral-summary") {
  const proposal = read(resolve(outputPath, "../discovery/post-patch-review.json"));
  const artifacts = read(resolve(outputPath, "artifacts.json"));
  const get = (id) => {
    const artifact = artifacts.find((a) => a.request.id === id);
    if (artifact?.status !== "completed") throw new Error(`Missing completed ${id}`);
    return artifact;
  };
  const logP = (id) => get(id).result.tokenLogProbs.reduce((x, y) => x + y, 0);
  const recognition = proposal.recognitionRequests.requests.map((item) => {
    const logA = logP(`${item.id}.A`), logB = logP(`${item.id}.B`);
    return {
      id: item.id, expected: item.expectedLabel,
      preferred: logA > logB ? "A" : logB > logA ? "B" : "tie",
      logA, logB, nominatedLabelMass: Math.exp(logA) + Math.exp(logB),
      correct: (logA > logB ? "A" : "B") === item.expectedLabel,
    };
  });
  const repairs = proposal.repairRequests.requests.map((item) => {
    const result = get(item.id).result;
    return {
      id: item.id, input: item.user, expected: item.expectedMinimalOutput,
      actual: result.text, stopReason: result.stopReason,
      exactMinimalEdit: result.text === item.expectedMinimalOutput,
      exactIgnoringOuterWhitespace: result.text.trim() === item.expectedMinimalOutput.trim(),
      changedInput: result.text !== item.user,
    };
  });
  const nounScores = proposal.bookDependencyRequests.requests.map((cell) => ({
    id: cell.id, variant: cell.variant, prefix: cell.assistantPrefix,
    booksLogP: logP(`${cell.id}.noun0`), corruptNounLogP: logP(`${cell.id}.noun1`),
    sheetsLogP: logP(`${cell.id}.noun2`),
    margin: logP(`${cell.id}.noun0`) - logP(`${cell.id}.noun1`),
  }));
  const byCell = new Map(nounScores.map((cell) => [cell.id, cell]));
  const effects = {
    numeralEffectClean: byCell.get("N_CLEAN_SHNEI").margin - byCell.get("N_CLEAN_SHTEI").margin,
    numeralEffectError: byCell.get("N_ERROR_SHNEI").margin - byCell.get("N_ERROR_SHTEI").margin,
    learnerEffectMasculine: byCell.get("N_CLEAN_SHNEI").margin - byCell.get("N_ERROR_SHNEI").margin,
    learnerEffectFeminine: byCell.get("N_CLEAN_SHTEI").margin - byCell.get("N_ERROR_SHTEI").margin,
  };
  const triggered = Object.values(effects).some((value) => Math.abs(value) >= 0.5) ||
    new Set(nounScores.map((cell) => Math.sign(cell.margin))).size > 1;
  const free = [];
  if (triggered) {
    for (const cell of proposal.bookDependencyRequests.requests) {
      const score = get(`${cell.id}.noun0`);
      free.push({
        id: `${cell.id}.free`, op: "replay", messages: score.request.messages,
        assistantPrefix: cell.assistantPrefix, maxNewTokens: 96, temperature: 0, seed: 23,
      });
    }
  }
  const summary = {
    recognition, repairs, nounScores, effects, freeStageTriggered: triggered,
    interpretation: "Recognition provides candidate answers; editing and forced numeral prefixes change the task/context. These are availability diagnostics, not a deployed NPC repair or evidence of an original-context native circuit.",
  };
  write(resolve(outputPath, "behavioral-summary.json"), summary);
  write(resolve(outputPath, "noun-free-requests.json"), free);
  console.log(JSON.stringify(summary, null, 2));
} else if (stage === "fork-cases") {
  const artifacts = read(resolve(outputPath, "artifacts.json"));
  const selected = process.argv.slice(4);
  const output = selected.map((id) => {
    const item = cases.find((c) => c.id === id);
    if (!item) throw new Error("Unknown case");
    const correct = artifacts.find((a) => a.request.id === `${id}.error.correct`);
    const incorrect = artifacts.find((a) => a.request.id === `${id}.error.incorrect`);
    if (!correct || !incorrect) throw new Error("Missing tokenization evidence");
    let fork = 0;
    while (fork < Math.min(correct.result.tokenIds.length, incorrect.result.tokenIds.length) &&
      correct.result.tokenIds[fork] === incorrect.result.tokenIds[fork]) fork += 1;
    const shared = correct.result.continuationPieces.slice(0, fork).join("");
    if (!item.correctContinuation.startsWith(shared) || !item.incorrectContinuation.startsWith(shared)) {
      throw new Error("Cannot reconstruct exact shared token prefix");
    }
    return {
      ...item, assistantPrefix: item.assistantPrefix + shared,
      correctContinuation: item.correctContinuation.slice(shared.length),
      incorrectContinuation: item.incorrectContinuation.slice(shared.length),
      prefixProvenance: {
        sharedTokenIds: correct.result.tokenIds.slice(0, fork),
        sourceArtifacts: [correct.id, incorrect.id],
        observation: "Scores isolate the first differing token; common prefix is held fixed, not itself a generated result of this request.",
      },
    };
  });
  write(resolve(outputPath, "fork-cases.json"), output);
  console.log(output.map((c) => ({ id: c.id, prefix: c.assistantPrefix })));
} else {
  mkdirSync(resolve(root, "research-results/mechanistic-repair"), { recursive: true });
  throw new Error("Usage: campaign.mjs prepare <requests.json> | summarize <run-dir> | captures <requests.json> <case ids...> | patches <run-dir> <case ids...>");
}
