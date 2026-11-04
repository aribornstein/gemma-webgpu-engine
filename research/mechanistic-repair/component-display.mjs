import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "../..");
const dir = resolve(root, "research-results/component-localization");
const read = (file) => JSON.parse(readFileSync(resolve(dir, file), "utf8"));
if (!existsSync(resolve(dir, "screen-effects.json"))) throw new Error("Complete component screen before building display data");
const screen = read("screen-effects.json");
const controls = existsSync(resolve(dir, "control-effects.json")) ? read("control-effects.json") : null;
const norms = read("component-norms.json");
const captures = read("components/artifacts.json").filter((a) => a.request.op === "capture");
const trajectories = captures.map((artifact) => ({
  id: artifact.id,
  points: artifact.result.captures.map((c) => ({
    layer: c.site.residual, stage: c.stage, tokenPosition: c.tokenPosition,
    norm: Math.hypot(...c.values), maxAbs: Math.max(...c.values.map(Math.abs)),
    nonFinite: c.values.some((v) => !Number.isFinite(v)),
  })),
}));
const data = {
  schemaVersion: 1,
  scope: "Selected-token discovery on two known failures; no native weight repair or final evaluation",
  screen: screen.map((r) => ({
    caseId: r.caseId, layer: r.layer, stage: r.stage, effect: r.effect,
    donorLogOdds: r.donor.logOdds, zeroLogOdds: r.zero.logOdds, signFlipLogOdds: r.flip.logOdds,
    text: r.donor.text, baselineText: r.zero.text, signFlipText: r.flip.text,
    targetPhrasePresent: r.donor.targetPhrasePresent,
    statedTotalPresent: r.donor.statedTotalPresent, complete: r.donor.stopReason === "end-token",
    shamExact: r.zero.identicalToBaseline, artifactId: r.donor.freeArtifact,
    ...norms.find((n) => n.caseId === r.caseId && n.layer === r.layer && n.stage === r.stage),
  })),
  controls: controls?.results.map((r) => ({
    caseId: r.caseId, layer: r.layer, stage: r.stage,
    observed: r.effect, null: r.randomControl,
    random: r.random.map((x) => ({
      id: x.id, effect: x.logOdds - screen.find((s) => s.caseId === r.caseId && s.layer === r.layer && s.stage === r.stage).zero.logOdds,
      target: x.targetPhrasePresent, text: x.text,
    })),
    transfer: r.transfer, targetOnlyOutcomeCounts: r.targetOnlyOutcomeCounts,
  })) ?? [],
  trajectories,
  warnings: [
    "Patch map color means change in nominated-token log odds, not quality or proof of a circuit.",
    "Attention/MLP differences are normalized residual writes; output-minus-MLP mixes PLE and layer scaling.",
    "Random draws are matched to each selected delta's L2 norm. Fifteen controls give a minimum empirical p of 0.0625.",
    "Selection used the same two discovery cases; tiny family count cannot establish generalization.",
    "Semantic preservation and 9/10 quality are not inferred from phrase or number presence.",
  ],
};
writeFileSync(resolve(root, "public/research/components.json"), JSON.stringify(data));
console.log(`Built ${data.screen.length} component cells and ${data.controls.length} random-control distributions for the UI.`);
