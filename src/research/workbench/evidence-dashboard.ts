import type { counterbalancedPreference, familyPairedEffect, judgeAgreement, randomControlTest, summarizeOutcomes } from "./statistics";

interface Historical {
  schemaVersion: number; dataset: string; outcomeDefinition: string; limitations: string[];
  outcomes: ReturnType<typeof summarizeOutcomes>;
  effects: (ReturnType<typeof familyPairedEffect> & { criterion: string; holmAdjustedP: number })[];
  recognitionOrderControl: ReturnType<typeof counterbalancedPreference> & { interpretation: string };
  reservedJudgeAgreement: ReturnType<typeof judgeAgreement> & { interpretation: string };
}
interface JudgeAudit {
  scope: string; warning: string; itemCount: number;
  preference: ReturnType<typeof counterbalancedPreference>;
  hiddenDuplicateRatings: ReturnType<typeof judgeAgreement>;
}
interface Cell {
  caseId: string; layer: number; stage: string; effect: number;
  donorLogOdds: number; zeroLogOdds: number; signFlipLogOdds: number;
  text: string; baselineText: string; signFlipText: string;
  deltaNorm: number; recipientNorm: number; ratio: number; shamExact: boolean;
  targetPhrasePresent: boolean; statedTotalPresent: boolean; complete: boolean;
}
interface Components {
  schemaVersion: number; scope: string; warnings: string[]; screen: Cell[];
  controls: {
    caseId: string; layer: number; stage: string; observed: number;
    null: ReturnType<typeof randomControlTest>;
    random: { id: string; effect: number; target: boolean; text: string }[];
    transfer: { recipient: string; baseline: string; text: string; identicalToBaseline: boolean; targetPhrasePresent: boolean }[];
  }[];
  trajectories: { id: string; points: { layer: number; stage: string; tokenPosition: number; norm: number; maxAbs: number; nonFinite: boolean }[] }[];
}

function node<K extends keyof HTMLElementTagNameMap>(tag: K, text = ""): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag); el.textContent = text; return el;
}
function details(title: string, lines: string[]): HTMLDetailsElement {
  const el = node("details"); el.append(node("summary", title));
  const list = node("ul");
  for (const line of lines) list.append(node("li", line));
  el.append(list); return el;
}
function number(value: number): string {
  if (!Number.isFinite(value)) throw new Error("Nonfinite evidence value");
  return value.toFixed(3);
}
function percent(value: number | null): string { return value === null ? "unknown" : `${number(100 * value)}%`; }
function tabulate(host: HTMLElement, headers: string[], rows: string[][]): void {
  const wrap = node("div"); wrap.className = "evidence-table";
  const table = node("table"), head = node("thead"), row = node("tr");
  for (const title of headers) row.append(node("th", title));
  head.append(row); table.append(head);
  const body = node("tbody");
  for (const values of rows) { const tr = node("tr"); for (const value of values) tr.append(node("td", value)); body.append(tr); }
  table.append(body); wrap.append(table); host.append(wrap);
}
function axis(host: HTMLElement, values: { value: number; label: string; donor?: boolean }[], min: number, max: number): void {
  const plot = node("div"); plot.className = "evidence-axis";
  const width = Math.max(max - min, 1e-6);
  for (const [i, point] of values.entries()) {
    const dot = node("span"); dot.className = point.donor ? "evidence-dot donor" : "evidence-dot";
    dot.style.left = `${5 + 90 * (point.value - min) / width}%`; dot.style.top = `${15 + (i % 3) * 12}px`;
    dot.title = `${point.label}: ${number(point.value)}`; dot.setAttribute("aria-label", dot.title); plot.append(dot);
  }
  host.append(plot, node("small", `Axis ${number(min)} to ${number(max)}. Each dot is one recorded control; amber = donor.`));
}
function renderHistorical(host: HTMLElement, data: Historical): void {
  if (data.schemaVersion !== 1 || !Array.isArray(data.effects)) throw new Error("Unsupported paired statistics");
  host.append(node("p", data.dataset), node("p", data.outcomeDefinition));
  const counts = node("div"); counts.className = "outcome-counts";
  const labels = { fixed: "Fixed", broken: "Broken", unchangedCorrect: "Still correct", unchangedWrong: "Still wrong", unknown: "Unknown" } as const;
  for (const key of Object.keys(labels) as (keyof typeof labels)[]) {
    const item = node("div"); item.append(node("strong", String(data.outcomes[key])), node("span", labels[key])); counts.append(item);
  }
  host.append(counts, node("p", `${data.outcomes.caseCount} matched episodes; ${data.outcomes.familyCount} family clusters, not independent seeds or tokens.`));
  const forest = node("div"); forest.className = "effect-forest"; forest.setAttribute("aria-label", "Family-level effect intervals");
  for (const effect of data.effects) {
    const row = node("div"); row.className = "effect-row";
    row.append(node("span", effect.criterion.replaceAll("_", " ")));
    const plot = node("div"); plot.className = "effect-range";
    // Historical scores are bounded 0..4, so paired differences lie in -4..4.
    const position = (v: number) => `${(v + 4) * 12.5}%`;
    const ci = effect.confidenceInterval95;
    if (ci) {
      const bar = node("span"); bar.className = "effect-interval";
      bar.style.left = position(ci[0]); bar.style.width = `${(ci[1] - ci[0]) * 12.5}%`; plot.append(bar);
    }
    const dot = node("span"); dot.className = "effect-point"; dot.style.left = position(effect.meanDelta); plot.append(dot);
    row.append(plot, node("small", `${number(effect.meanDelta)} ${ci ? `[${ci.map(number).join(", ")}]` : "(CI unavailable)"}`)); forest.append(row);
  }
  host.append(forest, node("p", "Effect axis -4 to +4 (center = no change). Negative = worse; bars = family-bootstrap 95% intervals. Original ordinal 0-4 scores, not 0-10 certification."));
  tabulate(host, ["Criterion", "Family delta", "Sign-flip p", "Holm p"], data.effects.map((e) =>
    [e.criterion, number(e.meanDelta), number(e.pValue), number(e.holmAdjustedP)]));
  host.append(details("Assumptions and limitations", data.limitations));
  host.append(details("Earlier recognition and judge diagnostics", [
    `${data.recognitionOrderControl.interpretation} Consistent identities: ${data.recognitionOrderControl.consistentCount}/${data.recognitionOrderControl.completeCount}.`,
    `${data.reservedJudgeAgreement.interpretation} Exact agreement: ${percent(data.reservedJudgeAgreement.exactFraction)}.`,
  ]));
}
function renderJudge(host: HTMLElement, data: JudgeAudit): void {
  if (!data.preference || !data.hiddenDuplicateRatings) throw new Error("Unsupported judge audit");
  host.append(node("p", data.scope));
  tabulate(host, ["Check", "Observed"], [
    ["Candidate identity after reversing order", `${data.preference.consistentCount}/${data.preference.completeCount} consistent`],
    ["Hidden duplicate criterion ratings", `${percent(data.hiddenDuplicateRatings.exactFraction)} exact; ${data.hiddenDuplicateRatings.completeCount} paired ratings`],
    ["Packets / abstentions", `${data.itemCount} / ${data.preference.abstentionCount + data.hiddenDuplicateRatings.abstentionCount}`],
  ]);
  host.append(node("p", `${data.warning} Consistency does not prove linguistic correctness, independence, or a 9/10 quality pass.`));
}
function renderComponents(host: HTMLElement, data: Components): void {
  if (data.schemaVersion !== 1 || !Array.isArray(data.screen) || !Array.isArray(data.controls)) throw new Error("Unsupported component evidence");
  host.append(node("p", data.scope), node("p", "Token-local intervention map: change in correct-minus-incorrect next-token log probability. Teal positive, red negative; numbers retain the exact scale. Click a site to inspect its actual text."));
  const map = node("div"); map.className = "component-map";
  const preview = node("div"); preview.id = "component-preview";
  function show(cell: Cell): void {
    preview.replaceChildren(node("h4", `${cell.caseId} / L${cell.layer} / ${cell.stage}`));
    tabulate(preview, ["Readout", "Value"], [
      ["Fork log odds: zero / donor / reverse-sign", [cell.zeroLogOdds, cell.donorLogOdds, cell.signFlipLogOdds].map(number).join(" / ")],
      ["Delta norm / receiver norm / ratio", `${number(cell.deltaNorm)} / ${number(cell.recipientNorm)} / ${percent(cell.ratio)}`],
      ["Zero token parity", String(cell.shamExact)],
      ["Target phrase / numeric total / end token", `${cell.targetPhrasePresent} / ${cell.statedTotalPresent} / ${cell.complete}`],
    ]);
    const outputs = node("div"); outputs.className = "comparison-columns";
    for (const [label, text] of [["Baseline (zero control)", cell.baselineText], ["Donor contribution delta", cell.text], ["Reverse sign", cell.signFlipText]]) {
      const card = node("article"); card.className = "comparison-card"; card.append(node("h4", label));
      const p = node("p", text); p.dir = "auto"; card.append(p); outputs.append(card);
    }
    preview.append(outputs, node("p", "These phrase/number checks are narrow observations, not semantic or full-panel passes."));
  }
  const largest = Math.max(1, ...data.screen.map((c) => Math.abs(c.effect)));
  for (const cell of data.screen) {
    const button = node("button", `${cell.caseId}\nL${cell.layer} ${cell.stage}\n${cell.effect >= 0 ? "+" : ""}${number(cell.effect)}`);
    button.className = "component-cell"; button.dataset.case = cell.caseId; button.dataset.stage = cell.stage; button.dataset.layer = String(cell.layer);
    button.style.borderColor = cell.effect > 0 ? "#77d7c6" : "#ffb6a4";
    button.style.backgroundColor = `rgba(${cell.effect > 0 ? "40,130,116" : "174,66,50"},${0.15 + 0.65 * Math.abs(cell.effect) / largest})`;
    button.addEventListener("click", () => show(cell)); map.append(button);
  }
  host.append(map, preview); if (data.screen.length) show(data.screen[0]);
  const controls = node("div"); controls.id = "random-controls";
  for (const control of data.controls) {
    controls.append(node("h4", `${control.caseId} / L${control.layer} ${control.stage}: random-direction comparison`));
    const values = [...control.random.map((r) => ({ value: r.effect, label: r.id })), { value: control.observed, label: "donor", donor: true }];
    axis(controls, values, Math.min(0, ...values.map((v) => v.value)), Math.max(0, ...values.map((v) => v.value)));
    controls.append(node("p", `Donor ${number(control.observed)}; random mean ${number(control.null.mean)}; rank ${control.null.rank}/${control.null.n + 1}; empirical p = ${control.null.pValue} (${control.null.n} controls). Same-case site selection: exploratory, not a significance or promotion claim.`));
    const all = node("details"); all.append(node("summary", "All random draws and outputs"));
    tabulate(all, ["Control", "Effect", "Target phrase", "Actual output"], control.random.map((r) => [r.id, number(r.effect), String(r.target), r.text])); controls.append(all);
    controls.append(node("h4", "Transfer without a new donor"));
    for (const transfer of control.transfer) {
      const item = node("details"); item.open = true; item.append(node("summary", transfer.recipient));
      tabulate(item, ["Unmodified receiver", "Same source delta applied"], [[transfer.baseline, transfer.text]]);
      item.append(node("p", `Target phrase: ${transfer.targetPhrasePresent}; token-identical to receiver baseline: ${transfer.identicalToBaseline}.`)); controls.append(item);
    }
  }
  if (!data.controls.length) controls.append(node("p", "No qualifying site; no follow-up control distribution."));
  host.append(controls);
  const norms = node("details"); norms.append(node("summary", "Captured-token geometry and finiteness"));
  tabulate(norms, ["Artifact", "Layer / stage", "Token", "L2 norm", "Max |value|", "Nonfinite"], data.trajectories.flatMap((t) => t.points.map((p) =>
    [t.id, `${p.layer} / ${p.stage}`, String(p.tokenPosition), number(p.norm), number(p.maxAbs), String(p.nonFinite)])));
  host.append(norms, details("What this map cannot establish", data.warnings));
}

export async function mountEvidenceDashboard(host: HTMLElement): Promise<void> {
  host.className = "evidence-dashboard";
  async function section<T>(id: string, title: string, file: string, render: (el: HTMLElement, data: T) => void): Promise<void> {
    const area = node("section"); area.id = id; area.append(node("h3", title)); host.append(area);
    const status = node("p", "Loading recorded evidence..."); status.setAttribute("role", "status"); area.append(status);
    try {
      const response = await fetch(new URL(`research/${file}`, document.baseURI));
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const body = node("div"); render(body, await response.json() as T);
      area.append(body); status.remove();
    } catch (error) {
      status.dataset.error = "true"; status.textContent = `Evidence unavailable (${file}): ${error instanceof Error ? error.message : String(error)}`;
    }
  }
  await Promise.all([
    section("paired-statistics", "Historical steering: repair versus damage", "paired-statistics.json", renderHistorical),
    section("judge-audit", "Counterbalanced judge stability", "judge-audit.json", renderJudge),
    section("component-evidence", "Within-layer component diagnostics", "components.json", renderComponents),
  ]);
}
