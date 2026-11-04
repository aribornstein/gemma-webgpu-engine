import { atLevel, systemMessage, type CefrLevel, type CefrScenario } from "../cefr/prompts";
import type { Artifact, ExperimentRequest } from "./runner";

interface Rating {
  criterion: string; score: number | null; scale: number;
  evidence: { span: string; finding: string }[]; judge: string;
}
interface Turn {
  player: string; text: string; tokenCount: number; stopReason: string;
  forcedPrefix?: string; tokenIds?: number[]; tokenLogProbs?: number[];
  firstDecision?: { text: string; logProbability: number; tokenId: number }[];
}
interface Arm {
  id: string; label: string; kind: string; sourceId: string; sourceFile: string;
  policy: string; seed: number; turns: Turn[]; ratings: Rating[]; warnings: string[];
}
interface Example {
  id: string; experiment: string; label: string; family: string; level: string; seed: number;
  metadata: unknown; arms: Arm[]; caveat: string; live?: { scenario: CefrScenario; patch: string };
}
interface Catalogue { schemaVersion: number; examples: Example[]; qualityClaim: string; sources: { path: string; sha256: string }[] }

function node<K extends keyof HTMLElementTagNameMap>(tag: K, text = ""): HTMLElementTagNameMap[K] {
  const element = document.createElement(tag); element.textContent = text; return element;
}
function select(label: string, id: string) {
  const wrapper = node("label", label);
  const input = node("select"); input.id = id; wrapper.append(input);
  return { wrapper, input };
}
function options(input: HTMLSelectElement, values: { value: string; label: string }[]) {
  input.replaceChildren(...values.map(({ value, label }) => {
    const option = node("option", label); option.value = value; return option;
  }));
}
function record(value: unknown): value is Record<string, unknown> { return typeof value === "object" && value !== null; }
function parseCatalogue(value: unknown): Catalogue {
  if (!record(value) || value.schemaVersion !== 1 || !Array.isArray(value.examples) || !Array.isArray(value.sources)) {
    throw new Error("Unsupported recorded comparison catalogue");
  }
  for (const example of value.examples) {
    if (!record(example) || typeof example.id !== "string" || !Array.isArray(example.arms) || example.arms.length < 2) throw new Error("Invalid recorded case");
    for (const arm of example.arms) {
      if (!record(arm) || typeof arm.id !== "string" || !Array.isArray(arm.turns) || !Array.isArray(arm.ratings) || !Array.isArray(arm.warnings)) throw new Error("Invalid comparison arm");
      for (const turn of arm.turns) {
        if (!record(turn) || typeof turn.text !== "string" || typeof turn.player !== "string") throw new Error("Invalid recorded turn");
      }
    }
  }
  if (typeof value.qualityClaim !== "string" || value.sources.some((source) =>
    !record(source) || typeof source.path !== "string" || typeof source.sha256 !== "string")) {
    throw new Error("Missing comparison provenance");
  }
  return { schemaVersion: 1, examples: value.examples as Example[], qualityClaim: value.qualityClaim, sources: value.sources as Catalogue["sources"] };
}

export async function mountComparisonDemo(
  host: HTMLElement,
  run: (request: ExperimentRequest) => Promise<Artifact>,
): Promise<void> {
  host.className = "comparison-demo";
  host.append(node("h2", "Before / after: recorded evidence"));
  const status = node("p", "Loading recorded comparisons..."); status.id = "comparison-status"; status.setAttribute("aria-live", "polite"); host.append(status);
  try {
    const response = await fetch(new URL("research/comparisons.json", document.baseURI));
    if (!response.ok) throw new Error(`Recorded comparisons unavailable (${response.status})`);
    const catalogue = parseCatalogue(await response.json());
    const toolbar = node("div"); toolbar.className = "comparison-toolbar";
    const experiment = select("Study", "comparison-study");
    const scenario = select("Scenario / level / seed", "comparison-case");
    const left = select("Before", "comparison-before");
    const right = select("After", "comparison-after");
    const protocol = select("Live protocol", "comparison-protocol");
    options(experiment.input, [
      { value: "cefr", label: "CEFR steering (quality regressions included)" },
      { value: "mechanistic", label: "Learner-error diagnostic (donor supplied)" },
    ]);
    options(protocol.input, [
      { value: "fixed-prefix", label: "Same-prefix first turn" },
      { value: "conversation", label: "Whole scripted conversation (own histories)" },
    ]);
    const blindLabel = node("label", "Hide arm identities");
    const blind = node("input"); blind.type = "checkbox"; blindLabel.append(blind);
    toolbar.append(experiment.wrapper, scenario.wrapper, left.wrapper, right.wrapper, protocol.wrapper, blindLabel);
    host.append(toolbar);
    const nav = node("div"); nav.className = "comparison-toolbar";
    const previous = node("button", "Previous turn"), next = node("button", "Next turn"), turnLabel = node("span");
    const live = node("button", "Run live baseline vs historical overlay");
    const exportButton = node("button", "Export displayed comparison");
    nav.append(previous, turnLabel, next, live, exportButton);
    host.append(nav);
    const caveat = node("p"); caveat.className = "notice"; host.append(caveat);
    const player = node("p"); player.dir = "auto"; player.className = "comparison-player"; host.append(player);
    const columns = node("div"); columns.className = "comparison-columns"; host.append(columns);
    const evidence = node("div"); evidence.className = "comparison-columns"; host.append(evidence);
    const details = node("details"), summary = node("summary", "Scenario, protocol and source hashes");
    const metadata = node("pre"); details.append(summary, metadata); host.append(details);
    let active: Example;
    let turn = 0;
    let liveExample: Example | null = null;
    let running = false;
    const selected = () => liveExample ?? catalogue.examples.find((e) => e.id === scenario.input.value)!;
    function render() {
      active = selected();
      const before = active.arms.find((a) => a.id === left.input.value) ?? active.arms[0];
      const after = active.arms.find((a) => a.id === right.input.value) ?? active.arms[1];
      const length = Math.max(before.turns.length, after.turns.length);
      turn = Math.min(turn, length - 1);
      const a = before.turns[turn], b = after.turns[turn];
      previous.disabled = turn === 0;
      next.disabled = turn + 1 >= length;
      live.disabled = !active.live || running;
      turnLabel.textContent = `Turn ${turn + 1} / ${length}`;
      const warn = [...new Set([active.caveat, ...before.warnings, ...after.warnings])];
      if (before.id === after.id) warn.push("The same arm is selected twice: this is not an intervention comparison.");
      caveat.textContent = `${catalogue.qualityClaim}. ${warn.join(" ")}`;
      player.textContent = a?.player === b?.player ? `Learner: ${a?.player ?? ""}` : "Different learner contexts: these arms are not a same-input intervention pair.";
      let shared = 0;
      const ta = a?.text ?? "", tb = b?.text ?? "";
      const wordsA = ta.split(/(\s+)/u), wordsB = tb.split(/(\s+)/u);
      while (shared < Math.min(wordsA.length, wordsB.length) && wordsA[shared] === wordsB[shared]) shared++;
      columns.replaceChildren();
      evidence.replaceChildren();
      for (const [index, arm] of [before, after].entries()) {
        const current = arm.turns[turn];
        const card = node("article"); card.className = "comparison-card";
        card.append(node("h3", blind.checked ? `Output ${index === 0 ? "A" : "B"}` : arm.label));
        if (a?.player !== b?.player) { const user = node("p", `Learner: ${current?.player ?? ""}`); user.dir = "auto"; card.append(user); }
        const output = node("p"); output.dir = "auto"; output.className = "comparison-output";
        const parts = (current?.text ?? "").split(/(\s+)/u);
        output.append(document.createTextNode(parts.slice(0, shared).join("")));
        const changed = node("mark", parts.slice(shared).join("")); output.append(changed); card.append(output);
        card.append(node("p", `Tokens: ${current?.tokenCount ?? "not recorded"} · Stop: ${current?.stopReason ?? "missing"} · Word-level first divergence: ${shared === Math.max(wordsA.length, wordsB.length) ? "none" : "highlighted (not a token alignment)"}`));
        if (current?.forcedPrefix) card.append(node("p", `Supplied common prefix: ${current.forcedPrefix}`));
        if (!blind.checked) card.append(node("small", `${arm.kind} · ${arm.policy} · seed ${arm.seed} · ${arm.sourceId}`));
        const trace = node("details"); trace.append(node("summary", "Token decisions"));
        trace.append(node("pre", current?.firstDecision?.length
          ? current.firstDecision.map((d) => `${JSON.stringify(d.text)}  log P=${d.logProbability.toFixed(4)}  id=${d.tokenId}`).join("\n")
          : current?.tokenLogProbs?.length ? current.tokenLogProbs.map((v, i) => `${current.tokenIds?.[i]}: ${v.toFixed(4)}`).join("\n")
          : "Per-token probabilities were not retained for this historical episode."));
        card.append(trace); columns.append(card);
        const grades = node("section"); grades.append(node("h3", blind.checked ? `Evidence ${index === 0 ? "A" : "B"}` : `${arm.label}: episode judgments`));
        if (!arm.ratings.length) grades.append(node("p", "Unassessed — no quality pass or 9/10 inference."));
        for (const rating of arm.ratings) {
          const item = node("details");
          item.append(node("summary", `${rating.criterion}: ${rating.score === null ? "abstain" : `${rating.score}/${rating.scale}`}`));
          for (const e of rating.evidence) {
            const quote = node("blockquote", e.span); quote.dir = "auto"; item.append(quote, node("p", e.finding));
          }
          item.append(node("small", `Judge identity: ${rating.judge}; exploratory, no human calibration`));
          grades.append(item);
        }
        evidence.append(grades);
      }
      metadata.textContent = JSON.stringify({ metadata: active.metadata, study: active.experiment, family: active.family,
        level: active.level, seed: active.seed, sources: catalogue.sources,
        limitation: "Same seed does not guarantee same history after outputs diverge; no new final judging on historical data." }, null, 2);
      status.textContent = liveExample ? "New live comparison — quality unassessed" : `${catalogue.examples.length} recorded comparisons available; no model required`;
    }
    function chooseCase() {
      liveExample = null;
      active = selected(); turn = 0;
      const defs = active.arms.map((a) => ({ value: a.id, label: a.label }));
      options(left.input, defs); options(right.input, defs);
      left.input.value = active.arms.find((a) => a.kind === "baseline")?.id ?? active.arms[0].id;
      right.input.value = active.arms.find((a) => a.kind === "residual-steering" || a.kind === "donor-diagnostic")?.id ??
        active.arms.find((a) => a.id !== left.input.value)!.id;
      render();
    }
    function chooseStudy() {
      options(scenario.input, catalogue.examples.filter((e) => e.experiment === experiment.input.value).map((e) => ({ value: e.id, label: e.label })));
      chooseCase();
    }
    experiment.input.addEventListener("change", chooseStudy);
    scenario.input.addEventListener("change", chooseCase);
    for (const input of [left.input, right.input, blind]) input.addEventListener("change", render);
    previous.addEventListener("click", () => { turn--; render(); });
    next.addEventListener("click", () => { turn++; render(); });
    live.addEventListener("click", async () => {
      if (!active.live || running) return;
      running = true; live.disabled = true;
      try {
        const preset = active;
        const response = await fetch(new URL(`research/${preset.live!.patch}`, document.baseURI));
        if (!response.ok) throw new Error(`Historical patch unavailable (${response.status})`);
        const patch: unknown = await response.json();
        if (!record(patch) || !Array.isArray(patch.overlays) || !record(patch.level_gate) ||
            typeof patch.level_gate[preset.level] !== "number") throw new Error("Invalid recorded patch");
        const gate = patch.level_gate[preset.level];
        if (typeof gate !== "number" || !Number.isFinite(gate)) throw new Error("Invalid level gate");
        const overlays = patch.overlays.map((o: unknown) => {
          if (!record(o) || !record(o.site) || typeof o.site.layer !== "number" || !Array.isArray(o.a) || !Array.isArray(o.b) ||
              !o.a.every((v) => typeof v === "number" && Number.isFinite(v)) || !o.b.every((v) => typeof v === "number" && Number.isFinite(v)) ||
              typeof o.scale !== "number") throw new Error("Invalid overlay vectors");
          return { layer: o.site.layer, a: o.a, b: o.b, coefficient: gate * o.scale };
        });
        const scenarioData = atLevel(preset.live!.scenario, preset.level as CefrLevel);
        const mode = protocol.input.value === "conversation" ? "conversation" : "fixed-prefix";
        const artifact = await run({
          id: `paired-demo-${Date.now()}`, op: "pairedReplay", protocol: mode,
          messages: [{ role: "system", content: systemMessage(scenarioData, "canonical") }],
          playerTurns: mode === "conversation" ? scenarioData.player_policy.player_turns_he : scenarioData.player_policy.player_turns_he.slice(0, 1),
          arms: [{ id: "base", label: "Unmodified metadata baseline" }, { id: "p6", label: "Historical p6 steering — unpromoted", overlays }],
          maxNewTokens: 96, temperature: 1, seed: preset.seed,
        });
        const result = artifact.result;
        if (!record(result) || !Array.isArray(result.arms) || result.complete !== true) throw new Error("Paired replay did not complete");
        liveExample = { ...preset, id: artifact.id, live: undefined, caveat: String(result.caveat),
          arms: result.arms.map((value: unknown) => {
            if (!record(value) || !Array.isArray(value.turns)) throw new Error("Invalid paired result");
            return {
              id: String(value.id), label: String(value.label), kind: String(value.intervention),
              sourceId: artifact.id, sourceFile: "live artifact export", policy: mode, seed: preset.seed,
              warnings: ["New full-logit/sequential run, not the historical sampling route. Not judged; no quality promotion."], ratings: [],
              turns: value.turns.map((t: Turn & { tokenIds: number[] }) => ({ ...t, tokenCount: t.tokenIds.length })),
            };
          }),
        };
        turn = 0;
        options(left.input, liveExample.arms.map((a) => ({ value: a.id, label: a.label })));
        options(right.input, liveExample.arms.map((a) => ({ value: a.id, label: a.label })));
        right.input.value = liveExample.arms[1].id;
        render();
      } catch (error) {
        status.textContent = `Live comparison failed: ${error instanceof Error ? error.message : String(error)}. Load the research session first; no result is assumed.`;
      } finally { running = false; live.disabled = !selected().live; }
    });
    exportButton.addEventListener("click", () => {
      const url = URL.createObjectURL(new Blob([JSON.stringify({ example: selected(), sources: catalogue.sources })], { type: "application/json" }));
      const a = node("a"); a.href = url; a.download = "paired-comparison.json"; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    chooseStudy();
  } catch (error) {
    status.textContent = `Comparison demo unavailable: ${error instanceof Error ? error.message : String(error)}`;
    status.dataset.error = "true";
  }
}
