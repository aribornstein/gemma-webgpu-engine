"""Length confound check: how much of each arm's judged A1/B2 level gap is predicted by turn length alone?

Fits estimated_level ~ a + b * log(words_per_turn) on the unedited arms (raw, strong prompt, best-of-2 candidates),
then reports, per arm, the observed level gap (B2 - A1, judged), the gap predicted from length, and the residual.

Usage: python3 research/cefr-agentic/length_check.py <panel-json> <out-json>
"""
import json
import math
import sys

panel_path, out_path = sys.argv[1:3]
episodes = [e for e in json.load(open(panel_path))["episodes"] if e.get("estimated_level_value") and e.get("words_per_turn")]
reference = [e for e in episodes if e["arm"] in ("raw_base", "strong_prompt", "best_of_2_candidate")]
xs = [math.log(e["words_per_turn"]) for e in reference]
ys = [e["estimated_level_value"] for e in reference]
mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
intercept = my - slope * mx
predicted = lambda e: intercept + slope * math.log(e["words_per_turn"])
ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
ss_tot = sum((y - my) ** 2 for y in ys)

report = {"fit": {"slope_per_log_word": round(slope, 3), "intercept": round(intercept, 3), "r2_on_unedited": round(1 - ss_res / ss_tot, 3), "n": len(reference)}, "arms": {}}
for arm in sorted({e["arm"] for e in episodes}):
    rows = [e for e in episodes if e["arm"] == arm]
    a1 = [e for e in rows if e["level"] == "A1"]
    b2 = [e for e in rows if e["level"] == "B2"]
    if not a1 or not b2:
        continue
    mean = lambda values: sum(values) / len(values)
    observed = mean([e["estimated_level_value"] for e in b2]) - mean([e["estimated_level_value"] for e in a1])
    from_length = mean([predicted(e) for e in b2]) - mean([predicted(e) for e in a1])
    residual_b2 = mean([e["estimated_level_value"] - predicted(e) for e in b2])
    residual_a1 = mean([e["estimated_level_value"] - predicted(e) for e in a1])
    report["arms"][arm] = {
        "observed_gap": round(observed, 2),
        "gap_predicted_by_length": round(from_length, 2),
        "gap_beyond_length": round(residual_b2 - residual_a1, 2),
        "a1_words": round(mean([e["words_per_turn"] for e in a1]), 1),
        "b2_words": round(mean([e["words_per_turn"] for e in b2]), 1),
    }
json.dump(report, open(out_path, "w"), indent=1)
print(json.dumps(report, indent=1))
