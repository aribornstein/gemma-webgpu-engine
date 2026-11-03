"""Paired cluster bootstrap of arm differences by scenario family (deterministic, seed 7).

Usage: python3 research/cefr-agentic/compare.py <panel-json> <out-json>
"""
import json
import random
import sys

panel_path, out_path = sys.argv[1:3]
episodes = json.load(open(panel_path))["episodes"]
ARMS = ["raw_base", "strong_prompt", "best_of_2", "agent_rank1", "random_rank1", "wrong_level_gate"]


def family_values(arm, metric):
    """Per family: metric value, where cefr_control = estimated B2 - estimated A1."""
    out = {}
    rows = [e for e in episodes if e["arm"] == arm]
    for family in sorted({e["family"] for e in rows}):
        fam = [e for e in rows if e["family"] == family]
        if metric == "cefr_control":
            a1 = [e["estimated_level_value"] for e in fam if e["level"] == "A1" and e.get("estimated_level_value")]
            b2 = [e["estimated_level_value"] for e in fam if e["level"] == "B2" and e.get("estimated_level_value")]
            if a1 and b2:
                out[family] = sum(b2) / len(b2) - sum(a1) / len(a1)
        elif metric == "complexity_gap":
            a1 = [e["complexity"] for e in fam if e["level"] == "A1"]
            b2 = [e["complexity"] for e in fam if e["level"] == "B2"]
            if a1 and b2:
                out[family] = sum(b2) / len(b2) - sum(a1) / len(a1)
        else:
            values = [e.get(metric) for e in fam]
            values = [float(v) for v in values if isinstance(v, (int, float, bool))]
            if values:
                out[family] = sum(values) / len(values)
    return out


def paired(arm, reference, metric, replicates=2000):
    a, b = family_values(arm, metric), family_values(reference, metric)
    families = sorted(set(a) & set(b))
    diffs = [a[f] - b[f] for f in families]
    if not diffs:
        return None
    rng = random.Random(7)
    boots = []
    for _ in range(replicates):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    return {
        "families": len(diffs),
        "mean_diff": round(sum(diffs) / len(diffs), 3),
        "ci95": [round(boots[int(0.025 * replicates)], 3), round(boots[int(0.975 * replicates) - 1], 3)],
        "per_family": {f: round(d, 3) for f, d in zip(families, diffs)},
    }


report = {"unit": "scenario family (paired), percentile cluster bootstrap 2000x", "comparisons": {}}
for metric in ["cefr_control", "complexity_gap", "joint_pass", "hard_pass", "morphosyntactic_correctness", "natural_hebrew", "cefr_suitability", "multi_turn_coherence"]:
    report["comparisons"][metric] = {}
    import os
    default_pairs = [
        ("agent_rank1", "raw_base"), ("agent_rank1", "random_rank1"), ("agent_rank1", "wrong_level_gate"),
        ("agent_rank1", "best_of_2"), ("agent_rank1", "strong_prompt"),
        ("strong_prompt", "raw_base"), ("best_of_2", "raw_base"),
    ]
    pairs = [tuple(p.split(":")) for p in os.environ["COMPARE_PAIRS"].split(",")] if os.environ.get("COMPARE_PAIRS") else default_pairs
    for arm, reference in pairs:
        if any(e["arm"] == arm for e in episodes) and any(e["arm"] == reference for e in episodes):
            report["comparisons"][metric][f"{arm} - {reference}"] = paired(arm, reference, metric)
json.dump(report, open(out_path, "w"), indent=1)
print(json.dumps({m: {k: (v["mean_diff"], v["ci95"]) for k, v in c.items() if v} for m, c in report["comparisons"].items()}, indent=1))
