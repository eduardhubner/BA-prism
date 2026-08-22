"""
Similarity-collapse threshold calibration dataset.

Reconstructs (and saves, traceably) the empirical analysis behind the collapse
threshold. The framework uses a single, post-AUC collapse: descriptions are
AUC-filtered first, then survivors with pairwise similarity > 0.70 are collapsed,
keeping the higher-AUC description per group. (A pre-AUC collapse at 0.75 was
considered to save AUC evaluations but was NOT adopted; it is not part of the
pipeline.)

The calibration question is: at each similarity level, how different are the two
descriptions' AUCs? Because collapse keeps the higher-AUC description, a pair is
safe to collapse when the two are redundant (AUC difference ~0); it is unsafe
when they are genuinely distinct (one description captures the feature and the
other does not, i.e. a large AUC difference) -- collapsing there would discard a
real, better description. This script builds every within-neuron description pair
(both texts, both AUCs, similarity), bins by similarity, and reports the
AUC-difference distribution per bin -- the evidence for placing the threshold at
0.70: distinct pairs with AUC difference up to ~0.97 still occur in [0.65, 0.70)
(so a looser threshold would over-collapse), while above 0.70 pairs are
predominantly redundant and become near-paraphrases (median AUC diff -> 0) above
~0.75.

Data source: the PRISM master table (descriptions + AUCs + pairwise similarities,
gte-Qwen2-1.5B embeddings, 239 neurons / 4 models).

Outputs:
  results/similarity_threshold_calibration.csv   -- all 2390 pairs (the "collection
      of description pairs": model/layer/unit, i, j, sim, auc_i, auc_j, auc_diff,
      description_i, description_j, sim_bin)
  results/similarity_threshold_calibration.json  -- percentiles + per-bin summary
      + curated distinct/paraphrase example pairs
"""

import sys
import json
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import case_study_analysis as cs

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"

# similarity bins for the calibration table (upper bound of last bin is open)
BINS = [(0.00, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 1.01)]
COLLAPSE_TAU = 0.70    # the single (post-AUC) collapse threshold actually used
PARAPHRASE_TAU = 0.75  # display band: pairs above this are near-paraphrases


def _bin_label(s):
    for lo, hi in BINS:
        if lo <= s < hi:
            return f"[{lo:.2f},{hi:.2f})"
    return "other"


def build_pairs():
    df = pd.read_csv(cs.PRISM_PATH)
    rows = []
    for _, r in df.iterrows():
        sm = cs.get_sim_matrix(r)
        aucs = [r[f"auc_{i}"] for i in range(1, 6)]
        descs = [r[f"description_{i}"] for i in range(1, 6)]
        for i, j in combinations(range(5), 2):
            rows.append({
                "model": r["model"], "layer": int(r["layer"]), "unit": int(r["unit"]),
                "i": i + 1, "j": j + 1, "sim": float(sm[i, j]),
                "auc_i": float(aucs[i]), "auc_j": float(aucs[j]),
                "auc_diff": float(abs(aucs[i] - aucs[j])),
                "description_i": str(descs[i]), "description_j": str(descs[j]),
            })
    p = pd.DataFrame(rows)
    p["sim_bin"] = p["sim"].map(_bin_label)
    return p


def bin_summary(p):
    out = []
    for lo, hi in BINS:
        b = p[(p["sim"] >= lo) & (p["sim"] < hi)]
        if len(b) == 0:
            continue
        out.append({
            "sim_bin": f"[{lo:.2f},{hi:.2f})", "n_pairs": int(len(b)),
            "median_auc_diff": round(float(b["auc_diff"].median()), 4),
            "mean_auc_diff": round(float(b["auc_diff"].mean()), 4),
            "p90_auc_diff": round(float(b["auc_diff"].quantile(0.90)), 4),
            "max_auc_diff": round(float(b["auc_diff"].max()), 4),
            "frac_auc_diff_gt_0.3": round(float((b["auc_diff"] > 0.3).mean()), 4),
        })
    return out


def example_pairs(p, n=4):
    """Curated illustrative pairs: the distinct-but-similar pairs that justify not
    lowering the threshold below 0.70, and clearly-redundant paraphrase pairs."""
    distinct = (p[(p["sim"] >= 0.65) & (p["sim"] < COLLAPSE_TAU)]
                .sort_values("auc_diff", ascending=False).head(n))
    paraphrase = (p[p["sim"] > PARAPHRASE_TAU].sort_values("sim", ascending=False)
                  .query("auc_diff < 0.05").head(n))

    def fmt(df):
        return [{"feature": f"{r.model} L{r.layer}/U{r.unit}", "sim": round(r.sim, 4),
                 "auc_diff": round(r.auc_diff, 4),
                 "desc_high_auc": r.description_i if r.auc_i >= r.auc_j else r.description_j,
                 "auc_high": round(max(r.auc_i, r.auc_j), 4),
                 "desc_low_auc": r.description_j if r.auc_i >= r.auc_j else r.description_i,
                 "auc_low": round(min(r.auc_i, r.auc_j), 4)}
                for r in df.itertuples()]
    return {"distinct_but_similar_0.65_0.70": fmt(distinct),
            "paraphrases_gt_0.75": fmt(paraphrase)}


def main():
    p = build_pairs()
    p.to_csv(RESULTS / "similarity_threshold_calibration.csv", index=False)

    summary = {
        "params": {"embedding": "gte-Qwen2-1.5B", "n_neurons": int(p[["model", "layer", "unit"]].drop_duplicates().shape[0]),
                   "n_pairs": int(len(p)), "collapse_tau": COLLAPSE_TAU,
                   "note": "single post-AUC collapse; no pre-AUC collapse in pipeline"},
        "similarity_percentiles": {q: round(float(p["sim"].quantile(q / 100)), 4)
                                   for q in (50, 75, 90, 95, 99)},
        "auc_diff_by_sim_bin": bin_summary(p),
        "example_pairs": example_pairs(p),
    }
    (RESULTS / "similarity_threshold_calibration.json").write_text(json.dumps(summary, indent=2))

    print(f"Similarity-threshold calibration ({summary['params']['n_pairs']} pairs, "
          f"{summary['params']['n_neurons']} neurons)")
    pc = summary["similarity_percentiles"]
    print(f"  sim percentiles: p50={pc[50]} p90={pc[90]} p95={pc[95]}")
    print("  AUC-difference by similarity bin:")
    for b in summary["auc_diff_by_sim_bin"]:
        print(f"    {b['sim_bin']}  n={b['n_pairs']:4d}  median|ΔAUC|={b['median_auc_diff']:.3f}  "
              f"max={b['max_auc_diff']:.3f}  %|ΔAUC|>0.3={b['frac_auc_diff_gt_0.3']*100:.0f}%")
    print(f"\nSaved:\n  {RESULTS/'similarity_threshold_calibration.csv'}\n  {RESULTS/'similarity_threshold_calibration.json'}")


if __name__ == "__main__":
    main()
