"""
Embedding-model robustness of the collapse decision.

Companion to the embedding-robustness matrices (results/embedding_robustness_*).
Those establish that pairwise cosine similarity is NOT robust in absolute terms
across embedding models (the scale shifts) but IS robust in rank (within-feature
Spearman 0.83-0.85). This script tests the consequence that matters for the
framework: whether that rank reshuffling actually moves the collapse decision
(sim > tau_1) or whether it is confined to pairs near the threshold, leaving the
clear redundant / distinct cases -- and therefore the collapse *separation* --
stable.

Method: for every within-neuron description pair we have the cosine similarity
under the baseline embedding model (gte-Qwen2-1.5B) and four alternatives. A
"flip" is a pair whose collapse decision (sim > tau_1) differs from baseline
under another model. We report:

  1. absolute similarity distribution per model (why a fixed tau_1 is not
     portable): mean / median / p90 / % of pairs above tau_1;
  2. pooled within-feature rank agreement vs baseline (Spearman) -- recomputed
     here so the script is self-contained (matches embedding_robustness_matrix.json);
  3. collapse-decision agreement vs baseline under a fixed tau_1;
  4. flip rate as a function of |sim - tau_1| in the baseline model -- the core
     result: flips concentrate at the boundary and vanish away from it;
  5. stability of the clearly-distinct (sim < clear_lo) and clearly-redundant
     (sim > clear_hi) cases;
  6. the share of all collapse disagreements that sit within `margin` of tau_1.

Input:  results/embedding_robustness_within.csv  (one row per within-neuron pair;
        columns: neuron_id, pair, then one cosine column per embedding model)
Output: results/embedding_collapse_robustness.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"

TAU1 = 0.70            # similarity-collapse threshold
BASELINE = "gte-Qwen2-1.5B (baseline)"
CLEAR_LO = 0.50        # "clearly distinct"  (well below tau_1)
CLEAR_HI = 0.85        # "clearly redundant" (well above tau_1)
MARGIN = 0.10          # "near the threshold" band, |sim - tau_1| < MARGIN
# distance-from-threshold bins for the flip-rate curve
DIST_BINS = [(0.00, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 1.00)]


def _models(df):
    return [c for c in df.columns if c not in ("neuron_id", "pair")]


def absolute_distribution(df, models):
    """Per-model absolute cosine distribution and collapse fraction at tau_1."""
    out = {}
    for m in models:
        v = df[m].dropna()
        out[m] = {
            "mean": round(float(v.mean()), 4),
            "median": round(float(v.median()), 4),
            "p90": round(float(v.quantile(0.90)), 4),
            "pct_pairs_gt_tau1": round(float((v > TAU1).mean() * 100), 2),
            "tau1_matched_recalibrated": round(float(v.quantile(1 - (df[BASELINE] > TAU1).mean())), 4),
        }
    return out


def rank_agreement(df, others):
    """Pooled within-feature Spearman vs baseline (self-contained recompute)."""
    out = {}
    for m in others:
        s = df.dropna(subset=[BASELINE, m])
        out[m] = round(float(spearmanr(s[BASELINE], s[m]).statistic), 4)
    return out


def collapse_agreement(df, others):
    """Fraction of pairs whose collapse decision matches baseline at fixed tau_1."""
    out = {}
    for m in others:
        s = df.dropna(subset=[BASELINE, m])
        base_dec, oth_dec = s[BASELINE] > TAU1, s[m] > TAU1
        out[m] = {
            "agreement_pct": round(float((base_dec == oth_dec).mean() * 100), 2),
            "baseline_collapse_pct": round(float(base_dec.mean() * 100), 2),
            "other_collapse_pct": round(float(oth_dec.mean() * 100), 2),
        }
    return out


def flip_rate_by_distance(df, others):
    """Mean collapse-flip rate (vs baseline, across the other models) for pairs
    binned by |baseline_sim - tau_1|. The central result: flips are local to the
    threshold."""
    d = df.dropna(subset=[BASELINE]).copy()
    d["dist"] = (d[BASELINE] - TAU1).abs()
    rows = []
    for lo, hi in DIST_BINS:
        sub = d[(d["dist"] >= lo) & (d["dist"] < hi)]
        if len(sub) == 0:
            continue
        flips = []
        for m in others:
            s = sub.dropna(subset=[m])
            flips.append(float(((s[BASELINE] > TAU1) != (s[m] > TAU1)).mean()))
        rows.append({"dist_lo": lo, "dist_hi": hi, "n_pairs": int(len(sub)),
                     "mean_flip_rate_pct": round(float(np.mean(flips) * 100), 2)})
    return rows


def clear_case_stability(df, others):
    """Flip rate for the clearly-distinct and clearly-redundant pairs."""
    d = df.dropna(subset=[BASELINE])
    out = {}
    for name, mask in [("clearly_distinct_lt_%.2f" % CLEAR_LO, d[BASELINE] < CLEAR_LO),
                       ("clearly_redundant_gt_%.2f" % CLEAR_HI, d[BASELINE] > CLEAR_HI)]:
        sub = d[mask]
        flips = []
        for m in others:
            s = sub.dropna(subset=[m])
            flips.append(float(((s[BASELINE] > TAU1) != (s[m] > TAU1)).mean()))
        out[name] = {"n_pairs": int(len(sub)),
                     "mean_flip_rate_pct": round(float(np.mean(flips) * 100), 2)}
    return out


def disagreements_near_threshold(df, others):
    """Of all collapse disagreements, the share with baseline sim within MARGIN
    of tau_1."""
    tot = near = 0
    for m in others:
        s = df.dropna(subset=[BASELINE, m])
        dis = (s[BASELINE] > TAU1) != (s[m] > TAU1)
        tot += int(dis.sum())
        near += int((dis & ((s[BASELINE] - TAU1).abs() < MARGIN)).sum())
    return {"n_disagreements": tot, "n_within_margin": near,
            "pct_within_margin": round(near / tot * 100, 2) if tot else None,
            "margin": MARGIN}


def main():
    df = pd.read_csv(RESULTS / "embedding_robustness_within.csv")
    models = _models(df)
    others = [m for m in models if m != BASELINE]

    out = {
        "params": {"tau1": TAU1, "baseline": BASELINE, "clear_lo": CLEAR_LO,
                   "clear_hi": CLEAR_HI, "margin": MARGIN,
                   "n_pairs": int(len(df)), "n_neurons": int(df["neuron_id"].nunique())},
        "absolute_distribution": absolute_distribution(df, models),
        "within_feature_rank_agreement_vs_baseline": rank_agreement(df, others),
        "collapse_decision_agreement_vs_baseline": collapse_agreement(df, others),
        "flip_rate_by_distance_from_threshold": flip_rate_by_distance(df, others),
        "clear_case_stability": clear_case_stability(df, others),
        "disagreements_near_threshold": disagreements_near_threshold(df, others),
    }
    (RESULTS / "embedding_collapse_robustness.json").write_text(json.dumps(out, indent=2))

    # console summary
    print("Embedding-model robustness of the collapse decision")
    print(f"  {out['params']['n_pairs']} within-neuron pairs, "
          f"{out['params']['n_neurons']} neurons, tau_1={TAU1}\n")
    print("  rank agreement vs baseline (within-feature Spearman):")
    for m, r in out["within_feature_rank_agreement_vs_baseline"].items():
        print(f"    {m:26s} {r:.3f}")
    print("\n  flip rate vs |sim - tau_1| (baseline):")
    for r in out["flip_rate_by_distance_from_threshold"]:
        print(f"    [{r['dist_lo']:.2f},{r['dist_hi']:.2f})  n={r['n_pairs']:5d}  "
              f"{r['mean_flip_rate_pct']:5.1f}%")
    cs = out["clear_case_stability"]
    for k, v in cs.items():
        print(f"  {k}: n={v['n_pairs']}  flip={v['mean_flip_rate_pct']}%")
    dn = out["disagreements_near_threshold"]
    print(f"  {dn['pct_within_margin']}% of collapse disagreements within "
          f"{MARGIN} of tau_1 ({dn['n_within_margin']}/{dn['n_disagreements']})")
    print(f"\nSaved to {RESULTS / 'embedding_collapse_robustness.json'}")


if __name__ == "__main__":
    main()
