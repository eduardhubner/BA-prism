"""
Quality-blindness check: PRISM score vs. description quality (AUROC).

This is the thesis's motivating result: PRISM's polysemanticity score (mean
pairwise similarity among a neuron's candidate descriptions) should not be
confounded with how good those descriptions are at discriminating the
neuron's activations (AUROC). If PRISM score correlated with mean AUROC, a
low PRISM score could just mean "the LLM wrote bad descriptions" rather than
"this neuron is genuinely polysemantic" -- so the absence of that correlation
is the core justification for treating PRISM score as a polysemanticity
measure rather than a description-quality artifact.

Computed directly from pairwise_similarities_dataset.csv (self-contained --
does not depend on filtered_evaluation.csv): prism_poly = mean_similarity
(pre-filter, pre-collapse); mean_auroc_pre = mean of auroc_1..auroc_5 for the
same neuron. All 239 neurons across all 4 models.

Outputs:
  results/quality_blindness_check.csv   -- per-neuron (prism_poly, mean_auroc_pre)
  results/quality_blindness_check.json  -- correlation summary

Run with: python src/quality_blindness_check.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).parent.parent
PAIRWISE_PATH = ROOT / "results" / "pairwise_similarities_dataset.csv"
OUT_CSV = ROOT / "results" / "quality_blindness_check.csv"
OUT_JSON = ROOT / "results" / "quality_blindness_check.json"

AUROC_COLS = ["auroc_1", "auroc_2", "auroc_3", "auroc_4", "auroc_5"]


def main():
    df = pd.read_csv(PAIRWISE_PATH)
    out = df[["model", "layer", "unit"]].copy()
    out["prism_poly"] = df["mean_similarity"]
    out["mean_auroc_pre"] = df[AUROC_COLS].mean(axis=1)

    out.to_csv(OUT_CSV, index=False)
    print(f"Saved per-neuron table to {OUT_CSV}  (n={len(out)})")

    spear = spearmanr(out["prism_poly"], out["mean_auroc_pre"])
    pear = pearsonr(out["prism_poly"], out["mean_auroc_pre"])

    by_model = {}
    for model, grp in out.groupby("model"):
        s = spearmanr(grp["prism_poly"], grp["mean_auroc_pre"])
        by_model[model] = {"n": len(grp), "spearman_r": round(float(s.statistic), 4),
                            "spearman_p": round(float(s.pvalue), 4)}

    summary = {
        "n_neurons": len(out),
        "spearman_r": round(float(spear.statistic), 4),
        "spearman_p": round(float(spear.pvalue), 4),
        "pearson_r": round(float(pear.statistic), 4),
        "pearson_p": round(float(pear.pvalue), 4),
        "by_model": by_model,
        "interpretation": "rho near 0 and not significant means PRISM score is not "
                           "explained by description quality (AUROC) -- a low PRISM "
                           "score reflects genuine polysemanticity, not bad descriptions. "
                           "This is the motivating result for treating PRISM score as a "
                           "polysemanticity measure rather than a quality artifact.",
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {OUT_JSON}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
