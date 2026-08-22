"""
Weighting-gap vs. survivor-AUROC variance.

The AUC-weighted mean similarity (WMS) differs from the plain (unweighted) mean
similarity only when the SURVIVING descriptions differ in AUROC: if all survivor
AUROCs are equal the pair weights are equal and WMS == unweighted mean. This
tests that directly -- |WMS - unweighted mean sim| should grow with the spread
of survivor AUROCs. A positive relationship is the mechanistic justification for
AUC-weighting: it diverges from the plain mean exactly (and only) when the
descriptions being averaged differ in reliability.

Restricted to n_s >= 3: at n_s == 2 there is a single pairwise similarity, so the
AUC-weighted mean of one value trivially equals the unweighted mean and the gap
is identically 0 regardless of AUROC spread.

Pipeline: filter-first (AUC filter, then similarity collapse), matching the
canonical filtered_evaluation.py / case_study_analysis.py.

Runs on the full k=5 dataset (239 neurons, 4 models) and the k=10 gpt2-xl set.
Reports Pearson and Spearman, plus partial correlations controlling for
n_surviving (since |gap| also shrinks with more survivors, we check that
survivor-AUROC spread predicts the gap independently of survivor count).

Output: results/weighting_gap_auroc_variance.json
"""

import json
import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path
import verification as V
from scipy.stats import pearsonr, spearmanr, rankdata, t as tdist, wilcoxon

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
SIM_THRESHOLD = V.SIM_THRESHOLD
AUC_THRESHOLD = V.AUC_THRESHOLD


def build_sim_matrix(row, n_desc, sep):
    """sep='' for k=5 columns (sim_12); sep='_' for k=10 (sim_1_2)."""
    return V.sim_matrix_from_row(row, n_desc=n_desc, sep=sep)


def survivors_filter_first(aucs, sim, n_desc):
    """AUROC filter then similarity collapse. Delegates to `verification`."""
    return list(V.filter_and_collapse(aucs, sim, sim_threshold=SIM_THRESHOLD,
                                      auc_threshold=AUC_THRESHOLD).surviving)


def neuron_gap_row(row, n_desc, sep):
    aucs = np.array([row[f"auc_{i}"] for i in range(1, n_desc + 1)])
    sim = build_sim_matrix(row, n_desc, sep)
    surv = survivors_filter_first(aucs, sim, n_desc)
    if len(surv) < 3:
        return None
    pairs = list(combinations(surv, 2))
    mean_sim = float(np.mean([sim[i, j] for i, j in pairs]))
    num = den = 0.0
    for i, j in pairs:
        w = ((aucs[i] - 0.5) / 0.5) * ((aucs[j] - 0.5) / 0.5)
        num += w * sim[i, j]
        den += w
    wms = num / den if den > 0 else np.nan
    surv_aucs = aucs[surv]
    return dict(n_s=len(surv), abs_gap=abs(wms - mean_sim),
                auc_std=float(np.std(surv_aucs)), auc_var=float(np.var(surv_aucs)))


def partial_corr(x, y, z, method="pearson"):
    """Partial correlation of x and y controlling for z. dof = n - 3."""
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    if method == "spearman":
        x, y, z = rankdata(x), rankdata(y), rankdata(z)
    rxy = np.corrcoef(x, y)[0, 1]
    rxz = np.corrcoef(x, z)[0, 1]
    ryz = np.corrcoef(y, z)[0, 1]
    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    r = (rxy - rxz * ryz) / denom if denom > 0 else np.nan
    n = len(x)
    dof = n - 3
    if np.isnan(r) or abs(r) >= 1 or dof <= 0:
        p = np.nan
    else:
        tval = r * np.sqrt(dof / (1 - r ** 2))
        p = float(2 * tdist.sf(abs(tval), dof))
    return float(r), p


def analyze(df, n_desc, sep, label):
    rows = [r for r in (neuron_gap_row(row, n_desc, sep) for _, row in df.iterrows()) if r]
    d = pd.DataFrame(rows)
    res = {"label": label, "n_neurons_ns_ge_3": int(len(d)),
           "mean_abs_gap": round(float(d.abs_gap.mean()), 4),
           "mean_survivor_auc_std": round(float(d.auc_std.mean()), 4)}
    for x in ["auc_std", "auc_var"]:
        pr = pearsonr(d[x], d.abs_gap)
        sp = spearmanr(d[x], d.abs_gap)
        ppr, pprp = partial_corr(d.abs_gap, d[x], d.n_s, "pearson")
        psp, pspp = partial_corr(d.abs_gap, d[x], d.n_s, "spearman")
        res[f"abs_gap_vs_{x}"] = {
            "pearson_r": round(float(pr.statistic), 4), "pearson_p": float(f"{pr.pvalue:.3g}"),
            "spearman_r": round(float(sp.statistic), 4), "spearman_p": float(f"{sp.pvalue:.3g}"),
            "partial_pearson_r_given_ns": round(ppr, 4), "partial_pearson_p": float(f"{pprp:.3g}"),
            "partial_spearman_r_given_ns": round(psp, 4), "partial_spearman_p": float(f"{pspp:.3g}"),
        }
    return res


def variance_stats(df, n_desc, sep):
    """Distribution of the std (and range) of survivor AUROCs, over neurons with
    n_s >= 2 -- quantifies how much 'quality spread' the weighting has to work
    with in the first place."""
    stds, rngs = [], []
    for _, row in df.iterrows():
        aucs = np.array([row[f"auc_{i}"] for i in range(1, n_desc + 1)])
        sim = build_sim_matrix(row, n_desc, sep)
        surv = survivors_filter_first(aucs, sim, n_desc)
        if len(surv) >= 2:
            sa = aucs[surv]
            stds.append(float(np.std(sa)))
            rngs.append(float(sa.max() - sa.min()))
    v = np.array(stds)
    return {
        "n_neurons_ns_ge_2": int(len(v)),
        "mean": round(float(v.mean()), 4), "median": round(float(np.median(v)), 4),
        "p25": round(float(np.percentile(v, 25)), 4), "p75": round(float(np.percentile(v, 75)), 4),
        "p90": round(float(np.percentile(v, 90)), 4), "max": round(float(v.max()), 4),
        "frac_near_homogeneous_std_lt_0p02": round(float((v < 0.02).mean()), 4),
        "mean_survivor_auc_range": round(float(np.mean(rngs)), 4),
    }


def matched_k_variance(k5_df, k10_df):
    """Paired k=5 vs k=10 survivor-AUROC std on the SAME gpt2-xl neurons (n_s>=2
    at both k) -- tests whether the quality spread grows with more candidates."""
    def by_neuron(df, n_desc, sep):
        out = {}
        for _, row in df.iterrows():
            aucs = np.array([row[f"auc_{i}"] for i in range(1, n_desc + 1)])
            sim = build_sim_matrix(row, n_desc, sep)
            surv = survivors_filter_first(aucs, sim, n_desc)
            if len(surv) >= 2:
                out[(row["layer"], row["unit"])] = (float(np.std(aucs[surv])), len(surv))
        return out
    a = by_neuron(k5_df[k5_df["model"] == "gpt2-xl"], 5, "")
    b = by_neuron(k10_df, 10, "_")
    common = sorted(set(a) & set(b))
    sa = np.array([a[k][0] for k in common]); sb = np.array([b[k][0] for k in common])
    na = np.array([a[k][1] for k in common]); nb = np.array([b[k][1] for k in common])
    return {
        "n_matched_gpt2xl": len(common),
        "mean_std_k5": round(float(sa.mean()), 4), "mean_std_k10": round(float(sb.mean()), 4),
        "delta_k10_minus_k5": round(float(sb.mean() - sa.mean()), 4),
        "wilcoxon_p": round(float(wilcoxon(sb, sa).pvalue), 4),
        "mean_ns_k5": round(float(na.mean()), 2), "mean_ns_k10": round(float(nb.mean()), 2),
    }


def main():
    out = {}
    k5 = pd.read_csv(RESULTS / "pairwise_similarities_dataset.csv")
    k10 = pd.read_csv(RESULTS / "k10_pairwise_similarities.csv")

    # Correlation |gap| vs survivor-AUROC spread. Pooled, the controlled
    # same-model k check (gpt2-xl at both k), and a per-model breakdown.
    out["k5_all_models"] = analyze(k5, 5, "", "k=5, all 4 models (239 neurons)")
    out["k5_gpt2xl"] = analyze(k5[k5["model"] == "gpt2-xl"], 5, "", "k=5, gpt2-xl only (matched to k=10)")
    out["k10_gpt2xl"] = analyze(k10, 10, "_", "k=10, gpt2-xl")
    out["k5_by_model"] = {m: analyze(k5[k5["model"] == m], 5, "", f"k=5, {m}")
                          for m in sorted(k5["model"].unique())}

    # How large is the survivor-AUROC spread, and does it grow with k?
    out["survivor_auc_variance"] = {
        "k5_all_models": variance_stats(k5, 5, ""),
        "k10_gpt2xl": variance_stats(k10, 10, "_"),
        "matched_k5_vs_k10_gpt2xl": matched_k_variance(k5, k10),
    }

    (RESULTS / "weighting_gap_auroc_variance.json").write_text(json.dumps(out, indent=2))

    def show(r):
        e = r["abs_gap_vs_auc_std"]
        print(f"  {r['label']:38s} n={r['n_neurons_ns_ge_3']:>3}  "
              f"Spearman rho={e['spearman_r']:+.3f} (p={e['spearman_p']:g})  "
              f"partial|n_s rho={e['partial_spearman_r_given_ns']:+.3f}")
    print("\n|gap| vs survivor-AUROC std (n_s>=3):")
    show(out["k5_all_models"]); show(out["k5_gpt2xl"]); show(out["k10_gpt2xl"])
    print("  --- per model (k=5) ---")
    for m in out["k5_by_model"]:
        show(out["k5_by_model"][m])
    mv = out["survivor_auc_variance"]
    print(f"\nsurvivor-AUROC std magnitude (k=5, n_s>=2, n={mv['k5_all_models']['n_neurons_ns_ge_2']}): "
          f"mean={mv['k5_all_models']['mean']} median={mv['k5_all_models']['median']} "
          f"IQR[{mv['k5_all_models']['p25']},{mv['k5_all_models']['p75']}] "
          f"near-homogeneous(<0.02)={mv['k5_all_models']['frac_near_homogeneous_std_lt_0p02']:.0%}")
    k = mv["matched_k5_vs_k10_gpt2xl"]
    print(f"matched gpt2-xl k5->k10 std: {k['mean_std_k5']}->{k['mean_std_k10']} "
          f"(delta={k['delta_k10_minus_k5']:+}, Wilcoxon p={k['wilcoxon_p']}); n_s {k['mean_ns_k5']}->{k['mean_ns_k10']}")
    print(f"\nSaved to {RESULTS / 'weighting_gap_auroc_variance.json'}")


if __name__ == "__main__":
    main()
