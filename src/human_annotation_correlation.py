"""
Human-annotation correspondence check.

Validates PRISM's score against human judgment, using the human study
described in the PRISM paper (Sec. 4 / Appendix A.8): 7 participants (PhD/
working students, no co-authors), 8 feature instances, 5 clusters of highly-
activating sentences per instance. Participants wrote a description per
cluster and rated pairwise similarity among their own 5 descriptions on an
11-point scale (0.0-1.0, where 1.0 = very high similarity = monosemantic).
The paper reports the average of these ratings as the "Human polysemanticity
score" -- despite the name, HIGH values mean monosemantic/coherent and LOW
values mean polysemantic/diverse, same direction as PRISM's own score.

Data: data/human_annotations/user_study_clean.csv stores this as an integer
0-10 scale (raw x10 of the paper's 0.0-1.0); divided by 10 here. Verified
against the paper's cited examples: GPT2-small-sae L5/U30319 reproduces
PRISM score 0.30 here near-exactly (0.3023), and human scores reproduce the
cited 0.40/0.90/0.80 closely (within ~0.01-0.03, likely because this "clean"
CSV underwent minor filtering relative to what produced the published figure).

"PRISM score" here is the plain mean pairwise similarity (prism_score =
mean_similarity in pairwise_similarities_dataset.csv) -- confirmed against
the paper's cited numbers to be the correct quantity, NOT diversity_score
(=1-weighted_mean_sim, which is inverted relative to how the paper defines
"PRISM score" in this comparison).

Two metrics are reported:
  1. Original PRISM score (prism_score) vs. human score -- defined for all 8
     neurons, n=8.
  2. Filtered metrics (mean_similarity_post, weighted_mean_similarity, from
     the same two-stage filter as filtered_evaluation.py) vs. human score --
     only defined where n_surviving >= 2, at the project-standard
     SIM_THRESHOLD=0.70. This excludes 4 of the 8 neurons: gpt2-small-sae
     L5/U30319 (n_surviving=0, the "all-fail" case -- every description's AUROC
     is below threshold) and three single-survivor neurons (Llama L0/U10095,
     gemma-scope-2b L10/U603, gpt2-small-sae L10/U9896). These exclusions are
     themselves part of the result: the framework declines to characterize
     these neurons rather than reporting a misleading similarity value.

IMPORTANT framing note: with n=4 surviving neurons, a high correlation
(Spearman/Pearson) is close to statistically content-free -- do not lead with
"rho improved from X to Y." The honest claim is narrower: on the neurons
where the filtered metric is even defined, it tracks human judgment at least
as well as the unfiltered PRISM score, and the neurons it excludes are
excluded for principled reasons (zero or single survivor), not arbitrarily.

Robustness check: one of the 4 excluded neurons (gemma-scope-2b L10/U603) is
a SIM_THRESHOLD boundary case -- sim(desc1,desc2)=0.7097, barely above the
project-standard tau=0.70, which collapses all 5 descriptions into one
component (n_surviving=1). To check whether the filtered-metric result
depends on this borderline case, the same correlation is recomputed at a
nearby threshold (tau=0.72), which flips L10/U603 to n_surviving=2 (and also
flips gemma-scope-2b L0/U10531 from 2 to 3 survivors -- this is a genuine
multi-neuron perturbation, not an isolated single-point change). This is a
sensitivity check, not a threshold search: tau=0.70 remains the reported
result; tau=0.72 is reported alongside it to show whether the result
survives the perturbation, not to find a threshold that gives a "prettier" n.

Outputs:
  results/human_annotation_correlation.csv            -- per-neuron paired
                                                           table (human score,
                                                           PRISM score)
  results/human_annotation_correlation_filtered.csv    -- per-neuron filtered
                                                           metrics at tau=0.70
                                                           and tau=0.72
  results/human_annotation_correlation.json            -- correlation summary
                                                           (all of the above)

Run with: python src/human_annotation_correlation.py
"""

import json
import numpy as np

import verification as V
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).parent.parent
ANNOTATIONS_PATH = ROOT / "data" / "human_annotations" / "user_study_clean.csv"
PAIRWISE_PATH = ROOT / "results" / "pairwise_similarities_dataset.csv"
OUT_CSV = ROOT / "results" / "human_annotation_correlation.csv"
OUT_FILTERED_CSV = ROOT / "results" / "human_annotation_correlation_filtered.csv"
OUT_JSON = ROOT / "results" / "human_annotation_correlation.json"

SIM_PAIRS = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
AUROC_THRESHOLD = 0.50
TAU_STANDARD = 0.70
TAU_ROBUSTNESS = 0.72


def build_sim_matrix(row):
    sim = np.zeros((5, 5))
    for i, j in SIM_PAIRS:
        v = row[f"sim_{i}{j}"]
        sim[i - 1, j - 1] = v
        sim[j - 1, i - 1] = v
    return sim


def components_collapse(indices, sim_matrix, threshold, aurocs):
    """Connected-components collapse. Delegates to `verification`.

    NOTE: this script applies the collapse BEFORE the AUROC filter, which is the
    non-canonical stage order (see verification.collapse_then_filter). Kept
    as-is so results are unchanged; see ARTIFACT_PLAN.md for the open decision.
    """
    return V.collapse_components(indices, sim_matrix, aurocs, sim_threshold=threshold)


def filtered_metrics_at_threshold(row, sim_threshold):
    sim = build_sim_matrix(row)
    aurocs = np.array([row[f"auroc_{i}"] for i in range(1, 6)])
    # Canonical stage order (AUROC filter, then collapse). Verified a no-op here:
    # the two orders agree on 0/239 neurons at both tau=0.70 and tau=0.72.
    stage2 = set(V.filter_and_collapse(aurocs, sim,
                                       sim_threshold=sim_threshold,
                                       auroc_threshold=AUROC_THRESHOLD).surviving)
    n_surviving = len(stage2)

    if n_surviving == 0:
        return dict(n_surviving=0, mean_sim_post=np.nan, weighted_mean_sim=np.nan,
                    exclusion_reason="uncharacterized (0 survivors, all-fail)")
    if n_surviving == 1:
        return dict(n_surviving=1, mean_sim_post=np.nan, weighted_mean_sim=np.nan,
                    exclusion_reason="single survivor (n_surviving=1, undefined pairwise similarity)")

    pairs = list(combinations(stage2, 2))
    mean_sim_post = float(np.mean([sim[i, j] for i, j in pairs]))
    num, den = 0.0, 0.0
    for i, j in pairs:
        w = ((aurocs[i] - 0.5) / 0.5) * ((aurocs[j] - 0.5) / 0.5)
        num += w * sim[i, j]
        den += w
    weighted_mean_sim = float(num / den) if den > 0 else np.nan
    return dict(n_surviving=n_surviving, mean_sim_post=mean_sim_post,
                weighted_mean_sim=weighted_mean_sim, exclusion_reason=None)


def correlate_filtered(human, pair_df, sim_threshold):
    rows = []
    for _, n in human[["model", "layer", "unit"]].iterrows():
        row = pair_df[(pair_df["model"] == n["model"]) & (pair_df["layer"] == n["layer"])
                       & (pair_df["unit"] == n["unit"])].iloc[0]
        m = filtered_metrics_at_threshold(row, sim_threshold)
        rows.append({"model": n["model"], "layer": n["layer"], "unit": n["unit"],
                      "sim_threshold": sim_threshold, **m})
    df = pd.DataFrame(rows).merge(human[["model", "layer", "unit", "human_poly_score"]],
                                   on=["model", "layer", "unit"])
    valid = df[df["n_surviving"] >= 2]
    result = {"sim_threshold": sim_threshold, "n_valid": int(len(valid)),
              "n_excluded": int(len(df) - len(valid))}
    if len(valid) >= 3:
        s1 = spearmanr(valid["human_poly_score"], valid["mean_sim_post"])
        s2 = spearmanr(valid["human_poly_score"], valid["weighted_mean_sim"])
        result["mean_similarity_post"] = {"spearman_r": round(float(s1.statistic), 4),
                                           "spearman_p": round(float(s1.pvalue), 4)}
        result["weighted_mean_similarity"] = {"spearman_r": round(float(s2.statistic), 4),
                                               "spearman_p": round(float(s2.pvalue), 4)}
    return df, result


def main():
    ann = pd.read_csv(ANNOTATIONS_PATH).rename(columns={"model_name": "model"})

    # human_poly_score is constant across the 5 cluster_id rows within each
    # (neuron, annotator) -- it is one neuron-level judgment per annotator,
    # not a per-cluster rating. Dedupe before averaging so n reflects the 7
    # independent annotators, not 7 x 5 duplicated cluster rows.
    per_annotator = ann.drop_duplicates(subset=["model", "layer", "unit", "user_name"])
    n_annotators = per_annotator.groupby(["model", "layer", "unit"])["user_name"].count()
    human = (per_annotator.groupby(["model", "layer", "unit"])["human_poly_score"]
                .agg(["mean", "std"])
                .rename(columns={"mean": "human_poly_score_raw_mean", "std": "human_poly_score_raw_std"})
                .reset_index())
    human["n_annotators"] = n_annotators.values
    human["human_poly_score"] = human["human_poly_score_raw_mean"] / 10.0  # raw 0-10 -> paper's 0.0-1.0

    pair_df = pd.read_csv(PAIRWISE_PATH)
    pair = pair_df[["model", "layer", "unit", "mean_similarity"]].rename(columns={"mean_similarity": "prism_score"})

    merged = human.merge(pair, on=["model", "layer", "unit"], how="left")
    missing = merged[merged["prism_score"].isna()]
    if len(missing) > 0:
        print(f"WARNING: {len(missing)} annotated neurons have no matching PRISM score:")
        print(missing[["model", "layer", "unit"]])
    merged = merged.dropna(subset=["prism_score"])

    merged.to_csv(OUT_CSV, index=False)
    print(f"Saved per-neuron paired table to {OUT_CSV}\n")
    print(merged[["model", "layer", "unit", "n_annotators",
                  "human_poly_score", "prism_score"]].to_string(index=False))

    n = len(merged)
    spear = spearmanr(merged["human_poly_score"], merged["prism_score"])
    pear = pearsonr(merged["human_poly_score"], merged["prism_score"])

    # ---- Filtered metrics at the standard threshold + robustness check ----
    df_std, result_std = correlate_filtered(human, pair_df, TAU_STANDARD)
    df_robust, result_robust = correlate_filtered(human, pair_df, TAU_ROBUSTNESS)

    filtered_detail = pd.concat([df_std, df_robust], ignore_index=True)
    filtered_detail.to_csv(OUT_FILTERED_CSV, index=False)
    print(f"\nSaved filtered-metric detail (tau={TAU_STANDARD} and tau={TAU_ROBUSTNESS}) to {OUT_FILTERED_CSV}\n")
    print(f"--- tau={TAU_STANDARD} (standard, reported result) ---")
    print(df_std[["model", "layer", "unit", "n_surviving", "exclusion_reason"]].to_string(index=False))
    print(f"\n--- tau={TAU_ROBUSTNESS} (robustness perturbation) ---")
    print(df_robust[["model", "layer", "unit", "n_surviving", "exclusion_reason"]].to_string(index=False))

    summary = {
        "original_prism_score": {
            "n_neurons": n,
            "spearman_r": round(float(spear.statistic), 4),
            "spearman_p": round(float(spear.pvalue), 4),
            "pearson_r": round(float(pear.statistic), 4),
            "pearson_p": round(float(pear.pvalue), 4),
        },
        "filtered_metrics_standard_tau_0_70": result_std,
        "filtered_metrics_robustness_tau_0_72": result_robust,
        "robustness_note": "tau=0.72 is a sensitivity check, not a threshold search -- tau=0.70 "
                            "is the reported result. The perturbation flips gemma-scope-2b L10/U603 "
                            "(boundary case, sim(desc1,desc2)=0.7097 vs tau=0.70) AND gemma-scope-2b "
                            "L0/U10531 (2->3 survivors) -- a genuine multi-neuron perturbation, not "
                            "an isolated single-point change. Correlation moves from rho=1.00 (n=4) "
                            "to rho=0.90 (n=5, p=0.037): the result is not an artifact of the "
                            "borderline neuron's exact classification.",
        "n_annotators_per_neuron": 7,
        "provenance": "PRISM paper's own human study (Sec. 4 / Appendix A.8): 7 participants "
                      "(PhD/working students, no co-authors), 8 instances, 5 clusters each. "
                      "Verified against paper-cited example values (see module docstring).",
        "human_score_scale": "11-point, 0.0-1.0 (paper's pairwise-similarity rating average); "
                              "stored locally as integer 0-10, divided by 10 here. "
                              "HIGH = monosemantic/coherent, LOW = polysemantic/diverse.",
        "prism_score_used": "prism_score = mean_similarity (plain mean pairwise cosine similarity, "
                             "pre-filter) from pairwise_similarities_dataset.csv -- same direction "
                             "as the human score (HIGH = monosemantic). NOT diversity_score.",
        "interpretation": "Do NOT lead with 'rho improved from 0.826 to 1.00' -- with n=4 surviving "
                           "neurons the filtered correlation is close to statistically content-free "
                           "(huge CI). The supportable claim: on neurons where the filtered metric is "
                           "defined, it tracks human judgment at least as well as the unfiltered PRISM "
                           "score, AND the framework's exclusions are principled (0 or 1 survivors), "
                           "not arbitrary -- gpt2-small-sae L5/U30319 is the all-fail/uncharacterized "
                           "case, the other 3 exclusions are single-survivor neurons. The flagging "
                           "behavior is the result; the near-perfect correlation is a too-small-n "
                           "curiosity reported without leaning on it.",
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {OUT_JSON}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
