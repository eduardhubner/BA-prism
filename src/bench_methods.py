"""
Cross-method reliability benchmark (both evaluation regimes, in parallel).

Compares feature-description methods by the framework's own reliability gate — the
fraction of a method's descriptions that clear the AUROC filter (AUROC > 0.50) — and
by mean AUROC. Methods are grouped by CoSy evaluation regime and never mixed within a
score: the old regime (evaluation samples from gemini-1.5-pro) and the post-rebuttal
regime (gemini-2.0-flash-lite) are shown side by side, following the parent PRISM
benchmark's treatment of gemini-2.0-flash-lite as the closest-available successor.

This is an aggregate per-method comparison (one reliability rate per method/target),
so a regime offset shifts a method uniformly rather than distorting a within-feature
weighting. Methods cover different targets; coverage is reported per bar.

Old-regime AUROCs are read from the pooled artifacts (results/pooled_*/descriptions.csv);
new-regime AUROCs from the raw CoSy evaluation CSVs. NB the Transluce file is named
"...flash..." but its evaluation samples are gemini-2.0-flash-lite (upstream naming
error), so it is assigned to the flash-lite regime.

RUN:  python src/bench_methods.py
Outputs: results/method_benchmark/{stats.json}, figures/fig_method_benchmark.{png,pdf,svg}
"""

import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import thesis_style as ts

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "method_benchmark"
OUT.mkdir(parents=True, exist_ok=True)

AUC_THR = 0.50
OLD, NEW = "gemini-1.5-pro", "gemini-2.0-flash-lite"
TARGETS = ["Llama-3.1-8B-Instruct", "gemma-scope-2b", "gpt2-small-sae", "gpt2-xl"]
TARGET_SHORT = {"Llama-3.1-8B-Instruct": "Llama", "gemma-scope-2b": "Gemma",
                "gpt2-small-sae": "GPT-2 SAE", "gpt2-xl": "GPT-2 XL"}
METHOD_LABEL = {"input-centric": "PRISM", "output-centric": "output-centric",
                "GPT-explain": "GPT-explain", "neuronpedia": "Neuronpedia",
                "Transluce": "Transluce", "max_act": "max-act"}
METHOD_ORDER = ["input-centric", "output-centric", "GPT-explain", "neuronpedia", "Transluce", "max_act"]
REGIME_COLOR = {OLD: "#4c9f70", NEW: "#d98a3d"}

_NEWFILES = {
    ("Transluce", "Llama-3.1-8B-Instruct"):
        "cosy-evaluation_Transluce_target-Llama-3.1-8B-Instruct_textgen-gemini-2-0-flash_mean_evalgen-gemini-2-0-flash_cosmopedia_1000.csv",
    ("max_act", "Llama-3.1-8B-Instruct"):
        "cosy-evaluation_max_act_10k_target-Llama-3.1-8B-Instruct_textgen-gemini-2-0-flash-lite_mean_evalgen-gemini-2-0-flash-lite_cosmopedia_1000.csv",
    ("max_act", "gemma-scope-2b"):
        "cosy-evaluation_max_act_10k_target-gemma-scope-2b_textgen-gemini-2-0-flash-lite_mean_evalgen-gemini-2-0-flash-lite_cosmopedia_1000.csv",
    ("max_act", "gpt2-small-sae"):
        "cosy-evaluation_max_act_10k_target-gpt2-small-sae_textgen-gemini-2-0-flash-lite_mean_evalgen-gemini-2-0-flash-lite_cosmopedia_1000.csv",
    ("max_act", "gpt2-xl"):
        "cosy-evaluation_max_act_10k_target-gpt2-xl_textgen-gemini-2-0-flash-lite_mean_evalgen-gemini-2-0-flash-lite_cosmopedia_1000.csv",
}


def _assemble():
    """Long frame: one row per description with method, target, regime, auc."""
    rows = []

    # old regime: reuse pooled artifacts (already gemini-1.5-pro).
    oc = pd.read_csv(RESULTS / "pooled_outputcentric" / "descriptions.csv")
    for r in oc.itertuples():
        rows.append((r.method, r.model, OLD, float(r.auc)))
    gx = pd.read_csv(RESULTS / "pooled_gpt2xl" / "descriptions.csv")
    for r in gx.itertuples():
        rows.append((r.method, "gpt2-xl", OLD, float(r.auc)))

    # new regime: raw CoSy CSVs (gemini-2.0-flash-lite).
    for (method, target), fname in _NEWFILES.items():
        d = pd.read_csv(RESULTS / fname)
        for auc in d["AUC"].to_numpy():
            rows.append((method, target, NEW, float(auc)))

    return pd.DataFrame(rows, columns=["method", "target", "regime", "auc"])


def _summarise(df):
    g = df.groupby(["method", "target", "regime"])["auc"]
    out = g.agg(n="size", mean_auc="mean",
                reliability=lambda a: float((a > AUC_THR).mean())).reset_index()
    out["mean_auc"] = out["mean_auc"].round(3)
    out["reliability"] = out["reliability"].round(3)
    return out


def _figure(summary):
    ts.apply()
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    xticks, xlabels, bar_i = [], [], 0
    gap = 1.0
    for target in TARGETS:
        sub = summary[summary["target"] == target]
        sub = sub.set_index("method").reindex([m for m in METHOD_ORDER if m in set(sub["method"])]).reset_index()
        start = bar_i
        for r in sub.itertuples():
            ax.bar(bar_i, 100 * r.reliability, width=0.85, color=REGIME_COLOR[r.regime],
                   edgecolor="black", linewidth=0.5)
            ax.text(bar_i, 100 * r.reliability + 1.5, METHOD_LABEL[r.method], rotation=90,
                    ha="center", va="bottom", fontsize=7, color="#333")
            ax.text(bar_i, 2, f"n={r.n}", rotation=90, ha="center", va="bottom", fontsize=6, color="white")
            bar_i += 1
        xticks.append((start + bar_i - 1) / 2); xlabels.append(TARGET_SHORT[target])
        bar_i += gap
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_ylabel(f"reliability: descriptions with AUROC > {AUC_THR:.2f}  (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Method reliability under the framework, by evaluation regime", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=REGIME_COLOR[OLD]),
               plt.Rectangle((0, 0), 1, 1, color=REGIME_COLOR[NEW])]
    ax.legend(handles, [f"old regime ({OLD})", f"post-rebuttal ({NEW})"],
              loc="upper right", fontsize=8, frameon=True, framealpha=0.95)
    ts.despine(ax)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.10)
    ts.save(fig, "fig_method_benchmark")
    plt.close(fig)


def main():
    df = _assemble()
    summary = _summarise(df)
    summary.to_csv(OUT / "method_reliability.csv", index=False)
    stats = {
        "auc_threshold": AUC_THR,
        "regimes": {"old": OLD, "new_post_rebuttal": NEW},
        "note": "aggregate per-method reliability (AUROC>thr); regimes shown in parallel, never mixed within a score",
        "by_method_target": summary.to_dict(orient="records"),
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    _figure(summary)

    print(f"Method reliability benchmark ({len(df)} descriptions):")
    for target in TARGETS:
        print(f"  {TARGET_SHORT[target]}")
        for r in summary[summary["target"] == target].itertuples():
            print(f"    {METHOD_LABEL[r.method]:14s} [{r.regime:22s}] "
                  f"reliability {100*r.reliability:5.1f}%  mean AUROC {r.mean_auc:.3f}  (n={r.n})")
    print(f"\nSaved → {OUT}/stats.json, method_reliability.csv; figures/fig_method_benchmark.*")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
