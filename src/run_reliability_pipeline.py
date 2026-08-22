"""
Reliability-aware profiling pipeline — single operational entry point.

This is the one command that turns raw PRISM outputs into the reliability-aware
per-neuron profile that this thesis proposes. It is a thin orchestrator: the
actual filter/collapse/metric logic lives in `case_study_analysis` (the single
source of truth) and is reused here, so this script never re-implements the
pipeline — it just runs it end to end and writes the profile.

Pipeline (per neuron):
  input: k candidate descriptions, each with an AUROC, plus their pairwise
         cosine similarities (the PRISM output)
    │
    ├─ Stage 1  AUROC filter   — drop descriptions with AUROC ≤ τ_auroc (0.50)
    ├─ Stage 2  similarity collapse — among survivors, collapse connected
    │            components at cosine sim > τ_sim (0.70); keep the highest-AUROC
    │            description per component
    └─ Metrics — n_filtered / n_collapsed / n_surviving, evidence quality Q,
                 weighted mean similarity σ̃, diversity 1−σ̃, characterisation status
    │
  output: one profile row per neuron.

Input contract (CSV; one row per neuron): columns `model, layer, unit`,
`auroc_1..auroc_k`, and pairwise `sim_ij` (e.g. `sim_12`). Defaults to the PRISM
master table `results/pairwise_similarities_dataset.csv`.

Usage:
    python src/run_reliability_pipeline.py
    python src/run_reliability_pipeline.py --input results/pairwise_similarities_dataset.csv \\
        --output results/reliability_profile.csv --sim-threshold 0.70 --auroc-threshold 0.50

Note: "AUROC" and "AUC" denote the same metric; the input columns keep the
legacy `auroc_` spelling (see the terminology note in RESULTS_INDEX.md).
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import case_study_analysis as cs

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"


def status_of(n_s):
    return "uncharacterised" if n_s == 0 else ("single" if n_s == 1 else "multi")


def render_mds(df, specs):
    """Render the framework MDS plot for each `model:layer:unit` spec, reusing the
    canonical renderer in case_study_analysis (saved under results/case_study/)."""
    for spec in specs:
        try:
            model, layer, unit = spec.rsplit(":", 2)
            layer, unit = int(layer), int(unit)
        except ValueError:
            print(f"  (bad --mds spec '{spec}', expected model:layer:unit)"); continue
        sub = df[(df["model"] == model) & (df["layer"] == layer) & (df["unit"] == unit)]
        if sub.empty:
            print(f"  (--mds: neuron not found: {spec})"); continue
        row = sub.iloc[0]
        aurocs = np.array([row[f"auroc_{i}"] for i in range(1, cs.N_DESC + 1)], dtype=float)
        descs = [row[f"description_{i}"] for i in range(1, cs.N_DESC + 1)]
        sm = cs.get_sim_matrix(row)
        dec = cs.filter_decisions(aurocs, sm)
        met = cs.compute_metrics(aurocs, sm, dec)
        cs.make_mds_plot(aurocs, sm, dec, descs, met, model, layer, unit)
        print(f"  MDS plot rendered → {cs.OUTPUT_DIR}/  ({model} L{layer}/U{unit})")


def profile_neuron(row):
    """Run the two-stage filter + metrics on one neuron; return a profile dict."""
    aurocs = np.array([row[f"auroc_{i}"] for i in range(1, cs.N_DESC + 1)], dtype=float)
    sm = cs.get_sim_matrix(row)
    dec = cs.filter_decisions(aurocs, sm)
    met = cs.compute_metrics(aurocs, sm, dec)

    n_s = met["n_surviving"]
    n_coll = dec.count("collapsed")
    n_filt = dec.count("removed (low AUROC)")
    # diversity only where a spread among ≥2 survivors is defined (σ̃=1 at n_s=1 is
    # a convention, not a measured spread — see RESULTS_INDEX terminology/σ̃ note).
    div = (1.0 - met["weighted_mean_sim"]) if n_s >= 2 else np.nan

    return {
        "model": row["model"], "layer": int(row["layer"]), "unit": int(row["unit"]),
        "n_descriptions": int(np.sum(~np.isnan(aurocs))),
        "n_filtered": n_filt, "n_collapsed": n_coll, "n_surviving": n_s,
        "status": status_of(n_s),
        "prism_score": round(met["prism_score"], 4),
        "evidence_quality": round(met["evidence_quality"], 4) if not np.isnan(met["evidence_quality"]) else np.nan,
        "weighted_mean_similarity": round(met["weighted_mean_sim"], 4) if not np.isnan(met["weighted_mean_sim"]) else np.nan,
        "diversity": round(div, 4) if not np.isnan(div) else np.nan,
        "mean_similarity_post": round(met["mean_sim_post"], 4) if not np.isnan(met["mean_sim_post"]) else np.nan,
        "effective_n": round(met["effective_n"], 4) if not np.isnan(met["effective_n"]) else np.nan,
        "uncharacterised": met["uncharacterized"],
    }


def main():
    ap = argparse.ArgumentParser(description="Run the reliability-aware profiling pipeline.")
    ap.add_argument("--input", default=str(cs.PRISM_PATH), help="PRISM output CSV (one row per neuron).")
    ap.add_argument("--output", default=str(RESULTS / "reliability_profile.csv"), help="Profile CSV to write.")
    ap.add_argument("--sim-threshold", type=float, default=cs.SIM_THRESHOLD, help="Collapse threshold τ_sim (default 0.70).")
    ap.add_argument("--auroc-threshold", type=float, default=cs.AUROC_THRESHOLD, help="Reliability filter τ_auroc (default 0.50).")
    ap.add_argument("--mds", nargs="*", metavar="model:layer:unit", default=[],
                    help="Also render the framework MDS plot for these neurons (repeatable).")
    args = ap.parse_args()

    # Apply thresholds to the canonical module (filter_decisions reads these globals).
    cs.SIM_THRESHOLD = args.sim_threshold
    cs.AUROC_THRESHOLD = args.auroc_threshold

    df = pd.read_csv(args.input)
    profile = pd.DataFrame([profile_neuron(r) for _, r in df.iterrows()])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(args.output, index=False)

    # Population summary (also written next to the profile as JSON).
    n = len(profile)
    counts = profile["status"].value_counts().to_dict()
    summary = {
        "input": args.input, "output": args.output,
        "tau_auroc": args.auroc_threshold, "tau_sim": args.sim_threshold,
        "n_neurons": n,
        "status_pct": {k: round(counts.get(k, 0) / n * 100, 1) for k in ("uncharacterised", "single", "multi")},
        "median_n_surviving": float(profile["n_surviving"].median()),
        "median_Q_ns_ge_1": round(float(profile.loc[profile.n_surviving >= 1, "evidence_quality"].median()), 4),
        "median_diversity_ns_ge_2": round(float(profile.loc[profile.n_surviving >= 2, "diversity"].median()), 4),
    }
    Path(args.output).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))

    print("Reliability-aware profiling pipeline")
    print(f"  input : {args.input}")
    print(f"  τ_auroc = {args.auroc_threshold}   τ_sim = {args.sim_threshold}   ({n} neurons)")
    print(f"  Stage 1 (AUROC filter) + Stage 2 (similarity collapse) → profile")
    sp = summary["status_pct"]
    print(f"  characterisation: uncharacterised {sp['uncharacterised']}%  "
          f"single {sp['single']}%  multi {sp['multi']}%")
    print(f"  median n_s={summary['median_n_surviving']:.0f}  "
          f"median Q={summary['median_Q_ns_ge_1']} (n_s≥1)  "
          f"median diversity={summary['median_diversity_ns_ge_2']} (n_s≥2)")
    print(f"  → {args.output}")
    print(f"  → {Path(args.output).with_suffix('.summary.json')}")

    if args.mds:
        print("  rendering MDS plots:")
        render_mds(df, args.mds)


if __name__ == "__main__":
    main()
