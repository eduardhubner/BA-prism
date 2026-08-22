"""
Cross-method pooling — Stage 2 (filter + metrics).

Runs the canonical pipeline on the pooled descriptions of Llama-3.1-8B /
gemma-scope-2b / gpt2-small-sae and quantifies method generality: whether
descriptions from methods other than input-centric (PRISM) are treated fairly and
whether they duplicate or extend the input-centric concepts.

Consumes results/pooled_outputcentric/{descriptions.csv, embeddings.npy}. Pipeline
identical to pool_gpt2xl_analyze (imported): AUROC filter > 0.50, then cosine-collapse
> 0.70 keeping the highest-AUROC representative.

Each non-input-centric description (output-centric on all three targets; neuronpedia
on gpt2-small-sae) is assigned a fate relative to the input-centric set:
  filtered   — removed by the AUROC filter (unreliable);
  converges  — passes the filter and lands in a collapsed component that also
               contains an input-centric description (same concept);
  distinct   — passes the filter but forms its own component (a concept the
               input-centric set did not cover).

RUN (default venv):
    python src/pool_outputcentric_analyze.py

Outputs (results/pooled_outputcentric/):
    profile.csv   per feature: metrics (n_s, Q, sigma_tilde, diversity, ...) plus a
                  {method}_outcome / {method}_auroc pair for each external method.
    stats.json    per-model and pooled: survival rate by method, and fate breakdown
                  for each external method.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from pool_gpt2xl_analyze import filter_pool, metrics   # canonical filter + metrics

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results" / "pooled_outputcentric"

MODELS = ["Llama-3.1-8B-Instruct", "gemma-scope-2b", "gpt2-small-sae"]
METHOD_ORDER = ["input-centric", "output-centric", "neuronpedia"]
BASELINE = "input-centric"


def analyse(desc, emb):
    aurocs_all = desc["auroc"].to_numpy()
    rows = []
    surv = {}   # (model, method) -> [n_survived, n_total]
    fate = {}   # (model, method) -> {filtered, converges, distinct}   (external methods only)
    for (model, layer, unit), g in desc.groupby(["model", "layer", "unit"]):
        idx = g.index.to_numpy()
        a = aurocs_all[idx]
        E = emb[idx]
        sim = E @ E.T
        dec, comps = filter_pool(a, sim)
        m, _ = metrics(a, sim, dec)
        methods = g["method"].to_numpy()

        for k in range(len(idx)):                       # survival tally, per method
            s = surv.setdefault((model, methods[k]), [0, 0])
            s[1] += 1
            if dec[k] == "survived":
                s[0] += 1

        row = {"model": model, "layer": int(layer), "unit": int(unit), **m}
        for k in range(len(idx)):                       # fate of each external description
            if methods[k] == BASELINE:
                continue
            if dec[k] == "removed":
                outcome = "filtered"
            else:
                comp = next((c for c in comps if k in c), [k])
                outcome = "converges" if any(methods[c] == BASELINE for c in comp) else "distinct"
            fate.setdefault((model, methods[k]), {"filtered": 0, "converges": 0, "distinct": 0})[outcome] += 1
            row[f"{methods[k]}_outcome"] = outcome
            row[f"{methods[k]}_auroc"] = float(a[k])
        rows.append(row)
    return pd.DataFrame(rows), surv, fate


def main():
    desc = pd.read_csv(OUT / "descriptions.csv")
    emb = np.load(OUT / "embeddings.npy")
    profile, surv, fate = analyse(desc, emb)
    profile.to_csv(OUT / "profile.csv", index=False)

    methods = [x for x in METHOD_ORDER if x in set(desc["method"])]
    external = [x for x in methods if x != BASELINE]

    def surv_agg(pairs):
        n_s = n_t = 0
        for key in pairs:
            s = surv.get(key)
            if s:
                n_s += s[0]; n_t += s[1]
        return None if n_t == 0 else {"n_survived": n_s, "n_total": n_t, "rate": round(n_s / n_t, 3)}

    def fate_agg(pairs):
        agg = {"filtered": 0, "converges": 0, "distinct": 0}
        for key in pairs:
            for k, v in fate.get(key, {}).items():
                agg[k] += v
        tot = sum(agg.values())
        if tot == 0:
            return None
        reliable = agg["converges"] + agg["distinct"]
        return {**agg, "n": tot,
                "pct_filtered": round(agg["filtered"] / tot * 100, 1),
                "pct_converges_of_reliable": round(agg["converges"] / reliable * 100, 1) if reliable else None,
                "pct_distinct_of_reliable": round(agg["distinct"] / reliable * 100, 1) if reliable else None}

    def present(model, method):
        return (model, method) in surv

    stats = {
        "params": {"auroc_threshold": 0.50, "sim_threshold": 0.70,
                   "n_features": int(profile.shape[0]), "n_descriptions": int(len(desc))},
        "methods": methods,
        "per_model": {
            md: {
                "survival": {meth: surv_agg([(md, meth)]) for meth in methods if present(md, meth)},
                "fate": {meth: fate_agg([(md, meth)]) for meth in external if present(md, meth)},
            } for md in MODELS
        },
        "pooled": {
            "survival": {meth: surv_agg([(md, meth) for md in MODELS]) for meth in methods},
            "fate": {meth: fate_agg([(md, meth) for md in MODELS]) for meth in external},
        },
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"Cross-method pooling ({stats['params']['n_features']} features, "
          f"{stats['params']['n_descriptions']} descriptions; methods: {', '.join(methods)})")
    for md in MODELS:
        b = stats["per_model"][md]
        surv_s = "  ".join(f"{meth.split('-')[0]}={b['survival'][meth]['rate']}"
                           for meth in methods if meth in b["survival"])
        print(f"  {md:24s} survival[{surv_s}]")
        for meth in external:
            ft = b["fate"].get(meth)
            if ft:
                print(f"      {meth:15s} filtered {ft['pct_filtered']}% | "
                      f"converges {ft['pct_converges_of_reliable']}% / distinct {ft['pct_distinct_of_reliable']}% (of reliable, n={ft['n']})")
    for meth in external:
        ft = stats["pooled"]["fate"][meth]
        print(f"  POOLED {meth:15s} of {ft['n']}: filtered {ft['pct_filtered']}%, "
              f"converges {ft['pct_converges_of_reliable']}% / distinct {ft['pct_distinct_of_reliable']}% (of reliable)")
    print(f"\nSaved → {OUT}/profile.csv, stats.json")


if __name__ == "__main__":
    main()
