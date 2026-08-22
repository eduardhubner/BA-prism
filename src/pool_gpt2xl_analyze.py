"""
Cross-method pooling — Stage 2: run the reliability filter + metrics on the
pooled gpt2-xl descriptions, method-agnostically, and quantify method generality.

Consumes results/pooled_gpt2xl/{descriptions.csv, embeddings.npy} from Stage 1.
Uses the canonical pipeline unchanged: AUROC filter (> 0.50) FIRST, then similarity
collapse (cosine > 0.70) via connected components keeping the highest-AUROC
representative. Metrics match the framework exactly (Q = 2·mean(survivor AUROC)−1;
σ̃ = AUROC-weighted mean pairwise similarity among survivors, weight
((a_i−0.5)/0.5)((a_j−0.5)/0.5); diversity = 1 − σ̃).

RUN (default venv):
    python src/pool_gpt2xl_analyze.py

Outputs (results/pooled_gpt2xl/):
    profile.csv   per feature: n_desc, n_filt, n_coll, n_s, Q, sigma_tilde,
                  diversity, uncharacterised, + survivor source composition
    stats.json    cross-source collapse rate, survival-by-source, and consistency
                  of Q/diversity between PRISM-only and the pooled set
"""

import json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results" / "pooled_gpt2xl"
AUROC_THRESHOLD = 0.50   # matches case_study_analysis
SIM_THRESHOLD = 0.70


def filter_pool(aurocs, sim):
    """Filter-first: AUROC filter, then connected-component collapse (highest-AUROC
    representative). Returns per-description decisions and the kept components."""
    n = len(aurocs)
    kept = [i for i in range(n) if aurocs[i] > AUROC_THRESHOLD]
    adj = {i: set() for i in kept}
    for a in range(len(kept)):
        for b in range(a + 1, len(kept)):
            i, j = kept[a], kept[b]
            if sim[i, j] > SIM_THRESHOLD:
                adj[i].add(j); adj[j].add(i)
    visited, comps = set(), []
    for s in kept:
        if s not in visited:
            comp, stack = [], [s]
            while stack:
                x = stack.pop()
                if x not in visited:
                    visited.add(x); comp.append(x); stack.extend(adj[x] - visited)
            comps.append(comp)
    reps = {max(c, key=lambda i: aurocs[i]) for c in comps}
    dec = ["removed" if aurocs[i] <= AUROC_THRESHOLD else ("survived" if i in reps else "collapsed")
           for i in range(n)]
    return dec, comps


def metrics(aurocs, sim, dec):
    surv = [i for i, d in enumerate(dec) if d == "survived"]
    n_s = len(surv)
    out = {"n_desc": len(aurocs), "n_filt": dec.count("removed"),
           "n_coll": dec.count("collapsed"), "n_s": n_s}
    if n_s == 0:
        out.update(Q=np.nan, sigma_tilde=np.nan, diversity=np.nan, uncharacterised=True)
        return out, surv
    sa = [aurocs[i] for i in surv]
    out["Q"] = float(2 * np.mean(sa) - 1)
    if n_s == 1:
        out.update(sigma_tilde=1.0, diversity=np.nan, uncharacterised=False)
        return out, surv
    num = den = 0.0
    for i, j in combinations(surv, 2):
        w = ((aurocs[i] - 0.5) / 0.5) * ((aurocs[j] - 0.5) / 0.5)
        num += w * sim[i, j]; den += w
    st = float(num / den) if den > 0 else np.nan
    out.update(sigma_tilde=st, diversity=(1 - st) if not np.isnan(st) else np.nan, uncharacterised=False)
    return out, surv


def profile_subset(sub_idx, aurocs, emb):
    """Run the pipeline on a subset of a feature's descriptions (global indices)."""
    a = aurocs[sub_idx]
    E = emb[sub_idx]
    sim = E @ E.T
    dec, comps = filter_pool(a, sim)
    m, surv_local = metrics(a, sim, dec)
    return m, dec, comps


def main():
    desc = pd.read_csv(OUT / "descriptions.csv")
    emb = np.load(OUT / "embeddings.npy")
    desc["src"] = desc["method"] + " / " + desc["generator"]
    aurocs_all = desc["auroc"].to_numpy()

    rows, comp_records, surv_src, tot_src = [], [], {}, {}
    q_prism, q_pool, div_prism, div_pool = [], [], [], []

    for (layer, unit), g in desc.groupby(["layer", "unit"]):
        idx = g.index.to_numpy()
        # pooled
        m, dec, comps = profile_subset(idx, aurocs_all, emb)
        methods = g["method"].to_numpy(); srcs = g["src"].to_numpy()
        surv_methods = sorted({methods[k] for k in range(len(idx)) if dec[k] == "survived"})
        rows.append({"layer": layer, "unit": unit, **m,
                     "survivor_methods": "|".join(surv_methods)})
        # survival by source (global tally)
        for k in range(len(idx)):
            s = srcs[k]
            tot_src[s] = tot_src.get(s, 0) + 1
            if dec[k] == "survived":
                surv_src[s] = surv_src.get(s, 0) + 1
        # collapse components (>=2 members = a real collapse group); comp indices
        # are local to this feature's description block.
        loc_method = {k: methods[k] for k in range(len(idx))}
        loc_src = {k: srcs[k] for k in range(len(idx))}
        for comp in comps:
            if len(comp) >= 2:
                cmeth = {loc_method[c] for c in comp}
                csrc = {loc_src[c] for c in comp}
                comp_records.append({"size": len(comp), "cross_method": len(cmeth) >= 2,
                                     "cross_source": len(csrc) >= 2})
        # PRISM-only (input-centric / gemini-1.5-pro) profile for consistency check
        pmask = (g["method"] == "input-centric") & (g["generator"] == "gemini-1.5-pro")
        pidx = g.index[pmask].to_numpy()
        if len(pidx) >= 1:
            mp, _, _ = profile_subset(pidx, aurocs_all, emb)
            if not np.isnan(m["Q"]) and not np.isnan(mp["Q"]):
                q_prism.append(mp["Q"]); q_pool.append(m["Q"])
            if (not m.get("uncharacterised", True) and m["n_s"] >= 2 and mp["n_s"] >= 2
                    and not np.isnan(m["diversity"]) and not np.isnan(mp["diversity"])):
                div_prism.append(mp["diversity"]); div_pool.append(m["diversity"])

    profile = pd.DataFrame(rows)
    profile.to_csv(OUT / "profile.csv", index=False)

    comp_df = pd.DataFrame(comp_records)
    n_collapse_groups = int(len(comp_df))
    stats = {
        "params": {"auroc_threshold": AUROC_THRESHOLD, "sim_threshold": SIM_THRESHOLD,
                   "n_features": int(profile.shape[0]), "n_descriptions": int(len(desc))},
        "pooled_profile_summary": {
            "median_n_desc": float(profile["n_desc"].median()),
            "median_n_s": float(profile["n_s"].median()),
            "pct_uncharacterised": round(float(profile["uncharacterised"].mean()) * 100, 1),
            "median_Q": round(float(profile.loc[profile.n_s >= 1, "Q"].median()), 4),
            "median_diversity": round(float(profile.loc[profile.n_s >= 2, "diversity"].median()), 4),
        },
        "cross_source_collapse": {
            "n_collapse_groups": n_collapse_groups,
            "pct_cross_method": round(float(comp_df["cross_method"].mean()) * 100, 1) if n_collapse_groups else None,
            "pct_cross_source": round(float(comp_df["cross_source"].mean()) * 100, 1) if n_collapse_groups else None,
        },
        "survival_by_source": {
            s: {"n_total": tot_src[s], "n_survived": surv_src.get(s, 0),
                "survival_rate": round(surv_src.get(s, 0) / tot_src[s], 3)}
            for s in sorted(tot_src)
        },
        "consistency_prism_vs_pooled": {
            "Q_spearman": round(float(spearmanr(q_prism, q_pool).statistic), 4),
            "Q_n": len(q_prism),
            "diversity_spearman": round(float(spearmanr(div_prism, div_pool).statistic), 4) if len(div_prism) >= 3 else None,
            "diversity_n": len(div_prism),
        },
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))

    s = stats
    print(f"Pooled profile ({s['params']['n_features']} gpt2-xl features, {s['params']['n_descriptions']} descriptions)")
    p = s["pooled_profile_summary"]
    print(f"  median n_desc={p['median_n_desc']:.0f}  n_s={p['median_n_s']:.0f}  "
          f"uncharacterised={p['pct_uncharacterised']}%  median Q={p['median_Q']}  median diversity={p['median_diversity']}")
    c = s["cross_source_collapse"]
    print(f"  collapse groups={c['n_collapse_groups']}  cross-method={c['pct_cross_method']}%  cross-source={c['pct_cross_source']}%")
    print("  survival rate by source:")
    for src, v in s["survival_by_source"].items():
        print(f"    {src:26s} {v['n_survived']:3d}/{v['n_total']:3d} = {v['survival_rate']:.2f}")
    cc = s["consistency_prism_vs_pooled"]
    print(f"  PRISM-only vs pooled: Q ρ={cc['Q_spearman']} (n={cc['Q_n']}), diversity ρ={cc['diversity_spearman']} (n={cc['diversity_n']})")
    print(f"\nSaved → {OUT}/profile.csv, stats.json")


if __name__ == "__main__":
    main()
