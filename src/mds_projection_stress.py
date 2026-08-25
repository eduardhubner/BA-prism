"""
Goodness-of-fit of the 2-D MDS projections used in the feature-case figures.

The panels project description dissimilarities Delta_ij = 1 - cos(s_i, s_j) into two
dimensions with metric MDS (SMACOF, raw-stress objective -- see toy_neurons.mds_coords).
This records how faithful each projection is, so the figures can be read with the right
amount of trust.

Two numbers per feature:
  raw stress        what SMACOF actually minimises,
                    sigma(Z) = sum_{i<j} ( ||z_i - z_j||_2 - Delta_ij )^2.
                    NOT scale-free, so it is not comparable across features.
  Kruskal stress-1  sqrt( sum (d - Delta)^2 / sum Delta^2 ) -- scale-free and therefore
                    the one to quote. Convention: < 0.10 good, 0.10-0.20 fair.

The projection is diagnostic only. No filtering or collapse decision uses it, and the axes
carry no intrinsic meaning.

Covers every feature plotted anywhere: the two main-figure cases, the six appendix cases,
and the standalone case_study/ set.

Output: results/mds_projection_stress.json
Run with: python src/mds_projection_stress.py
"""

import json
import sys
import warnings
from pathlib import Path

import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).parent))
import case_study_analysis as cs          # noqa: E402
import toy_neurons as toy                 # noqa: E402

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results" / "mds_projection_stress.json"

# every feature that appears in a plotted MDS panel
CASES = [
    ("gemma-scope-2b", 0, 13988),          # main figure
    ("gpt2-small-sae", 10, 9896),          # main figure
    ("gpt2-small-sae", 10, 900),           # appendix
    ("gpt2-xl", 20, 4268),                 # appendix
    ("gemma-scope-2b", 0, 10531),          # appendix
    ("gemma-scope-2b", 0, 2725),           # appendix + case_study
    ("gemma-scope-2b", 10, 15418),         # appendix
    ("Llama-3.1-8B-Instruct", 0, 3016),    # appendix
    ("gemma-scope-2b", 0, 13183),          # case_study
    ("gpt2-small-sae", 5, 30319),          # case_study
    ("gpt2-xl", 20, 5278),                 # case_study
    ("gpt2-xl", 40, 1516),                 # case_study
]


def main():
    warnings.filterwarnings("ignore")
    df = pd.read_csv(cs.PRISM_PATH)

    rows = []
    for model, layer, unit in CASES:
        sel = df[(df.model == model) & (df.layer == layer) & (df.unit == unit)]
        if sel.empty:
            print(f"  !! not found: {model} L{layer}/U{unit}")
            continue
        sim = cs.get_sim_matrix(sel.iloc[0])
        coords, raw = toy.mds_coords(sim, return_stress=True)
        rows.append({
            "model": model, "layer": int(layer), "unit": int(unit),
            "raw_stress": round(float(raw), 5),
            "kruskal_stress_1": round(toy.kruskal_stress(sim, coords), 4),
        })
    rows.sort(key=lambda r: r["kruskal_stress_1"])

    summary = {
        "note": "Goodness-of-fit of the 2-D metric-MDS projections in the feature-case "
                "figures. Quote Kruskal stress-1 (scale-free, comparable across features); "
                "raw stress is what SMACOF minimises but is not comparable. Convention: "
                "<0.10 good, 0.10-0.20 fair. Projections are visualisation only.",
        "params": {
            "algorithm": "SMACOF (metric MDS)",
            "objective": "raw stress",
            "dissimilarity": "1 - cosine(s_i, s_j)",
            "metric": "precomputed", "init": "random", "n_init": 4,
            "random_state": 42, "max_iter": 300, "eps": 1e-6,
            "sklearn_version": sklearn.__version__,
            "init_note": "init is pinned: scikit-learn changes the default to "
                         "'classical_mds' in 1.10, which would move every plotted "
                         "coordinate despite the fixed seed.",
        },
        "kruskal_stress_1_range": [rows[0]["kruskal_stress_1"], rows[-1]["kruskal_stress_1"]],
        "per_feature": rows,
    }
    OUT.write_text(json.dumps(summary, indent=2))

    print(f"{'feature':36s}{'raw':>10s}{'stress-1':>10s}")
    for r in rows:
        print(f"{r['model'] + ' L' + str(r['layer']) + '/U' + str(r['unit']):36s}"
              f"{r['raw_stress']:10.5f}{r['kruskal_stress_1']:10.4f}")
    lo, hi = summary["kruskal_stress_1_range"]
    print(f"\nKruskal stress-1 range: {lo}-{hi}   (worst: "
          f"{rows[-1]['model']} L{rows[-1]['layer']}/U{rows[-1]['unit']})")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
