"""
Clustering-degeneracy diagnostics: is the k=2 agglomerative solution real?

A two-cluster solution with one small group and one large residual does not, by
itself, prove failed clustering -- it could be a genuine minority concept. This
script tests whether it is instead degenerate, using three positive checks on the
per-neuron text-sample embeddings (gte-Qwen2, embeddings_qwen2/, 1001 x 1536 per
GPT-2 XL neuron, spherically normalised):

  1. Size degeneracy. If the "small" cluster is a handful of points (outliers),
     the k=2 split is outlier-peeling, not a semantic partition. We report the
     smaller-cluster fraction / count for agglomerative (cosine, average linkage).

  2. Null-model baseline. We destroy cross-dimension structure by permuting each
     embedding dimension independently across samples (preserves per-dimension
     marginals, removes any cluster structure) and re-cluster. If the real
     silhouette is close to this null, the apparent separation is ~noise.

  3. Balance contrast. Spherical k-means (which resists singleton clusters) is run
     at the same k=2; if it produces balanced clusters at a *lower* silhouette,
     the agglomerative advantage came only from peeling outliers.

Output: results/clustering_degeneracy_diagnostics.json
"""

import json
import glob
import os
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
EMB_DIR = ROOT / "embeddings_qwen2"
K = 2
NULL_SEED = 0


def _norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def _agglo_k2(X):
    lab = AgglomerativeClustering(n_clusters=K, metric="cosine", linkage="average").fit(X).labels_
    sizes = sorted(Counter(lab).values())
    sil = float(silhouette_score(X, lab, metric="cosine"))
    return sizes[0], sil


def _null_sil(X, rng):
    Xn = _norm(np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])]))
    lab = AgglomerativeClustering(n_clusters=K, metric="cosine", linkage="average").fit(Xn).labels_
    return float(silhouette_score(Xn, lab, metric="cosine"))


def _kmeans_k2(X):
    lab = KMeans(n_clusters=K, n_init=10, random_state=0).fit(X).labels_  # spherical: X is normalised
    sizes = sorted(Counter(lab).values())
    sil = float(silhouette_score(X, lab, metric="cosine"))
    return sizes[0], sil


def main():
    files = sorted(glob.glob(str(EMB_DIR / "gpt2-xl_*.npy")))
    rng = np.random.default_rng(NULL_SEED)
    rows = []
    for f in files:
        X = _norm(np.load(f).astype(float))
        n = len(X)
        ag_small, ag_sil = _agglo_k2(X)
        null_sil = _null_sil(X, rng)
        km_small, km_sil = _kmeans_k2(X)
        rows.append({
            "neuron": os.path.basename(f).replace(".npy", ""), "n_points": n,
            "agglo_smaller_count": ag_small, "agglo_smaller_frac": round(ag_small / n, 5),
            "agglo_silhouette": round(ag_sil, 4), "null_silhouette": round(null_sil, 4),
            "silhouette_minus_null": round(ag_sil - null_sil, 4),
            "kmeans_smaller_count": km_small, "kmeans_smaller_frac": round(km_small / n, 5),
            "kmeans_silhouette": round(km_sil, 4),
        })

    def agg(key):
        v = np.array([r[key] for r in rows], float)
        return {"mean": round(float(v.mean()), 4), "median": round(float(np.median(v)), 4),
                "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}

    summary = {
        "params": {"k": K, "linkage": "average", "metric": "cosine",
                   "embedding": "gte-Qwen2-1.5B", "null": "per-dimension permutation",
                   "null_seed": NULL_SEED, "n_neurons": len(rows),
                   "target_model": "gpt2-xl", "points_per_neuron": rows[0]["n_points"]},
        "agglo_smaller_count": agg("agglo_smaller_count"),
        "agglo_smaller_frac": agg("agglo_smaller_frac"),
        "agglo_silhouette": agg("agglo_silhouette"),
        "null_silhouette": agg("null_silhouette"),
        "silhouette_minus_null": agg("silhouette_minus_null"),
        "kmeans_smaller_frac": agg("kmeans_smaller_frac"),
        "kmeans_silhouette": agg("kmeans_silhouette"),
        "frac_neurons_agglo_smaller_le_3_points": round(
            float(np.mean([r["agglo_smaller_count"] <= 3 for r in rows])), 4),
        "frac_neurons_real_gt_null": round(
            float(np.mean([r["silhouette_minus_null"] > 0 for r in rows])), 4),
        "per_neuron": rows,
    }
    (RESULTS / "clustering_degeneracy_diagnostics.json").write_text(json.dumps(summary, indent=2))

    s = summary
    print(f"Clustering-degeneracy diagnostics (n={len(rows)} GPT-2 XL neurons, "
          f"{s['params']['points_per_neuron']} points each, k={K})")
    print(f"  agglomerative smaller cluster: median {s['agglo_smaller_count']['median']:.0f} points "
          f"({s['agglo_smaller_frac']['median']*100:.2f}% of points); "
          f"<=3 points for {s['frac_neurons_agglo_smaller_le_3_points']*100:.0f}% of neurons")
    print(f"  agglomerative silhouette: median {s['agglo_silhouette']['median']:.3f}  "
          f"vs null median {s['null_silhouette']['median']:.3f}  "
          f"(real-null median {s['silhouette_minus_null']['median']:.3f}, "
          f"real>null for {s['frac_neurons_real_gt_null']*100:.0f}% of neurons)")
    print(f"  spherical k-means (balanced): smaller cluster median "
          f"{s['kmeans_smaller_frac']['median']*100:.0f}% of points, "
          f"silhouette median {s['kmeans_silhouette']['median']:.3f}")
    print(f"\nSaved to {RESULTS / 'clustering_degeneracy_diagnostics.json'}")


if __name__ == "__main__":
    main()
