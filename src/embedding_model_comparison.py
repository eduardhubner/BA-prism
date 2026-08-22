"""
Compare embedding models on pairwise description similarity rankings.

For each alternative embedding model (hosted locally via LM Studio), recomputes
cosine similarities for two fixed sets of description pairs and measures how well
the rank ordering is preserved relative to the baseline (gte-Qwen2-1.5B-instruct).

Pair sets
---------
Within-neuron  : all 10 pairwise combinations of the 5 descriptions for each of
                 the 239 neurons (2390 pairs total). Covers the 0.3–1.0 similarity
                 range typical of same-neuron descriptions.
Cross-neuron   : a fixed random sample of description pairs drawn from different
                 neurons, giving a broader range down to near-zero similarity.
                 Sampled once with seed=42 and saved to
                 results/embedding_comparison/cross_neuron_pairs.csv so every
                 model is compared on identical pairs. Each model's similarities
                 are added as a separate column named after the model slug.

Usage
-----
    python src/embedding_model_comparison.py \\
        --model-id "nomic-embed-text" \\
        [--host http://localhost:1234] \\
        [--n-cross 500]

Output
------
    results/embedding_comparison/<slug>_within.csv
    results/embedding_comparison/cross_neuron_pairs.csv  (updated in-place, one column per model)
    Console summary with overall and per-target-model Spearman r.
"""

import argparse
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from scipy.stats import spearmanr

BASELINE_PATH   = Path(__file__).parent.parent / "results" / "pairwise_similarities_dataset.csv"
EMBEDDINGS_DIR  = Path(__file__).parent.parent / "results" / "description_embeddings"
OUTPUT_DIR      = Path(__file__).parent.parent / "results" / "embedding_comparison"
CROSS_PAIR_PATH = OUTPUT_DIR / "cross_neuron_pairs.csv"
OUTPUT_DIR.mkdir(exist_ok=True)

SIM_PAIRS  = [(1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)]
N_DESC     = 5
CROSS_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True,
                        help="Model identifier as shown in LM Studio")
    parser.add_argument("--host", default="http://localhost:1234",
                        help="LM Studio base URL (default: http://localhost:1234)")
    parser.add_argument("--n-cross", type=int, default=500,
                        help="Number of cross-neuron pairs to sample on first run (default: 500)")
    return parser.parse_args()


def get_embedding(text: str, model_id: str, host: str) -> np.ndarray:
    url = f"{host}/v1/embeddings"
    response = requests.post(url, json={"model": model_id, "input": text})
    response.raise_for_status()
    return np.array(response.json()["data"][0]["embedding"])


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def embed_all(texts: list[str], model_id: str, host: str) -> dict[str, np.ndarray]:
    """Embed all unique texts via LM Studio, return text -> embedding dict."""
    unique = list(dict.fromkeys(str(t) for t in texts))
    print(f"  Embedding {len(unique)} unique texts...")
    cache = {}
    for idx, text in enumerate(unique):
        if idx % 50 == 0:
            print(f"    {idx}/{len(unique)}")
        cache[text] = get_embedding(text, model_id, host)
    print(f"    Done. Dim={len(next(iter(cache.values())))}")
    return cache


def print_spearman_summary(label: str, rs: np.ndarray) -> None:
    print(f"\n{label} (n={len(rs)}):")
    print(f"  Spearman r  mean={rs.mean():.4f}  median={np.median(rs):.4f}"
          f"  <0.9: {(rs < 0.9).mean()*100:.1f}%  <0.7: {(rs < 0.7).mean()*100:.1f}%")


def per_neuron_spearman(df: pd.DataFrame) -> np.ndarray:
    """Compute per-neuron Spearman r from a within-neuron dataframe."""
    return (
        df.groupby("neuron_id")
        .apply(lambda g: spearmanr(g["baseline"], g["new"])[0], include_groups=False)
        .values
    )


# ---------------------------------------------------------------------------
# Baseline embeddings
# ---------------------------------------------------------------------------

def load_baseline_embeddings() -> dict[str, np.ndarray]:
    """Load saved Qwen2-1.5B embeddings keyed by description text."""
    manifest = pd.read_csv(EMBEDDINGS_DIR / "description_manifest.csv")
    npz = np.load(EMBEDDINGS_DIR / "embeddings.npz")
    emb_matrix = npz["embeddings"]   # (1195, 1536)
    return {str(row["description"]): emb_matrix[idx]
            for idx, (_, row) in enumerate(manifest.iterrows())}


# ---------------------------------------------------------------------------
# Cross-neuron pair sampling (one-time, fixed seed)
# ---------------------------------------------------------------------------

def build_cross_neuron_pairs(df: pd.DataFrame, n: int,
                             baseline_cache: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Sample n pairs from different neurons. Baseline cosine similarity is
    computed immediately from the saved Qwen2 embeddings.
    """
    rng = np.random.default_rng(CROSS_SEED)
    desc_cols = [f"description_{i}" for i in range(1, N_DESC + 1)]

    records = []
    for _, row in df.iterrows():
        for col in desc_cols:
            records.append({
                "neuron_id":    f"{row['model']}_L{row['layer']}_U{row['unit']}",
                "target_model": row["model"],
                "text":         str(row[col]),
            })
    flat = pd.DataFrame(records)

    indices = rng.choice(len(flat), size=(n * 3, 2), replace=True)
    pairs = []
    for a, b in indices:
        if flat.iloc[a]["neuron_id"] != flat.iloc[b]["neuron_id"]:
            ta, tb = flat.iloc[a]["text"], flat.iloc[b]["text"]
            ea = baseline_cache.get(ta)
            eb = baseline_cache.get(tb)
            baseline_sim = cosine_sim(ea, eb) if (ea is not None and eb is not None) else np.nan
            pairs.append({
                "neuron_a":     flat.iloc[a]["neuron_id"],
                "neuron_b":     flat.iloc[b]["neuron_id"],
                "model_a":      flat.iloc[a]["target_model"],
                "model_b":      flat.iloc[b]["target_model"],
                "text_a":       ta,
                "text_b":       tb,
                "baseline_sim": baseline_sim,
            })
        if len(pairs) == n:
            break

    return pd.DataFrame(pairs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    slug = args.model_id.replace("/", "_").replace(" ", "_")

    # --- Verify required files exist ---
    missing = [p for p in [BASELINE_PATH,
                            EMBEDDINGS_DIR / "description_manifest.csv",
                            EMBEDDINGS_DIR / "embeddings.npz"]
               if not p.exists()]
    if missing:
        print("ERROR: Required files not found:")
        for p in missing:
            print(f"  {p}")
        print("Run step1_generate_description_embeddings.py first.")
        return

    # --- Sanity check LM Studio ---
    try:
        r = requests.get(f"{args.host}/v1/models", timeout=5)
        avail = [m["id"] for m in r.json().get("data", [])]
        print(f"LM Studio reachable. Models: {avail}")
        if args.model_id not in avail:
            print(f"WARNING: '{args.model_id}' not in list above — double-check the model ID.")
    except Exception as e:
        print(f"ERROR: Cannot reach LM Studio at {args.host}: {e}")
        return

    # --- Load baseline dataset and embeddings ---
    df = pd.read_csv(BASELINE_PATH)
    print(f"Loaded {len(df)} neurons.")

    print("Loading baseline Qwen2 embeddings...")
    baseline_emb_cache = load_baseline_embeddings()
    print(f"  Loaded {len(baseline_emb_cache)} embeddings.")

    desc_cols = [f"description_{i}" for i in range(1, N_DESC + 1)]

    # --- Build or load cross-neuron pairs ---
    if CROSS_PAIR_PATH.exists():
        cross = pd.read_csv(CROSS_PAIR_PATH)
        print(f"Loaded {len(cross)} cross-neuron pairs from {CROSS_PAIR_PATH}")
    else:
        print(f"Sampling {args.n_cross} cross-neuron pairs (seed={CROSS_SEED})...")
        cross = build_cross_neuron_pairs(df, args.n_cross, baseline_emb_cache)
        cross.to_csv(CROSS_PAIR_PATH, index=False)
        print(f"Saved to {CROSS_PAIR_PATH}")

    # --- Embed all texts needed for this model ---
    within_texts = list(pd.unique(df[desc_cols].values.ravel()))
    cross_texts  = list(pd.unique(pd.concat([cross["text_a"], cross["text_b"]]).values))
    all_texts    = list(dict.fromkeys(str(t) for t in within_texts + cross_texts))

    print(f"\nEmbedding with '{args.model_id}'...")
    cache = embed_all(all_texts, args.model_id, args.host)

    # -----------------------------------------------------------------------
    # Within-neuron pairs
    # -----------------------------------------------------------------------
    print("\nComputing within-neuron similarities...")
    within_rows = []
    for _, row in df.iterrows():
        descs     = [str(row[col]) for col in desc_cols]
        embs      = [cache[d] for d in descs]
        neuron_id = f"{row['model']}_L{row['layer']}_U{row['unit']}"
        new_sim_dict = {(i,j): cosine_sim(embs[i-1], embs[j-1]) for i,j in SIM_PAIRS}
        for i, j in SIM_PAIRS:
            within_rows.append({
                "neuron_id":    neuron_id,
                "target_model": row["model"],
                "pair":         f"sim_{i}{j}",
                "baseline":     row[f"sim_{i}{j}"],
                "new":          new_sim_dict[(i, j)],
            })

    within_df = pd.DataFrame(within_rows)
    within_df["abs_diff"] = (within_df["new"] - within_df["baseline"]).abs()

    # -----------------------------------------------------------------------
    # Cross-neuron pairs — add column named after this model slug
    # -----------------------------------------------------------------------
    print("Computing cross-neuron similarities...")
    cross[f"sim_{slug}"] = [
        cosine_sim(cache[str(r["text_a"])], cache[str(r["text_b"])])
        for _, r in cross.iterrows()
    ]
    cross.to_csv(CROSS_PAIR_PATH, index=False)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"MODEL: {args.model_id}  |  {len(df)} neurons")
    print(f"{'='*65}")

    # Within-neuron overall
    neuron_rs = per_neuron_spearman(within_df)
    print_spearman_summary("Within-neuron (per-neuron Spearman r)", neuron_rs)
    print(f"  Mean abs diff:  {within_df['abs_diff'].mean():.4f}  std={within_df['abs_diff'].std():.4f}")
    print(f"  Mean baseline:  {within_df['baseline'].mean():.4f}   Mean new: {within_df['new'].mean():.4f}")

    # Within-neuron per target model
    print(f"\n  Per-target-model breakdown:")
    for tmodel, grp in within_df.groupby("target_model"):
        rs = per_neuron_spearman(grp)
        print(f"    {tmodel:<28}  mean={rs.mean():.4f}  median={np.median(rs):.4f}  n={len(rs)}")

    # Cross-neuron
    cross_col = f"sim_{slug}"
    valid = cross.dropna(subset=["baseline_sim", cross_col])
    if len(valid) > 0:
        cr, _ = spearmanr(valid["baseline_sim"], valid[cross_col])
        cross_abs = (valid[cross_col] - valid["baseline_sim"]).abs()
        print(f"\nCross-neuron ({len(valid)} pairs):")
        print(f"  Spearman r (global):  {cr:.4f}")
        print(f"  Mean abs diff:        {cross_abs.mean():.4f}  std={cross_abs.std():.4f}")

    # Save within results
    out_path = OUTPUT_DIR / f"{slug}_within.csv"
    within_df.to_csv(out_path, index=False)
    print(f"\nSaved within-neuron results to {out_path}")
    print(f"Cross-neuron pairs updated at {CROSS_PAIR_PATH}")


if __name__ == "__main__":
    main()
