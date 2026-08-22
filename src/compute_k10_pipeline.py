"""
Embedding + pairwise similarity pipeline for the k=10 ablation dataset.

Reads  : results/k10_ablation.csv
Writes : results/k10_pairwise_similarities.csv

Run with:
    venv_qwen2/bin/python src/compute_k10_pipeline.py
"""

import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

EMBEDDING_MODEL = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
BATCH_SIZE = 32
K = 10  # descriptions per neuron

ROOT = Path(__file__).parent.parent
INPUT_CSV  = ROOT / "results" / "k10_ablation.csv"
OUTPUT_CSV = ROOT / "results" / "k10_pairwise_similarities.csv"

# All C(10,2) = 45 ordered pairs
PAIRS = list(combinations(range(1, K + 1), 2))   # (1,2),(1,3),...,(9,10)
SIM_COLS = [f"sim_{i}_{j}" for i, j in PAIRS]


def embed(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    vecs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        batch = model.encode(
            texts[i : i + BATCH_SIZE],
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vecs.append(batch)
    return np.vstack(vecs)


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows ({df.groupby(['layer','unit']).ngroups} neurons, k={K})")

    # Sort within each neuron by AUROC descending — consistent with k=5 pipeline
    df = (
        df.sort_values(["layer", "unit", "AUC"], ascending=[True, True, False])
          .groupby(["layer", "unit"], group_keys=False)
          .apply(lambda g: g.reset_index(drop=True))
    )

    # Assign description index 1..10 within each neuron
    df["desc_idx"] = df.groupby(["layer", "unit"]).cumcount() + 1

    # ── Step 1: embed (with cache) ────────────────────────────────────────────
    cache = ROOT / "results" / "k10_embeddings_cache.npz"
    if cache.exists():
        print(f"\nLoading cached embeddings from {cache} …")
        vecs = np.load(cache)["vecs"]
    else:
        print(f"\nLoading {EMBEDDING_MODEL} …")
        model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
        model.max_seq_length = 512
        print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
        texts = df["explanation"].tolist()
        vecs = embed(texts, model)
        np.savez_compressed(cache, vecs=vecs)
        print(f"Cached to {cache}")
    print(f"Embeddings shape: {vecs.shape}")

    # Map (layer, unit, desc_idx) -> embedding row
    # Must use enumerate (not df index) because groupby/apply reset_index
    # leaves a repeated 0-9 index across groups.
    emb_index = {}
    for pos, (_, row) in enumerate(df.iterrows()):
        emb_index[(row.layer, row.unit, row.desc_idx)] = vecs[pos]

    # ── Step 2: pairwise similarities ─────────────────────────────────────────
    rows = []
    for (layer, unit), grp in tqdm(df.groupby(["layer", "unit"]), desc="Pairwise"):
        grp = grp.sort_values("desc_idx").reset_index(drop=True)
        assert len(grp) == K, f"Expected {K} rows, got {len(grp)} for L{layer}_U{unit}"

        row = {"layer": layer, "unit": unit}

        for pos, r in grp.iterrows():
            idx = r["desc_idx"]
            row[f"description_{idx}"] = r["explanation"]
            row[f"auroc_{idx}"]         = r["AUC"]
            row[f"mad_{idx}"]         = r["MAD"]
            row[f"p_{idx}"]           = r["p"]

        sims = []
        for (i, j), col in zip(PAIRS, SIM_COLS):
            vi = emb_index[(layer, unit, i)]
            vj = emb_index[(layer, unit, j)]
            s  = float(np.dot(vi, vj))   # already L2-normalised
            row[col] = s
            sims.append(s)

        sims_arr = np.array(sims)
        row["mean_similarity"] = sims_arr.mean()
        row["min_similarity"]  = sims_arr.min()
        row["max_similarity"]  = sims_arr.max()
        row["std_similarity"]  = sims_arr.std()

        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(out)} neurons → {OUTPUT_CSV}")
    print(f"Columns: {len(out.columns)}  ({len(SIM_COLS)} similarity columns)")
    print(f"\nMean PRISM (mean_similarity): {out['mean_similarity'].mean():.4f}")
    print(f"Std:                          {out['mean_similarity'].std():.4f}")


if __name__ == "__main__":
    main()
