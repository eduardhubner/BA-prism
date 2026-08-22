"""
Cross-method pooling — Stage 1: pool the gpt2-xl descriptions from every
AUROC-scored source and embed them (gte-Qwen2-1.5B), for the method-generality
experiment. Stage 2 (pool_gpt2xl_analyze.py) runs the filter + metrics.

Only committed, Gemini-1.5-evaluated CoSy files are used, so every description
carries an AUROC on the *same* reliability yardstick.

The input-centric (PRISM) embeddings are REUSED from the framework's cache
(results/description_embeddings/embeddings.npz) — byte-identical to what the
pipeline used, no re-embedding drift. Only the new sources (GPT-explain, phi-4,
deepseek, Qwen3 — ~960 descriptions) are embedded fresh, with the framework's
exact convention (normalize_embeddings=True, no instruction prefix), so cosine =
dot product and the 0.70 collapse threshold transfers.

Reproducibility: input-file SHA-256 hashes, the embedding model, and library
versions are written to provenance.json; embeddings are cached to embeddings.npy
(row-aligned to descriptions.csv) so Stage 2 is deterministic.

RUN IN venv_qwen2 (has sentence_transformers):
    venv_qwen2/bin/python src/pool_gpt2xl_embed.py

Outputs (results/pooled_gpt2xl/):
    descriptions.csv   layer, unit, method, generator, desc_idx, description, auroc
    embeddings.npy     float32 [N, 1536], row-aligned to descriptions.csv
    provenance.json
"""

import sys
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
CACHE = RESULTS / "description_embeddings"
OUT = RESULTS / "pooled_gpt2xl"
OUT.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
_E = "_mean_evalgen-gemini-1-5-pro_cosmopedia_1000.csv"
COLS = ["layer", "unit", "method", "generator", "desc_idx", "description", "auroc"]

# New sources to embed fresh (input-centric/gemini comes from the cache instead).
NEW_SOURCES = [
    ("GPT-explain",   "gemini-1.5-pro", f"cosy-evaluation_GPT-explain_target-gpt2-xl_textgen-gemini-1-5-pro{_E}"),
    ("input-centric", "phi-4",          f"cosy-evaluation_target-gpt2-xl_textgen-phi-4{_E}"),
    ("input-centric", "deepseek",       f"cosy-evaluation_target-gpt2-xl_textgen-deepseek-reasoner{_E}"),
    ("input-centric", "Qwen3-32B",      f"cosy-evaluation_target-gpt2-xl_textgen-Qwen3-32B{_E}"),
]


def _sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    records, vecs, prov_inputs = [], [], []

    # 1. input-centric (PRISM): reuse cached embeddings + description + AUROC.
    man = pd.read_csv(CACHE / "description_manifest.csv")
    z = np.load(CACHE / "embeddings.npz", allow_pickle=True)
    cache_emb = z["embeddings"]
    cache_ids = [d.decode() if isinstance(d, (bytes, bytearray)) else str(d) for d in z["desc_ids"]]
    id2row = {d: i for i, d in enumerate(cache_ids)}
    gx = man[man["model"] == "gpt2-xl"]
    for r in gx.itertuples():
        records.append((int(r.layer), int(r.unit), "input-centric", "gemini-1.5-pro",
                        int(r.desc_idx) - 1, str(r.description), float(r.auroc)))
        vecs.append(cache_emb[id2row[r.desc_id]])
    prov_inputs.append({"source": "input-centric / gemini-1.5-pro", "from": "cache",
                        "file": "description_embeddings/embeddings.npz",
                        "sha256": _sha256(CACHE / "embeddings.npz"), "n_desc": int(len(gx))})

    # 2. new sources: gather text, embed fresh.
    new_meta, new_text = [], []
    for method, generator, fname in NEW_SOURCES:
        f = RESULTS / fname
        d = pd.read_csv(f).sort_values(["layer", "unit"]).reset_index(drop=True)
        d["desc_idx"] = d.groupby(["layer", "unit"]).cumcount()
        for row in d.itertuples():
            new_meta.append((int(row.layer), int(row.unit), method, generator,
                             int(row.desc_idx), str(row.explanation), float(row.AUROC)))
            new_text.append(str(row.explanation))
        prov_inputs.append({"source": f"{method} / {generator}", "from": "embedded_fresh",
                            "file": fname, "sha256": _sha256(f), "n_desc": int(len(d))})

    from sentence_transformers import SentenceTransformer
    print(f"Loading {EMBEDDING_MODEL} ...")
    embedder = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    print(f"Embedding {len(new_text)} new-source descriptions (input-centric/gemini reused from cache) ...")
    new_emb = np.asarray(embedder.encode(new_text, convert_to_tensor=False,
                                         normalize_embeddings=True, show_progress_bar=True,
                                         batch_size=16), dtype=np.float32)
    records.extend(new_meta)
    vecs.extend(list(new_emb))

    # 3. assemble + deterministic ordering; keep embeddings row-aligned.
    desc = pd.DataFrame(records, columns=COLS)
    emb = np.vstack(vecs).astype(np.float32)
    order = desc.sort_values(["layer", "unit", "method", "generator", "desc_idx"]).index.to_numpy()
    desc = desc.iloc[order].reset_index(drop=True)
    emb = emb[order]

    desc.to_csv(OUT / "descriptions.csv", index=False)
    np.save(OUT / "embeddings.npy", emb)

    import sentence_transformers, torch, sklearn
    prov = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_model": "gpt2-xl",
        "n_features": int(desc[["layer", "unit"]].drop_duplicates().shape[0]),
        "n_descriptions": int(len(desc)), "embedding_dim": int(emb.shape[1]),
        "embedding_model": EMBEDDING_MODEL, "embedding_normalized": True,
        "inputs": prov_inputs,
        "versions": {"python": sys.version.split()[0],
                     "sentence_transformers": sentence_transformers.__version__,
                     "torch": torch.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"Pooled {len(desc)} descriptions over {prov['n_features']} features "
          f"({len(gx)} reused + {len(new_text)} fresh) → {OUT}")


if __name__ == "__main__":
    main()
