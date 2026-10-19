"""
Cross-method pooling — Stage 1 (embedding).

For each feature of Llama-3.1-8B / gemma-scope-2b / gpt2-small-sae, this pools
descriptions from multiple feature-description methods and embeds them into a common
space. Input-centric (PRISM, 5 per feature) and output-centric (1 per feature) are
pooled for all three targets; gpt2-small-sae additionally includes neuronpedia (1
per feature), giving that target a three-method pool.

Input-centric embeddings are reused unchanged from the framework cache
(results/description_embeddings/embeddings.npz). Output-centric (~179 across 3
targets) and neuronpedia (58, gpt2-small-sae only) descriptions are embedded here
with the framework convention (normalize_embeddings=True).

All AUROCs pooled here come from the gemini-1.5-pro CoSy evaluation regime, so they
share one reliability scale. Methods evaluated under the gemini-2.0-flash-lite regime
(Transluce, max_act) are excluded, because within-feature AUROC weighting assumes a
single evaluation regime. The regime is recorded per source in provenance.json.

RUN IN venv_qwen2:
    venv_qwen2/bin/python src/pool_outputcentric_embed.py

Outputs (results/pooled_outputcentric/):
    descriptions.csv   model, layer, unit, method, generator, desc_idx, description, auroc
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
OUT = RESULTS / "pooled_outputcentric"
OUT.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
_E = "_mean_evalgen-gemini-1-5-pro_cosmopedia_1000.csv"
MODELS = ["Llama-3.1-8B-Instruct", "gemma-scope-2b", "gpt2-small-sae"]
COLS = ["model", "layer", "unit", "method", "generator", "desc_idx", "description", "auroc"]
EVAL_REGIME = "gemini-1-5-pro"   # CoSy evaluation-sample generator shared by all pooled AUROCs
# Third external method, gpt2-small-sae only. Descriptions are imported as published;
# the textgen field in the upstream filename is a placeholder, so the source is
# labelled by its method name ("neuronpedia") rather than a generator model.
NEURONPEDIA_MODEL = "gpt2-small-sae"
NEURONPEDIA_FILE = f"cosy-evaluation_neuronpedia_target-gpt2-small-sae_textgen-gemini-1-5-pro{_E}"


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
    for model in MODELS:
        ic = man[man["model"] == model]
        for r in ic.itertuples():
            records.append((model, int(r.layer), int(r.unit), "input-centric", "gemini-1.5-pro",
                            int(r.desc_idx) - 1, str(r.description), float(r.auroc)))
            vecs.append(cache_emb[id2row[r.desc_id]])
        prov_inputs.append({"model": model, "source": "input-centric", "generator": "gemini-1.5-pro",
                            "eval_regime": EVAL_REGIME, "from": "cache", "n_desc": int(len(ic))})

    # 2. output-centric: embed fresh.
    new_meta, new_text = [], []
    for model in MODELS:
        f = RESULTS / f"cosy-evaluation_output-centric_target-{model}_textgen-gemini-1-5-pro{_E}"
        d = pd.read_csv(f).sort_values(["layer", "unit"]).reset_index(drop=True)
        for row in d.itertuples():
            new_meta.append((model, int(row.layer), int(row.unit), "output-centric", "gemini-1.5-pro",
                             0, str(row.explanation), float(row.AUROC)))
            new_text.append(str(row.explanation))
        prov_inputs.append({"model": model, "source": "output-centric", "generator": "gemini-1.5-pro",
                            "eval_regime": EVAL_REGIME, "from": "embedded_fresh",
                            "file": f.name, "sha256": _sha256(f), "n_desc": int(len(d))})

    # 3. neuronpedia (gpt2-small-sae only): third external method, embed fresh.
    f = RESULTS / NEURONPEDIA_FILE
    d = pd.read_csv(f).sort_values(["layer", "unit"]).reset_index(drop=True)
    for row in d.itertuples():
        new_meta.append((NEURONPEDIA_MODEL, int(row.layer), int(row.unit), "neuronpedia", "neuronpedia",
                         0, str(row.explanation), float(row.AUROC)))
        new_text.append(str(row.explanation))
    prov_inputs.append({"model": NEURONPEDIA_MODEL, "source": "neuronpedia", "generator": "neuronpedia",
                        "eval_regime": EVAL_REGIME, "from": "embedded_fresh",
                        "file": f.name, "sha256": _sha256(f), "n_desc": int(len(d))})

    from sentence_transformers import SentenceTransformer
    print(f"Loading {EMBEDDING_MODEL} ...")
    embedder = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    print(f"Embedding {len(new_text)} output-centric + neuronpedia descriptions "
          f"(input-centric reused from cache) ...")
    new_emb = np.asarray(embedder.encode(new_text, convert_to_tensor=False,
                                         normalize_embeddings=True, show_progress_bar=True,
                                         batch_size=16), dtype=np.float32)
    records.extend(new_meta)
    vecs.extend(list(new_emb))

    desc = pd.DataFrame(records, columns=COLS)
    emb = np.vstack(vecs).astype(np.float32)
    order = desc.sort_values(["model", "layer", "unit", "method", "desc_idx"]).index.to_numpy()
    desc = desc.iloc[order].reset_index(drop=True)
    emb = emb[order]

    desc.to_csv(OUT / "descriptions.csv", index=False)
    np.save(OUT / "embeddings.npy", emb)

    import sentence_transformers, torch, sklearn
    prov = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "models": MODELS,
        "n_features": int(desc[["model", "layer", "unit"]].drop_duplicates().shape[0]),
        "n_descriptions": int(len(desc)), "embedding_dim": int(emb.shape[1]),
        "embedding_model": EMBEDDING_MODEL, "embedding_normalized": True,
        "eval_regime": EVAL_REGIME,
        "methods": sorted(desc["method"].unique().tolist()),
        "inputs": prov_inputs,
        "versions": {"python": sys.version.split()[0],
                     "sentence_transformers": sentence_transformers.__version__,
                     "torch": torch.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"Pooled {len(desc)} descriptions over {prov['n_features']} features "
          f"({len(records) - len(new_text)} reused + {len(new_text)} fresh) → {OUT}")


if __name__ == "__main__":
    main()
