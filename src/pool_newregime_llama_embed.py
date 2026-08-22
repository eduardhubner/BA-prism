"""
New-regime cross-method pool (Llama) — Stage 1 (embedding).

Pools the two feature-description methods that were evaluated under the
post-rebuttal CoSy regime (evaluation samples from gemini-2.0-flash-lite) on
Llama-3.1-8B: Transluce and max-act (one description per feature each, 60 features).
Both descriptions per feature are embedded so the unchanged filter + collapse
pipeline can be run on them (Stage 2).

This is the parallel, new-regime counterpart to the gemini-1.5-pro pool in
pool_outputcentric_*. The two regimes are analysed separately and never mixed within
a feature; Llama is the only target with two methods on the new regime.

Both AUROC sets come from gemini-2.0-flash-lite evaluation samples. NB the Transluce
file is named "...flash..." but its evaluation samples are gemini-2.0-flash-lite
(upstream naming error), so it is recorded here under the flash-lite regime.
Descriptions are imported as published, so each method is labelled by its method name.

RUN IN venv_qwen2:
    venv_qwen2/bin/python src/pool_newregime_llama_embed.py

Outputs (results/pooled_newregime_llama/):
    descriptions.csv   model, layer, unit, method, generator, desc_idx, description, auc
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
OUT = RESULTS / "pooled_newregime_llama"
OUT.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
EVAL_REGIME = "gemini-2-0-flash-lite"
TARGET = "Llama-3.1-8B-Instruct"
COLS = ["model", "layer", "unit", "method", "generator", "desc_idx", "description", "auc"]
SOURCES = {   # method -> file (both evaluated under gemini-2.0-flash-lite)
    "Transluce": "cosy-evaluation_Transluce_target-Llama-3.1-8B-Instruct_textgen-gemini-2-0-flash_mean_evalgen-gemini-2-0-flash_cosmopedia_1000.csv",
    "max_act":   "cosy-evaluation_max_act_10k_target-Llama-3.1-8B-Instruct_textgen-gemini-2-0-flash-lite_mean_evalgen-gemini-2-0-flash-lite_cosmopedia_1000.csv",
}


def _sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    records, texts, prov_inputs = [], [], []
    for method, fname in SOURCES.items():
        f = RESULTS / fname
        d = pd.read_csv(f).sort_values(["layer", "unit"]).reset_index(drop=True)
        for row in d.itertuples():
            records.append((TARGET, int(row.layer), int(row.unit), method, method,
                            0, str(row.explanation), float(row.AUC)))
            texts.append(str(row.explanation))
        prov_inputs.append({"model": TARGET, "source": method, "generator": method,
                            "eval_regime": EVAL_REGIME, "from": "embedded_fresh",
                            "file": f.name, "sha256": _sha256(f), "n_desc": int(len(d))})

    from sentence_transformers import SentenceTransformer
    print(f"Loading {EMBEDDING_MODEL} ...")
    embedder = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    print(f"Embedding {len(texts)} Transluce + max-act descriptions ...")
    emb = np.asarray(embedder.encode(texts, convert_to_tensor=False, normalize_embeddings=True,
                                     show_progress_bar=True, batch_size=16), dtype=np.float32)

    desc = pd.DataFrame(records, columns=COLS)
    order = desc.sort_values(["model", "layer", "unit", "method", "desc_idx"]).index.to_numpy()
    desc = desc.iloc[order].reset_index(drop=True)
    emb = emb[order]

    desc.to_csv(OUT / "descriptions.csv", index=False)
    np.save(OUT / "embeddings.npy", emb)

    import sentence_transformers, torch, sklearn
    prov = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": TARGET,
        "n_features": int(desc[["model", "layer", "unit"]].drop_duplicates().shape[0]),
        "n_descriptions": int(len(desc)), "embedding_dim": int(emb.shape[1]),
        "embedding_model": EMBEDDING_MODEL, "embedding_normalized": True,
        "eval_regime": EVAL_REGIME,
        "methods": sorted(desc["method"].unique().tolist()),
        "transluce_filename_note": "file says 'flash' but evaluation samples are flash-lite (upstream naming error)",
        "inputs": prov_inputs,
        "versions": {"python": sys.version.split()[0],
                     "sentence_transformers": sentence_transformers.__version__,
                     "torch": torch.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"Pooled {len(desc)} descriptions over {prov['n_features']} features → {OUT}")


if __name__ == "__main__":
    main()
