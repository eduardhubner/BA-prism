"""
Visualizations for filtered_evaluation.csv results.
Saves all plots to results/figures/filtered_evaluation/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from scipy.stats import gaussian_kde

FILTERED_PATH = Path(__file__).parent.parent / "results" / "filtered_evaluation.csv"
PAIRWISE_PATH = Path(__file__).parent.parent / "results" / "pairwise_similarities_dataset.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "figures" / "filtered_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load and join ---
df_filt = pd.read_csv(FILTERED_PATH)
df_pair = pd.read_csv(PAIRWISE_PATH)[
    ["model", "layer", "unit", "mean_similarity",
     "auc_1", "auc_2", "auc_3", "auc_4", "auc_5"]
]
df_pair = df_pair.rename(columns={"mean_similarity": "prism_score"})

auc_cols = ["auc_1", "auc_2", "auc_3", "auc_4", "auc_5"]
def auc_std_survivors(row):
    aucs = [row[c] for c in auc_cols if row[c] > 0.5]
    return float(np.std(aucs)) if len(aucs) >= 2 else np.nan
df_pair["auc_std_survivors"] = df_pair.apply(auc_std_survivors, axis=1)

df = df_filt.merge(df_pair[["model", "layer", "unit", "prism_score", "auc_std_survivors"]],
                   on=["model", "layer", "unit"], how="left")
df_valid = df[df["n_surviving"] > 0].copy()
df_wms = df_valid.dropna(subset=["mean_similarity_post"]).copy()

MODELS = sorted(df["model"].unique())

# Okabe-Ito colorblind-safe palette
MODEL_COLORS = {
    "Llama-3.1-8B-Instruct": "#0072B2",  # blue
    "gemma-scope-2b":         "#E69F00",  # orange
    "gpt2-small-sae":         "#009E73",  # bluish green
    "gpt2-xl":                "#D55E00",  # vermillion
}


def save(fig, name):
    fig.savefig(OUTPUT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 1. Prism score vs post-filter weighted mean similarity ─────────────────
df_wms_r = df_wms.copy()
df_wms_r["rank_prism"] = df_wms_r["prism_score"].rank(ascending=False)
df_wms_r["rank_wms"]   = df_wms_r["weighted_mean_similarity"].rank(ascending=False)
df_wms_r["abs_rank_change"] = (df_wms_r["rank_prism"] - df_wms_r["rank_wms"]).abs()

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(df_wms_r["prism_score"], df_wms_r["weighted_mean_similarity"],
                c=df_wms_r["abs_rank_change"], cmap="viridis", alpha=0.8, s=40)
plt.colorbar(sc, ax=ax, label="|Rank change|")
lims = [0, 1.05]
ax.plot(lims, lims, "k--", lw=1, alpha=0.4, label="y=x")
ax.set_xlabel("PRISM score (mean pairwise similarity, pre-filter)")
ax.set_ylabel("Weighted mean similarity (post-filter)")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.legend(fontsize=8)
spear = spearmanr(df_wms_r["prism_score"], df_wms_r["weighted_mean_similarity"]).statistic
ax.set_title(f"PRISM score vs post-filter weighted mean similarity\n"
             f"(Spearman ρ={spear:.2f}, n={len(df_wms_r)}, colored by |rank change|)")
plt.tight_layout()
save(fig, "01_prism_vs_weighted_mean_sim.png")


# ── 2. Rank change: prism score rank vs weighted mean sim rank ─────────────
df_wms2 = df_wms.copy()
df_wms2["rank_prism"] = df_wms2["prism_score"].rank(ascending=False)
df_wms2["rank_wms"]   = df_wms2["weighted_mean_similarity"].rank(ascending=False)

fig, ax = plt.subplots(figsize=(7, 6))
for model, group in df_wms2.groupby("model"):
    ax.scatter(group["rank_prism"], group["rank_wms"],
               label=model, color=MODEL_COLORS[model], alpha=0.7, s=40)
lims = [0, len(df_wms2) + 5]
ax.plot(lims, lims, "k--", lw=1, alpha=0.4)
ax.set_xlabel("Rank by PRISM score")
ax.set_ylabel("Rank by weighted mean similarity (post-filter)")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.legend(fontsize=8)
mean_change = (df_wms2["rank_prism"] - df_wms2["rank_wms"]).abs().mean()
ax.set_title(f"Rank change: PRISM score vs weighted mean similarity\n(mean |Δrank|={mean_change:.1f}, n={len(df_wms2)})")
plt.tight_layout()
save(fig, "02_rank_change_prism_vs_wms.png")


# ── 3. Distribution of n_surviving by model ────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)
axes = axes.flatten()
bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
for ax, model in zip(axes, MODELS):
    group = df[df["model"] == model]["n_surviving"]
    ax.hist(group, bins=bins, color=MODEL_COLORS[model], edgecolor="white", alpha=0.85)
    ax.set_title(model, fontsize=9)
    ax.set_xlabel("Surviving descriptions")
    ax.set_ylabel("Neurons")
    ax.set_xticks(range(6))
fig.suptitle("Distribution of surviving descriptions after filtering", fontsize=11)
plt.tight_layout()
save(fig, "03_n_surviving_by_model.png")


# ── 4. Weighted mean similarity distribution (KDE per model) ───────────────
fig, ax = plt.subplots(figsize=(7, 5))
x_grid = np.linspace(0, 1, 300)
for model, group in df_wms.groupby("model"):
    vals = group["weighted_mean_similarity"].dropna().values
    if len(vals) > 2:
        kde = gaussian_kde(vals, bw_method=0.3)
        ax.plot(x_grid, kde(x_grid), label=model, color=MODEL_COLORS[model], lw=2)
        ax.fill_between(x_grid, kde(x_grid), alpha=0.1, color=MODEL_COLORS[model])
ax.set_xlabel("Weighted mean similarity")
ax.set_ylabel("Density")
ax.legend(fontsize=8)
ax.set_title("Distribution of weighted mean similarity (post-filter, KDE)")
plt.tight_layout()
save(fig, "04_weighted_mean_similarity_kde.png")


# ── 5. Effective N distribution (KDE per model) ────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
x_grid = np.linspace(0, 5, 300)
for model, group in df_valid.groupby("model"):
    vals = group["effective_n_descriptions"].dropna().values
    if len(vals) > 2:
        kde = gaussian_kde(vals, bw_method=0.3)
        ax.plot(x_grid, kde(x_grid), label=model, color=MODEL_COLORS[model], lw=2)
        ax.fill_between(x_grid, kde(x_grid), alpha=0.1, color=MODEL_COLORS[model])
ax.set_xlabel("Effective number of descriptions")
ax.set_ylabel("Density")
ax.legend(fontsize=8)
ax.set_title("Distribution of effective N (post-filter, KDE)")
plt.tight_layout()
save(fig, "05_effective_n_kde.png")


# ── 6. Mean similarity (post-filter) vs effective N ───────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
for model, group in df_wms.groupby("model"):
    ax.scatter(group["mean_similarity_post"], group["effective_n_descriptions"],
               label=model, color=MODEL_COLORS[model], alpha=0.7, s=40)
ax.set_xlabel("Mean similarity (post-filter, unweighted)")
ax.set_ylabel("Effective N")
ax.legend(fontsize=8)
ax.set_title("Mean similarity vs effective N (post-filter)")
plt.tight_layout()
save(fig, "06_mean_sim_vs_effN.png")


# ── 10. Zero survivors breakdown by model ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
zero_rates = df.groupby("model").apply(
    lambda g: g["uncharacterized"].mean() * 100
).reindex(MODELS)
bars = ax.bar(range(len(zero_rates)), zero_rates.values,
              color=[MODEL_COLORS[m] for m in MODELS], edgecolor="white", alpha=0.85)
ax.set_xticks(range(len(zero_rates)))
ax.set_xticklabels([m.replace("-", "\n") for m in MODELS], fontsize=8)
ax.set_ylabel("% neurons with 0 survivors")
ax.set_title("Zero-survivor rate by model")
for bar, val in zip(bars, zero_rates.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", fontsize=9)
plt.tight_layout()
save(fig, "10_zero_survivors_by_model.png")


# ── 11. Prism score vs mean AUC (all models) ──────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
for model, group in df.groupby("model"):
    ax.scatter(group["prism_score"], group["mean_auc_pre"],
               label=model, color=MODEL_COLORS[model], alpha=0.7, s=40)
ax.set_xlabel("PRISM score (mean pairwise similarity)")
ax.set_ylabel("Mean AUC (pre-filter)")
ax.legend(fontsize=8)
spear = spearmanr(df["prism_score"].dropna(),
                  df.loc[df["prism_score"].notna(), "mean_auc_pre"]).statistic
ax.set_title(f"PRISM score vs mean AUC\n(Spearman ρ={spear:.2f}, n={df['prism_score'].notna().sum()})")
plt.tight_layout()
save(fig, "11_prism_score_vs_mean_auc.png")


# ── 15. Prism score vs post-filter mean similarity (unweighted) ───────────
fig, ax = plt.subplots(figsize=(7, 6))
for model, group in df_wms.groupby("model"):
    ax.scatter(group["prism_score"], group["mean_similarity_post"],
               label=model, color=MODEL_COLORS[model], alpha=0.7, s=40)
lims = [0, 1.05]
ax.plot(lims, lims, "k--", lw=1, alpha=0.4, label="y=x")
ax.set_xlabel("PRISM score (mean similarity, pre-filter)")
ax.set_ylabel("Mean similarity (post-filter, unweighted)")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.legend(fontsize=8)
spear = spearmanr(df_wms["prism_score"], df_wms["mean_similarity_post"]).statistic
ax.set_title(f"PRISM score vs post-filter mean similarity\n(Spearman ρ={spear:.2f}, n={len(df_wms)})")
plt.tight_layout()
save(fig, "15_prism_vs_mean_sim_post.png")


# ── 17. Zero-survivor neurons: prism score and mean AUC distributions ──────
df_zero    = df[df["uncharacterized"]].copy()
df_nonzero = df[~df["uncharacterized"]].copy()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col, label in zip(axes,
                           ["prism_score", "mean_auc_pre"],
                           ["PRISM score", "Mean AUC (pre-filter)"]):
    x_grid = np.linspace(0, 1, 300)
    for subset, name, color in [
        (df_zero,    "0 survivors",  "#D55E00"),
        (df_nonzero, "≥1 survivor",  "#0072B2"),
    ]:
        vals = subset[col].dropna().values
        if len(vals) > 2:
            kde = gaussian_kde(vals, bw_method=0.3)
            ax.plot(x_grid, kde(x_grid), lw=2, color=color,
                    label=f"{name} (n={len(vals)})")
            ax.fill_between(x_grid, kde(x_grid), alpha=0.15, color=color)
    ax.set_xlabel(label)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.set_title(f"{label}: zero vs non-zero survivors")

fig.suptitle("Are zero-survivor neurons systematically different?", fontsize=11)
plt.tight_layout()
save(fig, "17_zero_survivor_characteristics.png")


# ── 18. AUC variance vs weighted/unweighted similarity divergence ──────────
df_div = df_wms.dropna(subset=["weighted_mean_similarity", "auc_std_survivors"]).copy()
df_div["sim_divergence"] = (
    df_div["weighted_mean_similarity"] - df_div["mean_similarity_post"]
).abs()

fig, ax = plt.subplots(figsize=(7, 6))
for model, group in df_div.groupby("model"):
    ax.scatter(group["auc_std_survivors"], group["sim_divergence"],
               label=model, color=MODEL_COLORS[model], alpha=0.7, s=40)
ax.axvline(0.05, color="gray", lw=1, linestyle="--", alpha=0.6, label="std=0.05")
ax.axvline(0.10, color="gray", lw=1, linestyle=":",  alpha=0.6, label="std=0.10")
ax.set_xlabel("AUC std among survivors")
ax.set_ylabel("|Weighted sim − Unweighted sim|")
ax.legend(fontsize=8)
spear = spearmanr(df_div["auc_std_survivors"], df_div["sim_divergence"]).statistic
pct_above = (df_div["auc_std_survivors"] > 0.05).mean() * 100
ax.set_title(f"AUC variance drives weighted/unweighted divergence\n"
             f"(Spearman ρ={spear:.2f}, {pct_above:.0f}% of neurons have std>0.05, n={len(df_div)})")
plt.tight_layout()
save(fig, "18_auc_variance_vs_sim_divergence.png")


print(f"Saved figures to {OUTPUT_DIR}")
