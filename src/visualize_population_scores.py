"""
Visualize population-level polysemanticity scores.

Creates:
1. Overall histograms: weighted diversity, n_eff, total reliability mass, median sim unweighted
2. Violin plots by model: same 4 metrics
3. Scatter plots: unweighted diversity vs reliability mass, unweighted diversity vs n_eff
4. Individual AUC distribution (all 1,195 AUC values)
5. Within-neuron similarity spread (std of pairwise similarities)

Usage:
    python3 src/visualize_population_scores.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# Paths
RESULTS_DIR = Path(__file__).parent.parent / "results"
POLY_FILE = RESULTS_DIR / "polysemanticity_experiments" / "polysemanticity_scores_auc_direct.csv"
PAIRWISE_FILE = RESULTS_DIR / "pairwise_similarities_dataset.csv"
OUTPUT_DIR = RESULTS_DIR / "polysemanticity_experiments" / "population_plots"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Model display names and colors
MODEL_LABELS = {
    'gpt2-xl': 'GPT2-XL',
    'Llama-3.1-8B-Instruct': 'Llama-3.1-8B',
    'gemma-scope-2b': 'Gemma-2B',
    'gpt2-small-sae': 'GPT2-SAE',
}
MODEL_ORDER = ['gpt2-xl', 'Llama-3.1-8B-Instruct', 'gemma-scope-2b', 'gpt2-small-sae']
MODEL_COLORS = {
    'gpt2-xl': '#1f77b4',
    'Llama-3.1-8B-Instruct': '#ff7f0e',
    'gemma-scope-2b': '#2ca02c',
    'gpt2-small-sae': '#d62728',
}


def load_data():
    """Load polysemanticity scores and pairwise similarities."""
    poly_df = pd.read_csv(POLY_FILE)
    pw_df = pd.read_csv(PAIRWISE_FILE)
    return poly_df, pw_df


# ============================================================
# 1. Overall Histograms
# ============================================================

def plot_histograms(poly_df):
    """Create overall histograms for key metrics."""
    metrics = [
        ('diversity_score', 'Weighted Diversity Score', (0, 1)),
        ('n_eff', 'Effective Number of Descriptions (n_eff)', (1, 5)),
        ('sum_weights', 'Total Reliability Mass', (0, 5)),
        ('median_sim_unweighted', 'Median Pairwise Similarity (Unweighted)', (0, 1)),
    ]

    for col, title, xlim in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))

        data = poly_df[col].dropna()
        ax.hist(data, bins=30, color='steelblue', edgecolor='black', alpha=0.7)

        ax.set_xlabel(title, fontsize=12)
        ax.set_ylabel('Number of Neurons', fontsize=12)
        ax.set_title(f'Distribution of {title}\n(N = {len(data)} neurons)', fontsize=14)
        ax.set_xlim(xlim)

        # Add mean line
        mean_val = data.mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean = {mean_val:.3f}')
        ax.legend(fontsize=11)

        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        save_path = OUTPUT_DIR / f"histogram_{col}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()


# ============================================================
# 2. Violin Plots by Model
# ============================================================

def plot_violins(poly_df):
    """Create violin plots grouped by model for key metrics."""
    metrics = [
        ('diversity_score', 'Weighted Diversity Score', (0, 1)),
        ('n_eff', 'Effective Number of Descriptions (n_eff)', (1, 5)),
        ('sum_weights', 'Total Reliability Mass', (0, 5)),
        ('median_sim_unweighted', 'Median Pairwise Similarity (Unweighted)', (0, 1)),
    ]

    for col, title, ylim in metrics:
        fig, ax = plt.subplots(figsize=(9, 6))

        # Prepare data per model
        data_per_model = []
        positions = []
        labels = []
        for i, model in enumerate(MODEL_ORDER):
            model_data = poly_df[poly_df['model'] == model][col].dropna().values
            data_per_model.append(model_data)
            positions.append(i + 1)
            labels.append(MODEL_LABELS[model])

        # Create violin plot
        parts = ax.violinplot(data_per_model, positions=positions, showmeans=True, showmedians=True)

        # Style violins
        for i, pc in enumerate(parts['bodies']):
            model = MODEL_ORDER[i]
            pc.set_facecolor(MODEL_COLORS[model])
            pc.set_alpha(0.6)

        # Style lines
        for partname in ('cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes'):
            if partname in parts:
                parts[partname].set_edgecolor('black')
                parts[partname].set_linewidth(1)

        # Overlay individual points (jittered)
        for i, model in enumerate(MODEL_ORDER):
            model_data = poly_df[poly_df['model'] == model][col].dropna().values
            jitter = np.random.default_rng(42).normal(0, 0.04, size=len(model_data))
            ax.scatter(
                positions[i] + jitter, model_data,
                color=MODEL_COLORS[model], alpha=0.3, s=15, zorder=3
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title} by Model', fontsize=14)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        save_path = OUTPUT_DIR / f"violin_{col}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()


# ============================================================
# 3. Scatter Plots
# ============================================================

def plot_scatters(poly_df):
    """Create scatter plots of unweighted diversity vs other metrics."""
    scatter_configs = [
        ('sum_weights', 'Total Reliability Mass'),
        ('n_eff', 'Effective Number of Descriptions (n_eff)'),
    ]

    for x_col, x_label in scatter_configs:
        fig, ax = plt.subplots(figsize=(8, 6))

        for model in MODEL_ORDER:
            subset = poly_df[poly_df['model'] == model]
            ax.scatter(
                subset[x_col], subset['unweighted_diversity'],
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
                alpha=0.6, s=40, edgecolors='black', linewidths=0.3
            )

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Unweighted Diversity Score', fontsize=12)
        ax.set_title(f'Unweighted Diversity vs {x_label}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = OUTPUT_DIR / f"scatter_unweighted_diversity_vs_{x_col}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()


# ============================================================
# 4. Individual AUC Distribution
# ============================================================

def plot_auc_distribution(pw_df):
    """Plot distribution of all individual AUC scores."""
    auc_cols = [f'auc_{i}' for i in range(1, 6)]
    all_aucs = pw_df[auc_cols].values.flatten()
    all_aucs = all_aucs[~np.isnan(all_aucs)]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(all_aucs, bins=50, color='steelblue', edgecolor='black', alpha=0.7)

    ax.set_xlabel('AUC Score', fontsize=12)
    ax.set_ylabel('Number of Descriptions', fontsize=12)
    ax.set_title(f'Distribution of Individual AUC Scores\n(N = {len(all_aucs)} descriptions)', fontsize=14)
    ax.set_xlim(0, 1)

    # Add reference lines
    mean_val = np.mean(all_aucs)
    ax.axvline(0.5, color='gray', linestyle=':', linewidth=1.5, label='Chance level (0.5)')
    ax.axvline(0.55, color='orange', linestyle='--', linewidth=1.5, label='Reliability threshold (0.55)')
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean = {mean_val:.3f}')
    ax.legend(fontsize=10)

    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    save_path = OUTPUT_DIR / "histogram_individual_aucs.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()

    # Also per model
    fig, ax = plt.subplots(figsize=(9, 6))

    for model in MODEL_ORDER:
        subset = pw_df[pw_df['model'] == model]
        model_aucs = subset[auc_cols].values.flatten()
        model_aucs = model_aucs[~np.isnan(model_aucs)]
        ax.hist(model_aucs, bins=40, color=MODEL_COLORS[model],
                alpha=0.4, label=MODEL_LABELS[model], edgecolor='black', linewidth=0.3)

    ax.axvline(0.5, color='gray', linestyle=':', linewidth=1.5, label='Chance level')
    ax.set_xlabel('AUC Score', fontsize=12)
    ax.set_ylabel('Number of Descriptions', fontsize=12)
    ax.set_title('Distribution of Individual AUC Scores by Model', fontsize=14)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    save_path = OUTPUT_DIR / "histogram_individual_aucs_by_model.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


# ============================================================
# 5. Within-Neuron Similarity Spread
# ============================================================

def plot_similarity_spread(pw_df):
    """Plot distribution of within-neuron similarity spread."""
    fig, ax = plt.subplots(figsize=(8, 5))

    std_sims = pw_df['std_similarity'].dropna()

    ax.hist(std_sims, bins=30, color='steelblue', edgecolor='black', alpha=0.7)

    ax.set_xlabel('Std. Dev. of Pairwise Similarities (within neuron)', fontsize=12)
    ax.set_ylabel('Number of Neurons', fontsize=12)
    ax.set_title(f'Within-Neuron Similarity Spread\n(N = {len(std_sims)} neurons)', fontsize=14)

    mean_val = std_sims.mean()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean = {mean_val:.3f}')
    ax.legend(fontsize=11)

    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    save_path = OUTPUT_DIR / "histogram_similarity_spread.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("POPULATION-LEVEL POLYSEMANTICITY SCORE VISUALIZATION")
    print("=" * 80)

    poly_df, pw_df = load_data()
    print(f"\nLoaded {len(poly_df)} neurons from polysemanticity scores")
    print(f"Loaded {len(pw_df)} neurons from pairwise similarities")
    print(f"Models: {sorted(poly_df['model'].unique())}")

    print("\n--- 1. Overall Histograms ---")
    plot_histograms(poly_df)

    print("\n--- 2. Violin Plots by Model ---")
    plot_violins(poly_df)

    print("\n--- 3. Scatter Plots ---")
    plot_scatters(poly_df)

    print("\n--- 4. Individual AUC Distribution ---")
    plot_auc_distribution(pw_df)

    print("\n--- 5. Within-Neuron Similarity Spread ---")
    plot_similarity_spread(pw_df)

    print(f"\n{'=' * 80}")
    print("ALL PLOTS GENERATED")
    print(f"{'=' * 80}")
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
