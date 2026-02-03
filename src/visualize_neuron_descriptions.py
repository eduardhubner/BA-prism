"""
Visualize neuron description similarity and reliability.

Creates two types of visualizations:
1. Network graph: Nodes = descriptions (color = AUC), edges = similarity
2. MDS scatter plot: 2D embedding where distance ≈ dissimilarity (1 - similarity)

Usage:
    python3 src/visualize_neuron_descriptions.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from sklearn.manifold import MDS
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings('ignore')

# Paths
RESULTS_DIR = Path(__file__).parent.parent / "results"
PAIRWISE_FILE = RESULTS_DIR / "pairwise_similarities_dataset.csv"
OUTPUT_DIR = RESULTS_DIR / "polysemanticity_experiments" / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Pair ordering (same as polysemanticity_scoring.py)
PAIRS = [(1,2), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5)]


def load_neuron_data(model: str, layer: int, unit: int) -> dict:
    """
    Load pairwise similarities and AUCs for a specific neuron.

    Returns:
        dict with 'pairwise_sims' (10 values), 'aucs' (5 values), 'metadata'
    """
    df = pd.read_csv(PAIRWISE_FILE)

    # Find neuron
    neuron = df[(df['model'] == model) &
                (df['layer'] == layer) &
                (df['unit'] == unit)]

    if len(neuron) == 0:
        raise ValueError(f"Neuron not found: {model} L{layer}_U{unit}")

    row = neuron.iloc[0]

    # Extract pairwise similarities
    sim_cols = [f'sim_{i}{j}' for i, j in PAIRS]
    pairwise_sims = np.array([row[col] for col in sim_cols])

    # Extract AUCs
    aucs = np.array([row[f'auc_{i}'] for i in range(1, 6)])

    # Metadata
    metadata = {
        'model': model,
        'layer': layer,
        'unit': unit,
        'mean_similarity': row['mean_similarity'],
        'diversity': 1 - row['mean_similarity'],
    }

    return {
        'pairwise_sims': pairwise_sims,
        'aucs': aucs,
        'metadata': metadata,
    }


def create_similarity_matrix(pairwise_sims: np.ndarray) -> np.ndarray:
    """
    Convert pairwise similarities to full 5×5 similarity matrix.

    Args:
        pairwise_sims: Array of 10 pairwise similarities (upper triangle)

    Returns:
        5×5 symmetric similarity matrix
    """
    n = 5
    sim_matrix = np.eye(n)  # Diagonal = 1 (self-similarity)

    # Fill upper triangle
    for idx, (i, j) in enumerate(PAIRS):
        sim_matrix[i-1, j-1] = pairwise_sims[idx]
        sim_matrix[j-1, i-1] = pairwise_sims[idx]  # Symmetric

    return sim_matrix


def visualize_network_graph(neuron_data: dict, save_path: Path = None, show: bool = True):
    """
    Visualize descriptions as network graph.

    Nodes: Descriptions (color-coded by AUC)
    Edges: Similarity (thickness = similarity strength)
    """
    pairwise_sims = neuron_data['pairwise_sims']
    aucs = neuron_data['aucs']
    metadata = neuron_data['metadata']

    # Create graph
    G = nx.Graph()

    # Add nodes (descriptions 1-5)
    for i in range(1, 6):
        G.add_node(i, auc=aucs[i-1])

    # Add edges (with weight = similarity)
    for idx, (i, j) in enumerate(PAIRS):
        sim = pairwise_sims[idx]
        G.add_edge(i, j, weight=sim)

    # Layout: spring layout (force-directed)
    # High similarity → nodes pulled together
    pos = nx.spring_layout(G, weight='weight', seed=42, k=0.5, iterations=50)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw edges (thickness = similarity)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    # Normalize edge widths
    min_width, max_width = 0.5, 5.0
    edge_widths = [min_width + (w * (max_width - min_width)) for w in weights]

    # Edge colors (based on similarity)
    edge_colors = plt.cm.Greys([0.3 + 0.6 * w for w in weights])

    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.6,
        ax=ax
    )

    # Draw nodes (color = AUC)
    node_colors = aucs
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        cmap='RdYlGn',  # Red (low AUC) → Yellow → Green (high AUC)
        vmin=0.0,
        vmax=1.0,
        node_size=1000,
        edgecolors='black',
        linewidths=2,
        ax=ax
    )

    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        labels={i: f"D{i}" for i in range(1, 6)},
        font_size=14,
        font_weight='bold',
        ax=ax
    )

    # Colorbar for AUC
    cbar = plt.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('AUC (Prediction Accuracy)', rotation=270, labelpad=20, fontsize=12)

    # Title
    title = (f"{metadata['model']} Layer {metadata['layer']} Unit {metadata['unit']}\n"
             f"Network Graph: Edge thickness ∝ Similarity\n"
             f"Mean Similarity: {metadata['mean_similarity']:.3f} | "
             f"Diversity: {metadata['diversity']:.3f}")
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_mds_scatter(neuron_data: dict, save_path: Path = None, show: bool = True):
    """
    Visualize descriptions using MDS (Multidimensional Scaling).

    2D embedding where Euclidean distance ≈ dissimilarity (1 - similarity)
    Nodes: Descriptions (color-coded by AUC, size = AUC)
    """
    pairwise_sims = neuron_data['pairwise_sims']
    aucs = neuron_data['aucs']
    metadata = neuron_data['metadata']

    # Create dissimilarity matrix (distance = 1 - similarity)
    sim_matrix = create_similarity_matrix(pairwise_sims)
    dissim_matrix = 1 - sim_matrix

    # Apply MDS to embed in 2D
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    coords = mds.fit_transform(dissim_matrix)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot connections (all pairs)
    for idx, (i, j) in enumerate(PAIRS):
        x_vals = [coords[i-1, 0], coords[j-1, 0]]
        y_vals = [coords[i-1, 1], coords[j-1, 1]]
        sim = pairwise_sims[idx]

        # Line thickness = similarity
        linewidth = 0.5 + 4 * sim
        # Line color = similarity (darker = more similar)
        color = plt.cm.Greys(0.3 + 0.6 * sim)

        ax.plot(x_vals, y_vals,
                color=color,
                linewidth=linewidth,
                alpha=0.5,
                zorder=1)

    # Plot nodes (descriptions)
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=aucs,
        cmap='RdYlGn',  # Red (low) → Yellow → Green (high)
        vmin=0.0,
        vmax=1.0,
        s=1000,  # Size based on AUC
        edgecolors='black',
        linewidths=2,
        zorder=2
    )

    # Add labels
    for i in range(5):
        ax.text(coords[i, 0], coords[i, 1], f"D{i+1}",
                ha='center', va='center',
                fontsize=12, fontweight='bold',
                color='black',
                zorder=3)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('AUC (Prediction Accuracy)', rotation=270, labelpad=20, fontsize=12)

    # Title
    title = (f"{metadata['model']} Layer {metadata['layer']} Unit {metadata['unit']}\n"
             f"MDS Plot: Euclidean distance ≈ Dissimilarity (1 - Similarity)\n"
             f"Mean Similarity: {metadata['mean_similarity']:.3f} | "
             f"Diversity: {metadata['diversity']:.3f}")
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    ax.set_xlabel('MDS Dimension 1', fontsize=12)
    ax.set_ylabel('MDS Dimension 2', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def create_combined_visualization(neuron_data: dict, save_path: Path = None, show: bool = True):
    """
    Create side-by-side visualization: Network graph + MDS scatter.
    """
    pairwise_sims = neuron_data['pairwise_sims']
    aucs = neuron_data['aucs']
    metadata = neuron_data['metadata']

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # ============================================================
    # LEFT: Network Graph
    # ============================================================

    G = nx.Graph()
    for i in range(1, 6):
        G.add_node(i, auc=aucs[i-1])
    for idx, (i, j) in enumerate(PAIRS):
        G.add_edge(i, j, weight=pairwise_sims[idx])

    pos = nx.spring_layout(G, weight='weight', seed=42, k=0.5, iterations=50)

    # Edges
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    edge_widths = [0.5 + 5.0 * w for w in weights]
    edge_colors = plt.cm.Greys([0.3 + 0.6 * w for w in weights])

    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.6, ax=ax1)

    # Nodes
    nodes1 = nx.draw_networkx_nodes(
        G, pos,
        node_color=aucs,
        cmap='RdYlGn',
        vmin=0.0, vmax=1.0,
        node_size=1000,
        edgecolors='black',
        linewidths=2,
        ax=ax1
    )

    # Labels
    nx.draw_networkx_labels(
        G, pos,
        labels={i: f"D{i}" for i in range(1, 6)},
        font_size=14,
        font_weight='bold',
        ax=ax1
    )

    ax1.set_title('Network Graph\nEdge thickness ∝ Similarity', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # ============================================================
    # RIGHT: MDS Scatter
    # ============================================================

    sim_matrix = create_similarity_matrix(pairwise_sims)
    dissim_matrix = 1 - sim_matrix

    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    coords = mds.fit_transform(dissim_matrix)

    # Edges
    for idx, (i, j) in enumerate(PAIRS):
        x_vals = [coords[i-1, 0], coords[j-1, 0]]
        y_vals = [coords[i-1, 1], coords[j-1, 1]]
        sim = pairwise_sims[idx]
        linewidth = 0.5 + 4 * sim
        color = plt.cm.Greys(0.3 + 0.6 * sim)
        ax2.plot(x_vals, y_vals, color=color, linewidth=linewidth, alpha=0.5, zorder=1)

    # Nodes
    scatter = ax2.scatter(
        coords[:, 0], coords[:, 1],
        c=aucs,
        cmap='RdYlGn',
        vmin=0.0, vmax=1.0,
        s=1000,
        edgecolors='black',
        linewidths=2,
        zorder=2
    )

    # Labels
    for i in range(5):
        ax2.text(coords[i, 0], coords[i, 1], f"D{i+1}",
                ha='center', va='center',
                fontsize=12, fontweight='bold',
                color='black',
                zorder=3)

    ax2.set_title('MDS Plot\nDistance ≈ Dissimilarity', fontsize=14, fontweight='bold')
    ax2.set_xlabel('MDS Dimension 1', fontsize=12)
    ax2.set_ylabel('MDS Dimension 2', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')

    # ============================================================
    # Shared colorbar
    # ============================================================

    cbar = fig.colorbar(scatter, ax=[ax1, ax2], fraction=0.046, pad=0.04)
    cbar.set_label('AUC (Prediction Accuracy)', rotation=270, labelpad=20, fontsize=12)

    # ============================================================
    # Overall title
    # ============================================================

    fig.suptitle(
        f"{metadata['model']} Layer {metadata['layer']} Unit {metadata['unit']}\n"
        f"Mean Similarity: {metadata['mean_similarity']:.3f} | Diversity: {metadata['diversity']:.3f}",
        fontsize=16, fontweight='bold', y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def print_neuron_stats(neuron_data: dict):
    """Print detailed statistics for a neuron."""
    pairwise_sims = neuron_data['pairwise_sims']
    aucs = neuron_data['aucs']
    metadata = neuron_data['metadata']

    print(f"\n{'='*60}")
    print(f"Neuron: {metadata['model']} Layer {metadata['layer']} Unit {metadata['unit']}")
    print(f"{'='*60}")

    print(f"\nAUC Scores (Prediction Accuracy):")
    for i, auc in enumerate(aucs, 1):
        reliability = "✓" if auc >= 0.5 else "✗"
        print(f"  Description {i}: {auc:.4f} {reliability}")
    print(f"  Max AUC: {aucs.max():.4f}")
    print(f"  Mean AUC: {aucs.mean():.4f}")
    print(f"  Descriptions above chance (≥0.5): {np.sum(aucs >= 0.5)}/5")

    print(f"\nPairwise Similarities:")
    for idx, (i, j) in enumerate(PAIRS):
        sim = pairwise_sims[idx]
        print(f"  D{i} ↔ D{j}: {sim:.4f}")

    print(f"\nSummary Statistics:")
    print(f"  Mean Similarity: {metadata['mean_similarity']:.4f}")
    print(f"  Diversity (1 - Mean Sim): {metadata['diversity']:.4f}")
    print(f"  Min Similarity: {pairwise_sims.min():.4f}")
    print(f"  Max Similarity: {pairwise_sims.max():.4f}")
    print(f"  Range: {pairwise_sims.max() - pairwise_sims.min():.4f}")


def main():
    """
    Visualize example neurons with different characteristics.
    """
    print("="*80)
    print("NEURON DESCRIPTION VISUALIZATION")
    print("="*80)

    # Example neurons to visualize (selected from actual data)
    examples = [
        # Example 1: Highest diversity + reliable (polysemantic)
        {"model": "gpt2-xl", "layer": 40, "unit": 4808, "label": "highly_polysemantic"},

        # Example 2: Low diversity + reliable (monosemantic)
        {"model": "gemma-scope-2b", "layer": 0, "unit": 13988, "label": "monosemantic"},

        # Example 3: Moderate diversity + high reliability
        {"model": "Llama-3.1-8B-Instruct", "layer": 20, "unit": 8476, "label": "moderate_polysemantic"},
    ]

    for example in examples:
        model = example['model']
        layer = example['layer']
        unit = example['unit']
        label = example['label']

        print(f"\n{'='*60}")
        print(f"Processing: {model} L{layer}_U{unit} ({label})")
        print(f"{'='*60}")

        try:
            # Load neuron data
            neuron_data = load_neuron_data(model, layer, unit)

            # Print statistics
            print_neuron_stats(neuron_data)

            # Create visualizations
            print(f"\nCreating visualizations...")

            # 1. Network graph
            network_path = OUTPUT_DIR / f"{model}_L{layer}_U{unit}_network.png"
            visualize_network_graph(neuron_data, save_path=network_path, show=False)

            # 2. MDS scatter
            mds_path = OUTPUT_DIR / f"{model}_L{layer}_U{unit}_mds.png"
            visualize_mds_scatter(neuron_data, save_path=mds_path, show=False)

            # 3. Combined view
            combined_path = OUTPUT_DIR / f"{model}_L{layer}_U{unit}_combined.png"
            create_combined_visualization(neuron_data, save_path=combined_path, show=False)

            print(f"\n✓ Visualizations complete for {model} L{layer}_U{unit}")

        except Exception as e:
            print(f"Error processing {model} L{layer}_U{unit}: {e}")
            continue

    print(f"\n{'='*80}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nFiles generated:")
    print("  *_network.png   - Network graph visualization")
    print("  *_mds.png       - MDS scatter plot visualization")
    print("  *_combined.png  - Side-by-side comparison")
    print("\nTo visualize custom neurons:")
    print("  1. Edit the 'examples' list in main()")
    print("  2. Add {'model': '...', 'layer': X, 'unit': Y, 'label': '...'}")
    print("  3. Run: python3 src/visualize_neuron_descriptions.py")


if __name__ == "__main__":
    main()
