#!/usr/bin/env python3
"""
Compute AUC for a description from generated evaluation texts.

This script takes generated text samples (one per line) and computes the AUC
score by comparing activations against control activations for a specific neuron.

Usage:
    python3 src/compute_auc_from_texts.py \
        --texts path/to/generated_texts.txt \
        --model gpt2-xl \
        --layer 0 \
        --unit 5551 \
        [--description "Optional description for logging"]

The texts file can use one of two formats:
1. One text sample per line (default)
2. Texts separated by a delimiter (use --separator '---')

The second format allows you to paste LLM output directly without reformatting.
"""

import argparse
import csv
import gc
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import set_seed

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import constants, data, helper_modules, models

set_seed(42)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Model-specific configurations
MODEL_CONFIGS = {
    "gpt2-xl": {
        "hook_id": "mlp.hook_post",
    },
    "Llama-3.1-8B-Instruct": {
        "hook_id": "mlp.hook_post",
    },
    "gemma-scope-2b": {
        "hook_id": "hook_resid_post",
        "trained_layer": "res",
        "width": 16,
    },
    "gpt2-small-sae": {
        "hook_id": "hook_resid_post",
        "version": "v5_32k",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute AUC for a description from generated evaluation texts."
    )
    parser.add_argument(
        "--texts",
        type=str,
        required=True,
        help="Path to text file with generated samples (one per line)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Target model name",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer ID",
    )
    parser.add_argument(
        "--unit",
        type=int,
        required=True,
        help="Unit/neuron ID",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Optional description text for logging",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to use (default: cpu)",
    )
    parser.add_argument(
        "--control-size",
        type=int,
        default=1000,
        help="Number of control samples (default: 1000)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum text length (default: 512)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output",
    )
    parser.add_argument(
        "--separator",
        type=str,
        default=None,
        help="Separator between texts (e.g., '---' or '==='). If not specified, assumes one text per line.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output CSV file for results. If not specified, saves to results.csv in the same folder as the texts file.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file, only print to terminal.",
    )
    return parser.parse_args()


def load_texts(text_file: str, separator: str = None) -> list[str]:
    """Load text samples from file.

    Supports two formats:
    1. One text per line (default) - each line is a separate text sample
    2. Separator-based - texts separated by a delimiter (e.g., '---' or '===')
       Use --separator flag to specify the delimiter

    When using separator mode, line breaks within each text are preserved
    and converted to spaces.
    """
    with open(text_file, "r", encoding="utf-8") as f:
        content = f.read()

    if separator:
        # Split by separator, clean up each text
        raw_texts = content.split(separator)
        texts = []
        for t in raw_texts:
            # Replace line breaks with spaces, collapse multiple spaces
            cleaned = " ".join(t.split())
            if cleaned:
                texts.append(cleaned)
    else:
        # One per line mode
        texts = [line.strip() for line in content.split("\n") if line.strip()]

    return texts


def get_or_compute_control_activations(
    model,
    tokenizer,
    layer_id: int,
    unit_id: int,
    hook_id: str,
    device: torch.device,
    control_size: int,
    max_length: int,
    model_name: str,
    verbose: bool = False,
) -> torch.Tensor:
    """Load cached control activations or compute them."""
    # Define cache path
    activations_path = Path(constants.ACTIVATIONS_PATH) / model_name
    activations_path.mkdir(parents=True, exist_ok=True)

    control_tensor_path = (
        activations_path / f"control_{model_name}_layer{layer_id}_mean_cosmopedia_{control_size}.pt"
    )

    if control_tensor_path.exists():
        if verbose:
            print(f"Loading cached control activations from {control_tensor_path}")
        return torch.load(control_tensor_path, weights_only=True, map_location="cpu")

    if verbose:
        print(f"Computing control activations ({control_size} samples)...")

    # Load control dataset
    control_dataset = data.get_dataset(
        data_name="cosmopedia",
        data_files=None,
        split="train",
        streaming=True,
    )
    subset_samples = data.collect_samples(control_dataset, control_size)

    control_ds = Dataset.from_pandas(pd.DataFrame(subset_samples))
    del subset_samples

    control_ds = control_ds.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        ),
        batched=True,
    )
    control_ds = control_ds.with_format("torch")
    control_loader = DataLoader(control_ds, batch_size=64, shuffle=False)

    # Get activations
    activations = models.get_activations(
        model=model,
        data_loader=control_loader,
        tensor_path=str(control_tensor_path),
        unit_id=None,
        hook_name=f"blocks.{layer_id}.{hook_id}",
        agg_method="mean",
        save=True,
    )
    activations = activations.cpu()

    del control_ds, control_loader
    gc.collect()

    return activations


class TextListDataset(torch.utils.data.Dataset):
    """Simple dataset for a list of text strings."""

    def __init__(self, texts: list[str], tokenizer, max_length: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )
        return {
            "input_ids": torch.tensor(tokens["input_ids"]),
            "attention_mask": torch.tensor(tokens["attention_mask"]),
        }


def get_explanation_activations(
    model,
    tokenizer,
    texts: list[str],
    layer_id: int,
    unit_id: int,
    hook_id: str,
    device: torch.device,
    max_length: int,
    verbose: bool = False,
) -> torch.Tensor:
    """Get activations for explanation texts."""
    if verbose:
        print(f"Computing activations for {len(texts)} explanation texts...")

    # Create dataset from texts using our simple class
    explain_dataset = TextListDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    explain_loader = DataLoader(explain_dataset, batch_size=2, shuffle=False)

    # Get activations (no caching for explanation texts)
    activations = models.get_activations(
        model=model,
        data_loader=explain_loader,
        tensor_path=None,
        unit_id=unit_id,
        hook_name=f"blocks.{layer_id}.{hook_id}",
        agg_method="mean",
        save=False,
    )
    activations = activations.cpu()

    del explain_dataset, explain_loader
    gc.collect()

    return activations


def compute_auc(
    control_activations: torch.Tensor,
    explanation_activations: torch.Tensor,
    unit_id: int,
) -> dict:
    """Compute AUC and related statistics."""
    # Extract unit activations from control
    activ_non_concept = control_activations[:, unit_id]
    activ_concept = explanation_activations

    # Create labels: 0 for control, 1 for explanation
    labels = torch.cat([
        torch.zeros(activ_non_concept.shape[0]),
        torch.ones(activ_concept.shape[0]),
    ])

    # Concatenate activations
    all_activations = torch.cat([activ_non_concept.cpu(), activ_concept.cpu()])

    # Compute AUC
    auc = roc_auc_score(labels.numpy(), all_activations.numpy())

    # Compute Mann-Whitney U test
    U1, p = mannwhitneyu(activ_non_concept.numpy(), activ_concept.numpy())

    # Compute MAD (Mean Activation Difference normalized by std)
    if activ_non_concept.std().item() == 0:
        mad = 0.0
    else:
        mad = (
            activ_concept.mean().item() - activ_non_concept.mean().item()
        ) / activ_non_concept.std().item()

    return {
        "auc": auc,
        "U1": U1,
        "p_value": p,
        "mad": mad,
        "mean_control": activ_non_concept.mean().item(),
        "mean_explanation": activ_concept.mean().item(),
        "std_control": activ_non_concept.std().item(),
        "std_explanation": activ_concept.std().item(),
        "n_control": len(activ_non_concept),
        "n_explanation": len(activ_concept),
    }


def main():
    args = parse_args()

    print("=" * 60)
    print("COMPUTE AUC FROM GENERATED TEXTS")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Layer: {args.layer}")
    print(f"Unit: {args.unit}")
    print(f"Texts file: {args.texts}")
    if args.description:
        print(f"Description: {args.description}")
    print("=" * 60)

    # Validate texts file
    if not Path(args.texts).exists():
        print(f"ERROR: Texts file not found: {args.texts}")
        sys.exit(1)

    # Load texts
    texts = load_texts(args.texts, separator=args.separator)
    print(f"\nLoaded {len(texts)} text samples")
    if args.separator:
        print(f"(using separator: '{args.separator}')")
    if args.verbose:
        for i, t in enumerate(texts[:3]):
            print(f"  Sample {i+1}: {t[:80]}...")

    # Get model config
    model_config = MODEL_CONFIGS[args.model]
    hook_id = model_config["hook_id"]

    # Setup device
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    print(f"\nDevice: {device}")

    # Load model and tokenizer
    print(f"\nLoading model: {args.model}...")

    # Set config values (required by get_activations and get_model)
    from utils import config
    config.TARGET_MODEL_NAME = args.model
    config.DEVICE = args.device
    config.LAYER_ID = args.layer
    config.HOOK_ID = hook_id

    if args.model == "gemma-scope-2b":
        config.TRAINED_LAYER = model_config["trained_layer"]
        config.WIDTH = model_config["width"]
    elif args.model == "gpt2-small-sae":
        config.VERSION = model_config["version"]

    model = models.get_model(args.model)
    model.to(device)
    model.eval()

    tokenizer = models.get_tokenizer(args.model)
    print("Model and tokenizer loaded")

    # Get control activations
    print("\n--- Control Activations ---")
    control_activations = get_or_compute_control_activations(
        model=model,
        tokenizer=tokenizer,
        layer_id=args.layer,
        unit_id=args.unit,
        hook_id=hook_id,
        device=device,
        control_size=args.control_size,
        max_length=args.max_length,
        model_name=args.model,
        verbose=args.verbose,
    )
    print(f"Control activations shape: {control_activations.shape}")

    # Get explanation activations
    print("\n--- Explanation Activations ---")
    explanation_activations = get_explanation_activations(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        layer_id=args.layer,
        unit_id=args.unit,
        hook_id=hook_id,
        device=device,
        max_length=args.max_length,
        verbose=args.verbose,
    )
    print(f"Explanation activations shape: {explanation_activations.shape}")

    # Compute AUC
    print("\n--- Results ---")
    results = compute_auc(control_activations, explanation_activations, args.unit)

    print(f"\n{'='*40}")
    print(f"AUC: {results['auc']:.4f}")
    print(f"{'='*40}")

    # Interpret AUC
    if results['auc'] >= 0.55:
        interpretation = "RELIABLE (AUC >= 0.55)"
    elif results['auc'] >= 0.5:
        interpretation = "MARGINAL (0.5 <= AUC < 0.55)"
    else:
        interpretation = "BELOW CHANCE (AUC < 0.5)"
    print(f"Interpretation: {interpretation}")

    if args.verbose:
        print(f"\nDetailed Statistics:")
        print(f"  Mann-Whitney U: {results['U1']:.2f}")
        print(f"  p-value: {results['p_value']:.6f}")
        print(f"  MAD: {results['mad']:.4f}")
        print(f"  Mean (control): {results['mean_control']:.4f}")
        print(f"  Mean (explanation): {results['mean_explanation']:.4f}")
        print(f"  Std (control): {results['std_control']:.4f}")
        print(f"  Std (explanation): {results['std_explanation']:.4f}")
        print(f"  N (control): {results['n_control']}")
        print(f"  N (explanation): {results['n_explanation']}")

    # Save results to CSV
    if not args.no_save:
        if args.output:
            output_path = Path(args.output)
        else:
            # Save to results.csv in the same folder as the texts file
            output_path = Path(args.texts).parent / "results.csv"

        # Check if file exists to determine if we need to write header
        file_exists = output_path.exists()

        row = {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "layer": args.layer,
            "unit": args.unit,
            "description": args.description or "",
            "texts_file": Path(args.texts).name,
            "n_texts": len(texts),
            "auc": results["auc"],
            "mad": results["mad"],
            "p_value": results["p_value"],
            "mean_control": results["mean_control"],
            "mean_explanation": results["mean_explanation"],
        }

        with open(output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        print(f"\nResults saved to: {output_path}")

    # Cleanup
    del model, tokenizer, control_activations, explanation_activations
    helper_modules.clear_gpu_cache()
    gc.collect()

    print("\nDone.")
    return results


if __name__ == "__main__":
    main()
