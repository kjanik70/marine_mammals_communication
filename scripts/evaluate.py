#!/usr/bin/env python3
"""Evaluate a trained marine mammal communication model."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import create_symbolic_datasets
from src.data.symbolic_tokenizer import SymbolicVocab, decode_token_sequence
from src.evaluation.metrics import compute_perplexity, compute_accuracy, analyze_generated_sequences
from src.evaluation.visualize import plot_training_curves, plot_coda_distribution
from src.model.transformer import CausalTransformer


def main():
    parser = argparse.ArgumentParser(description="Evaluate marine mammal LLM")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--dataset-type", type=str, default="coda", choices=["coda", "dialogue"])
    parser.add_argument("--n-generate", type=int, default=50, help="Number of sequences to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Load model
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=args.device)
    cfg = ckpt["config"]
    model = CausalTransformer(cfg).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded: {model.count_parameters():,} params, val_loss={ckpt.get('val_loss', '?')}")

    vocab = SymbolicVocab()

    # Output dir
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets
    datasets = create_symbolic_datasets(
        "data/raw/ceti/DominicaCodas.csv",
        "data/raw/ceti/sperm-whale-dialogues.csv",
        max_seq_len=cfg.max_seq_len,
        dialogue_max_seq_len=cfg.max_seq_len,
    )

    if args.dataset_type == "coda":
        test_ds = datasets["coda_test"]
    else:
        test_ds = datasets["dialogue_test"]

    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    # Perplexity
    print("\n=== Perplexity ===")
    ppl = compute_perplexity(model, test_loader, args.device)
    print(f"Test perplexity: {ppl:.2f}")

    # Accuracy
    print("\n=== Accuracy ===")
    acc = compute_accuracy(model, test_loader, vocab, args.device)
    print(f"Top-1 accuracy: {acc['top1_accuracy']:.4f}")
    print(f"Top-5 accuracy: {acc['top5_accuracy']:.4f}")
    print(f"Per-type accuracy:")
    for t, a in sorted(acc["per_type_accuracy"].items(), key=lambda x: -x[1])[:10]:
        print(f"  {t}: {a:.3f}")

    # Generate sequences
    print(f"\n=== Generating {args.n_generate} sequences ===")
    bos = torch.tensor([[vocab.BOS]], device=args.device)
    generated = []
    for i in range(args.n_generate):
        gen = model.generate(
            bos, max_new_tokens=50,
            temperature=args.temperature, top_k=20,
        )
        tokens = gen[0].tolist()
        generated.append(tokens)

    # Analyze generated
    analysis = analyze_generated_sequences(generated, vocab)
    print(f"Avg codas per sequence: {analysis['avg_codas_per_seq']:.1f}")
    print("Generated coda distribution:")
    for t, p in list(analysis["coda_distribution"].items())[:10]:
        print(f"  {t}: {p:.3f}")

    # Compute real distribution for comparison
    from collections import Counter
    import pandas as pd
    codas = pd.read_csv("data/raw/ceti/DominicaCodas.csv", encoding="utf-8-sig")
    real_counts = Counter(codas[~codas["CodaType"].str.contains("NOISE")]["CodaType"])
    total = sum(real_counts.values())
    real_dist = {k: v / total for k, v in real_counts.most_common()}

    # Plot
    fig = plot_coda_distribution(real_dist, analysis["coda_distribution"], output_dir / "coda_dist.png")
    print(f"\nCoda distribution plot saved to {output_dir / 'coda_dist.png'}")

    # Plot training curves if log exists
    log_file = Path(args.checkpoint).parent / "training_log.jsonl"
    if log_file.exists():
        plot_training_curves(log_file, output_dir / "training_curves.png")
        print(f"Training curves saved to {output_dir / 'training_curves.png'}")

    # Save results
    results = {
        "perplexity": ppl,
        "top1_accuracy": acc["top1_accuracy"],
        "top5_accuracy": acc["top5_accuracy"],
        "generated_analysis": {
            "n_sequences": analysis["n_sequences"],
            "avg_codas_per_seq": analysis["avg_codas_per_seq"],
        },
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print some generated examples
    print("\n=== Sample generated sequences ===")
    for i in range(min(5, len(generated))):
        decoded = decode_token_sequence(generated[i], vocab)
        if "<eos>" in decoded:
            decoded = decoded[:decoded.index("<eos>") + 1]
        print(f"  {i+1}: {' '.join(decoded)}")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
