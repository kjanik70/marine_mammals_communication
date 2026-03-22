"""Evaluation metrics for marine mammal communication models."""

import math
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.model.transformer import CausalTransformer
from src.data.symbolic_tokenizer import SymbolicVocab


@torch.no_grad()
def compute_perplexity(
    model: CausalTransformer,
    dataloader: DataLoader,
    device: str = "cuda",
) -> float:
    """Compute perplexity on a dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        targets = batch["target_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        with torch.autocast(device, dtype=torch.bfloat16):
            output = model(input_ids, attention_mask=mask, targets=targets)

        # Count non-pad tokens
        n_tokens = (targets != 0).sum().item()
        total_loss += output["loss"].item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)


@torch.no_grad()
def compute_accuracy(
    model: CausalTransformer,
    dataloader: DataLoader,
    vocab: SymbolicVocab,
    device: str = "cuda",
    top_k: int = 5,
) -> dict:
    """Compute next-token prediction accuracy.

    Returns dict with top-1 accuracy, top-K accuracy, and per-type accuracy.
    """
    model.eval()
    correct_top1 = 0
    correct_topk = 0
    total = 0
    per_type_correct = Counter()
    per_type_total = Counter()

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        targets = batch["target_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        with torch.autocast(device, dtype=torch.bfloat16):
            output = model(input_ids, attention_mask=mask)

        logits = output["logits"]  # (B, T, V)
        preds_top1 = logits.argmax(dim=-1)  # (B, T)
        preds_topk = logits.topk(top_k, dim=-1).indices  # (B, T, K)

        # Only count non-pad targets that are coda tokens
        for b in range(targets.size(0)):
            for t in range(targets.size(1)):
                tgt = targets[b, t].item()
                if tgt == 0:  # PAD
                    continue
                total += 1
                if preds_top1[b, t].item() == tgt:
                    correct_top1 += 1
                if tgt in preds_topk[b, t].tolist():
                    correct_topk += 1

                if vocab.is_coda_token(tgt):
                    type_name = vocab.decode_token(tgt)
                    per_type_total[type_name] += 1
                    if preds_top1[b, t].item() == tgt:
                        per_type_correct[type_name] += 1

    top1_acc = correct_top1 / max(total, 1)
    topk_acc = correct_topk / max(total, 1)

    per_type_acc = {}
    for t in per_type_total:
        per_type_acc[t] = per_type_correct[t] / per_type_total[t]

    return {
        "top1_accuracy": top1_acc,
        f"top{top_k}_accuracy": topk_acc,
        "total_tokens": total,
        "per_type_accuracy": per_type_acc,
    }


def analyze_generated_sequences(
    sequences: list[list[int]],
    vocab: SymbolicVocab,
) -> dict:
    """Analyze properties of generated token sequences."""
    coda_counts = Counter()
    seq_lengths = []
    whale_transitions = Counter()

    for seq in sequences:
        codas_in_seq = [t for t in seq if vocab.is_coda_token(t)]
        seq_lengths.append(len(codas_in_seq))
        for t in codas_in_seq:
            coda_counts[vocab.decode_token(t)] += 1

        # Count whale-to-whale transitions
        prev_whale = None
        for t in seq:
            if vocab.is_whale_token(t):
                whale_name = vocab.decode_token(t)
                if prev_whale is not None:
                    whale_transitions[(prev_whale, whale_name)] += 1
                prev_whale = whale_name

    total_codas = sum(coda_counts.values())
    coda_dist = {k: v / max(total_codas, 1) for k, v in coda_counts.most_common()}

    return {
        "n_sequences": len(sequences),
        "avg_codas_per_seq": np.mean(seq_lengths) if seq_lengths else 0,
        "coda_distribution": coda_dist,
        "whale_transitions": dict(whale_transitions.most_common()),
    }
