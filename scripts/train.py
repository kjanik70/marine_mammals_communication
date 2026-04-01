#!/usr/bin/env python3
"""Training script for marine mammal communication LLM."""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import create_symbolic_datasets, AudioTokenDataset
from src.model.config import get_config
from src.model.transformer import CausalTransformer
from src.training.trainer import Trainer, TrainConfig


def main():
    parser = argparse.ArgumentParser(description="Train marine mammal LLM")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--init-from", type=str, default=None,
                        help="Initialize model weights from a checkpoint (e.g., runs/.../best_model.pt)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"Config: {args.config}")
    print(f"Device: {args.device}")

    # Create datasets
    print("\n=== Loading data ===")
    dataset_type = cfg["data"]["dataset_type"]

    if dataset_type == "audio":
        # Audio token dataset (Track 2)
        token_dir = cfg["data"]["token_dir"]
        augment = cfg["data"].get("augment", False)
        vocab_size = cfg["model"]["vocab_size"]
        concat = cfg["data"].get("concat", False)
        sep_token = cfg["data"].get("sep_token", None)

        # Common dataset kwargs
        ds_kwargs = dict(
            max_seq_len=cfg["data"].get("max_seq_len", 512),
            vocab_size=vocab_size,
            concat=concat,
            sep_token=sep_token,
        )

        # Build train set (with augmentation if enabled)
        full_ds_noaug = AudioTokenDataset(
            token_dir, augment=False, **ds_kwargs,
        )
        # Split 80/20 train/val
        n_total = len(full_ds_noaug)
        n_train = int(n_total * 0.8)
        n_val = n_total - n_train

        # Get split indices
        g = torch.Generator().manual_seed(42)
        indices = torch.randperm(n_total, generator=g).tolist()
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        if augment:
            # Rebuild with augmentation for training
            full_ds_aug = AudioTokenDataset(
                token_dir,
                augment=True,
                token_noise_prob=cfg["data"].get("token_noise_prob", 0.05),
                token_mask_prob=cfg["data"].get("token_mask_prob", 0.02),
                time_stretch_prob=cfg["data"].get("time_stretch_prob", 0.3),
                **ds_kwargs,
            )
            train_ds = torch.utils.data.Subset(full_ds_aug, train_indices)
        else:
            train_ds = torch.utils.data.Subset(full_ds_noaug, train_indices)

        val_ds = torch.utils.data.Subset(full_ds_noaug, val_indices)

        print(f"Dataset: audio tokens from {token_dir}")
        print(f"Total windows: {n_total} (train: {n_train}, val: {n_val})")
        print(f"Augmentation: {augment}, Concat: {concat}")
        print(f"Vocab size: {vocab_size}")
    else:
        # Symbolic datasets (Track 1)
        datasets = create_symbolic_datasets(
            cfg["data"]["codas_csv"],
            cfg["data"]["dialogues_csv"],
            max_seq_len=cfg["data"].get("max_seq_len", 128),
            dialogue_max_seq_len=cfg["data"].get("max_seq_len", 256),
        )
        if dataset_type == "coda":
            train_ds = datasets["coda_train"]
            val_ds = datasets["coda_val"]
        elif dataset_type == "dialogue":
            train_ds = datasets["dialogue_train"]
            val_ds = datasets["dialogue_val"]
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")

        vocab_size = datasets["vocab"].vocab_size
        print(f"Dataset: {dataset_type}")
        print(f"Train: {len(train_ds)} samples")
        print(f"Val: {len(val_ds)} samples")
        print(f"Vocab size: {vocab_size}")

    batch_size = cfg["training"]["batch_size"]
    loader_kwargs = dict(
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=4,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    # Create model
    print("\n=== Creating model ===")
    # Pass all model overrides from YAML (dropout, vocab_size, max_seq_len, etc.)
    model_overrides = {k: v for k, v in cfg["model"].items() if k != "preset"}
    model_cfg = get_config(cfg["model"]["preset"], **model_overrides)
    model = CausalTransformer(model_cfg)
    n_params = model.count_parameters()
    print(f"Model: {cfg['model']['preset']} ({n_params:,} parameters)")

    if args.init_from:
        print(f"Initializing weights from {args.init_from}")
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded checkpoint (step {ckpt.get('step', '?')}, val_loss {ckpt.get('val_loss', '?')})")

    # Training config
    train_cfg = TrainConfig(
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        batch_size=batch_size,
        grad_accumulation_steps=cfg["training"].get("grad_accumulation_steps", 1),
        num_epochs=cfg["training"]["num_epochs"],
        warmup_steps=cfg["training"]["warmup_steps"],
        min_lr_ratio=cfg["training"].get("min_lr_ratio", 0.1),
        log_interval=cfg["training"]["log_interval"],
        eval_interval=cfg["training"]["eval_interval"],
        save_interval=cfg["training"]["save_interval"],
        patience=cfg["training"].get("patience", 20),
        output_dir=cfg["training"]["output_dir"],
    )

    # Train
    print(f"\n=== Training ===")
    print(f"Output: {train_cfg.output_dir}")
    trainer = Trainer(model, train_loader, val_loader, train_cfg, device=args.device)
    trainer.train()


if __name__ == "__main__":
    main()
