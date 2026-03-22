"""Training loop for the causal transformer."""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.transformer import CausalTransformer


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    batch_size: int = 32
    grad_accumulation_steps: int = 1

    # Schedule
    num_epochs: int = 100
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1  # min LR = lr * min_lr_ratio

    # Logging & checkpointing
    log_interval: int = 10       # Log every N steps
    eval_interval: int = 50      # Evaluate every N steps
    save_interval: int = 200     # Save checkpoint every N steps
    save_top_k: int = 3          # Keep top K checkpoints by val loss
    output_dir: str = "runs/default"

    # Early stopping
    patience: int = 20           # Stop after N evals without improvement
    patience_counter: int = 0


def get_lr(step: int, config: TrainConfig, total_steps: int = 0) -> float:
    """Cosine annealing with warmup."""
    if step < config.warmup_steps:
        return config.learning_rate * step / max(config.warmup_steps, 1)
    # Cosine decay
    if total_steps <= 0:
        total_steps = config.num_epochs * 1000  # fallback estimate
    decay_ratio = (step - config.warmup_steps) / max(total_steps - config.warmup_steps, 1)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.learning_rate * (config.min_lr_ratio + (1 - config.min_lr_ratio) * coeff)


class Trainer:
    """Training loop with bf16, gradient accumulation, and checkpointing."""

    def __init__(
        self,
        model: CausalTransformer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: TrainConfig,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        param_groups = [
            {"params": [p for n, p in model.named_parameters() if "norm" not in n and p.requires_grad],
             "weight_decay": config.weight_decay},
            {"params": [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad],
             "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(param_groups, lr=config.learning_rate, betas=(0.9, 0.95))

        # Logging
        self.log_file = self.output_dir / "training_log.jsonl"
        self.best_val_loss = float("inf")
        self.saved_checkpoints = []  # (val_loss, path) tuples

    def train(self):
        """Run the full training loop."""
        step = 0
        self.model.train()

        log_entries = []
        start_time = time.time()
        total_steps = self.config.num_epochs * len(self.train_loader)

        for epoch in range(self.config.num_epochs):
            for batch in self.train_loader:
                # Update learning rate
                lr = get_lr(step, self.config, total_steps)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

                # Forward pass with bf16
                input_ids = batch["input_ids"].to(self.device)
                targets = batch["target_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                with torch.autocast(self.device, dtype=torch.bfloat16):
                    output = self.model(input_ids, attention_mask=attention_mask, targets=targets)
                    loss = output["loss"] / self.config.grad_accumulation_steps

                loss.backward()

                if (step + 1) % self.config.grad_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Logging
                if step % self.config.log_interval == 0:
                    elapsed = time.time() - start_time
                    entry = {
                        "step": step,
                        "epoch": epoch,
                        "train_loss": (loss * self.config.grad_accumulation_steps).item(),
                        "lr": lr,
                        "elapsed_s": round(elapsed, 1),
                    }
                    log_entries.append(entry)
                    print(
                        f"Step {step:5d} | Epoch {epoch:3d} | "
                        f"Loss {entry['train_loss']:.4f} | LR {lr:.2e} | "
                        f"Time {elapsed:.0f}s"
                    )

                # Evaluation
                if self.val_loader and step > 0 and step % self.config.eval_interval == 0:
                    val_loss = self.evaluate()
                    print(f"  -> Val loss: {val_loss:.4f} (best: {self.best_val_loss:.4f})")

                    log_entries[-1]["val_loss"] = val_loss

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.config.patience_counter = 0
                        self.save_checkpoint(step, val_loss, is_best=True)
                    else:
                        self.config.patience_counter += 1

                    if self.config.patience_counter >= self.config.patience:
                        print(f"Early stopping at step {step}")
                        self._write_logs(log_entries)
                        return

                    self.model.train()

                # Periodic save
                if step > 0 and step % self.config.save_interval == 0:
                    self.save_checkpoint(step, self.best_val_loss)

                step += 1

        # Final evaluation and save
        if self.val_loader:
            val_loss = self.evaluate()
            print(f"Final val loss: {val_loss:.4f}")

        self.save_checkpoint(step, self.best_val_loss, is_best=True)
        self._write_logs(log_entries)
        print(f"Training complete. Best val loss: {self.best_val_loss:.4f}")

    @torch.no_grad()
    def evaluate(self) -> float:
        """Compute average loss on validation set."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            targets = batch["target_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with torch.autocast(self.device, dtype=torch.bfloat16):
                output = self.model(input_ids, attention_mask=attention_mask, targets=targets)

            total_loss += output["loss"].item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def save_checkpoint(self, step: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        ckpt_path = self.output_dir / f"checkpoint_step{step}.pt"
        torch.save({
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.model.config,
            "val_loss": val_loss,
        }, ckpt_path)

        self.saved_checkpoints.append((val_loss, ckpt_path))
        self.saved_checkpoints.sort(key=lambda x: x[0])

        # Keep only top K
        while len(self.saved_checkpoints) > self.config.save_top_k:
            _, old_path = self.saved_checkpoints.pop()
            if old_path.exists() and old_path != ckpt_path:
                old_path.unlink()

        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save({
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "config": self.model.config,
                "val_loss": val_loss,
            }, best_path)

    def _write_logs(self, entries: list[dict]):
        with open(self.log_file, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
