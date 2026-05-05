#!/usr/bin/env python3
"""Watch training log for epoch 0 completion, then stop training and run generation."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

TRAIN_LOG   = Path("runs/audio_medium_swa_moe_sanctsound_humpback_dac_9cb_32k/training_log.jsonl")
CHECKPOINT  = Path("runs/audio_medium_swa_moe_sanctsound_humpback_dac_9cb_32k/best_model.pt")
EPOCH0_STEPS = 505060
POLL_SECS    = 60


def scan_log(log_path: Path) -> tuple[dict | None, float, dict | None]:
    """Return (latest_entry, best_val_loss_overall, latest_epoch1_val_entry)."""
    last = None
    best_val = float("inf")
    latest_e1_val = None
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                last = entry
                if "val_loss" in entry:
                    vl = entry["val_loss"]
                    if vl < best_val:
                        best_val = vl
                    if entry.get("epoch", 0) >= 1:
                        latest_e1_val = entry
    except OSError:
        pass
    return last, best_val, latest_e1_val


def kill_training():
    result = subprocess.run(
        ["pgrep", "-f", "train.py.*audio_medium_swa_moe_sanctsound_humpback_dac_9cb_32k"],
        capture_output=True, text=True,
    )
    pids = [int(p) for p in result.stdout.strip().split() if p]
    if not pids:
        print("[watcher] No matching training process found — already stopped?")
        return
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"[watcher] Sent SIGTERM to PID {pid}")
        except ProcessLookupError:
            pass
    time.sleep(5)
    # SIGKILL any survivors
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
            print(f"[watcher] Sent SIGKILL to PID {pid}")
        except ProcessLookupError:
            pass


def run_generation():
    cmd = [
        sys.executable, "scripts/generate_dac_9cb_prompted.py",
        "--checkpoint",    str(CHECKPOINT),
        "--token-dir",     "data/tokenized/sanctsound_humpback_dac",
        "--prompt-seconds", "5.0",
        "--n-samples",      "5",
        "--temperature",    "0.85",
        "--top-k",          "80",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    log_path = Path("logs/generate_medium_32k.log")
    log_path.parent.mkdir(exist_ok=True)
    print(f"[watcher] Launching generation → logging to {log_path}")
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    print(f"[watcher] Generation PID {proc.pid} — tailing {log_path}")
    proc.wait()
    rc = proc.returncode
    print(f"[watcher] Generation finished (exit code {rc})")
    return rc


def main():
    print(f"[watcher] Monitoring {TRAIN_LOG}")
    print(f"[watcher] Will wait for epoch >= 1 AND a new best val_loss before stopping.")

    epoch1_seen = False
    best_at_epoch0_end = float("inf")

    while True:
        entry, best_val_overall, latest_e1_val = scan_log(TRAIN_LOG)

        if entry:
            step  = entry.get("step", 0)
            epoch = entry.get("epoch", 0)
            vl    = entry.get("val_loss", None)
            vl_str = f"  val_loss={vl:.4f}" if vl else ""
            print(f"[watcher] step={step}  epoch={epoch}  best_so_far={best_val_overall:.4f}{vl_str}")

            # First time we see epoch >= 1: record the best val at end of epoch 0
            if epoch >= 1 and not epoch1_seen:
                epoch1_seen = True
                best_at_epoch0_end = best_val_overall
                print(f"[watcher] Epoch 0 done. Best val at epoch 0 end: {best_at_epoch0_end:.4f}")
                print(f"[watcher] Waiting for first epoch 1 eval that beats {best_at_epoch0_end:.4f}...")

            # Once in epoch 1, check if a new best has been logged
            if epoch1_seen and latest_e1_val is not None:
                e1_vl = latest_e1_val["val_loss"]
                e1_step = latest_e1_val["step"]
                print(f"[watcher] Epoch 1 val at step {e1_step}: {e1_vl:.4f}  "
                      f"(need < {best_at_epoch0_end:.4f})")
                if e1_vl < best_at_epoch0_end:
                    print(f"[watcher] New best {e1_vl:.4f} < {best_at_epoch0_end:.4f} — "
                          f"checkpoint saved. Stopping training.")
                    time.sleep(5)  # let trainer finish writing checkpoint
                    kill_training()
                    time.sleep(3)
                    print(f"[watcher] best_model.pt mtime: "
                          f"{time.ctime(CHECKPOINT.stat().st_mtime) if CHECKPOINT.exists() else 'MISSING'}")
                    rc = run_generation()
                    sys.exit(0 if rc == 0 else 1)

        time.sleep(POLL_SECS)


if __name__ == "__main__":
    main()
