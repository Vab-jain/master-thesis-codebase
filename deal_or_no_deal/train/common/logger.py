from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict

from torch.utils.tensorboard import SummaryWriter  # type: ignore


class TrainLogger:
    def __init__(self, out_dir: Path, enable_tb: bool = False, save_csv: bool = True) -> None:
        self.out_dir = out_dir
        self.enable_tb = enable_tb
        self.save_csv = save_csv
        self.tb = SummaryWriter(log_dir=str(out_dir)) if enable_tb else None
        self.returns_csv = out_dir / "train_returns.csv"
        self.losses_csv = out_dir / "losses.csv"
        if save_csv:
            if not self.returns_csv.exists():
                with self.returns_csv.open("w", newline="") as f:
                    csv.writer(f).writerow(["step", "episode", "return", "loss"])  # include step for x-axis
            if not self.losses_csv.exists():
                with self.losses_csv.open("w", newline="") as f:
                    csv.writer(f).writerow(["update", "policy_loss", "value_loss", "entropy", "steps"])  # include steps if provided

    def log_hparams(self, cfg: Dict[str, Any]) -> None:
        if self.tb:
            # Log basic hyperparams: flatten a subset
            flat = {
                "algo": str(cfg.get("algo")),
                "seed": int(cfg.get("seed", 0)),
                "hint_mode": str(cfg.get("hints", {}).get("mode", "none")),
            }
            self.tb.add_hparams(flat, {})

    def log_episode(self, episode_idx: int, step: int, loss: float, return_: float) -> None:
        if self.save_csv:
            with self.returns_csv.open("a", newline="") as f:
                csv.writer(f).writerow([int(step), int(episode_idx), float(return_), float(loss)])
        if self.tb:
            self.tb.add_scalar("episode/return", float(return_), int(step))
            self.tb.add_scalar("episode/loss", float(loss), int(step))

    def log_update(self, update_idx: int, policy_loss: float, value_loss: float, entropy: float, steps: int | None = None) -> None:
        if self.save_csv:
            with self.losses_csv.open("a", newline="") as f:
                csv.writer(f).writerow([int(update_idx), float(policy_loss), float(value_loss), float(entropy), int(steps) if steps is not None else ""])
        if self.tb:
            x = int(steps) if steps is not None else int(update_idx)
            self.tb.add_scalar("loss/policy", float(policy_loss), x)
            self.tb.add_scalar("loss/value", float(value_loss), x)
            self.tb.add_scalar("loss/entropy", float(entropy), x)

    def plot_curves(self, rolling_window: int = 100) -> None:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            return
        # Plot rolling mean return (x-axis: steps if available)
        try:
            steps = []
            returns = []
            with self.returns_csv.open("r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    step_v = row.get("step")
                    steps.append(int(step_v) if step_v not in (None, "") else len(steps) + 1)
                    returns.append(float(row.get("return", 0.0)))
            if len(returns) > 0:
                rw = max(1, int(rolling_window))
                rolling = []
                acc = 0.0
                from collections import deque
                dq = deque(maxlen=rw)
                for v in returns:
                    dq.append(v)
                    rolling.append(float(sum(dq) / len(dq)))
                plt.figure(figsize=(6, 3))
                plt.plot(steps, rolling)
                plt.title(f"Rolling Return (window={rw})")
                plt.xlabel("Steps")
                plt.ylabel("Return")
                plt.tight_layout()
                plt.savefig(self.out_dir / "rolling_return.png")
                plt.close()
        except Exception:
            pass
        # Plot training losses
        try:
            xs = []
            pl = []
            vl = []
            ent = []
            with self.losses_csv.open("r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    step_col = row.get("steps")
                    xs.append(int(step_col) if step_col not in (None, "") else int(row.get("update", len(xs) + 1)))
                    pl.append(float(row.get("policy_loss", 0.0)))
                    vl.append(float(row.get("value_loss", 0.0)))
                    ent.append(float(row.get("entropy", 0.0)))
            if len(xs) > 0:
                plt.figure(figsize=(6, 3))
                plt.plot(xs, pl, label="policy")
                plt.plot(xs, vl, label="value")
                plt.plot(xs, ent, label="entropy")
                plt.legend()
                plt.title("Losses")
                plt.xlabel("Steps")
                plt.tight_layout()
                plt.savefig(self.out_dir / "losses.png")
                plt.close()
        except Exception:
            pass


