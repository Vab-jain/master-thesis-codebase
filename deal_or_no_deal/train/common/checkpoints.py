from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(path: Path, model: Any, optimizer: Any | None, scheduler: Any | None, config: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
        "rng": {
            "torch": torch.get_rng_state(),
        },
    }
    torch.save(state, path)


def load_checkpoint(path: Path, model: Any, optimizer: Any | None = None, scheduler: Any | None = None) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])  # type: ignore[arg-type]
    if optimizer is not None and state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])  # type: ignore[arg-type]
    if scheduler is not None and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])  # type: ignore[arg-type]
    if "rng" in state and "torch" in state["rng"]:
        torch.set_rng_state(state["rng"]["torch"])  # type: ignore[arg-type]
    return state.get("config", {})


