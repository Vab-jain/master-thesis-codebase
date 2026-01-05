#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from train.common.cli import build_arg_parser, merge_config_with_cli, validate_config
from train.common.logger import TrainLogger
from train.common.utils import set_seed
from train.common.hints_adapter import make_env_with_hints, HintMode
from train.algorithms.ppo import train as ppo_train
from train.algorithms.reinforce import train as reinforce_train


def load_yaml_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    base_cfg = load_yaml_config(args.config)
    cfg = merge_config_with_cli(base_cfg, vars(args))
    validate_config(cfg)

    # Ensure output dir exists and snapshot config
    out_dir = Path(cfg["logging"]["output_dir"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    set_seed(int(cfg.get("seed", 42)))

    # Logger
    logger = TrainLogger(out_dir, enable_tb=bool(cfg["logging"].get("tensorboard", False)), save_csv=bool(cfg["logging"].get("save_csv", True)))
    logger.log_hparams(cfg)

    # Env factories
    hint_mode = HintMode(cfg["hints"]["mode"]) if cfg.get("hints") and cfg["hints"].get("mode") else HintMode.none
    env_fn, eval_env_fn, hint_adapter = make_env_with_hints(cfg)

    # Dispatch algorithm
    algo = str(cfg["algo"]).lower()
    if algo == "ppo":
        ppo_train(cfg, env_fn, hint_adapter=hint_adapter, logger=logger, eval_env_fn=eval_env_fn)
    elif algo == "reinforce":
        reinforce_train(cfg, env_fn, hint_adapter=hint_adapter, logger=logger, eval_env_fn=eval_env_fn)
    else:
        raise ValueError(f"Unsupported --algo: {cfg['algo']}")

    # Save status
    # Plot curves at end of training
    try:
        rw = int(cfg.get("training", {}).get("rolling_window", 100))
        logger.plot_curves(rolling_window=rw)
    except Exception:
        pass

    (out_dir / "status.json").write_text(json.dumps({"status": "OK"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


