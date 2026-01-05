from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified RL training entry point")
    # High-level
    p.add_argument("--algo", type=str, choices=["ppo", "reinforce"], required=True)
    p.add_argument("--hint", type=str, choices=["none", "random", "llm", "expert"], required=True)
    p.add_argument("--config", type=str, default=None, help="YAML config path; CLI overrides it")
    p.add_argument("--run_name", type=str, default=None)

    # Env
    p.add_argument("--env", type=str, default="DealOrNoDialog-v0")
    p.add_argument("--max_turns", type=int, default=None)

    # Training common
    p.add_argument("--num_train_steps", type=int, default=None)
    p.add_argument("--num_train_episodes", type=int, default=None)  # backward-compat
    p.add_argument("--max_steps_per_episode", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--update_every", type=int, default=None)
    p.add_argument("--ent_coef", type=float, default=None)

    # PPO-specific
    p.add_argument("--clip_eps", type=float, default=None)
    p.add_argument("--gae_lambda", type=float, default=None)
    p.add_argument("--update_epochs", type=int, default=None)
    p.add_argument("--vf_coef", type=float, default=None)
    p.add_argument("--max_grad_norm", type=float, default=None)

    # Logging
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--tensorboard", type=str, choices=["true", "false"], default=None)
    p.add_argument("--save_csv", type=str, choices=["true", "false"], default=None)
    p.add_argument("--save_every", type=int, default=None)
    p.add_argument("--eval_every", type=int, default=None)

    # Resuming
    p.add_argument("--resume_from", type=str, default=None)

    # Hints
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--p", type=float, default=None)
    p.add_argument("--prompt_path", type=str, default=None)
    p.add_argument("--provider", type=str, default=None)
    p.add_argument("--expert_ckpt", type=str, default=None)

    # LLM model IDs (passed through)
    p.add_argument("--groq_model", type=str, default=None)
    p.add_argument("--local_model", type=str, default=None)

    return p


def _truthy(val: str | None) -> bool | None:
    if val is None:
        return None
    return val.lower() in ("1", "true", "yes", "y", "on")


def merge_config_with_cli(cfg: Dict[str, Any], cli: Dict[str, Any]) -> Dict[str, Any]:
    # Establish canonical structure
    merged: Dict[str, Any] = {
        "algo": cfg.get("algo", cli.get("algo")),
        # Prefer CLI seed when provided; fall back to YAML, then default 42
        "seed": (cli.get("seed") if cli.get("seed") is not None else cfg.get("seed", 42)),
        "env": {
            "id": cfg.get("env", {}).get("id", cli.get("env", "DealOrNoDialog-v0")),
            "max_turns": cfg.get("env", {}).get("max_turns", cli.get("max_turns")),
        },
        "training": cfg.get("training", {}),
        "logging": cfg.get("logging", {}),
        "hints": cfg.get("hints", {}),
        "resume_from": cfg.get("resume_from", cli.get("resume_from")),
    }

    # Training defaults and overrides
    tr = merged["training"] = {
        **cfg.get("training", {}),
    }
    # common
    if cli.get("num_train_steps") is not None:
        tr["num_train_steps"] = int(cli["num_train_steps"]) 
    if cli.get("num_train_episodes") is not None:  # backward-compat
        tr["num_train_episodes"] = int(cli["num_train_episodes"]) 
    if cli.get("max_steps_per_episode") is not None:
        tr["max_steps_per_episode"] = int(cli["max_steps_per_episode"])
    if cli.get("lr") is not None:
        tr.setdefault("ppo", {})
        tr.setdefault("reinforce", {})
        tr["ppo"]["learning_rate"] = float(cli["lr"])  # keep parity with PPO naming
        tr["reinforce"]["learning_rate"] = float(cli["lr"])  # reinforce uses same
    if cli.get("gamma") is not None:
        tr.setdefault("ppo", {})
        tr.setdefault("reinforce", {})
        tr["ppo"]["gamma"] = float(cli["gamma"]) ; tr["reinforce"]["gamma"] = float(cli["gamma"]) 
    if cli.get("batch_size") is not None:
        tr.setdefault("ppo", {})
        tr["ppo"]["minibatch_size"] = int(cli["batch_size"])  # PPO mini-batch
    if cli.get("update_every") is not None:
        tr.setdefault("ppo", {})
        tr["ppo"]["rollout_steps"] = int(cli["update_every"])  # reuse name semantics
    if cli.get("ent_coef") is not None:
        tr.setdefault("ppo", {})
        tr.setdefault("reinforce", {})
        tr["ppo"]["ent_coef"] = float(cli["ent_coef"]) ; tr["reinforce"]["ent_coef"] = float(cli["ent_coef"]) 

    # PPO specifics
    ppo = tr.setdefault("ppo", {})
    if cli.get("clip_eps") is not None:
        ppo["clip_coef"] = float(cli["clip_eps"])
    if cli.get("gae_lambda") is not None:
        ppo["gae_lambda"] = float(cli["gae_lambda"])
    if cli.get("update_epochs") is not None:
        ppo["update_epochs"] = int(cli["update_epochs"])
    if cli.get("vf_coef") is not None:
        ppo["vf_coef"] = float(cli["vf_coef"])
    if cli.get("max_grad_norm") is not None:
        ppo["max_grad_norm"] = float(cli["max_grad_norm"])

    # Logging
    log = merged["logging"] = {
        **cfg.get("logging", {}),
    }
    if cli.get("output_dir") is not None:
        log["output_dir"] = str(cli["output_dir"]) 
    if cli.get("run_name") is not None:
        log["run_name"] = str(cli["run_name"]) 
    tb = _truthy(cli.get("tensorboard"))
    if tb is not None:
        log["tensorboard"] = bool(tb)
    sc = _truthy(cli.get("save_csv"))
    if sc is not None:
        log["save_csv"] = bool(sc)
    if cli.get("save_every") is not None:
        log["save_every"] = int(cli["save_every"]) 
    if cli.get("eval_every") is not None:
        log["eval_every"] = int(cli["eval_every"]) 

    # Hints
    hints = merged["hints"] = {
        **cfg.get("hints", {}),
        "mode": cli.get("hint") if cli.get("hint") is not None else cfg.get("hints", {}).get("mode", "none"),
    }
    if cli.get("k") is not None:
        hints["k"] = int(cli["k"]) 
    if cli.get("p") is not None:
        hints["p"] = float(cli["p"]) 
    if cli.get("prompt_path") is not None:
        hints["prompt_path"] = str(cli["prompt_path"]) 
    if cli.get("provider") is not None:
        hints["provider"] = str(cli["provider"]) 
    if cli.get("expert_ckpt") is not None:
        hints["expert_ckpt"] = str(cli["expert_ckpt"]) 
    if cli.get("groq_model") is not None:
        hints.setdefault("groq", {}) ; hints["groq"]["model"] = str(cli["groq_model"]) 
    if cli.get("local_model") is not None:
        hints.setdefault("local", {}) ; hints["local"]["model"] = str(cli["local_model"]) 

    merged["algo"] = cli.get("algo", merged.get("algo"))
    return merged


def validate_config(cfg: Dict[str, Any]) -> None:
    # Basic checks
    if cfg.get("algo") not in ("ppo", "reinforce"):
        raise ValueError("algo must be one of {ppo,reinforce}")
    mode = cfg.get("hints", {}).get("mode", "none")
    if mode not in ("none", "random", "llm", "expert"):
        raise ValueError("--hint must be one of {none,random,llm,expert}")
    if mode == "expert":
        # Allow auto-discovery of PPO checkpoints by default; only validate if a path is provided
        ck = cfg.get("hints", {}).get("expert_ckpt")
        if ck:
            p = Path(str(ck)).expanduser()
            if not p.exists():
                raise FileNotFoundError(f"expert checkpoint not found: {ck}")


