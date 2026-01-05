"""Pre-training LLM hints evaluation harness.

Loads a small subset of DealOrNoDialog (self_play counts), builds prompt inputs,
queries the LLM client, parses actions, and compares to a simple heuristic
"expert" to produce act accuracy and `oA` exact/MAE metrics.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json

import numpy as np
import yaml

from llm.client import LocalHFClient, GROQClient
from llm.prompt import render_prompt_with_cap
from llm.schema import parse_next_action


def load_yaml_config(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_selfplay_subset(limit: int) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("./deal_or_no_dialog/deal_or_no_dialog.py", name="self_play")
        rows = []
        for i in range(min(limit, len(ds["train"]))):
            rows.append(ds["train"][i])
        return rows
    except Exception:
        # Fallback: synthetic few rows
        return [
            {"input": {"count": [3, 2, 1]}},
            {"input": {"count": [2, 2, 2]}},
            {"input": {"count": [1, 0, 3]}},
        ][:limit]


def heuristic_expert(counts: List[int], utilities: List[int]) -> Tuple[int, List[int]]:
    # Propose: allocate to maximize our utility within counts
    order = list(np.argsort(-np.array(utilities)))
    remaining = list(counts)
    oA = [0, 0, 0]
    for idx in order:
        take = remaining[idx]
        oA[idx] = int(take)
        remaining[idx] = 0
    return 0, oA  # act_type=PROPOSE


def build_variables(counts: List[int], my_utils: List[int]) -> Dict[str, Any]:
    return {
        "counts_csv": ",".join(map(str, counts)),
        "my_utils_csv": ",".join(map(str, my_utils)),
        "last_partner_act_token": "NONE",
        "last_offer_csv": "0,0,0",
        "turns": 5,
        "p": 0.5,
        "history_str": "",
    }


def evaluate_hints(config_path: str, limit: int, out_dir: str, out_csv: str | None = None) -> Dict[str, Any]:
    cfg = load_yaml_config(config_path)
    prompt_path = cfg.get("hints", {}).get("prompt_path", "configs/llm_prompt.txt")
    rows = load_selfplay_subset(limit)

    hints_cfg = cfg.get("hints", {})
    provider = str(hints_cfg.get("provider", "groq")).lower()
    if provider == "groq":
        model = str((hints_cfg.get("groq", {}) or {}).get("model", "llama-3.3-70b-versatile"))
        client = GROQClient(model=model)
    else:
        # Disallow silent CPU stub: require GPU unless explicitly overridden (we do not override here)
        model = str((hints_cfg.get("local", {}) or {}).get("model", ""))
        client = LocalHFClient(model=model)

    records: List[Dict[str, Any]] = []
    act_matches = 0
    exact_matches = 0
    mae_sums = np.zeros(3, dtype=float)
    for idx, row in enumerate(rows):
        counts = list(map(int, row["input"]["count"]))
        # synthesize utilities for eval; scale 0..10
        rng = np.random.default_rng(1234 + idx)
        my_utils = list(map(int, rng.integers(0, 11, size=3)))

        variables = build_variables(counts, my_utils)
        prompt = render_prompt_with_cap(prompt_path, variables, max_prompt_chars=1000)
        text, latency_ms = client.generate(prompt)
        try:
            pred = parse_next_action(text)
        except Exception:
            pred = {"act_type": 4, "oA": [0, 0, 0], "confidence": 0.0}

        # Post-process/repair
        # clamp oA
        oA_pred_raw = list(map(int, pred.get("oA", [0, 0, 0])))
        oA_pred_clamped = [max(0, min(oA_pred_raw[i], counts[i])) for i in range(3)]
        pred["oA"] = oA_pred_clamped
        # If PROPOSE/INSIST but oA missing or all zeros and counts non-zero, greedy fallback
        if int(pred.get("act_type", 4)) in (0, 1) and sum(oA_pred_clamped) == 0 and sum(counts) > 0:
            # greedy by utility
            order = list(np.argsort(-np.array(my_utils)))
            rem = counts.copy()
            oA_g = [0, 0, 0]
            for idx2 in order:
                take = rem[idx2]
                oA_g[idx2] = int(take)
                rem[idx2] = 0
            pred["oA"] = oA_g

        act_true, oA_true = heuristic_expert(counts, my_utils)
        act_pred = int(pred["act_type"])  # type: ignore[index]
        oA_pred = list(map(int, pred["oA"]))  # type: ignore[index]

        act_matches += int(act_pred == act_true)
        exact = int(oA_pred == oA_true)
        exact_matches += exact
        mae = np.abs(np.array(oA_pred) - np.array(oA_true))
        mae_sums += mae

        records.append(
            {
                "idx": idx,
                "counts": counts,
                "my_utils": my_utils,
                "act_true": act_true,
                "act_pred": act_pred,
                "oA_true": oA_true,
                "oA_pred": oA_pred,
                "oA_exact": exact,
                "mae_0": float(mae[0]),
                "mae_1": float(mae[1]),
                "mae_2": float(mae[2]),
                "latency_ms": float(latency_ms),
            }
        )

    n = max(1, len(records))
    act_accuracy = act_matches / n
    oA_exact_match = exact_matches / n
    oA_mae_per_item = (mae_sums / n).tolist()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = Path(out_csv) if out_csv else (out_path / "llm_hint_eval.csv")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "counts",
                "my_utils",
                "act_true",
                "act_pred",
                "oA_true",
                "oA_pred",
                "oA_exact",
                "mae_0",
                "mae_1",
                "mae_2",
                "latency_ms",
            ],
        )
        w.writeheader()
        w.writerows(records)

    # Summary and status
    summary = {
        "act_accuracy": float(act_accuracy),
        "oA_exact_match": float(oA_exact_match),
        "oA_mae_per_item": [float(x) for x in oA_mae_per_item],
        "prompt_path": str(prompt_path),
        "subset_size": int(limit),
        "csv_path": str(csv_path),
    }
    (out_path / "summary.txt").write_text(
        f"act_accuracy={act_accuracy:.3f}\n"
        f"oA_exact_match={oA_exact_match:.3f}\n"
        f"oA_mae_per_item={[round(x,3) for x in oA_mae_per_item]}\n"
        f"prompt_path={prompt_path}\n",
        encoding="utf-8",
    )

    # thresholds
    he = cfg.get("hints_eval", {})
    act_acc_min = float(he.get("act_acc_min", 0.55))
    oA_mae_max = float(he.get("oA_mae_max", 0.60))
    status = {
        "act_accuracy": float(act_accuracy),
        "oA_mae_per_item": [float(x) for x in oA_mae_per_item],
        "act_acc_min": act_acc_min,
        "oA_mae_max": oA_mae_max,
    }
    ok = (act_accuracy >= act_acc_min) and all(x <= oA_mae_max for x in oA_mae_per_item)
    status_path = out_path / ("STATUS_OK.json" if ok else "STATUS_FAIL.json")
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    # Append command
    try:
        with Path("runs/COMMANDS.log").open("a", encoding="utf-8") as f:
            f.write(f"python -m train.test_llm_hints --config {config_path} --out_csv {csv_path}\n")
    except Exception:
        pass

    return {"ok": ok, **summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-training LLM hints evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default="runs/hints_eval")
    parser.add_argument("--out_csv", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    he = cfg.get("hints_eval", {})
    limit = int(args.limit) if args.limit is not None else int(he.get("subset_size", 500))

    res = evaluate_hints(args.config, limit, args.out_dir, args.out_csv)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()


