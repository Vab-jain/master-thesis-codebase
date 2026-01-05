"""Hint injection wrapper for Gymnasium envs.

Injects next-action hints into the `info` dict at a fixed cadence `k`.
On reset: `t=0` and a hint is immediately injected.
On each step: after env.step, if `(t+1) % k == 0` and not done, inject a hint;
otherwise provide neutral (zero) features.

Features stored under `info['hint_features']` as:
  - act_onehot: List[float] size 5
  - oA_norm: List[float] size 3
  - confidence: float
  - h_avail: int (1 if hint available, else 0)

Also exposes counters on the wrapper instance:
  - total_hint_calls
  - failed_hint_calls

Passes through `info['action_mask']` and `info['oA_max']` if provided by env.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, List
import numpy as np  # type: ignore

try:
    import gymnasium as gym
except Exception:  # pragma: no cover - gym present in runtime, but tests are minimal
    gym = None  # type: ignore


class HintInjectorEnv:
    """A lightweight wrapper that injects next-action hint features.

    Parameters
    - env: underlying environment
    - client: object with `.generate(prompt) -> (text, latency_ms)`
    - k: cadence for hint calls (int >= 1)
    - prompt_path: path to template used by llm.prompt
    - max_prompt_chars: history truncation cap for prompt rendering
    """

    def __init__(
        self,
        env: Any,
        client: Any | None = None,
        k: int = 1,
        prompt_path: str = "configs/llm_prompt.txt",
        max_prompt_chars: int = 2000,
        provider: str | None = None,
        expert_ckpt: str | None = None,
    ) -> None:
        self._env = env
        self._client = client
        self._k = max(1, int(k))
        self._prompt_path = prompt_path
        self._max_prompt_chars = max_prompt_chars
        self._provider = (provider or "local_hf").lower()
        self._expert_ckpt = expert_ckpt
        self.t = 0
        self.total_hint_calls = 0
        self.failed_hint_calls = 0
        self.total_retries = 0
        self.illegal_suggestions = 0
        self.hint_available_successes = 0
        # lazy expert model
        self._expert_model = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def reset(self, *args: Any, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        observation, info = self._env.reset(*args, **kwargs)
        info = dict(info or {})
        self.t = 0
        # Inject immediately on reset
        features, log = self._maybe_inject_hint(observation, info, force=True)
        info.update(log)
        info["hint_features"] = features
        return observation, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        info = dict(info or {})
        do_call = ((self.t + 1) % self._k == 0) and not (terminated or truncated)
        features, log = self._maybe_inject_hint(obs, info, force=do_call)
        info.update(log)
        info["hint_features"] = features
        self.t += 1
        return obs, reward, terminated, truncated, info

    def _maybe_inject_hint(self, observation: Any, info: Dict[str, Any], force: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Always pass through existing fields
        passthrough = {}
        if "action_mask" in info:
            passthrough["action_mask"] = info["action_mask"]
        if "oA_max" in info:
            passthrough["oA_max"] = info["oA_max"]

        if not force:
            # Neutral features when not calling
            features = self._neutral_features(info)
            return features, {**passthrough, "hint_call_success": False, "hint_latency_ms": 0.0, "hint_called": False, "total_hint_calls": self.total_hint_calls, "failed_hint_calls": self.failed_hint_calls, "hint_source": self._provider, "retries_used": 0, "hint_available": False}

        self.total_hint_calls += 1
        start_ns = self._now_ns()
        try:
            # Branch by provider type
            if self._provider in ("random",):
                parsed = self._random_hint(observation, info)
                features = self._to_features(parsed, info)
                total_ms = (self._now_ns() - start_ns) / 1e6
                log = {**passthrough, "hint_call_success": True, "hint_latency_ms": float(total_ms), "hint_called": True, "total_hint_calls": self.total_hint_calls, "failed_hint_calls": self.failed_hint_calls, "hint_source": self._provider}
                return features, log
            if self._provider in ("expert",):
                parsed = self._expert_hint(observation, info)
                features = self._to_features(parsed, info)
                total_ms = (self._now_ns() - start_ns) / 1e6
                log = {**passthrough, "hint_call_success": True, "hint_latency_ms": float(total_ms), "hint_called": True, "total_hint_calls": self.total_hint_calls, "failed_hint_calls": self.failed_hint_calls, "hint_source": self._provider}
                return features, log
            # default: externally provided client (e.g., local_hf or groq)
            prompt = self._build_prompt(observation, info)
            # Exponential backoff with up to 5 retries (6 attempts total)
            attempts = 0
            retries_used = 0
            text = ""
            latency_ms = 0.0
            last_exc: Exception | None = None
            while True:
                attempts += 1
                try:
                    text, latency_ms = self._client.generate(prompt) if self._client is not None else ("", 0.0)
                    if not text or not isinstance(text, str):
                        raise RuntimeError("empty llm text")
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    # Retry only on 429/5xx/timeout
                    msg = str(exc).lower()
                    retriable = ("http 429" in msg) or ("http 5" in msg) or ("timeout" in msg)
                    if attempts >= 6 or not retriable:
                        raise
                    # backoff: 0.5,1,2,4,8 seconds
                    delay_s = float(0.5 * (2 ** (attempts - 1)))
                    import time as _t
                    _t.sleep(min(8.0, delay_s))
                    retries_used += 1
            self.total_retries += int(retries_used)
            if not text or not isinstance(text, str):
                raise RuntimeError("empty llm text")
            from llm.schema import parse_next_action  # local import to avoid cycle in tests
            parsed = parse_next_action(text)
            # Validate legality of suggested action against current mask if available
            try:
                act_type = int(parsed.get("act_type", 4))
                mask_cur = info.get("action_mask")
                if mask_cur is not None:
                    mask_list = list(mask_cur)
                    if act_type < 0 or act_type >= len(mask_list) or int(mask_list[act_type]) == 0:
                        self.illegal_suggestions += 1
                        raise RuntimeError("illegal act suggestion per mask")
            except Exception:
                self.failed_hint_calls += 1
                total_ms = latency_ms if latency_ms else (self._now_ns() - start_ns) / 1e6
                features = self._neutral_features(info)
                log = {**passthrough, "hint_call_success": False, "hint_latency_ms": float(total_ms), "hint_called": True, "total_hint_calls": self.total_hint_calls, "failed_hint_calls": self.failed_hint_calls, "hint_source": self._provider, "retries_used": retries_used, "hint_available": False}
                return features, log
            features = self._to_features(parsed, info)
            total_ms = latency_ms if latency_ms else (self._now_ns() - start_ns) / 1e6
            self.hint_available_successes += 1
            log = {**passthrough, "hint_call_success": True, "hint_latency_ms": float(total_ms), "hint_called": True, "total_hint_calls": self.total_hint_calls, "failed_hint_calls": self.failed_hint_calls, "hint_source": self._provider, "retries_used": retries_used, "hint_available": True}
            return features, log
        except Exception:  # noqa: BLE001 - convert any provider/parser error to neutral features
            self.failed_hint_calls += 1
            total_ms = (self._now_ns() - start_ns) / 1e6
            features = self._neutral_features(info)
            log = {**passthrough, "hint_call_success": False, "hint_latency_ms": float(total_ms), "hint_called": True, "total_hint_calls": self.total_hint_calls, "failed_hint_calls": self.failed_hint_calls, "hint_source": self._provider, "retries_used": 0, "hint_available": False}
            return features, log

    @staticmethod
    def _now_ns() -> int:
        import time

        return time.time_ns()

    def _build_prompt(self, observation: Dict[str, Any], info: Dict[str, Any]) -> str:
        """Build prompt strictly from current observation and public info."""
        from llm.prompt import render_prompt_with_cap

        def _act_token(val: int) -> str:
            mapping = {0: "PROPOSE", 1: "INSIST", 2: "AGREE", 3: "DISAGREE", 4: "END"}
            try:
                return mapping.get(int(val), "NONE")
            except Exception:
                return "NONE"

        counts = observation.get("counts", [0, 0, 0])
        my_utils = observation.get("my_utilities", [0, 0, 0])
        last_offer = observation.get("last_partner_offer_for_me", [0, 0, 0])
        last_act = observation.get("last_partner_act", 4)
        turns = observation.get("turns_remaining", 0)
        mask = info.get("action_mask", [1, 1, 1, 1, 1])

        variables = {
            "counts_csv": ",".join(str(int(x)) for x in list(counts)),
            "my_utils_csv": ",".join(str(int(x)) for x in list(my_utils)),
            "last_partner_act_token": _act_token(last_act),
            "last_offer_csv": ",".join(str(int(x)) for x in list(last_offer)),
            "turns": int(turns),
            "legal_mask_csv": ",".join(str(int(x)) for x in list(mask)),
            "p": 0.5,
            "history_str": "",
        }
        return render_prompt_with_cap(self._prompt_path, variables, self._max_prompt_chars)

    @staticmethod
    def _neutral_features(info: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "act_onehot": [0.0] * 5,
            "oA_norm": [0.0] * 3,
            "confidence": 0.0,
            "h_avail": 0,
            "hint_available": False,
        }

    @staticmethod
    def _to_features(parsed: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        act_type = int(parsed["act_type"])  # 0..4
        act = [0.0] * 5
        if 0 <= act_type < 5:
            act[act_type] = 1.0
        oA = [int(x) for x in parsed["oA"]]
        oA_max: List[int] = list(info.get("oA_max", [1, 1, 1]))
        oA_norm: List[float] = []
        for i in range(3):
            m = float(oA_max[i]) if i < len(oA_max) else 1.0
            if m <= 0:
                oA_norm.append(0.0)
            else:
                oA_norm.append(min(1.0, max(0.0, float(oA[i]) / m)))
        return {
            "act_onehot": act,
            "oA_norm": oA_norm,
            "confidence": float(parsed["confidence"]),
            "h_avail": 1,
            "hint_available": True,
        }

    # --- provider implementations ---
    @staticmethod
    def _uniform_randint(low: int, high_incl: int) -> int:
        import random
        return int(random.randint(low, high_incl))

    def _random_hint(self, observation: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        # Sample an act strictly from the current legal set (action_mask)
        mask = info.get("action_mask")
        if mask is not None:
            legal_indices = [i for i, m in enumerate(list(mask)) if int(m) == 1]
            if len(legal_indices) == 0:
                # Fallback to END if no legal found (shouldn't happen with our env)
                act = 4
            else:
                import random as _r
                act = int(_r.choice(legal_indices))
        else:
            act = self._uniform_randint(0, 4)

        # Only propose/insist require oA; otherwise return zeros
        if act in (0, 1):
            caps = info.get("oA_max", [1, 1, 1])
            oA = [self._uniform_randint(0, int(caps[i])) for i in range(3)]
        else:
            oA = [0, 0, 0]

        return {"act_type": int(act), "oA": [int(x) for x in oA], "confidence": 0.5}

    def _expert_hint(self, observation: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        # Try to load a PPO ActorCritic checkpoint by default; fallback to supervised expert; else greedy
        try:
            import torch  # type: ignore
            import glob
            from pathlib import Path as _Path
            from train.common.nn import ActorCritic  # type: ignore

            if self._expert_model is None:
                ckpt_path = self._expert_ckpt
                if not ckpt_path:
                    # Prefer user-provided expert model location first
                    preferred = _Path("runs/expert_model/ppo_seed123/expert_final.pt")
                    if preferred.exists():
                        ckpt_path = str(preferred)
                    else:
                        ckpt_path = None
                if ckpt_path and _Path(ckpt_path).exists():
                    state = torch.load(ckpt_path, map_location="cpu")
                    # Support various save formats
                    if isinstance(state, dict) and "model" in state:
                        sd = state["model"]
                    elif isinstance(state, dict) and "policy" in state:
                        sd = state["policy"]
                    else:
                        sd = state
                    # Infer input dim from first linear layer
                    first_weight = sd.get("trunk.0.weight") or sd.get("shared.0.weight")
                    if first_weight is None:
                        raise RuntimeError("unsupported checkpoint format for expert PPO")
                    input_dim = int(first_weight.shape[1])
                    mdl = ActorCritic(input_dim=input_dim, hidden=64)
                    mdl.load_state_dict(sd, strict=False)
                    mdl.eval()
                    self._expert_model = mdl
        except Exception:
            # attempt supervised expert as before
            try:
                if self._expert_model is None:
                    from train.supervised_expert_train import ExpertNet  # type: ignore
                    import torch  # type: ignore
                    import glob
                    ckpt_path = self._expert_ckpt
                    if not ckpt_path:
                        cands = glob.glob("runs/expert_*/expert_best.pt") + glob.glob("runs/**/expert_best.pt", recursive=True)
                        ckpt_path = cands[0] if cands else None
                    if ckpt_path and Path(ckpt_path).exists():
                        mdl = ExpertNet()
                        state = torch.load(ckpt_path, map_location="cpu")
                        mdl.load_state_dict(state)
                        mdl.eval()
                        self._expert_model = mdl
            except Exception:
                pass

        # If we have a loaded model (PPO ActorCritic or ExpertNet), use it
        try:
            if self._expert_model is not None:
                import torch  # type: ignore
                # Build base-only feature vector compatible with PPO ActorCritic default
                counts = np.array(observation.get("counts", [0, 0, 0]), dtype=np.float32)
                my_utils = np.array(observation.get("my_utilities", [0, 0, 0]), dtype=np.float32)
                partner_utils = np.array(observation.get("partner_utilities", [0, 0, 0]), dtype=np.float32)
                last_act = int(observation.get("last_partner_act", 4))
                last_act_oh = np.zeros(5, dtype=np.float32)
                last_act_oh[min(4, max(0, last_act))] = 1.0
                last_offer = np.array(observation.get("last_partner_offer_for_me", [0, 0, 0]), dtype=np.float32)
                turns = float(observation.get("turns_remaining", 0.0))
                x_np = np.concatenate([counts, my_utils, partner_utils, last_act_oh, last_offer, np.array([turns], dtype=np.float32)])
                x = torch.from_numpy(x_np).unsqueeze(0).float()
                out = self._expert_model(x)  # supports ActorCritic forward
                if isinstance(out, tuple):  # ExpertNet path
                    act_logits, o_heads = out
                    act_type = int(torch.argmax(act_logits, dim=-1).item())
                    oA = [int(torch.argmax(h, dim=-1).item()) for h in o_heads]
                else:  # Dict from ActorCritic
                    act_logits = out["action_logits"]
                    o_heads = [out["oA_logits_0"], out["oA_logits_1"], out["oA_logits_2"]]
                    act_type = int(torch.argmax(act_logits, dim=-1).item())
                    oA = [int(torch.argmax(h, dim=-1).item()) for h in o_heads]
                caps = info.get("oA_max", [1, 1, 1])
                oA = [max(0, min(oA[i], int(caps[i]))) for i in range(3)]
                return {"act_type": int(act_type), "oA": [int(x) for x in oA], "confidence": 0.9}
        except Exception:
            pass
        # Greedy fallback
        caps = info.get("oA_max", [1, 1, 1])
        u = observation.get("my_utilities", [1, 1, 1])
        # allocate all counts to highest-utility items first to agent
        order = list(np.argsort(-np.array(u)))  # type: ignore[name-defined]
        oA = [0, 0, 0]
        remaining = list(observation.get("counts", [0, 0, 0]))
        for idx in order:
            take = int(remaining[idx])
            oA[idx] = max(0, min(take, int(caps[idx])))
        return {"act_type": 0, "oA": [int(x) for x in oA], "confidence": 0.8}


