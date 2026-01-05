"""LLM client implementations with exponential backoff and metrics.

Provides:
- `GROQClient` for remote GROQ models
- `LocalHFClient` for local HuggingFace models
- `LLMClient` facade for scaffold usage

Both clients use a shared backoff mechanism with jitter. Calls return
`(text, latency_ms)` or raise `LLMCallError`.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Tuple
import requests
from dotenv import load_dotenv

from .schema import HintRequest, HintResponse

# Load environment variables from .env at import time (for GROQ_API_KEY, etc.)
load_dotenv()


class LLMCallError(RuntimeError):
    pass


@dataclass
class BackoffConfig:
    initial_delay_s: float = 0.5
    max_delay_s: float = 8.0
    retries: int = 3


def _with_backoff(  # small utility wrapper
    func: Callable[[], Tuple[str, float]],
    backoff: BackoffConfig,
) -> Tuple[str, float]:
    """Run `func` with exponential backoff and jitter.

    Jitter is implemented by multiplying delay by a random factor in [0.5, 1.5].
    Delay grows as `delay = min(max_delay, delay * 2)` after each failure.
    """
    delay = max(0.0, backoff.initial_delay_s)
    last_exc: Exception | None = None
    for attempt in range(backoff.retries + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - surface provider errors safely
            last_exc = exc
            if attempt >= backoff.retries:
                break
            jitter = random.uniform(0.5, 1.5)
            sleep_s = min(backoff.max_delay_s, delay) * jitter
            time.sleep(max(0.0, sleep_s))
            delay = min(backoff.max_delay_s, max(delay * 2.0, backoff.initial_delay_s))
    raise LLMCallError(str(last_exc) if last_exc else "unknown error")


class Metrics:
    def __init__(self) -> None:
        self.success_count = 0
        self.fail_count = 0

    def record(self, success: bool) -> None:
        if success:
            self.success_count += 1
        else:
            self.fail_count += 1


class GROQClient:
    """GROQ client stub with backoff; replace call body with actual SDK later."""

    def __init__(self, model: str, api_key_env: str = "GROQ_API_KEY", backoff: BackoffConfig | None = None) -> None:
        self.model = model
        self.api_key = os.environ.get(api_key_env, "")
        self.backoff = backoff or BackoffConfig()
        self.metrics = Metrics()

    def generate(self, prompt: str) -> Tuple[str, float]:
        start_ns = time.time_ns()

        def _call() -> Tuple[str, float]:
            if not self.api_key:
                raise RuntimeError("missing GROQ_API_KEY")
            # OpenAI-compatible chat completions endpoint for Groq
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            # Encourage strict JSON line output
            sys_msg = (
                "Return ONLY a single-line JSON: "
                "{\"act_type\": int 0-4, \"oA\": [int,int,int], \"confidence\": float 0-1}"
            )
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt},
            ]
            body = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 48,
                "stop": ["\n"],
            }
            resp = requests.post(url, headers=headers, json=body, timeout=60)
            if resp.status_code >= 300:
                raise RuntimeError(f"groq http {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            try:
                text = data["choices"][0]["message"]["content"].strip()
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"groq response parse error: {exc}")
            latency_ms = (time.time_ns() - start_ns) / 1e6
            return text, latency_ms

        try:
            result = _with_backoff(_call, self.backoff)
            self.metrics.record(True)
            return result
        except Exception as exc:  # noqa: BLE001
            self.metrics.record(False)
            raise LLMCallError(str(exc))


class LocalHFClient:
    """Local HuggingFace client stub with backoff.

    Note: For the scaffold, we do not load real models to keep tests lightweight.
    Replace body with `transformers` pipeline/model loading when implementing.
    """

    def __init__(self, model: str, device: str | None = None, backoff: BackoffConfig | None = None, allow_cpu: bool = True) -> None:
        self.model = model
        self.device = device or ("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
        self.backoff = backoff or BackoffConfig()
        self.metrics = Metrics()
        # Optional CPU fallback for lightweight runs
        _ = allow_cpu

    def generate(self, prompt: str) -> Tuple[str, float]:
        # Lightweight placeholder that validates HF_API_KEY presence and returns a simple valid JSON
        start_ns = time.time_ns()
        api_key = os.environ.get("HF_API_KEY", "")
        if not api_key:
            raise LLMCallError("Missing HF_API_KEY in environment/.env for local_hf provider")
        # Produce a deterministic suggestion based on prompt hash
        try:
            h = abs(hash(prompt))
        except Exception:
            h = 0
        act = int(h % 5)
        oA = [int((h // (i + 1)) % 3) for i in range(3)]
        text = json.dumps({"act_type": act, "oA": oA, "confidence": 0.6})
        latency_ms = (time.time_ns() - start_ns) / 1e6
        self.metrics.record(True)
        return text, float(latency_ms)


class LLMClient:
    """Facade used by the scaffold demo to produce a hint string."""

    def __init__(self, provider: str = "local", model: str = "llama3.1:latest") -> None:
        self.provider = provider
        self.model = model
        self._client: Any
        if provider.lower() == "groq":
            self._client = GROQClient(model=model)
        else:
            self._client = LocalHFClient(model=model)

    def generate_hint(self, request: HintRequest) -> HintResponse:
        _ = request  # not used in scaffold
        text, _latency_ms = self._client.generate(prompt="dummy")
        # For the scaffold hint, just return a fixed text.
        return HintResponse(hint_text=text)


