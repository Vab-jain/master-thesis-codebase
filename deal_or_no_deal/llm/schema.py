"""LLM schema definitions, including strict JSON parsing for next action.

Round-trip example (mock):
    Text: '{"act_type": 2, "oA": [0, 1, 0], "confidence": 0.85}'
    Parsed: {"act_type": 2, "oA": [0, 1, 0], "confidence": 0.85}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Any, List, TypedDict


@dataclass
class HintRequest:
    state_summary: str
    metadata: Dict[str, Any] | None = None


@dataclass
class HintResponse:
    hint_text: str
    reasoning: str | None = None


def is_valid_hint(response: HintResponse) -> bool:
    """Return True if the response contains a non-empty hint string."""
    return isinstance(response.hint_text, str) and len(response.hint_text.strip()) > 0


class ParseError(ValueError):
    pass


class NextActionDict(TypedDict):
    act_type: int
    oA: List[int]
    confidence: float


def parse_next_action(json_line: str) -> NextActionDict:
    """Parse and validate an action JSON line.

    Strict checks:
    - act_type ∈ {0..4}
    - oA is a list of length=3, non-negative ints
    - confidence ∈ [0,1]
    Raises ParseError on any violation.
    """
    try:
        obj = json.loads(json_line)
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"invalid json: {exc}")

    if not isinstance(obj, dict):
        raise ParseError("expected object")

    # act_type
    act_type = obj.get("act_type")
    if not isinstance(act_type, int):
        raise ParseError("act_type must be int")
    if act_type < 0 or act_type > 4:
        raise ParseError("act_type out of range [0,4]")

    # oA
    oA = obj.get("oA")
    if not isinstance(oA, list):
        raise ParseError("oA must be list")
    if len(oA) != 3:
        raise ParseError("oA must have length 3")
    for i, v in enumerate(oA):
        if not isinstance(v, int):
            raise ParseError(f"oA[{i}] must be int")
        if v < 0:
            raise ParseError(f"oA[{i}] must be non-negative")

    # confidence
    conf = obj.get("confidence")
    if not (isinstance(conf, float) or isinstance(conf, int)):
        raise ParseError("confidence must be number")
    conf_f = float(conf)
    if conf_f < 0.0 or conf_f > 1.0:
        raise ParseError("confidence out of range [0,1]")

    return {"act_type": act_type, "oA": [int(x) for x in oA], "confidence": conf_f}

