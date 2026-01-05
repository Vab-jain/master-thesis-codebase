"""Prompt rendering utilities.

Provides helpers to load the editable prompt template and render it with
placeholder substitution. Includes a simple cap on history length for safety.

TODO: Replace dummy variable generation with real state summarization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def load_prompt_template(template_path: str) -> str:
    """Load prompt template text from `template_path`."""
    return Path(template_path).read_text(encoding="utf-8")


def _apply_history_cap(variables: Dict[str, Any], max_prompt_chars: int) -> Dict[str, Any]:
    """Return a copy of variables with `history_str` truncated to the cap."""
    capped = dict(variables)
    history = str(capped.get("history_str", ""))
    if max_prompt_chars is not None and max_prompt_chars >= 0 and len(history) > max_prompt_chars:
        capped["history_str"] = history[:max_prompt_chars]
    else:
        capped["history_str"] = history
    return capped


def render_prompt(template_path: str, variables: Dict[str, Any]) -> str:
    """Render a template by Python `str.format(**variables)`.

    Unknown fields will be left as `{field}` placeholders if not present.
    """
    text = load_prompt_template(template_path)
    # For safety, only substitute provided keys; others remain verbatim
    return text.format(**{k: variables.get(k, "{" + k + "}") for k in variables})


def render_prompt_with_cap(template_path: str, variables: Dict[str, Any], max_prompt_chars: int) -> str:
    """Render the template after truncating `history_str` to `max_prompt_chars`."""
    capped_vars = _apply_history_cap(variables, max_prompt_chars)
    return render_prompt(template_path, capped_vars)


def build_dummy_variables(turns: int = 3, p: float = 0.5) -> Dict[str, Any]:
    """Return a minimal set of dummy variables for the template."""
    return {
        "counts_csv": "agree:1,disagree:2",
        "my_utils_csv": "u1:0.7,u2:0.3",
        "last_partner_act_token": "OFFER",
        "last_offer_csv": "price:1000,terms:NA",
        "turns": turns,
        "p": p,
        "history_str": "YOU: hi | THEM: hello | YOU: offer=1000",
    }


def generate_dummy_prompt(template_path: str, max_prompt_chars: int = 2000) -> str:
    """Load template and render with dummy variables, capping history length."""
    variables = build_dummy_variables()
    return render_prompt_with_cap(template_path, variables, max_prompt_chars)


