"""
TG-034: Structured generation prompt guard.

When using `processor.apply_chat_template(..., add_generation_prompt=True)`, the
TranslateGemma template appends a generation wrapper like:

    <end_of_turn>\n<start_of_turn>model\n
If truncation removes these markers, the model may continue the user text.
This module provides a lightweight check to detect that condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


class StructuredTruncationError(RuntimeError):
    pass


@dataclass(frozen=True)
class _GuardResult:
    ok: bool
    reason: Optional[str] = None


def _decode_tail(tokenizer: Any, input_ids: Any, tail_tokens: int = 96) -> str:
    if input_ids is None:
        return ""
    ids = input_ids[0] if getattr(input_ids, "dim", lambda: 0)() == 2 else input_ids
    if ids is None:
        return ""
    try:
        ids = ids[-int(tail_tokens) :].tolist()
    except Exception:
        return ""
    try:
        return tokenizer.decode(ids, skip_special_tokens=False)
    except Exception:
        return ""


def _check_markers_in_tail(tail_text: str) -> _GuardResult:
    if not tail_text:
        return _GuardResult(ok=False, reason="empty_tail")

    # The official TG template ends with `<start_of_turn>model\n` when generation prompt is added.
    if "<start_of_turn>model" not in tail_text:
        return _GuardResult(ok=False, reason="missing_start_of_turn_model")

    # Prefer a stronger check: the tail should end with the generation wrapper (allow trailing whitespace).
    trimmed = tail_text.rstrip()
    if not trimmed.endswith("<start_of_turn>model"):
        # Some tokenizers may keep the newline separately; accept if wrapper is near the end.
        if "<start_of_turn>model" not in trimmed[-64:]:
            return _GuardResult(ok=False, reason="start_of_turn_model_not_at_end")

    return _GuardResult(ok=True)


def check_structured_generation_prompt(tokenizer: Any, input_ids: Any, debug: bool = False) -> bool:
    """
    Return True if the structured prompt likely still contains the generation wrapper.
    """
    tail = _decode_tail(tokenizer, input_ids)
    result = _check_markers_in_tail(tail)
    if debug and not result.ok:
        print(f"[TranslateGemma] TG-034 guard failed: {result.reason} (tail_preview={tail[-80:]!r})")
    return bool(result.ok)


def assert_structured_generation_prompt(tokenizer: Any, input_ids: Any, debug: bool = False) -> None:
    """
    Raise StructuredTruncationError if TG-034 guard fails.
    """
    if not check_structured_generation_prompt(tokenizer, input_ids, debug=debug):
        raise StructuredTruncationError(
            "Structured truncation removed the required generation wrapper "
            "('<start_of_turn>model'). Increase max_input_tokens, disable truncation, "
            "or use prompt_mode=plain."
        )

