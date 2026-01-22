"""
Debug utilities for TranslateGemma (TG-041).

This module provides:
- Runtime capability detection (print once per process)
- Stop reason inference and logging
- Consistent path/text redaction across text and image paths

Non-goal: Change translation behavior. Debug output is for diagnostics only.
"""

import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# =============================================================================
# TG-028 / TG-041: Privacy / Redaction
# =============================================================================

def is_verbose_debug() -> bool:
    """Check if verbose debug mode is enabled (TG-028)."""
    return os.environ.get("TRANSLATEGEMMA_VERBOSE_DEBUG", "").strip() == "1"


def redact_path(path: str) -> str:
    """
    Redact full path to basename only for non-verbose debug (TG-028).

    In verbose mode, returns the full path for troubleshooting.
    In normal debug mode, only returns the filename to protect privacy.
    """
    if is_verbose_debug():
        return path
    return os.path.basename(path) if path else path


def redact_text(text: str, max_len: int = 50) -> str:
    """
    Redact text content for non-verbose debug (TG-028).

    In verbose mode, returns the full text.
    In normal debug mode, returns only the character count.
    """
    if is_verbose_debug():
        return text
    if not text:
        return "(empty)"
    return f"[{len(text)} chars]"


# =============================================================================
# TG-041: Stop Reason Detection
# =============================================================================

class StopReason(Enum):
    """Possible reasons for generation stopping."""
    UNKNOWN = "unknown"
    EOS_TOKEN = "eos_token"
    END_OF_TURN = "end_of_turn"
    REPEAT_GUARD = "repeat_guard"
    MAX_TOKENS = "max_new_tokens"


@dataclass
class GenerationMeta:
    """Metadata about a generation run for debugging."""
    stop_reason: StopReason = StopReason.UNKNOWN
    generated_tokens: int = 0
    max_new_tokens: int = 0
    repeat_guard_triggered: bool = False
    eos_token_ids: list = field(default_factory=list)
    end_of_turn_id: Optional[int] = None


def infer_stop_reason(
    output_ids: Any,
    prompt_len: int,
    max_new_tokens: int,
    eos_token_ids: list,
    end_of_turn_id: Optional[int],
    stopping_criteria: Any = None,
) -> GenerationMeta:
    """
    Infer why generation stopped based on outputs and configuration.

    This is a best-effort inference since HF generate() doesn't expose stop reason.

    Args:
        output_ids: Generated token IDs (full sequence including prompt)
        prompt_len: Length of the prompt in tokens
        max_new_tokens: Maximum tokens that could be generated
        eos_token_ids: List of EOS token IDs used
        end_of_turn_id: The <end_of_turn> token ID if available
        stopping_criteria: The stopping criteria list (to check repeat guard)

    Returns:
        GenerationMeta with inferred stop reason
    """
    meta = GenerationMeta(
        max_new_tokens=max_new_tokens,
        eos_token_ids=list(eos_token_ids) if eos_token_ids else [],
        end_of_turn_id=end_of_turn_id,
    )

    if output_ids is None:
        return meta

    # Get the generated portion
    try:
        if hasattr(output_ids, "shape"):
            total_len = int(output_ids.shape[-1]) if output_ids.dim() >= 1 else 0
        else:
            total_len = len(output_ids)
        meta.generated_tokens = max(0, total_len - prompt_len)
    except Exception:
        return meta

    if meta.generated_tokens == 0:
        return meta

    # Check if repeat guard was triggered
    if stopping_criteria is not None:
        try:
            for criterion in stopping_criteria:
                if hasattr(criterion, "triggered") and criterion.triggered:
                    meta.stop_reason = StopReason.REPEAT_GUARD
                    meta.repeat_guard_triggered = True
                    return meta
        except Exception:
            pass

    # Check if max_new_tokens was exhausted
    if meta.generated_tokens >= max_new_tokens:
        meta.stop_reason = StopReason.MAX_TOKENS
        return meta

    # Check if the last token is an EOS/end-of-turn token
    try:
        if hasattr(output_ids, "dim") and output_ids.dim() >= 1:
            last_token = int(output_ids[0, -1] if output_ids.dim() == 2 else output_ids[-1])
        else:
            last_token = int(output_ids[-1])

        if end_of_turn_id is not None and last_token == end_of_turn_id:
            meta.stop_reason = StopReason.END_OF_TURN
            return meta

        if eos_token_ids and last_token in eos_token_ids:
            meta.stop_reason = StopReason.EOS_TOKEN
            return meta
    except Exception:
        pass

    return meta


def log_stop_reason(meta: GenerationMeta, debug: bool = False) -> None:
    """Log the stop reason if debug is enabled."""
    if not debug:
        return

    reason_str = meta.stop_reason.value
    details = f"generated_tokens={meta.generated_tokens}, max_new_tokens={meta.max_new_tokens}"

    if meta.stop_reason == StopReason.REPEAT_GUARD:
        details += " (pathological repetition detected)"
    elif meta.stop_reason == StopReason.MAX_TOKENS:
        details += " (output may be truncated)"

    print(f"[TranslateGemma] Stop reason: {reason_str} ({details})")


# =============================================================================
# TG-041: Runtime Capability Detection (Print Once)
# =============================================================================

_capability_logged = False
_capability_lock = threading.Lock()


@dataclass
class RuntimeCapabilities:
    """Detected runtime capabilities for the current environment."""
    transformers_version: str = "unknown"
    has_apply_chat_template: bool = False
    has_chat_template: bool = False
    structured_prompt_available: bool = False
    image_input_mode: str = "unknown"  # "in-memory", "url-only", "unsupported"
    truncation_strategy: str = "unknown"  # "structured", "plain"
    tokenizer_mutation_required: bool = False


def detect_runtime_capabilities(
    processor: Any,
    tokenizer: Any,
    debug: bool = False,
) -> RuntimeCapabilities:
    """
    Detect runtime capabilities for the current processor/tokenizer.

    This is called once per process to log environment info for debugging.

    Args:
        processor: The model processor
        tokenizer: The tokenizer (may be same as processor)
        debug: Whether to print capability info

    Returns:
        RuntimeCapabilities dataclass
    """
    caps = RuntimeCapabilities()

    # Detect transformers version
    try:
        import transformers
        caps.transformers_version = getattr(transformers, "__version__", "unknown")
    except Exception:
        pass

    # Check for apply_chat_template
    caps.has_apply_chat_template = hasattr(processor, "apply_chat_template")

    # Check for chat_template in tokenizer
    try:
        chat_template = getattr(tokenizer, "chat_template", None)
        caps.has_chat_template = chat_template is not None and len(str(chat_template)) > 0
    except Exception:
        pass

    # Structured prompt is available if both conditions are met
    caps.structured_prompt_available = caps.has_apply_chat_template and caps.has_chat_template

    # Tokenizer mutation is required when truncation_side needs to be changed
    try:
        truncation_side = getattr(tokenizer, "truncation_side", None)
        caps.tokenizer_mutation_required = truncation_side is not None and truncation_side != "right"
    except Exception:
        pass

    return caps


def log_runtime_capabilities_once(
    processor: Any,
    tokenizer: Any,
    debug: bool = False,
) -> RuntimeCapabilities:
    """
    Detect and log runtime capabilities (print once per process).

    Args:
        processor: The model processor
        tokenizer: The tokenizer
        debug: Whether to print capability info

    Returns:
        RuntimeCapabilities dataclass
    """
    global _capability_logged

    caps = detect_runtime_capabilities(processor, tokenizer, debug)

    if not debug:
        return caps

    with _capability_lock:
        if _capability_logged:
            return caps
        _capability_logged = True

    # Print capability summary
    print(
        f"[TranslateGemma] Runtime capabilities (TG-041): "
        f"transformers={caps.transformers_version}, "
        f"apply_chat_template={caps.has_apply_chat_template}, "
        f"chat_template={caps.has_chat_template}, "
        f"structured_available={caps.structured_prompt_available}, "
        f"tokenizer_mutation_required={caps.tokenizer_mutation_required}"
    )

    return caps


def log_truncation_event(
    *,
    path_used: str,
    raw_input_tokens: int,
    actual_input_tokens: int,
    max_input_tokens: int,
    overhead_tokens: Optional[int] = None,
    guard_outcome: Optional[str] = None,
    debug: bool = False,
) -> None:
    """
    Log a truncation event with consistent format (TG-041/TG-042).

    Args:
        path_used: "structured" or "plain"
        raw_input_tokens: Original input token count
        actual_input_tokens: Token count after truncation
        max_input_tokens: Maximum allowed input tokens
        overhead_tokens: Template overhead (structured only)
        guard_outcome: TG-034 guard result ("pass", "fallback", "error")
        debug: Whether to print
    """
    if not debug:
        return

    if raw_input_tokens <= actual_input_tokens:
        return  # No truncation occurred

    msg = (
        f"[TranslateGemma] Truncation: "
        f"path={path_used}, "
        f"raw_tokens={raw_input_tokens}, "
        f"actual_tokens={actual_input_tokens}, "
        f"max_tokens={max_input_tokens}"
    )

    if overhead_tokens is not None:
        msg += f", overhead_tokens={overhead_tokens}"

    if guard_outcome is not None:
        msg += f", guard_outcome={guard_outcome}"

    print(msg)


def log_image_input_mode(mode: str, debug: bool = False) -> None:
    """
    Log the image input mode used (TG-041).

    Args:
        mode: "in-memory(images=...)", "in-message(image=...)", "url-only(file)"
        debug: Whether to print
    """
    if debug:
        print(f"[TranslateGemma] Image input mode: {mode}")
