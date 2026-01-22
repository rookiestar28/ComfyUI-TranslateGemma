"""
Truncation compatibility layer for TranslateGemma (TG-042).

This module encapsulates:
- Tokenizer truncation_side mutation workaround (HF API limitation)
- Structured truncation with template-safe logic
- Unified truncation warning format

Non-goal: Change truncation behavior. This is a readability/maintainability refactor.

HF API Limitation Note:
-----------------------
As of transformers<=4.57, the tokenizer does not support per-call truncation_side.
We must mutate tokenizer.truncation_side temporarily and restore it after tokenization.
This workaround can be removed when HF supports per-call truncation side configuration.
"""

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from .runtime_constants import (
    STRUCTURED_TRUNCATION_MAX_ITERATIONS,
    STRUCTURED_TRUNCATION_ITERATION_BUFFER,
    TOKEN_COUNT_UNLIMITED,
)
from .debug_utils import log_truncation_event


# Global lock for tokenizer mutation (shared with translate_node.py)
_TOKENIZER_MUTATION_LOCK = threading.RLock()


@dataclass
class TruncationMeta:
    """Metadata about a truncation operation."""
    path_used: str  # "structured" or "plain"
    raw_input_tokens: int
    actual_input_tokens: int
    max_input_tokens: int
    overhead_tokens: Optional[int] = None  # Template overhead (structured only)
    was_truncated: bool = False
    guard_outcome: Optional[str] = None  # "pass", "fallback", "error" (TG-034)


@contextmanager
def tokenizer_truncation_side(tokenizer: Any, side: str = "right"):
    """
    Context manager to temporarily set tokenizer.truncation_side.

    HF API Limitation:
    ------------------
    The tokenizer does not support per-call truncation_side as of transformers<=4.57.
    We must mutate the tokenizer's truncation_side attribute and restore it after use.
    This is thread-safe via _TOKENIZER_MUTATION_LOCK.

    Usage:
        with tokenizer_truncation_side(tokenizer, "right"):
            inputs = tokenizer(text, truncation=True, max_length=max_len)

    Args:
        tokenizer: The tokenizer to modify
        side: The truncation side ("left" or "right")

    Yields:
        The tokenizer with modified truncation_side
    """
    with _TOKENIZER_MUTATION_LOCK:
        old_side = getattr(tokenizer, "truncation_side", None)
        needs_restore = old_side is not None and old_side != side

        if needs_restore:
            tokenizer.truncation_side = side

        try:
            yield tokenizer
        finally:
            if needs_restore and old_side is not None:
                tokenizer.truncation_side = old_side


def truncate_text_tokens_to_fit(
    tokenizer: Any,
    text: str,
    max_tokens: int,
) -> Tuple[str, int, bool]:
    """
    Truncate text to fit within max_tokens.

    This function tokenizes the text without special tokens, truncates if needed,
    and decodes back to text. This preserves the template wrapper when used with
    structured prompts.

    Args:
        tokenizer: The tokenizer to use
        text: The text to potentially truncate
        max_tokens: Maximum number of tokens allowed

    Returns:
        Tuple of (truncated_text, original_token_count, was_truncated)
    """
    text_ids = tokenizer(text, add_special_tokens=False).get("input_ids", [])
    original_count = len(text_ids)

    if original_count <= int(max_tokens):
        return text, original_count, False

    truncated_ids = text_ids[: int(max_tokens)]
    truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
    return truncated_text, original_count, True


def build_structured_inputs_with_truncation(
    processor: Any,
    tokenizer: Any,
    input_text: str,
    target_lang_code: str,
    source_lang_code: str,
    max_input_tokens: int,
    truncate_input: bool,
    build_messages_fn: Any,
    debug: bool = False,
) -> Tuple[Any, TruncationMeta]:
    """
    Build structured inputs with template-safe truncation.

    This function:
    1. Measures template overhead (empty message)
    2. Truncates only user text to preserve template markers
    3. Iteratively refines truncation if needed
    4. Returns inputs and metadata

    Args:
        processor: The processor with apply_chat_template
        tokenizer: The tokenizer for text truncation
        input_text: User-provided text
        target_lang_code: Target language code
        source_lang_code: Source language code
        max_input_tokens: Maximum input tokens
        truncate_input: Whether to truncate
        build_messages_fn: Function to build structured messages
        debug: Enable debug logging

    Returns:
        Tuple of (inputs dict, TruncationMeta)
    """
    meta = TruncationMeta(
        path_used="structured",
        raw_input_tokens=0,
        actual_input_tokens=0,
        max_input_tokens=max_input_tokens,
    )

    kwargs_base = dict(
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    with _TOKENIZER_MUTATION_LOCK:
        # Step 1: Measure template overhead
        overhead_messages = build_messages_fn(
            input_text="",
            target_lang_code=target_lang_code,
            source_lang_code=source_lang_code,
        )
        overhead_inputs = processor.apply_chat_template(overhead_messages, **kwargs_base)
        overhead_len = int(overhead_inputs["input_ids"].shape[1])
        meta.overhead_tokens = overhead_len
        available_for_text = int(max_input_tokens) - overhead_len

        # Step 2: Measure raw text tokens
        _, raw_text_tokens, _ = truncate_text_tokens_to_fit(
            tokenizer=tokenizer,
            text=input_text,
            max_tokens=TOKEN_COUNT_UNLIMITED,
        )
        meta.raw_input_tokens = overhead_len + int(raw_text_tokens)

        # Step 3: Check if truncation is feasible
        if truncate_input and int(max_input_tokens) <= overhead_len:
            raise RuntimeError(
                "max_input_tokens is too low for the official structured chat template. "
                f"(max_input_tokens={max_input_tokens}, template_overhead_tokens={overhead_len}) "
                "Increase max_input_tokens or use prompt_mode=plain."
            )

        # Step 4: Truncate text if needed
        truncated_text = input_text
        if truncate_input and meta.raw_input_tokens > int(max_input_tokens) and available_for_text > 0:
            truncated_text, _, _ = truncate_text_tokens_to_fit(
                tokenizer=tokenizer,
                text=input_text,
                max_tokens=available_for_text,
            )
            meta.was_truncated = True

        # Step 5: Iteratively refine truncation
        inputs = None
        for _ in range(STRUCTURED_TRUNCATION_MAX_ITERATIONS):
            messages = build_messages_fn(
                input_text=truncated_text,
                target_lang_code=target_lang_code,
                source_lang_code=source_lang_code,
            )
            inputs = processor.apply_chat_template(messages, **kwargs_base)
            actual_input_len = int(inputs["input_ids"].shape[1])

            if actual_input_len <= int(max_input_tokens) or not truncate_input:
                meta.actual_input_tokens = actual_input_len
                break

            overflow = actual_input_len - int(max_input_tokens)
            available_for_text = max(available_for_text - overflow - STRUCTURED_TRUNCATION_ITERATION_BUFFER, 0)
            if available_for_text <= 0:
                meta.actual_input_tokens = actual_input_len
                break

            truncated_text, _, _ = truncate_text_tokens_to_fit(
                tokenizer=tokenizer,
                text=input_text,
                max_tokens=available_for_text,
            )
            meta.was_truncated = True

        if inputs is not None and meta.actual_input_tokens == 0:
            meta.actual_input_tokens = int(inputs["input_ids"].shape[1])

    return inputs, meta


def log_truncation_if_occurred(meta: TruncationMeta, debug: bool = False) -> None:
    """
    Log truncation event if truncation occurred.

    Args:
        meta: TruncationMeta from a truncation operation
        debug: Enable debug logging
    """
    if not meta.was_truncated and meta.raw_input_tokens <= meta.actual_input_tokens:
        return

    log_truncation_event(
        path_used=meta.path_used,
        raw_input_tokens=meta.raw_input_tokens,
        actual_input_tokens=meta.actual_input_tokens,
        max_input_tokens=meta.max_input_tokens,
        overhead_tokens=meta.overhead_tokens,
        guard_outcome=meta.guard_outcome,
        debug=debug,
    )
