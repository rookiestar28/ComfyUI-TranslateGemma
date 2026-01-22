"""
Context and token limit utilities for TranslateGemma.

Provides helpers for:
- Determining reliable context limits from tokenizer
- Clamping generation length to avoid exceeding context window

Reference: TG-003 Implementation Plan
"""

import os
from typing import Optional

# Default context limit when tokenizer reports unreliable value
DEFAULT_CONTEXT_LIMIT = 8192

# Threshold for detecting unreliable model_max_length values
MAX_REASONABLE_CONTEXT = 100_000

DEFAULT_AUTO_MAX_NEW_TOKENS = 1024


def _get_env_int(name: str, default: int, min_val: int, max_val: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
        return max(min_val, min(max_val, val))
    except ValueError:
        return default


def suggest_max_new_tokens(
    input_len: int,
    min_suggested: int = 64,
    max_suggested: Optional[int] = None,
) -> int:
    """
    Suggest a reasonable max_new_tokens value for translation-style tasks.

    This heuristic aims to avoid overly large generation windows (which increase
    latency and can amplify repetition) while leaving enough room for typical
    translations. Users can always override by setting max_new_tokens > 0.
    """
    if max_suggested is None:
        max_suggested = DEFAULT_AUTO_MAX_NEW_TOKENS
    safe_input_len = max(int(input_len or 0), 0)
    suggested = int(safe_input_len * 1.25 + 32)
    return max(min_suggested, min(max_suggested, suggested))


def get_context_limit(tokenizer, fallback: int = DEFAULT_CONTEXT_LIMIT) -> int:
    """
    Determine a reliable context limit from the tokenizer.
    
    Some tokenizers report model_max_length as an extremely large sentinel value
    (e.g., 1e30). This function returns a reasonable limit.
    
    Args:
        tokenizer: HuggingFace tokenizer
        fallback: Fallback value if tokenizer limit is unreliable
        
    Returns:
        Context limit in tokens
    """
    if not hasattr(tokenizer, "model_max_length"):
        return fallback
    
    limit = tokenizer.model_max_length
    
    # Check for unreliable sentinel values
    if limit is None or limit > MAX_REASONABLE_CONTEXT:
        return fallback
    
    return int(limit)


def clamp_max_new_tokens(
    context_limit: int,
    input_len: int,
    requested: int,
    min_output: int = 1,
) -> int:
    """
    Clamp max_new_tokens to avoid exceeding context window.
    
    Args:
        context_limit: Maximum context length (input + output)
        input_len: Current input token count
        requested: User-requested max_new_tokens
        min_output: Minimum output tokens (default 1)
        
    Returns:
        Clamped max_new_tokens value
    """
    available = max(context_limit - input_len, 0)
    
    if available < min_output:
        return 0  # No room for generation
    
    return min(requested, available)


def compute_effective_limits(
    tokenizer,
    input_len: int,
    max_new_tokens: int,
    max_input_tokens: int,
    strict_context_limit: bool = True,
    truncate_input: bool = True,
    debug: bool = False,
) -> dict:
    """
    Compute effective token limits for generation.
    
    Args:
        tokenizer: HuggingFace tokenizer
        input_len: Current input token count
        max_new_tokens: User-requested max output tokens
        max_input_tokens: User-requested max input tokens
        strict_context_limit: Whether to clamp output to fit context
        truncate_input: Whether input truncation is enabled
        debug: Enable debug logging
        
    Returns:
        Dict with:
        - context_limit: Detected context limit
        - effective_max_input: Effective max input tokens
        - effective_max_new_tokens: Clamped max output tokens
        - input_truncated: Whether input exceeds limit
        - output_clamped: Whether output was clamped
    """
    context_limit = get_context_limit(tokenizer)
    
    # Determine effective max input
    effective_max_input = min(max_input_tokens, context_limit - 1)  # Leave room for at least 1 output token
    
    # Check if input would be truncated
    input_truncated = input_len > effective_max_input
    
    # Clamp max_new_tokens if strict mode
    if strict_context_limit:
        # Important: only assume truncated length if truncate_input is enabled
        # Otherwise use actual input length for accurate context calculation
        if truncate_input:
            actual_input_len = min(input_len, effective_max_input)
        else:
            actual_input_len = input_len
        
        effective_max_new_tokens = clamp_max_new_tokens(
            context_limit, actual_input_len, max_new_tokens
        )
        output_clamped = effective_max_new_tokens < max_new_tokens
    else:
        effective_max_new_tokens = max_new_tokens
        output_clamped = False
    
    # Always warn on truncation (not just debug mode)
    if input_truncated and truncate_input:
        print(f"[TranslateGemma] Input will be truncated from {input_len} to {effective_max_input} tokens")
    
    if debug:
        print(f"[TranslateGemma] Context limit: {context_limit}")
        print(f"[TranslateGemma] Input tokens: {input_len}, max_input: {effective_max_input}")
        print(f"[TranslateGemma] Max new tokens: {max_new_tokens} -> {effective_max_new_tokens}")
        if output_clamped:
            print(f"[TranslateGemma] Output clamped from {max_new_tokens} to {effective_max_new_tokens}")
    
    return {
        "context_limit": context_limit,
        "effective_max_input": effective_max_input,
        "effective_max_new_tokens": effective_max_new_tokens,
        "input_truncated": input_truncated,
        "output_clamped": output_clamped,
    }
