"""
Runtime constants and heuristics for TranslateGemma (TG-040).

This module centralizes magic numbers and heuristics used throughout the codebase.
Each constant includes a rationale and optional environment variable override for
advanced tuning.

Non-goal: Change default behavior. All defaults preserve existing behavior.

Environment Variable Naming Convention
--------------------------------------
All environment variable overrides use the prefix ``TRANSLATEGEMMA_`` followed by
the constant name in SCREAMING_SNAKE_CASE. For example:

    TRANSLATEGEMMA_GENERATION_HEURISTIC_MULTIPLIER=3.0
    TRANSLATEGEMMA_REPEAT_GUARD_WINDOW=16

Environment Variable Clamp Behavior
-----------------------------------
All numeric overrides are clamped to safe ranges to prevent misconfiguration:

    | Env Var                                       | Type  | Default | Min  | Max   |
    |-----------------------------------------------|-------|---------|------|-------|
    | TRANSLATEGEMMA_GENERATION_HEURISTIC_MULTIPLIER| float | 2.0     | 1.0  | 10.0  |
    | TRANSLATEGEMMA_GENERATION_HEADROOM            | int   | 64      | 0    | 1024  |
    | TRANSLATEGEMMA_REPEAT_GUARD_WINDOW            | int   | 12      | 4    | 64    |
    | TRANSLATEGEMMA_REPEAT_GUARD_MIN_GEN           | int   | 32      | 8    | 256   |

Values outside the range are silently clamped; invalid (non-numeric) values fall
back to the default.
"""

import os
from typing import Optional


# =============================================================================
# Vision / Image Constants
# =============================================================================

# TranslateGemma optimal vision input size (from model card).
# The vision encoder expects 896×896 input for best grounding accuracy.
VISION_TARGET_SIZE: int = 896


# =============================================================================
# Generation Heuristics
# =============================================================================

# Headroom tokens added to heuristic generation cap.
# Prevents output from being cut off mid-sentence on short inputs.
# Formula: heuristic_cap = input_len * GENERATION_HEURISTIC_MULTIPLIER + GENERATION_HEADROOM_TOKENS
GENERATION_HEADROOM_TOKENS: int = 64

# Multiplier for heuristic generation cap.
# Translation rarely exceeds 2x input length; this prevents runaway generation
# when the model fails to emit EOS.
GENERATION_HEURISTIC_MULTIPLIER: float = 2.0

# Maximum tokens for image extraction pass (two-pass mode).
# Extraction (source→source) should be short; cap to reduce rambling.
IMAGE_EXTRACTION_MAX_TOKENS: int = 256

# Minimum input tokens for image translation.
# Values below this can truncate visual tokens, causing hallucinated outputs.
IMAGE_MIN_INPUT_TOKENS: int = 2048


def get_generation_heuristic_cap(input_len: int) -> int:
    """
    Compute heuristic generation cap based on input length.

    Override via TRANSLATEGEMMA_GENERATION_HEURISTIC_MULTIPLIER (float)
    and TRANSLATEGEMMA_GENERATION_HEADROOM (int) for long-form translation.

    Example: For 500 input tokens with defaults:
        cap = 500 * 2.0 + 64 = 1064 tokens
    """
    multiplier = _get_env_float(
        "TRANSLATEGEMMA_GENERATION_HEURISTIC_MULTIPLIER",
        default=GENERATION_HEURISTIC_MULTIPLIER,
        min_val=1.0,
        max_val=10.0,
    )
    headroom = _get_env_int(
        "TRANSLATEGEMMA_GENERATION_HEADROOM",
        default=GENERATION_HEADROOM_TOKENS,
        min_val=0,
        max_val=1024,
    )
    return int(input_len * multiplier + headroom)


# =============================================================================
# Repeat-Token Guard (Degeneracy Stopping)
# =============================================================================

# Window size for repeat-token detection.
# Checks if the last N tokens are all identical (e.g., "리리리리...").
REPEAT_GUARD_WINDOW: int = 12

# Minimum generated tokens before repeat guard activates.
# Prevents false positives on short outputs.
REPEAT_GUARD_MIN_GEN: int = 32


def get_repeat_guard_params() -> tuple[int, int]:
    """
    Get repeat-token guard parameters (window, min_gen).

    Override via TRANSLATEGEMMA_REPEAT_GUARD_WINDOW (int)
    and TRANSLATEGEMMA_REPEAT_GUARD_MIN_GEN (int) for debugging edge cases.
    """
    window = _get_env_int(
        "TRANSLATEGEMMA_REPEAT_GUARD_WINDOW",
        default=REPEAT_GUARD_WINDOW,
        min_val=4,
        max_val=64,
    )
    min_gen = _get_env_int(
        "TRANSLATEGEMMA_REPEAT_GUARD_MIN_GEN",
        default=REPEAT_GUARD_MIN_GEN,
        min_val=8,
        max_val=256,
    )
    return window, min_gen


# =============================================================================
# Structured Truncation Constants
# =============================================================================

# Maximum iterations for structured truncation fitting loop.
# Prevents infinite loops when overhead estimation is inaccurate.
STRUCTURED_TRUNCATION_MAX_ITERATIONS: int = 8

# Extra tokens to subtract per iteration when truncation overshoots.
# Accounts for tokenizer boundary effects.
STRUCTURED_TRUNCATION_ITERATION_BUFFER: int = 8

# Sentinel value representing "no limit" for token counting.
# Used when probing raw input length without truncation.
TOKEN_COUNT_UNLIMITED: int = 10**9


# =============================================================================
# Environment Variable Helpers
# =============================================================================

def _get_env_int(name: str, default: int, min_val: int, max_val: int) -> int:
    """Get integer from environment with clamping."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
        return max(min_val, min(max_val, val))
    except ValueError:
        return default


def _get_env_float(name: str, default: float, min_val: float, max_val: float) -> float:
    """Get float from environment with clamping."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = float(raw)
        return max(min_val, min(max_val, val))
    except ValueError:
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    """Get boolean from environment (1/true/yes = True, 0/false/no = False)."""
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes")


# =============================================================================
# Debug / Logging Constants
# =============================================================================

# Log prefix for all TranslateGemma messages.
LOG_PREFIX: str = "[TranslateGemma]"
