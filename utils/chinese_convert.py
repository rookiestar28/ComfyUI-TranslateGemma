"""
TG-038: Chinese variant conversion utility.

Provides OpenCC-based conversion for Chinese Simplified ↔ Traditional
without loading any translation model.

This module is intentionally isolated from translation codepaths to keep
behavior explicit and avoid entangling model load/inference logic.
"""

from __future__ import annotations

from typing import Optional

# Lazy-loaded OpenCC converters (cached per config)
_OPENCC_CONVERTERS: dict[str, object] = {}
_OPENCC_IMPORT_ERROR: Optional[str] = None


def _get_opencc_converter(config: str, debug: bool = False):
    """
    Get or create a cached OpenCC converter for the given config.

    Args:
        config: OpenCC config string (e.g., "s2t", "t2s").
        debug: Whether to print debug messages.

    Returns:
        OpenCC converter instance.

    Raises:
        ImportError: If OpenCC is not installed.
        RuntimeError: If converter initialization fails.
    """
    global _OPENCC_IMPORT_ERROR

    if config in _OPENCC_CONVERTERS:
        return _OPENCC_CONVERTERS[config]

    if _OPENCC_IMPORT_ERROR is not None:
        raise ImportError(
            f"OpenCC is required for chinese_conversion_only mode but is not available.\n"
            f"Install with: pip install opencc-python-reimplemented\n"
            f"(Import error: {_OPENCC_IMPORT_ERROR})"
        )

    try:
        from opencc import OpenCC  # type: ignore

        _OPENCC_CONVERTERS[config] = OpenCC(config)
        if debug:
            print(f"[TranslateGemma] Initialized OpenCC converter: {config}")
        return _OPENCC_CONVERTERS[config]
    except Exception as e:
        _OPENCC_IMPORT_ERROR = f"{type(e).__name__}: {e}"
        raise ImportError(
            f"OpenCC is required for chinese_conversion_only mode but is not available.\n"
            f"Install with: pip install opencc-python-reimplemented\n"
            f"(Import error: {_OPENCC_IMPORT_ERROR})"
        ) from e


def _compute_change_count(original: str, converted: str) -> int:
    """
    Compute the number of character differences between original and converted text.

    This is a simple proxy for how much the conversion changed the text.
    """
    if len(original) != len(converted):
        # Length changed, count as significant
        return max(len(original), len(converted))
    return sum(1 for a, b in zip(original, converted) if a != b)


def infer_chinese_variant(
    text: str,
    *,
    debug: bool = False,
) -> Optional[str]:
    """
    Infer whether input text is Simplified or Traditional Chinese.

    Uses two signals:
    1. TG-035 character-variant heuristic (simplified/traditional character counts)
    2. OpenCC change score: compare how much s2t vs t2s changes the text

    Args:
        text: Input text to analyze.
        debug: Whether to print debug messages.

    Returns:
        "zh" if Simplified Chinese is detected.
        "zh_Hant" if Traditional Chinese is detected.
        None if the variant is ambiguous or cannot be determined.

    Raises:
        ImportError: If OpenCC is not installed (needed for change score).
    """
    if not text or not text.strip():
        return None

    # Import TG-035 detection helper
    from .language_detect import _detect_zh_variant

    # Signal A: TG-035 character-variant heuristic
    signal_a = _detect_zh_variant(text)

    # Signal B: OpenCC change score
    # Convert in both directions and see which one changes more
    try:
        s2t_converter = _get_opencc_converter("s2t", debug=False)
        t2s_converter = _get_opencc_converter("t2s", debug=False)

        s2t_result = s2t_converter.convert(text)
        t2s_result = t2s_converter.convert(text)

        s2t_changes = _compute_change_count(text, s2t_result)
        t2s_changes = _compute_change_count(text, t2s_result)

        # If s2t changes more, input is likely Simplified
        # If t2s changes more, input is likely Traditional
        signal_b: Optional[str] = None
        min_threshold = 2  # Require at least 2 character changes for confidence
        margin_ratio = 1.5  # Require winner to have 1.5x more changes

        if s2t_changes >= min_threshold and t2s_changes >= min_threshold:
            # Both directions change the text, need clear winner
            if s2t_changes > t2s_changes * margin_ratio:
                signal_b = "zh"  # Input is Simplified (s2t changes more)
            elif t2s_changes > s2t_changes * margin_ratio:
                signal_b = "zh_Hant"  # Input is Traditional (t2s changes more)
        elif s2t_changes >= min_threshold and t2s_changes < min_threshold:
            signal_b = "zh"  # Only s2t changes → input is Simplified
        elif t2s_changes >= min_threshold and s2t_changes < min_threshold:
            signal_b = "zh_Hant"  # Only t2s changes → input is Traditional

        if debug:
            print(
                f"[TranslateGemma] infer_chinese_variant: "
                f"signal_a(TG-035)={signal_a}, signal_b(OpenCC)={signal_b}, "
                f"s2t_changes={s2t_changes}, t2s_changes={t2s_changes}"
            )

    except ImportError:
        # OpenCC not available, rely solely on signal_a
        signal_b = None
        if debug:
            print(
                f"[TranslateGemma] infer_chinese_variant: "
                f"OpenCC not available, using signal_a only: {signal_a}"
            )

    # Combine signals
    if signal_a is not None and signal_b is not None:
        if signal_a == signal_b:
            # Both signals agree
            return signal_a
        else:
            # Signals conflict → ambiguous
            if debug:
                print(
                    f"[TranslateGemma] infer_chinese_variant: "
                    f"signals conflict (signal_a={signal_a}, signal_b={signal_b}), returning None"
                )
            return None
    elif signal_a is not None:
        return signal_a
    elif signal_b is not None:
        return signal_b
    else:
        # Neither signal has confidence
        return None


def convert_chinese_variants(
    text: str,
    *,
    target_lang_code: str,
    debug: bool = False,
    source_lang_code: Optional[str] = None,
) -> str:
    """
    Convert Chinese text between Simplified and Traditional variants using OpenCC.

    This function performs fast, deterministic script conversion without any
    LLM inference. It uses OpenCC's phrase-level longest-match dictionaries
    followed by character-level fallback (the default OpenCC behavior).

    Args:
        text: Input text to convert.
        target_lang_code: Target language code. Must be "zh_Hant" (Traditional)
            or "zh" / "zh_Hans" (Simplified) to trigger conversion.
        debug: Whether to print debug messages.
        source_lang_code: Source language code (e.g., "zh", "zh_Hant").
            Used for logging/diagnostics; not strictly required as OpenCC
            handles mixed input gracefully.

    Returns:
        Converted text on success.

    Raises:
        ValueError: If target_lang_code is not a Chinese variant.
        ImportError: If OpenCC is not installed.
        RuntimeError: If OpenCC conversion fails.
    """
    if not text or not text.strip():
        return text

    # Determine conversion config based on target
    # zh_Hant -> convert TO Traditional (s2t)
    # zh / zh_Hans -> convert TO Simplified (t2s)
    config: Optional[str] = None
    if target_lang_code == "zh_Hant":
        config = "s2t"
    elif target_lang_code in {"zh", "zh_Hans"}:
        config = "t2s"

    if config is None:
        # Not a Chinese variant target
        raise ValueError(
            f"chinese_conversion_only requires a Chinese target language. "
            f"Got target_lang_code='{target_lang_code}'. "
            f"Expected 'zh_Hant' (Traditional) or 'zh'/'zh_Hans' (Simplified)."
        )

    # Get converter (handles lazy init and import errors)
    converter = _get_opencc_converter(config, debug=debug)

    # Perform conversion
    try:
        converted = converter.convert(text)
        if debug:
            direction = "Simplified → Traditional" if config == "s2t" else "Traditional → Simplified"
            if converted != text:
                print(f"[TranslateGemma] Conversion ({direction}): text changed")
            else:
                print(f"[TranslateGemma] Conversion ({direction}): no changes (text may already be in target variant)")
        return converted
    except Exception as e:
        if debug:
            print(f"[TranslateGemma] OpenCC conversion failed: {e}")
        raise RuntimeError(f"OpenCC conversion failed: {e}") from e


def is_chinese_variant_code(lang_code: str) -> bool:
    """
    Check if a language code represents a Chinese variant.

    Args:
        lang_code: Language code to check.

    Returns:
        True if the code is zh, zh_Hant, or zh_Hans.
    """
    return lang_code in {"zh", "zh_Hant", "zh_Hans"}
