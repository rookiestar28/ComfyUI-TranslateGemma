"""
TG-038: Chinese variant conversion utilities.

This module provides best-effort Simplified ↔ Traditional conversion via OpenCC,
plus a lightweight heuristic to infer the likely variant of an input string.
"""

from __future__ import annotations

from typing import Optional

_OPENCC_S2T: Optional[object] = None
_OPENCC_T2S: Optional[object] = None
_OPENCC_IMPORT_ERROR: Optional[str] = None


def is_chinese_variant_code(lang_code: str) -> bool:
    """
    Return True if `lang_code` represents a Chinese variant code we handle.

    TranslateGemma uses `zh` (Simplified) and `zh_Hant` (Traditional).
    We also accept minor variants like `zh-Hant`/`zh_Hans` for robustness.
    """
    if not lang_code:
        return False
    normalized = str(lang_code).replace("-", "_")
    return normalized == "zh" or normalized.startswith("zh_")


def _get_opencc_converters(debug: bool = False) -> tuple[object, object]:
    global _OPENCC_S2T, _OPENCC_T2S, _OPENCC_IMPORT_ERROR

    if _OPENCC_IMPORT_ERROR is not None:
        raise ImportError(
            "OpenCC is not available. Install `opencc-python-reimplemented` to enable "
            "chinese_conversion_only / Chinese variant conversion. "
            f"(import error: {_OPENCC_IMPORT_ERROR})"
        )

    if _OPENCC_S2T is None or _OPENCC_T2S is None:
        try:
            from opencc import OpenCC  # type: ignore

            _OPENCC_S2T = OpenCC("s2t")
            _OPENCC_T2S = OpenCC("t2s")
        except Exception as e:
            _OPENCC_IMPORT_ERROR = f"{type(e).__name__}: {e}"
            if debug:
                print(
                    "[TranslateGemma] NOTE: OpenCC is not available. "
                    "Install `opencc-python-reimplemented` to enable Chinese variant conversion. "
                    f"(import error: {_OPENCC_IMPORT_ERROR})"
                )
            raise ImportError(
                "OpenCC is not available. Install `opencc-python-reimplemented` to enable "
                "chinese_conversion_only / Chinese variant conversion."
            ) from e

    assert _OPENCC_S2T is not None and _OPENCC_T2S is not None
    return _OPENCC_S2T, _OPENCC_T2S


def _diff_score(a: str, b: str) -> int:
    if a == b:
        return 0
    n = min(len(a), len(b))
    score = sum(1 for i in range(n) if a[i] != b[i])
    score += abs(len(a) - len(b))
    return score


def infer_chinese_variant(text: str, debug: bool = False) -> Optional[str]:
    """
    Infer whether `text` is more likely Simplified (`zh`) or Traditional (`zh_Hant`).

    Returns:
        "zh" for Simplified, "zh_Hant" for Traditional, or None if ambiguous.
    """
    if not text or not text.strip():
        return None

    s2t, t2s = _get_opencc_converters(debug=debug)
    try:
        as_traditional = s2t.convert(text)
        as_simplified = t2s.convert(text)
    except Exception:
        return None

    # If one conversion is a no-op while the other changes, prefer the no-op side.
    if as_traditional == text and as_simplified != text:
        return "zh_Hant"
    if as_simplified == text and as_traditional != text:
        return "zh"

    # Otherwise compare change magnitude.
    simplified_score = _diff_score(text, as_traditional)  # how much it changes when s→t
    traditional_score = _diff_score(text, as_simplified)  # how much it changes when t→s

    if debug:
        print(
            "[TranslateGemma] Chinese variant inference: "
            f"simplified_score={simplified_score}, traditional_score={traditional_score}"
        )

    if simplified_score == traditional_score:
        return None
    return "zh" if simplified_score > traditional_score else "zh_Hant"


def convert_chinese_variants(
    text: str,
    target_lang_code: str,
    debug: bool = False,
) -> str:
    """
    Convert Chinese text to the requested script variant via OpenCC.

    Args:
        text: Input text to convert.
        target_lang_code: "zh" for Simplified, "zh_Hant" for Traditional
            (also accepts "zh-Hant"/"zh_Hans" variants).
        debug: Enable debug logging.

    Returns:
        Converted text.
    """
    if text is None:
        return ""
    if not target_lang_code:
        raise ValueError("target_lang_code is required for Chinese conversion.")

    normalized = str(target_lang_code).replace("-", "_")
    s2t, t2s = _get_opencc_converters(debug=debug)

    if normalized in {"zh_Hant", "zh_Hant_TW", "zh_Hant_HK", "zh_Hant_MO"} or "Hant" in target_lang_code:
        out = s2t.convert(text)
        if debug and out != text:
            print("[TranslateGemma] OpenCC conversion: s2t (→ Traditional)")
        return out

    if normalized in {"zh", "zh_Hans", "zh_Hans_CN", "zh_Hans_SG"} or "Hans" in target_lang_code:
        out = t2s.convert(text)
        if debug and out != text:
            print("[TranslateGemma] OpenCC conversion: t2s (→ Simplified)")
        return out

    raise ValueError(
        f"Unsupported target_lang_code for Chinese conversion: {target_lang_code!r}. "
        "Expected 'zh' or 'zh_Hant'."
    )

