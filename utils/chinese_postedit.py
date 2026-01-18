"""
TG-036: Traditional Chinese post-edit utility.

Provides OpenCC-based Simplified → Traditional conversion for consistent
Traditional Chinese output when targeting `zh_Hant`.
"""

from __future__ import annotations

import os
from typing import Optional

# Lazy-loaded OpenCC converter (cached)
_OPENCC_CONVERTER: Optional[object] = None
_OPENCC_IMPORT_ERROR: Optional[str] = None


def postedit_traditional_chinese(
    text: str,
    target_lang_code: str,
    debug: bool = False,
) -> str:
    """
    Best-effort Traditional Chinese post-edit.

    TranslateGemma may output mixed Simplified/Traditional even when targeting `zh_Hant`.
    If OpenCC is available, convert Simplified → Traditional when target is `zh_Hant`.

    Args:
        text: Output text to post-edit.
        target_lang_code: Target language code (e.g., "zh_Hant").
        debug: Whether to print debug messages.

    Returns:
        Post-edited text (Traditional Chinese) or original text if conversion
        is not applicable or not available.
    """
    global _OPENCC_CONVERTER, _OPENCC_IMPORT_ERROR

    if not text or target_lang_code != "zh_Hant":
        return text

    # Check env toggle (default: enabled)
    enabled = os.environ.get("TRANSLATEGEMMA_TRADITIONAL_POSTEDIT", "1").strip()
    if enabled in {"0", "false", "False"}:
        return text

    # Lazy import OpenCC
    if _OPENCC_CONVERTER is None and _OPENCC_IMPORT_ERROR is None:
        try:
            from opencc import OpenCC  # type: ignore

            _OPENCC_CONVERTER = OpenCC("s2t")
        except Exception as e:
            _OPENCC_IMPORT_ERROR = f"{type(e).__name__}: {e}"
            if debug:
                print(
                    "[TranslateGemma] NOTE: Traditional post-edit is enabled but OpenCC is not available. "
                    "To enable Simplified→Traditional conversion, install `opencc-python-reimplemented`. "
                    f"(import error: {_OPENCC_IMPORT_ERROR})"
                )
            return text

    # If import previously failed, skip
    if _OPENCC_IMPORT_ERROR is not None:
        return text

    # Perform conversion
    try:
        assert _OPENCC_CONVERTER is not None
        converted = _OPENCC_CONVERTER.convert(text)
        if debug and converted != text:
            print("[TranslateGemma] Post-edit: converted output to Traditional Chinese (OpenCC s2t)")
        return converted
    except Exception:
        return text
