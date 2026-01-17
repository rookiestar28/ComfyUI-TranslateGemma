"""
Language auto-detection helpers for TranslateGemma.

TranslateGemma's official chat template requires an explicit `source_lang_code`.
The model card does not mention an "auto" source code, and the template will
raise if an unsupported code is provided. To keep the UX of "Auto Detect" in the
node UI, we perform best-effort local detection for text inputs.
"""

from __future__ import annotations

from typing import Optional

from .language_utils import LANGUAGE_MAP


def _detect_by_script(text: str) -> Optional[str]:
    # Japanese: Hiragana / Katakana
    for ch in text:
        o = ord(ch)
        if 0x3040 <= o <= 0x309F or 0x30A0 <= o <= 0x30FF:
            return "ja"
        # Korean: Hangul syllables
        if 0xAC00 <= o <= 0xD7AF:
            return "ko"
        # Arabic
        if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x08A0 <= o <= 0x08FF:
            return "ar"
        # Hebrew
        if 0x0590 <= o <= 0x05FF:
            return "he"
        # Greek
        if 0x0370 <= o <= 0x03FF:
            return "el"
        # Thai
        if 0x0E00 <= o <= 0x0E7F:
            return "th"
        # Devanagari (Hindi/Marathi)
        if 0x0900 <= o <= 0x097F:
            return "hi"
        # Bengali
        if 0x0980 <= o <= 0x09FF:
            return "bn"
        # Gujarati
        if 0x0A80 <= o <= 0x0AFF:
            return "gu"
        # Gurmukhi (Punjabi)
        if 0x0A00 <= o <= 0x0A7F:
            return "pa"
        # Tamil
        if 0x0B80 <= o <= 0x0BFF:
            return "ta"
        # Telugu
        if 0x0C00 <= o <= 0x0C7F:
            return "te"
        # Kannada
        if 0x0C80 <= o <= 0x0CFF:
            return "kn"
        # Malayalam
        if 0x0D00 <= o <= 0x0D7F:
            return "ml"
        # Burmese (Myanmar)
        if 0x1000 <= o <= 0x109F:
            return "my"
        # Khmer
        if 0x1780 <= o <= 0x17FF:
            return "km"
        # Lao
        if 0x0E80 <= o <= 0x0EFF:
            return "lo"
        # Cyrillic (Russian/Ukrainian/Bulgarian/Serbian, etc.)
        if 0x0400 <= o <= 0x04FF:
            return "ru"
        # CJK ideographs (Chinese/Japanese)
        if 0x4E00 <= o <= 0x9FFF:
            # If we reached here, we did not see kana, so prefer Chinese.
            return "zh"

    return None


def _normalize_lang_code(lang: str) -> str:
    lang = (lang or "").strip().lower()
    if not lang:
        return ""

    # Common aliases
    if lang == "iw":
        return "he"
    if lang == "in":
        return "id"
    if lang in ("fil", "tl"):
        return "tl"

    # Normalize separators / casing for regional codes.
    lang = lang.replace("-", "_")
    parts = lang.split("_", 1)
    if len(parts) == 2 and parts[1]:
        lang = f"{parts[0]}_{parts[1].upper()}"
    return lang


def detect_source_lang_code(text: str, fallback: str = "en") -> str:
    """
    Best-effort language detection for text inputs.

    Returns a code that is present in LANGUAGE_MAP when possible. If no reliable
    detection is available, returns `fallback` (default: "en").
    """
    if not text or not text.strip():
        return fallback

    supported = set(LANGUAGE_MAP.keys())

    by_script = _detect_by_script(text)
    if by_script and by_script in supported:
        return by_script

    # Optional statistical classifier (preferred when installed).
    # Keep dependency optional at runtime; requirements can include it for best UX.
    try:
        import langid  # type: ignore
    except Exception:
        langid = None

    if langid is not None:
        try:
            lang, _score = langid.classify(text)
            lang = _normalize_lang_code(lang)
            if lang in supported:
                return lang
            # Try base language for regional variants (e.g., pt_BR -> pt).
            base = lang.split("_", 1)[0]
            if base in supported:
                return base
        except Exception:
            pass

    return fallback

