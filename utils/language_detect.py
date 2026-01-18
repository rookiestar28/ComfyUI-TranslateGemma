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


_ZH_VARIANT_PAIRS: tuple[tuple[str, str], ...] = (
    # Common simplified/traditional character pairs (not exhaustive).
    ("这", "這"),
    ("来", "來"),
    ("为", "為"),
    ("后", "後"),
    ("发", "發"),
    ("复", "復"),
    ("台", "臺"),
    ("里", "裡"),
    ("面", "麵"),
    ("国", "國"),
    ("汉", "漢"),
    ("语", "語"),
    ("书", "書"),
    ("马", "馬"),
    ("云", "雲"),
    ("网", "網"),
    ("单", "單"),
    ("东", "東"),
    ("门", "門"),
    ("开", "開"),
    ("关", "關"),
    ("见", "見"),
    ("车", "車"),
    ("长", "長"),
    ("风", "風"),
    ("鱼", "魚"),
    ("鸟", "鳥"),
    ("龙", "龍"),
    ("万", "萬"),
    ("与", "與"),
    ("体", "體"),
    ("画", "畫"),
    ("广", "廣"),
    ("极", "極"),
    ("线", "線"),
    ("爱", "愛"),
    ("学", "學"),
    ("写", "寫"),
    ("译", "譯"),
    ("际", "際"),
    ("间", "間"),
    ("过", "過"),
    ("进", "進"),
    ("当", "當"),
    ("时", "時"),
    ("问", "問"),
    ("难", "難"),
    ("应", "應"),
    ("经", "經"),
    ("现", "現"),
    ("术", "術"),
    ("业", "業"),
    ("统", "統"),
    ("动", "動"),
    ("传", "傳"),
    ("优", "優"),
    ("价", "價"),
    ("专", "專"),
    ("两", "兩"),
    ("决", "決"),
    ("别", "別"),
    ("删", "刪"),
    ("听", "聽"),
    ("员", "員"),
    ("图", "圖"),
    ("报", "報"),
    ("处", "處"),
    ("备", "備"),
    ("对", "對"),
    ("导", "導"),
    ("实", "實"),
    ("审", "審"),
    ("开", "開"),
    ("断", "斷"),
    ("换", "換"),
    ("显", "顯"),
    ("标", "標"),
    ("测", "測"),
    ("环", "環"),
    ("确", "確"),
    ("缩", "縮"),
    ("续", "續"),
    ("维", "維"),
    ("网", "網"),
    ("设", "設"),
    ("调", "調"),
    ("资", "資"),
    ("输", "輸"),
    ("错", "錯"),
)

_ZH_SIMPLIFIED_CHARS = {simp for simp, _trad in _ZH_VARIANT_PAIRS}
_ZH_TRADITIONAL_CHARS = {trad for _simp, trad in _ZH_VARIANT_PAIRS}


def _detect_zh_variant(text: str) -> Optional[str]:
    """
    Heuristic detection for Simplified vs Traditional Chinese.

    Returns:
        - "zh_Hant" if Traditional Chinese is more likely
        - "zh" if Simplified Chinese is more likely
        - None if undecidable
    """
    if not text:
        return None

    simplified = 0
    traditional = 0
    for ch in text:
        if ch in _ZH_SIMPLIFIED_CHARS:
            simplified += 1
        elif ch in _ZH_TRADITIONAL_CHARS:
            traditional += 1

    if simplified == 0 and traditional == 0:
        return None

    if traditional > simplified and traditional >= 2:
        return "zh_Hant"
    if simplified > traditional and simplified >= 2:
        return "zh"

    # Low-signal tie: do not guess.
    return None


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
        if by_script == "zh":
            return _detect_zh_variant(text) or by_script
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
            # Prefer explicit region mapping when classifier provides it.
            if lang.startswith("zh_"):
                region = lang.split("_", 1)[1]
                if region in {"TW", "HK", "MO"} and "zh_Hant" in supported:
                    return "zh_Hant"
                if region in {"CN", "SG", "MY"} and "zh" in supported:
                    return "zh"
            if lang in supported:
                return lang
            # Try base language for regional variants (e.g., pt_BR -> pt).
            base = lang.split("_", 1)[0]
            if base in supported:
                if base == "zh":
                    return _detect_zh_variant(text) or base
                return base
        except Exception:
            pass

    return fallback
