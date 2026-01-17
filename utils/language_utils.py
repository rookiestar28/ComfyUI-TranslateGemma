# Language code to language name mapping for TranslateGemma
# Supports 55 languages

LANGUAGE_MAP = {
    "en": "English",
    # TranslateGemma's chat template supports script-specific codes like `zh-Hant`.
    # Prefer `zh_Hant` for Traditional Chinese to better enforce Traditional output.
    "zh": "Chinese (Simplified)",
    "zh_Hant": "Chinese (Traditional)",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "tl": "Tagalog",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "cs": "Czech",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "el": "Greek",
    "he": "Hebrew",
    "hu": "Hungarian",
    "ro": "Romanian",
    "bg": "Bulgarian",
    "sk": "Slovak",
    "hr": "Croatian",
    "sl": "Slovenian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "et": "Estonian",
    "sr": "Serbian",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu",
    "fa": "Persian",
    "sw": "Swahili",
    "am": "Amharic",
    "my": "Burmese",
    "km": "Khmer",
    "lo": "Lao",
    "ne": "Nepali",
}

# Reverse mapping: language name to code
LANGUAGE_NAME_TO_CODE = {v: k for k, v in LANGUAGE_MAP.items()}

# TG-010: Normalized code set for validation (template uses `-`, we may use `_`)
SUPPORTED_CODES_NORMALIZED = {k.replace("_", "-") for k in LANGUAGE_MAP.keys()}


def normalize_lang_code(code: str) -> str:
    """
    Normalize a language code for TranslateGemma template compatibility (TG-010).
    
    The official chat_template.jinja normalizes `_` to `-`.
    Example: `zh_Hant` → `zh-Hant`
    """
    return code.replace("_", "-") if code else code


# Get list of language names for ComfyUI dropdown
def get_language_names() -> list[str]:
    """Return sorted list of language names for UI dropdown."""
    return sorted(LANGUAGE_MAP.values())


def get_language_code(language_name: str) -> str:
    """
    Convert language name to ISO code (TG-010).
    
    Lookup order:
    1. Exact match in LANGUAGE_NAME_TO_CODE (e.g., "English" → "en")
    2. If input looks like a code and is supported after normalization, accept it
    3. Otherwise: warn and return "en" (compat fallback)
    
    Note: Set env TRANSLATEGEMMA_STRICT_LANG=1 to raise instead of fallback.
    """
    import os
    
    # 1. Exact name match
    if language_name in LANGUAGE_NAME_TO_CODE:
        return LANGUAGE_NAME_TO_CODE[language_name]
    
    # 2. Check if input is already a supported code (with normalization)
    normalized = normalize_lang_code(language_name)
    if normalized in SUPPORTED_CODES_NORMALIZED:
        # Find the original key that matches after normalization
        for code in LANGUAGE_MAP.keys():
            if normalize_lang_code(code) == normalized:
                return code
    
    # 3. Unknown input: warn and handle based on strict mode
    strict_mode = os.environ.get("TRANSLATEGEMMA_STRICT_LANG", "").strip() == "1"
    
    warning_msg = (
        f"[TranslateGemma] WARNING: Unknown language '{language_name}'. "
        f"Expected one of: {', '.join(sorted(LANGUAGE_NAME_TO_CODE.keys())[:5])}... "
        f"(total {len(LANGUAGE_NAME_TO_CODE)} supported). "
    )
    
    if strict_mode:
        raise ValueError(
            f"{warning_msg}Set TRANSLATEGEMMA_STRICT_LANG=0 or remove it to fallback to 'en'."
        )
    
    print(f"{warning_msg}Defaulting to 'en'. Set TRANSLATEGEMMA_STRICT_LANG=1 to raise instead.")
    return "en"


def get_language_name(language_code: str) -> str:
    """
    Convert ISO code to language name (TG-010).
    
    Supports both `_` and `-` variants (e.g., `zh_Hant` and `zh-Hant`).
    """
    # Try exact match first
    if language_code in LANGUAGE_MAP:
        return LANGUAGE_MAP[language_code]
    
    # Try normalized lookup
    normalized = normalize_lang_code(language_code)
    for code, name in LANGUAGE_MAP.items():
        if normalize_lang_code(code) == normalized:
            return name
    
    return "English"

