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

# Get list of language names for ComfyUI dropdown
def get_language_names() -> list[str]:
    """Return sorted list of language names for UI dropdown."""
    return sorted(LANGUAGE_MAP.values())

def get_language_code(language_name: str) -> str:
    """Convert language name to ISO code."""
    return LANGUAGE_NAME_TO_CODE.get(language_name, "en")

def get_language_name(language_code: str) -> str:
    """Convert ISO code to language name."""
    return LANGUAGE_MAP.get(language_code, "English")
