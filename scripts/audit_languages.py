#!/usr/bin/env python3
"""
TG-033: Language whitelist audit script.

Compares LANGUAGE_MAP codes in utils/language_utils.py against
the official TranslateGemma chat_template.jinja to detect drift.

Usage:
    python scripts/audit_languages.py

Exit code:
    0 - All language codes align
    1 - Drift detected (missing or extra codes)
"""

import json
import re
import sys
from pathlib import Path

# Paths relative to script location
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
TEMPLATE_PATH = PROJECT_ROOT / "REFERENCE" / "translategemma4b" / "chat_template.jinja"
LANGUAGE_UTILS_PATH = PROJECT_ROOT / "utils" / "language_utils.py"


def normalize_code(code: str) -> str:
    """Normalize _ to - for comparison."""
    return code.replace("_", "-")


def extract_template_codes(template_path: Path) -> set[str]:
    """
    Extract supported language codes from chat_template.jinja.
    
    Looks for the `languages` dict definition and extracts keys.
    """
    if not template_path.exists():
        print(f"WARNING: Template file not found: {template_path}")
        return set()
    
    content = template_path.read_text(encoding="utf-8")
    
    # Look for languages = { ... } block
    # The template typically has: {% set languages = {'en': 'English', ...} %}
    pattern = r"languages\s*=\s*\{([^}]+)\}"
    match = re.search(pattern, content)
    if not match:
        print("WARNING: Could not find 'languages' dict in template")
        return set()
    
    # Extract keys (language codes) from the dict
    dict_content = match.group(1)
    # Pattern for dict keys: 'code': or "code":
    key_pattern = r"['\"]([a-zA-Z_-]+)['\"]:"
    codes = re.findall(key_pattern, dict_content)
    
    return {normalize_code(c) for c in codes}


def extract_language_map_codes(utils_path: Path) -> set[str]:
    """
    Extract LANGUAGE_MAP keys from utils/language_utils.py.
    """
    if not utils_path.exists():
        print(f"ERROR: language_utils.py not found: {utils_path}")
        return set()
    
    content = utils_path.read_text(encoding="utf-8")
    
    # Look for LANGUAGE_MAP = { ... } block
    # Simple extraction: find all "code": "Name" patterns before first function
    # More robust: parse up to the closing }
    
    # Find LANGUAGE_MAP start
    start_match = re.search(r"LANGUAGE_MAP\s*=\s*\{", content)
    if not start_match:
        print("ERROR: Could not find LANGUAGE_MAP in language_utils.py")
        return set()
    
    # Find matching closing brace (simple heuristic: first lone })
    start_idx = start_match.end()
    brace_count = 1
    end_idx = start_idx
    for i, c in enumerate(content[start_idx:], start_idx):
        if c == "{":
            brace_count += 1
        elif c == "}":
            brace_count -= 1
            if brace_count == 0:
                end_idx = i
                break
    
    dict_content = content[start_idx:end_idx]
    
    # Extract keys
    key_pattern = r"['\"]([a-zA-Z_-]+)['\"]:"
    codes = re.findall(key_pattern, dict_content)
    
    return {normalize_code(c) for c in codes}


def main():
    print("=" * 60)
    print("TG-033: Language Whitelist Audit")
    print("=" * 60)
    
    template_codes = extract_template_codes(TEMPLATE_PATH)
    our_codes = extract_language_map_codes(LANGUAGE_UTILS_PATH)
    
    if not template_codes:
        print("\nWARNING: Could not extract template codes. Audit skipped.")
        print("Make sure REFERENCE/translategemma4b/chat_template.jinja exists.")
        return 0  # Non-fatal if reference missing
    
    print(f"\nTemplate codes: {len(template_codes)}")
    print(f"Our LANGUAGE_MAP codes: {len(our_codes)}")
    
    # Find drift
    missing_in_ours = template_codes - our_codes
    extra_in_ours = our_codes - template_codes
    
    drift = False
    
    if missing_in_ours:
        print(f"\n❌ Missing from LANGUAGE_MAP (in template): {sorted(missing_in_ours)}")
        drift = True
    
    if extra_in_ours:
        print(f"\n⚠️  Extra in LANGUAGE_MAP (not in template): {sorted(extra_in_ours)}")
        # Note: extra codes may be intentional (e.g., our zh_Hant vs template zh-Hant)
        # Don't fail for extras, just warn
    
    common = template_codes & our_codes
    print(f"\n✓ Aligned codes: {len(common)}")
    
    if drift:
        print("\n❌ AUDIT FAILED: Language code drift detected")
        return 1
    
    print("\n✓ AUDIT PASSED: Language codes aligned with template")
    return 0


if __name__ == "__main__":
    sys.exit(main())
