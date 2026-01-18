#!/usr/bin/env python3
"""
TG-034: Structured truncation regression check script.

Validates that template-safe truncation preserves the required chat template
wrapper (specifically `<start_of_turn>model`) to prevent target-language drift.

Usage:
    python scripts/tg034_structured_truncation_check.py

Requirements:
    - transformers>=4.57.0
    - Network access (first run) or cached model files

Exit code:
    0 - PASS: all checks pass
    1 - FAIL: wrapper missing after truncation
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    print("=" * 60)
    print("TG-034: Structured Truncation Regression Check")
    print("=" * 60)

    try:
        from transformers import AutoProcessor
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers>=4.57.0")
        return 1

    # IMPORTANT: Do not import `utils` as a package here.
    # `utils/__init__.py` is designed for ComfyUI runtime and imports `folder_paths`.
    # Load `utils/template_guard.py` directly by file path so this script can run
    # outside ComfyUI.
    try:
        import importlib.util

        guard_path = PROJECT_ROOT / "utils" / "template_guard.py"
        spec = importlib.util.spec_from_file_location("tg034_template_guard", guard_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load spec for {guard_path}")
        guard_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard_mod)

        check_structured_generation_prompt = guard_mod.check_structured_generation_prompt
        REQUIRED_TAIL_MARKER = guard_mod.REQUIRED_TAIL_MARKER
    except Exception as e:
        print(f"ERROR: Could not import template guard module: {e}")
        return 1

    # Load processor (tokenizer + template)
    model_id = "google/translategemma-4b-it"
    print(f"\nLoading processor from {model_id}...")
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"ERROR: Could not load processor: {e}")
        print("Try running with network access or pre-downloading the model.")
        return 1

    tokenizer = getattr(processor, "tokenizer", processor)
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

    def build_structured_messages(input_text: str, target_lang_code: str, source_lang_code: str):
        # Keep this script ComfyUI-independent: do not import `utils.prompt_builder`,
        # because it may require ComfyUI runtime modules (e.g. `folder_paths`).
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": source_lang_code,
                        "target_lang_code": target_lang_code,
                        "text": input_text,
                    }
                ],
            }
        ]

    # Build a long structured message
    long_text = "This is a test sentence. " * 200  # ~800 tokens
    messages = build_structured_messages(
        input_text=long_text,
        target_lang_code="en",
        source_lang_code="zh",
    )

    # Apply chat template
    print("\nBuilding structured prompt...")
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    full_len = int(inputs["input_ids"].shape[1])
    print(f"Full prompt length: {full_len} tokens")

    # Check wrapper in full prompt
    wrapper_ok = check_structured_generation_prompt(tokenizer, inputs["input_ids"])
    print(f"Wrapper present in full prompt: {wrapper_ok}")
    if not wrapper_ok:
        print(f"FAIL: Required marker '{REQUIRED_TAIL_MARKER}' not found in full prompt!")
        return 1

    # Simulate truncation by slicing input_ids
    # This tests the guard, not the actual truncation logic.
    test_lengths = [512, 256, 128, 64]
    any_triggered = False

    print("\n--- Truncation simulation ---")
    for trunc_len in test_lengths:
        if trunc_len >= full_len:
            continue
        truncated_ids = inputs["input_ids"][:, :trunc_len]
        wrapper_ok = check_structured_generation_prompt(tokenizer, truncated_ids)
        guard_triggers = not wrapper_ok
        any_triggered = any_triggered or guard_triggers
        status = "PASS" if guard_triggers else "FAIL (wrapper still present)"
        print(f"  {trunc_len} tokens -> guard triggers: {guard_triggers} [{status}]")

    # The guard should detect missing wrapper when heavily truncated
    heavily_truncated = inputs["input_ids"][:, :50]
    guard_triggers = not check_structured_generation_prompt(tokenizer, heavily_truncated)
    if not any_triggered or not guard_triggers:
        print("\nFAIL: Guard did not reliably trigger under truncation.")
        print(f"- any_triggered={any_triggered}")
        print(f"- heavy_truncation_50_tokens_triggered={guard_triggers}")
        return 1

    print("\n" + "=" * 60)
    print("RESULT: PASS - Guard correctly detects wrapper presence/absence")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
