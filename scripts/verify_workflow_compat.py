#!/usr/bin/env python3
"""Verify legacy/named workflow serialization without exposing fixture values."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


EXPECTED_FILES = {
    "basic_text_translation.json",
    "external_text_override_template.json",
    "chinese_conversion_only.json",
    "image_translation_explicit_source.json",
    "long_text_segmented.json",
}
EXPECTED_INPUTS = (
    ("text", "STRING", True),
    ("target_language", "COMBO", True),
    ("model_size", "COMBO", True),
    ("device", "COMBO", True),
    ("image", "IMAGE", False),
    ("image_enhance", "BOOLEAN", True),
    ("image_resize_mode", "COMBO", True),
    ("image_two_pass", "BOOLEAN", True),
    ("source_language", "COMBO", True),
    ("external_text", "STRING", False),
    ("prompt_mode", "COMBO", True),
    ("max_new_tokens", "INT", True),
    ("max_input_tokens", "INT", True),
    ("truncate_input", "BOOLEAN", True),
    ("strict_context_limit", "BOOLEAN", True),
    ("keep_model_loaded", "BOOLEAN", True),
    ("debug", "BOOLEAN", True),
    ("chinese_conversion_only", "BOOLEAN", True),
    ("chinese_conversion_direction", "COMBO", True),
    ("long_text_strategy", "COMBO", True),
    ("quantization", "COMBO", True),
)
WIDGET_NAMES = tuple(name for name, _kind, widget in EXPECTED_INPUTS if widget)
FORBIDDEN = re.compile(
    r"(?:\.planning|reference/docs|REFERENCE/|COMMAND_LOG|IMPLEMENTATION_RECORD|"
    r"HF_TOKEN|HUGGINGFACE_HUB_TOKEN|\bTG-?0*63\b|"
    r"(?<![A-Za-z])[A-Za-z]:[\\/]|https?://(?:localhost|127\.0\.0\.1|10\.|192\.168\.))",
    re.IGNORECASE,
)


def verify_workflow(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    workflow = json.loads(raw.decode("utf-8"))
    if workflow.get("version") != 0.4:
        raise ValueError(f"{path.name}: workflow version must remain 0.4")
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != 1:
        raise ValueError(f"{path.name}: expected exactly one node")
    node = nodes[0]
    if node.get("type") != "TranslateGemma":
        raise ValueError(f"{path.name}: unexpected node type")

    inputs = node.get("inputs")
    actual_inputs = [
        (
            item.get("name"),
            item.get("type"),
            item.get("widget", {}).get("name") == item.get("name"),
        )
        for item in inputs or []
    ]
    if actual_inputs != list(EXPECTED_INPUTS):
        raise ValueError(f"{path.name}: public input name/order/type/widget contract drift")

    positional = node.get("widgets_values")
    named = node.get("widgets_values_named")
    if not isinstance(positional, list) or len(positional) != len(WIDGET_NAMES):
        raise ValueError(f"{path.name}: positional widget values must contain 19 entries")
    if not isinstance(named, dict) or tuple(named) != WIDGET_NAMES:
        raise ValueError(f"{path.name}: named widget keys/order do not match the official mapping")
    if named != dict(zip(WIDGET_NAMES, positional, strict=True)):
        raise ValueError(f"{path.name}: named and positional widget values differ")
    if positional[0] != "" or named["text"] != "":
        raise ValueError(f"{path.name}: fixture text must be empty")
    if not isinstance(workflow.get("links"), list):
        raise ValueError(f"{path.name}: links must remain an array")

    searchable = json.dumps(workflow, ensure_ascii=False)
    if FORBIDDEN.search(searchable):
        raise ValueError(f"{path.name}: fixture contains forbidden private/internal content")

    return {
        "file": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "widget_count": len(positional),
        "version": workflow["version"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow_dir", type=Path)
    args = parser.parse_args()
    paths = sorted(args.workflow_dir.glob("*.json"))
    names = {path.name for path in paths}
    if names != EXPECTED_FILES:
        missing = sorted(EXPECTED_FILES - names)
        extra = sorted(names - EXPECTED_FILES)
        raise SystemExit(f"workflow file set mismatch; missing={missing}; extra={extra}")

    results = [verify_workflow(path) for path in paths]
    print(json.dumps({"status": "PASS", "workflows": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
