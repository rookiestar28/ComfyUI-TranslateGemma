import json
import re
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / "example_workflows"
WORKFLOW_FILES = (
    "basic_text_translation.json",
    "external_text_override_template.json",
    "chinese_conversion_only.json",
    "image_translation_explicit_source.json",
    "long_text_segmented.json",
)
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
EXPECTED_WIDGET_NAMES = tuple(name for name, _type, widget in EXPECTED_INPUTS if widget)


def _load(name):
    return json.loads((WORKFLOW_DIR / name).read_text(encoding="utf-8"))


def _legacy_restore(node, consumer_order=None):
    order = consumer_order or EXPECTED_WIDGET_NAMES
    return dict(zip(order, node["widgets_values"], strict=True))


def _named_restore(node, consumer_order=None):
    order = consumer_order or EXPECTED_WIDGET_NAMES
    named = node["widgets_values_named"]
    return {name: named[name] for name in order}


class WorkflowSerializationTests(unittest.TestCase):
    def test_dual_fields_are_exact_and_positional_is_never_removed(self):
        for name in WORKFLOW_FILES:
            with self.subTest(name=name):
                workflow = _load(name)
                self.assertEqual(workflow["version"], 0.4)
                node = workflow["nodes"][0]
                positional = node["widgets_values"]
                named = node["widgets_values_named"]
                self.assertEqual(len(positional), 19)
                self.assertEqual(tuple(named), EXPECTED_WIDGET_NAMES)
                self.assertEqual(named, dict(zip(EXPECTED_WIDGET_NAMES, positional, strict=True)))

    def test_input_name_order_type_and_widget_binding_stay_stable(self):
        expected = [
            (name, input_type, widget)
            for name, input_type, widget in EXPECTED_INPUTS
        ]
        for name in WORKFLOW_FILES:
            with self.subTest(name=name):
                node = _load(name)["nodes"][0]
                actual = [
                    (
                        item["name"],
                        item["type"],
                        item.get("widget", {}).get("name") == item["name"],
                    )
                    for item in node["inputs"]
                ]
                self.assertEqual(actual, expected)
                self.assertEqual(node["type"], "TranslateGemma")

    def test_legacy_and_modern_restore_are_identical_without_named_only_dependency(self):
        for name in WORKFLOW_FILES:
            with self.subTest(name=name):
                node = _load(name)["nodes"][0]
                legacy = _legacy_restore(node)
                modern = _named_restore(node)
                self.assertEqual(legacy, modern)

                positional_only = dict(node)
                positional_only.pop("widgets_values_named")
                self.assertEqual(_legacy_restore(positional_only), modern)

                reordered = tuple(reversed(EXPECTED_WIDGET_NAMES))
                self.assertEqual(
                    _named_restore(node, reordered),
                    {widget: modern[widget] for widget in reordered},
                )

    def test_fixtures_are_prompt_free_and_public_safe(self):
        forbidden_tokens = (
            ".planning",
            "reference/docs",
            "REFERENCE/",
            "COMMAND_LOG",
            "IMPLEMENTATION_RECORD",
            "HF_TOKEN",
            "HUGGINGFACE_HUB_TOKEN",
        )
        windows_path = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")
        private_url = re.compile(r"https?://(?:localhost|127\.0\.0\.1|10\.|192\.168\.)", re.I)
        validation_id = re.compile(r"\bTG-?0*63\b", re.I)

        for name in WORKFLOW_FILES:
            with self.subTest(name=name):
                workflow = _load(name)
                node = workflow["nodes"][0]
                self.assertEqual(node["widgets_values"][0], "")
                self.assertEqual(node["widgets_values_named"]["text"], "")
                payload = json.dumps(workflow, ensure_ascii=False)
                for token in forbidden_tokens:
                    self.assertNotIn(token, payload)
                self.assertIsNone(windows_path.search(payload))
                self.assertIsNone(private_url.search(payload))
                self.assertIsNone(validation_id.search(payload))

    def test_targeted_verifier_exists(self):
        verifier = ROOT / "scripts" / "verify_workflow_compat.py"
        self.assertTrue(verifier.is_file())

    def test_release_metadata_includes_public_workflow_templates(self):
        manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
        self.assertIn("recursive-include example_workflows *.json *.md", manifest)
        with (ROOT / "pyproject.toml").open("rb") as handle:
            package_data = tomllib.load(handle)["tool"]["setuptools"]["package-data"]
        declared = package_data["comfyui_translategemma"]
        self.assertIn("example_workflows/*.json", declared)
        self.assertIn("example_workflows/*.md", declared)


if __name__ == "__main__":
    unittest.main()
