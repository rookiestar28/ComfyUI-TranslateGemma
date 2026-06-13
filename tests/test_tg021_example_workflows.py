import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / "example_workflows"
WORKFLOW_FILES = [
    "basic_text_translation.json",
    "external_text_override_template.json",
    "chinese_conversion_only.json",
    "image_translation_explicit_source.json",
    "long_text_segmented.json",
]
EXPECTED_WIDGET_COUNT = 19
EXPECTED_INPUT_NAMES = [
    "text",
    "target_language",
    "model_size",
    "device",
    "image",
    "image_enhance",
    "image_resize_mode",
    "image_two_pass",
    "source_language",
    "external_text",
    "prompt_mode",
    "max_new_tokens",
    "max_input_tokens",
    "truncate_input",
    "strict_context_limit",
    "keep_model_loaded",
    "debug",
    "chinese_conversion_only",
    "chinese_conversion_direction",
    "long_text_strategy",
    "quantization",
]
FORBIDDEN_INTERNAL_TOKENS = (
    ".planning",
    "reference/docs",
    "REFERENCE/",
    "COMMAND_LOG",
    "IMPLEMENTATION_RECORD",
)
FORBIDDEN_SECRET_PATTERNS = (
    re.compile(r"\bHF_TOKEN\s*="),
    re.compile(r"\bHUGGINGFACE_HUB_TOKEN\s*="),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
)


class ExampleWorkflowTests(unittest.TestCase):
    def _load_workflow(self, name):
        path = WORKFLOW_DIR / name
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_expected_example_files_exist(self):
        self.assertTrue((WORKFLOW_DIR / "README.md").exists())
        for name in WORKFLOW_FILES:
            self.assertTrue((WORKFLOW_DIR / name).exists(), name)

    def test_workflows_parse_and_use_only_translategemma_nodes(self):
        for name in WORKFLOW_FILES:
            with self.subTest(name=name):
                workflow = self._load_workflow(name)
                nodes = workflow.get("nodes", [])
                self.assertGreaterEqual(len(nodes), 1)
                self.assertTrue(any(node.get("type") == "TranslateGemma" for node in nodes))
                self.assertTrue(all(node.get("type") == "TranslateGemma" for node in nodes))
                self.assertIsInstance(workflow.get("links"), list)
                self.assertEqual(workflow.get("version"), 0.4)

    def test_workflows_match_current_widget_contract(self):
        for name in WORKFLOW_FILES:
            with self.subTest(name=name):
                workflow = self._load_workflow(name)
                node = workflow["nodes"][0]
                input_names = [item["name"] for item in node.get("inputs", [])]
                self.assertEqual(input_names, EXPECTED_INPUT_NAMES)
                self.assertEqual(len(node.get("widgets_values", [])), EXPECTED_WIDGET_COUNT)
                self.assertEqual(node["widgets_values"][3], "default")
                self.assertEqual(node["widgets_values"][-1], "none")

    def test_workflows_and_example_docs_are_public_safe(self):
        paths = [WORKFLOW_DIR / "README.md", ROOT / "README.md"]
        paths.extend(WORKFLOW_DIR / name for name in WORKFLOW_FILES)
        windows_abs = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")

        for path in paths:
            with self.subTest(path=str(path)):
                text = path.read_text(encoding="utf-8")
                for token in FORBIDDEN_INTERNAL_TOKENS:
                    self.assertNotIn(token, text)
                for pattern in FORBIDDEN_SECRET_PATTERNS:
                    self.assertIsNone(pattern.search(text), pattern.pattern)
                self.assertIsNone(windows_abs.search(text))

    def test_root_readme_links_all_examples(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for name in WORKFLOW_FILES:
            self.assertIn(f"example_workflows/{name}", readme)


if __name__ == "__main__":
    unittest.main()
