import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FrontendHelpDocsTests(unittest.TestCase):
    def test_embedded_docs_match_node_mapping_key(self):
        node_source = (ROOT / "nodes" / "translate_node.py").read_text(encoding="utf-8")
        docs_path = ROOT / "web" / "docs" / "TranslateGemma.md"

        self.assertIn('"TranslateGemma": TranslateGemmaNode', node_source)
        self.assertTrue(docs_path.exists())

    def test_embedded_docs_cover_current_user_facing_controls(self):
        docs = (ROOT / "web" / "docs" / "TranslateGemma.md").read_text(encoding="utf-8")

        for required_text in (
            "`device`",
            "`quantization`",
            "`image_two_pass`",
            "`long_text_strategy`",
            "`chinese_conversion_only`",
            "Image translation requires an explicit source language",
            "Base installation does not install BitsAndBytes",
        ):
            self.assertIn(required_text, docs)

    def test_embedded_docs_do_not_expose_internal_paths(self):
        docs = (ROOT / "web" / "docs" / "TranslateGemma.md").read_text(encoding="utf-8")

        forbidden = (
            ".planning",
            "reference/docs",
            "REFERENCE/",
            "COMMAND_LOG",
            "IMPLEMENTATION_RECORD",
        )
        for token in forbidden:
            self.assertNotIn(token, docs)

    def test_frontend_extension_has_renderer_independent_label_fallback(self):
        js = (ROOT / "web" / "extensions" / "translategemma_help.js").read_text(encoding="utf-8")
        contract = (
            ROOT / "web" / "extensions" / "translategemma_help_contract.js"
        ).read_text(encoding="utf-8")

        self.assertIn("nodeCreated(node)", js)
        self.assertIn("applyWidgetLabelFallbacks(node)", js)
        self.assertNotIn("beforeRegisterNodeDef", js)
        self.assertNotIn("onDrawForeground", js)
        self.assertNotIn("innerHTML", js)
        self.assertIn('export const NODE_NAME = "TranslateGemma";', contract)
        self.assertIn("export const LABEL_FALLBACKS", contract)
        self.assertIn("isMeaningfulLabel", contract)


if __name__ == "__main__":
    unittest.main()
