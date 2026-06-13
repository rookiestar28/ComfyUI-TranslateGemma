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

    def test_frontend_extension_has_patch_guard_and_label_fallback(self):
        js = (ROOT / "web" / "extensions" / "translategemma_help.js").read_text(encoding="utf-8")

        self.assertIn("const NODE_NAME = \"TranslateGemma\";", js)
        self.assertIn("const PATCH_FLAG = \"__tgHelpPatched\";", js)
        self.assertIn("if (!isTranslateGemmaNode(nodeData)) return;", js)
        self.assertIn("nodeType.prototype[PATCH_FLAG]", js)
        self.assertIn("const LABEL_FALLBACKS", js)
        self.assertIn("existingLabel && existingLabel !== w.name", js)
        self.assertIn("optional <code>bitsandbytes</code>", js)
        self.assertIn("<code>device</code>", js)


if __name__ == "__main__":
    unittest.main()
