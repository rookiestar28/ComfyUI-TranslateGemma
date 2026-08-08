import ast
import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "nodes" / "translate_node.py"

EXPECTED_REQUIRED = ("text", "target_language", "model_size", "device")
EXPECTED_OPTIONAL = (
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
)
EXPECTED_ADVANCED = {
    "image_enhance",
    "image_resize_mode",
    "image_two_pass",
    "prompt_mode",
    "max_new_tokens",
    "max_input_tokens",
    "truncate_input",
    "strict_context_limit",
    "keep_model_loaded",
    "debug",
    "long_text_strategy",
    "quantization",
}
EXPECTED_SCHEMA_WITHOUT_ADVANCED_SHA256 = (
    "ada2dbaf072e3ded31a43d2eee53f58c34c058a07210895db4b1124235886126"  # pragma: allowlist secret
)


def _input_schema_ast():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    node_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TranslateGemmaNode"
    )
    method = next(
        node
        for node in node_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "INPUT_TYPES"
    )
    return next(node.value for node in ast.walk(method) if isinstance(node, ast.Return))


def _dict_items(node):
    if not isinstance(node, ast.Dict):
        raise AssertionError(f"expected a dictionary AST node, got {type(node).__name__}")
    result = {}
    for key, value in zip(node.keys, node.values, strict=True):
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            raise AssertionError("INPUT_TYPES keys must stay static strings")
        result[key.value] = value
    return result


def _schema_sections():
    root = _dict_items(_input_schema_ast())
    return {name: _dict_items(root[name]) for name in ("required", "optional")}


def _option_dict(spec):
    if not isinstance(spec, ast.Tuple) or len(spec.elts) != 2:
        raise AssertionError("every public input spec must remain a two-item tuple")
    return _dict_items(spec.elts[1])


class _StripAdvanced(ast.NodeTransformer):
    def visit_Dict(self, node):  # noqa: N802 - ast.NodeTransformer contract
        self.generic_visit(node)
        pairs = [
            (key, value)
            for key, value in zip(node.keys, node.values, strict=True)
            if not (isinstance(key, ast.Constant) and key.value == "advanced")
        ]
        node.keys = [key for key, _value in pairs]
        node.values = [value for _key, value in pairs]
        return node


class AdvancedInputMetadataTests(unittest.TestCase):
    def test_exact_component_classification_uses_boolean_true(self):
        sections = _schema_sections()
        self.assertEqual(tuple(sections["required"]), EXPECTED_REQUIRED)
        self.assertEqual(tuple(sections["optional"]), EXPECTED_OPTIONAL)

        actual_advanced = set()
        for section in sections.values():
            for name, spec in section.items():
                options = _option_dict(spec)
                if "advanced" not in options:
                    continue
                self.assertIsInstance(options["advanced"], ast.Constant, name)
                self.assertIs(options["advanced"].value, True, name)
                actual_advanced.add(name)

        self.assertEqual(actual_advanced, EXPECTED_ADVANCED)

    def test_essential_widgets_and_force_link_sockets_are_not_advanced(self):
        sections = _schema_sections()
        essential = {
            *EXPECTED_REQUIRED,
            "source_language",
            "chinese_conversion_only",
            "chinese_conversion_direction",
        }
        for name in essential:
            section = "required" if name in EXPECTED_REQUIRED else "optional"
            self.assertNotIn("advanced", _option_dict(sections[section][name]), name)

        for name in ("image", "external_text"):
            options = _option_dict(sections["optional"][name])
            self.assertNotIn("advanced", options, name)
            self.assertIs(options["forceInput"].value, True, name)

    def test_every_non_advanced_schema_detail_is_frozen(self):
        schema = _StripAdvanced().visit(_input_schema_ast())
        normalized = ast.dump(
            ast.fix_missing_locations(schema),
            annotate_fields=True,
            include_attributes=False,
        )
        actual = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        self.assertEqual(actual, EXPECTED_SCHEMA_WITHOUT_ADVANCED_SHA256)

    def test_no_custom_frontend_advanced_ui_is_introduced(self):
        extension = (ROOT / "web" / "extensions" / "translategemma_help.js").read_text(
            encoding="utf-8"
        )
        for forbidden in (
            "Show Advanced",
            "Hide Advanced",
            "showAdvanced",
            "AlwaysShowAdvancedWidgets",
            "innerHTML",
            "createElement",
        ):
            self.assertNotIn(forbidden, extension)


if __name__ == "__main__":
    unittest.main()
