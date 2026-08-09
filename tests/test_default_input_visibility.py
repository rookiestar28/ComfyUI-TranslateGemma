import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "nodes" / "translate_node.py"

EXPECTED_WIDGETS = (
    "text",
    "target_language",
    "model_size",
    "device",
    "image_enhance",
    "image_resize_mode",
    "image_two_pass",
    "source_language",
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


def _dict_items(node: ast.AST) -> dict[str, ast.AST]:
    if not isinstance(node, ast.Dict):
        raise AssertionError(f"expected a dictionary AST node, got {type(node).__name__}")
    result = {}
    for key, value in zip(node.keys, node.values, strict=True):
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            raise AssertionError("INPUT_TYPES keys must stay static strings")
        result[key.value] = value
    return result


def _schema_sections() -> dict[str, dict[str, ast.AST]]:
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
    returned = next(node.value for node in ast.walk(method) if isinstance(node, ast.Return))
    root = _dict_items(returned)
    return {name: _dict_items(root[name]) for name in ("required", "optional")}


def _options(spec: ast.AST) -> dict[str, ast.AST]:
    if not isinstance(spec, ast.Tuple) or len(spec.elts) != 2:
        raise AssertionError("every public input spec must remain a two-item tuple")
    return _dict_items(spec.elts[1])


class DefaultInputVisibilityRegressionTests(unittest.TestCase):
    def test_all_established_widgets_are_visible_without_an_advanced_action(self):
        sections = _schema_sections()
        widgets = tuple(
            name
            for section in sections.values()
            for name in section
            if name not in {"image", "external_text"}
        )
        advanced = tuple(
            name
            for section in sections.values()
            for name, spec in section.items()
            if "advanced" in _options(spec)
        )

        self.assertEqual(widgets, EXPECTED_WIDGETS)
        self.assertEqual(
            advanced,
            (),
            "established inputs must not require a renderer-specific advanced reveal action",
        )


if __name__ == "__main__":
    unittest.main()
