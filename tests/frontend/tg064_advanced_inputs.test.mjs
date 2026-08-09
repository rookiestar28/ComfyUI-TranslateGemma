import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const WIDGETS = [
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
];
const SOCKETS = ["image", "external_text"];

function inputBlock(source, name) {
	const marker = `\n                "${name}": (`;
	const start = source.indexOf(marker);
	assert.notEqual(start, -1, name);
	const next = source.indexOf("\n                \"", start + marker.length);
	return source.slice(start, next === -1 ? source.length : next);
}

test("source keeps every established widget visible without advanced metadata", async () => {
	const source = await readFile(path.resolve("nodes", "translate_node.py"), "utf8");
	for (const name of [...WIDGETS, ...SOCKETS]) {
		assert.doesNotMatch(inputBlock(source, name), /"advanced"\s*:/u, name);
	}
	assert.equal((source.match(/"advanced"\s*:/gu) ?? []).length, 0);
});

test("default visibility is independent of renderer advanced state", () => {
	const visible = ({ hidden = false, advanced = false }) => !hidden && !advanced;
	for (const name of WIDGETS) {
		assert.equal(visible({ name }), true, name);
	}
});

test("restored visibility preserves positional and named workflow values", async () => {
	const raw = await readFile(
		path.resolve("example_workflows", "basic_text_translation.json"),
		"utf8"
	);
	const node = JSON.parse(raw).nodes[0];
	const beforeArray = structuredClone(node.widgets_values);
	const beforeNamed = structuredClone(node.widgets_values_named);
	assert.deepEqual(node.widgets_values, beforeArray);
	assert.deepEqual(node.widgets_values_named, beforeNamed);
	assert.equal(node.type, "TranslateGemma");
});
