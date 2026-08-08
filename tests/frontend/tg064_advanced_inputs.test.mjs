import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const ADVANCED = [
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
];
const ESSENTIAL = [
	"text",
	"target_language",
	"model_size",
	"device",
	"source_language",
	"chinese_conversion_only",
	"chinese_conversion_direction",
];
const SOCKETS = ["image", "external_text"];

function inputBlock(source, name) {
	const marker = `\n                "${name}": (`;
	const start = source.indexOf(marker);
	assert.notEqual(start, -1, name);
	const next = source.indexOf("\n                \"", start + marker.length);
	return source.slice(start, next === -1 ? source.length : next);
}

test("source declares only the planned host-native advanced inputs", async () => {
	const source = await readFile(path.resolve("nodes", "translate_node.py"), "utf8");
	for (const name of ADVANCED) {
		assert.match(inputBlock(source, name), /"advanced": True,/u, name);
	}
	for (const name of [...ESSENTIAL, ...SOCKETS]) {
		assert.doesNotMatch(inputBlock(source, name), /"advanced"\s*:/u, name);
	}
	assert.equal((source.match(/"advanced": True,/gu) ?? []).length, ADVANCED.length);
});

test("native visibility semantics retain linked advanced widgets", () => {
	const visible = ({ hidden = false, advanced = false }, showAdvanced, linked = false) =>
		!hidden && (!advanced || showAdvanced || linked);
	assert.equal(visible({ advanced: true }, false), false);
	assert.equal(visible({ advanced: true }, true), true);
	assert.equal(visible({ advanced: true }, false, true), true);
	assert.equal(visible({ advanced: true, hidden: true }, true, true), false);
});

test("advanced editor state never changes positional or named workflow values", async () => {
	const raw = await readFile(
		path.resolve("example_workflows", "basic_text_translation.json"),
		"utf8"
	);
	const node = JSON.parse(raw).nodes[0];
	const beforeArray = structuredClone(node.widgets_values);
	const beforeNamed = structuredClone(node.widgets_values_named);
	node.showAdvanced = false;
	node.showAdvanced = true;
	assert.deepEqual(node.widgets_values, beforeArray);
	assert.deepEqual(node.widgets_values_named, beforeNamed);
	assert.equal(node.type, "TranslateGemma");
});
