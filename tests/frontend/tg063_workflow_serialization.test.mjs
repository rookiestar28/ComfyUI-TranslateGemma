import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const WORKFLOW_FILES = [
	"basic_text_translation.json",
	"external_text_override_template.json",
	"chinese_conversion_only.json",
	"image_translation_explicit_source.json",
	"long_text_segmented.json",
];

const WIDGET_NAMES = [
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

async function loadNode(name) {
	const payload = await readFile(path.resolve("example_workflows", name), "utf8");
	return JSON.parse(payload).nodes[0];
}

function restoreLegacy(node, widgetNames = WIDGET_NAMES) {
	return Object.fromEntries(
		widgetNames.map((name, index) => [name, node.widgets_values[index]])
	);
}

function restoreModern(node, widgetNames = WIDGET_NAMES) {
	return Object.fromEntries(
		widgetNames.map((name) => [name, node.widgets_values_named[name]])
	);
}

test("dual-field fixtures preserve the legacy array and exact named values", async () => {
	for (const name of WORKFLOW_FILES) {
		const node = await loadNode(name);
		assert.equal(node.widgets_values.length, 19, name);
		assert.deepEqual(Object.keys(node.widgets_values_named), WIDGET_NAMES, name);
		assert.deepEqual(restoreLegacy(node), restoreModern(node), name);
	}
});

test("1.49.2+ named restore is order-independent while legacy remains usable", async () => {
	const reorderedNames = [...WIDGET_NAMES].reverse();
	for (const name of WORKFLOW_FILES) {
		const node = await loadNode(name);
		const expected = restoreModern(node);
		const reordered = restoreModern(node, reorderedNames);
		for (const widgetName of reorderedNames) {
			assert.deepEqual(reordered[widgetName], expected[widgetName], `${name}:${widgetName}`);
		}
		assert.deepEqual(restoreLegacy(node), expected, name);
	}
});

test("workflow fixtures keep version 0.4 and contain no prompt content", async () => {
	for (const name of WORKFLOW_FILES) {
		const payload = await readFile(path.resolve("example_workflows", name), "utf8");
		const workflow = JSON.parse(payload);
		assert.equal(workflow.version, 0.4, name);
		assert.equal(workflow.nodes[0].widgets_values[0], "", name);
		assert.equal(workflow.nodes[0].widgets_values_named.text, "", name);
		assert.doesNotMatch(payload, /(?:\.planning|reference\/docs|COMMAND_LOG|TG-?0*63)/i);
	}
});
