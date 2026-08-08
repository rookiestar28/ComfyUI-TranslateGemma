import assert from "node:assert/strict";
import test from "node:test";

import {
	ADVANCED_WIDGET_NAMES,
	LABEL_FALLBACKS,
	applyAdvancedWidgetMetadataFallbacks,
	applyWidgetLabelFallbacks,
	isTranslateGemmaNode,
} from "../../web/extensions/translategemma_help_contract.js";

test("repairs the 1.48.x Classic advanced metadata bridge without changing values", () => {
	const node = legacyNode();
	node.widgets[0].options.advanced = true;
	node.widgets[1].options.advanced = true;
	const beforeNames = node.widgets.map((widget) => widget.name);
	const beforeValues = node.widgets.map((widget) => widget.value);
	const beforeOptions = structuredClone(node.widgets.map((widget) => widget.options));

	assert.equal(applyAdvancedWidgetMetadataFallbacks(node), 2);
	assert.equal(node.widgets[0].advanced, true);
	assert.equal(node.widgets[1].advanced, true);
	assert.equal(node.widgets[2].advanced, undefined);
	assert.deepEqual(node.widgets.map((widget) => widget.name), beforeNames);
	assert.deepEqual(node.widgets.map((widget) => widget.value), beforeValues);
	assert.deepEqual(node.widgets.map((widget) => widget.options), beforeOptions);
	assert.equal(applyAdvancedWidgetMetadataFallbacks(node), 0);
	assert.ok(ADVANCED_WIDGET_NAMES.includes("max_new_tokens"));
});

test("advanced bridge accepts only exact backend boolean metadata", () => {
	const node = legacyNode();
	node.widgets[0].options.advanced = "true";
	node.widgets[1].options.advanced = false;
	assert.equal(applyAdvancedWidgetMetadataFallbacks(node), 0);
	assert.equal(node.widgets[0].advanced, undefined);
	assert.equal(node.widgets[1].advanced, undefined);
});

function legacyNode() {
	return {
		comfyClass: "TranslateGemma",
		widgets: [
			{
				name: "max_new_tokens",
				value: 512,
				label: "max_new_tokens",
				options: { min: 0, max: 8192, step: 1, label: "max_new_tokens" },
			},
			{
				name: "max_input_tokens",
				value: 2048,
				options: { min: 0, max: 32768, step: 1 },
			},
			{ name: "device", value: "default", options: { values: ["default", "cpu"] } },
		],
	};
}

test("recognizes only the TranslateGemma node identity variants", () => {
	assert.equal(isTranslateGemmaNode({ comfyClass: "TranslateGemma" }), true);
	assert.equal(
		isTranslateGemmaNode({ constructor: { nodeData: { name: "TranslateGemma" } } }),
		true
	);
	assert.equal(isTranslateGemmaNode({ type: "TranslateGemma" }), true);
	assert.equal(isTranslateGemmaNode({ comfyClass: "OtherNode" }), false);
	assert.equal(isTranslateGemmaNode(null), false);
});

for (const frontendVersion of ["1.43.18", "1.48.7"]) {
	test(`repairs raw legacy labels without changing workflow data on ${frontendVersion}`, () => {
		const node = legacyNode();
		const names = node.widgets.map((widget) => widget.name);
		const values = node.widgets.map((widget) => widget.value);
		const unrelatedOptions = structuredClone(node.widgets[2].options);

		assert.equal(applyWidgetLabelFallbacks(node), 2);
		assert.equal(node.widgets[0].label, LABEL_FALLBACKS.max_new_tokens);
		assert.equal(node.widgets[0].options.label, LABEL_FALLBACKS.max_new_tokens);
		assert.equal(node.widgets[1].label, LABEL_FALLBACKS.max_input_tokens);
		assert.equal(node.widgets[1].options.label, LABEL_FALLBACKS.max_input_tokens);
		assert.deepEqual(node.widgets.map((widget) => widget.name), names);
		assert.deepEqual(node.widgets.map((widget) => widget.value), values);
		assert.deepEqual(node.widgets[2].options, unrelatedOptions);
		assert.equal(node.widgets[0].options.max, 8192);
		assert.equal(node.widgets[1].options.max, 32768);
	});
}

test("is idempotent when frontend 1.50.2+ already resolved backend labels", () => {
	const node = legacyNode();
	node.widgets[0].label = LABEL_FALLBACKS.max_new_tokens;
	node.widgets[0].options.label = LABEL_FALLBACKS.max_new_tokens;
	node.widgets[1].label = LABEL_FALLBACKS.max_input_tokens;
	node.widgets[1].options.label = LABEL_FALLBACKS.max_input_tokens;

	const before = structuredClone(node);
	assert.equal(applyWidgetLabelFallbacks(node), 0);
	assert.equal(applyWidgetLabelFallbacks(node), 0);
	assert.deepEqual(node, before);
	assert.equal(node.widgets[0].label.match(/0 = auto/g).length, 1);
	assert.equal(node.widgets[1].label.match(/0 = auto/g).length, 1);
});

test("normalizes mixed raw and resolved label fields without duplicating suffixes", () => {
	const node = legacyNode();
	node.widgets[0].options.label = LABEL_FALLBACKS.max_new_tokens;
	node.widgets[1].label = LABEL_FALLBACKS.max_input_tokens;
	node.widgets[1].options.label = "max_input_tokens";

	assert.equal(applyWidgetLabelFallbacks(node), 2);
	for (const widget of node.widgets.slice(0, 2)) {
		assert.equal(widget.label, LABEL_FALLBACKS[widget.name]);
		assert.equal(widget.options.label, LABEL_FALLBACKS[widget.name]);
		assert.equal(widget.label.match(/0 = auto/g).length, 1);
	}
});

test("preserves meaningful backend or localized labels", () => {
	const node = legacyNode();
	node.widgets[0].label = "Output budget (automatic at zero)";
	node.widgets[0].options.label = "Output budget (automatic at zero)";
	node.widgets[1].label = "輸入權杖上限（0 為自動）";
	node.widgets[1].options.label = "輸入權杖上限（0 為自動）";

	assert.equal(applyWidgetLabelFallbacks(node), 0);
	assert.equal(node.widgets[0].label, "Output budget (automatic at zero)");
	assert.equal(node.widgets[0].options.label, "Output budget (automatic at zero)");
	assert.equal(node.widgets[1].label, "輸入權杖上限（0 為自動）");
	assert.equal(node.widgets[1].options.label, "輸入權杖上限（0 為自動）");
});

test("normalizes raw label fields to an existing meaningful label", () => {
	const node = legacyNode();
	node.widgets[0].options.label = "Output budget (automatic at zero)";
	node.widgets[1].label = "輸入權杖上限（0 為自動）";
	node.widgets[1].options.label = "max_input_tokens";

	assert.equal(applyWidgetLabelFallbacks(node), 2);
	assert.equal(node.widgets[0].label, "Output budget (automatic at zero)");
	assert.equal(node.widgets[0].options.label, "Output budget (automatic at zero)");
	assert.equal(node.widgets[1].label, "輸入權杖上限（0 為自動）");
	assert.equal(node.widgets[1].options.label, "輸入權杖上限（0 為自動）");
});

test("does not copy malicious backend metadata into labels or a DOM surface", () => {
	const attack = '<img data-tg-xss src=x onerror="globalThis.__tgXss=1">';
	const node = legacyNode();
	node.constructor = {
		nodeData: {
			name: "TranslateGemma",
			description: attack,
			input: { required: { max_new_tokens: ["INT", { tooltip: attack }] } },
		},
	};

	assert.equal(applyWidgetLabelFallbacks(node), 2);
	assert.equal(node.widgets[0].label, LABEL_FALLBACKS.max_new_tokens);
	assert.equal(node.widgets[0].label.includes("<"), false);
	assert.equal(typeof globalThis.__tgXss, "undefined");
});
