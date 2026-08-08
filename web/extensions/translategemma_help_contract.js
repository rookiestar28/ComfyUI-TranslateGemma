export const NODE_NAME = "TranslateGemma";

export const LABEL_FALLBACKS = Object.freeze({
	max_new_tokens: "max_new_tokens (0 = auto)",
	max_input_tokens: "max_input_tokens (0 = auto)",
});

export const ADVANCED_WIDGET_NAMES = Object.freeze([
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
]);

const ADVANCED_WIDGET_NAME_SET = new Set(ADVANCED_WIDGET_NAMES);

export function isTranslateGemmaNode(node) {
	const nodeName =
		node?.comfyClass ??
		node?.constructor?.nodeData?.name ??
		node?.type ??
		"";
	return nodeName === NODE_NAME;
}

function isMeaningfulLabel(label, rawName, fallbackLabel) {
	return Boolean(label && label !== rawName && label !== fallbackLabel);
}

export function applyWidgetLabelFallbacks(node) {
	if (!Array.isArray(node?.widgets)) return 0;

	let changed = 0;
	for (const widget of node.widgets) {
		if (!widget?.name) continue;

		const fallbackLabel = LABEL_FALLBACKS[widget.name];
		if (!fallbackLabel) continue;

		const candidateLabels = [widget.label, widget.options?.label];
		const meaningfulLabel = candidateLabels.find((label) =>
			isMeaningfulLabel(label, widget.name, fallbackLabel)
		);
		if (meaningfulLabel) {
			if (
				widget.label !== meaningfulLabel ||
				widget.options?.label !== meaningfulLabel
			) {
				widget.label = meaningfulLabel;
				widget.options = { ...(widget.options || {}), label: meaningfulLabel };
				changed += 1;
			}
			continue;
		}
		if (candidateLabels.includes(fallbackLabel)) {
			if (
				widget.label !== fallbackLabel ||
				widget.options?.label !== fallbackLabel
			) {
				widget.label = fallbackLabel;
				widget.options = { ...(widget.options || {}), label: fallbackLabel };
				changed += 1;
			}
			continue;
		}

		// Keep widget.name/value and every unrelated option unchanged; name is
		// the backend prompt and workflow serialization key.
		widget.label = fallbackLabel;
		widget.options = { ...(widget.options || {}), label: fallbackLabel };
		changed += 1;
	}

	return changed;
}

export function applyAdvancedWidgetMetadataFallbacks(node) {
	if (!Array.isArray(node?.widgets)) return 0;

	let changed = 0;
	for (const widget of node.widgets) {
		if (!ADVANCED_WIDGET_NAME_SET.has(widget?.name)) continue;
		if (widget.options?.advanced !== true || widget.advanced === true) continue;

		// IMPORTANT: frontend 1.48.x stores backend metadata in options, while
		// Classic LiteGraph reads the native widget property directly.
		widget.advanced = true;
		changed += 1;
	}
	return changed;
}
