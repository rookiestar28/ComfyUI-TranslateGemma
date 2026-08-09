export const NODE_NAME = "TranslateGemma";

export const LABEL_FALLBACKS = Object.freeze({
	max_new_tokens: "max_new_tokens (0 = auto)",
	max_input_tokens: "max_input_tokens (0 = auto)",
});

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
