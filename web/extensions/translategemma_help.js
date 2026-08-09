// This file is served from the custom-node extension directory, so three
// parent traversals are required to reach ComfyUI's application module.
import { app } from "../../../scripts/app.js";
import {
	applyWidgetLabelFallbacks,
	isTranslateGemmaNode,
} from "./translategemma_help_contract.js";

const EXT_NAME = "TranslateGemma.HelpCompatibility";

app.registerExtension({
	name: EXT_NAME,
	nodeCreated(node) {
		if (!isTranslateGemmaNode(node)) return;

		// IMPORTANT: keep this per-instance hook renderer-independent. Canvas
		// prototype draw/mouse patches are not invoked by ComfyUI Vue Nodes.
		applyWidgetLabelFallbacks(node);
	},
});
