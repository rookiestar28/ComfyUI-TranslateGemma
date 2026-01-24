// NOTE: This file is served at:
//   /extensions/ComfyUI-TranslateGemma/extensions/translategemma_help.js
// so we need three ".." to reach /scripts/app.js.
import { app } from "../../../scripts/app.js";

const EXT_NAME = "TranslateGemma.HelpButton";

function escapeHtml(text) {
	if (text === null || text === undefined) return "";
	return String(text)
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/"/g, "&quot;")
		.replace(/'/g, "&#039;");
}

function applyWidgetLabelOverrides(node) {
	if (!node?.widgets) return;
	for (const w of node.widgets) {
		if (!w || !w.name) continue;
		if (w.name === "max_new_tokens") {
			// Keep w.name unchanged (used as the backend prompt key).
			w.label = "max_new_tokens (0 = auto)";
			w.options = { ...(w.options || {}), label: w.label };
		} else if (w.name === "max_input_tokens") {
			w.label = "max_input_tokens (0 = auto)";
			w.options = { ...(w.options || {}), label: w.label };
		}
	}
}

function buildHelpHtml(nodeData) {
	const input = nodeData?.input || {};
	const required = input.required || {};
	const optional = input.optional || {};
	const hasParam = (name) => Object.prototype.hasOwnProperty.call(required, name) || Object.prototype.hasOwnProperty.call(optional, name);
	const sections = [
		{ title: "Required Inputs", data: required },
		{ title: "Optional Inputs", data: optional },
	];

	const rows = [];
	for (const section of sections) {
		const keys = Object.keys(section.data || {});
		if (!keys.length) continue;
		rows.push(`<h3 style="margin:12px 0 6px;">${escapeHtml(section.title)}</h3>`);
		// Use fixed table layout + aggressive wrapping to avoid ComfyUI global styles (e.g. code { white-space: pre })
		// causing long tooltips to overflow and get visually truncated.
		rows.push(`<table style="width:100%; border-collapse:collapse; font-size:14px; table-layout:fixed;">`);
		rows.push(
			`<tr>
				<th style="width:220px; text-align:left; padding:6px; border-bottom:1px solid #333;">Param</th>
				<th style="text-align:left; padding:6px; border-bottom:1px solid #333;">Details</th>
			</tr>`
		);
		for (const key of keys) {
			const spec = section.data[key];
			const type = Array.isArray(spec) ? spec[0] : spec;
			const cfg = Array.isArray(spec) ? spec[1] : {};
			const tooltip = cfg?.tooltip ? String(cfg.tooltip) : "";
			const defVal = cfg?.default;
			const minVal = cfg?.min;
			const maxVal = cfg?.max;
			const stepVal = cfg?.step;

			const extra = [];
			if (defVal !== undefined) extra.push(`default=${escapeHtml(defVal)}`);
			if (minVal !== undefined) extra.push(`min=${escapeHtml(minVal)}`);
			if (maxVal !== undefined) extra.push(`max=${escapeHtml(maxVal)}`);
			if (stepVal !== undefined) extra.push(`step=${escapeHtml(stepVal)}`);

			if ((key === "max_new_tokens" || key === "max_input_tokens") && Number(minVal) === 0) {
				extra.unshift("0 = Auto");
			}

			const details = [
				`<div style="color:#bdbdbd; margin-bottom:4px; overflow-wrap:anywhere; word-break:break-word;"><code style="white-space:normal; overflow-wrap:anywhere; word-break:break-word;">${escapeHtml(type)}</code>${extra.length ? ` — <span style="overflow-wrap:anywhere; word-break:break-word;">${extra.join(", ")}</span>` : ""}</div>`,
				tooltip
					? `<div style="white-space:pre-wrap; line-height:1.35; overflow-wrap:anywhere; word-break:break-word;">${escapeHtml(tooltip)}</div>`
					: "",
			].join("");

			rows.push(
				`<tr>
					<td style="vertical-align:top; padding:6px; border-bottom:1px solid #222; overflow-wrap:anywhere; word-break:break-word;"><code style="white-space:normal; overflow-wrap:anywhere; word-break:break-word;">${escapeHtml(
						key
					)}</code></td>
					<td style="vertical-align:top; padding:6px; border-bottom:1px solid #222; overflow-wrap:anywhere; word-break:break-word;">${details}</td>
				</tr>`
			);
		}
		rows.push(`</table>`);
	}

	return `
		<div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
			<div style="font-size:16px; font-weight:600;">TranslateGemma — Parameter Reference</div>
			<button id="tg-help-close" style="background:#2a2a2a; color:#fff; border:1px solid #444; border-radius:6px; padding:6px 10px; cursor:pointer;">Close</button>
		</div>
		<div style="margin-top:10px; color:#bdbdbd; font-size:14px;">
			Quick tips: <code>max_new_tokens=0</code> and <code>max_input_tokens=0</code> enable Auto sizing.
		</div>
		<div style="margin-top:10px; padding:10px; border:1px solid #222; border-radius:10px; background:#0d0d0d; color:#cfcfcf; font-size:14px; line-height:1.4; overflow-wrap:anywhere; word-break:break-word;">
			<div style="font-weight:600; margin-bottom:6px;">Feature Notes</div>
			<ul style="margin:0; padding-left:18px;">
				${hasParam("quantization") ? `<li><code>quantization</code>: BitsAndBytes VRAM reduction (<code>bnb-8bit</code>/<code>bnb-4bit</code>) requires a CUDA GPU + <code>bitsandbytes</code>. Use <code>none</code> if unsupported.</li>` : ""}
				${hasParam("chinese_conversion_only") ? `<li><code>chinese_conversion_only</code>: OpenCC Simplified↔Traditional conversion without loading the model (text-only).</li>` : ""}
				${hasParam("long_text_strategy") ? `<li><code>long_text_strategy</code>: Use <code>auto-continue</code> or <code>segmented</code> for long documents to reduce early-stop cases.</li>` : ""}
				<li>If model downloads are slow/unreliable, you can manually copy the model snapshot into your ComfyUI models folder and restart (see README for exact folder structure).</li>
			</ul>
		</div>
		${rows.join("")}
	`;
}

function showHelpModal(nodeData) {
	const existing = document.getElementById("tg-help-overlay");
	if (existing) existing.remove();

	const overlay = document.createElement("div");
	overlay.id = "tg-help-overlay";
	Object.assign(overlay.style, {
		position: "fixed",
		inset: "0",
		background: "rgba(0,0,0,0.65)",
		zIndex: 9999,
		display: "flex",
		alignItems: "center",
		justifyContent: "center",
		padding: "16px",
	});

	const panel = document.createElement("div");
	Object.assign(panel.style, {
		width: "min(980px, 96vw)",
		maxHeight: "min(760px, 92vh)",
		overflow: "auto",
		background: "#111",
		border: "1px solid #333",
		borderRadius: "10px",
		padding: "14px",
		boxShadow: "0 10px 40px rgba(0,0,0,0.45)",
	});

	panel.innerHTML = buildHelpHtml(nodeData);
	overlay.appendChild(panel);

	const close = () => overlay.remove();
	overlay.addEventListener("click", (e) => {
		if (e.target === overlay) close();
	});
	panel.querySelector("#tg-help-close")?.addEventListener("click", close);

	document.body.appendChild(overlay);
}

app.registerExtension({
	name: EXT_NAME,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData?.name !== "TranslateGemma") return;

		// Pad computed width slightly so the title-bar '?' doesn't overlap title text.
		const originalComputeSize = nodeType.prototype.computeSize;
		if (originalComputeSize) {
			nodeType.prototype.computeSize = function () {
				const size = originalComputeSize.apply(this, arguments);
				try {
					const titleH = (globalThis.LiteGraph?.NODE_TITLE_HEIGHT ?? 30);
					if (this?.title) {
						size[0] = Math.max(size[0], 120 + titleH * 2);
					}
				} catch {}
				return size;
			};
		}

		const onDrawForeground = nodeType.prototype.onDrawForeground;
		const onMouseDown = nodeType.prototype.onMouseDown;

		nodeType.prototype.onDrawForeground = function (ctx) {
			onDrawForeground?.apply(this, arguments);

			if (!this.__tgLabelsApplied) {
				applyWidgetLabelOverrides(this);
				this.__tgLabelsApplied = true;
			}

			if (this?.flags?.collapsed) return;

			// Draw '?' in the title bar (negative y in LiteGraph coords).
			// Note: TranslateGemma shows an output slot on the title row, so we
			// offset the icon left by one title-height to avoid overlapping the
			// output connector/label and keep it clickable.
			const titleH = globalThis.LiteGraph?.NODE_TITLE_HEIGHT ?? 30;
			const helpX = this.size[0] - 17 - titleH;
			ctx.save();
			ctx.font = "bold 20px Arial";
			ctx.fillStyle = "#fff";
			ctx.fillText("?", helpX, -8);
			ctx.restore();
		};

		nodeType.prototype.onMouseDown = function (e, pos, canvas) {
			if (!this?.flags?.collapsed) {
				// Title-bar click region (VideoHelperSuite-inspired) but uses a
				// tighter hitbox so it doesn't conflict with the output slot.
				const titleH = globalThis.LiteGraph?.NODE_TITLE_HEIGHT ?? 30;
				const helpX = this.size[0] - 17 - titleH;
				const hitLeft = helpX - 14;
				const hitRight = helpX + 14;
				const hitTop = -titleH;
				const hitBottom = 0;
				if (
					pos?.[0] >= hitLeft &&
					pos?.[0] <= hitRight &&
					pos?.[1] >= hitTop &&
					pos?.[1] <= hitBottom
				) {
					showHelpModal(nodeData);
					return true;
				}
			}
			return onMouseDown?.apply(this, arguments);
		};
	},
});
