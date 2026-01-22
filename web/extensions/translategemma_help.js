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
	const sections = [
		{ title: "Required Inputs", data: input.required || {} },
		{ title: "Optional Inputs", data: input.optional || {} },
	];

	const rows = [];
	for (const section of sections) {
		const keys = Object.keys(section.data || {});
		if (!keys.length) continue;
		rows.push(`<h3 style="margin:12px 0 6px;">${escapeHtml(section.title)}</h3>`);
		rows.push(`<table style="width:100%; border-collapse:collapse; font-size:12px;">`);
		rows.push(
			`<tr>
				<th style="text-align:left; padding:6px; border-bottom:1px solid #333;">Param</th>
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
				`<div style="color:#bdbdbd; margin-bottom:4px;"><code>${escapeHtml(type)}</code>${extra.length ? ` — <span>${extra.join(", ")}</span>` : ""}</div>`,
				tooltip ? `<div style="white-space:pre-wrap; line-height:1.35;">${escapeHtml(tooltip)}</div>` : "",
			].join("");

			rows.push(
				`<tr>
					<td style="vertical-align:top; padding:6px; border-bottom:1px solid #222;"><code>${escapeHtml(
						key
					)}</code></td>
					<td style="vertical-align:top; padding:6px; border-bottom:1px solid #222;">${details}</td>
				</tr>`
			);
		}
		rows.push(`</table>`);
	}

	return `
		<div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
			<div style="font-size:14px; font-weight:600;">TranslateGemma — Parameter Reference</div>
			<button id="tg-help-close" style="background:#2a2a2a; color:#fff; border:1px solid #444; border-radius:6px; padding:6px 10px; cursor:pointer;">Close</button>
		</div>
		<div style="margin-top:10px; color:#bdbdbd; font-size:12px;">
			Quick tips: <code>max_new_tokens=0</code> and <code>max_input_tokens=0</code> enable Auto sizing.
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

		const onDrawForeground = nodeType.prototype.onDrawForeground;
		const onMouseDown = nodeType.prototype.onMouseDown;

		nodeType.prototype.onDrawForeground = function (ctx) {
			onDrawForeground?.apply(this, arguments);

			if (!this.__tgLabelsApplied) {
				applyWidgetLabelOverrides(this);
				this.__tgLabelsApplied = true;
			}

			const size = 14;
			const margin = 6;
			const x = this.size[0] - size - margin;
			const y = margin;
			this.__tgHelpRect = [x, y, size, size];

			ctx.save();
			ctx.fillStyle = "rgba(0,0,0,0.35)";
			ctx.beginPath();
			ctx.arc(x + size / 2, y + size / 2, size / 2, 0, Math.PI * 2);
			ctx.fill();
			ctx.fillStyle = "#fff";
			ctx.font = "bold 12px sans-serif";
			ctx.textAlign = "center";
			ctx.textBaseline = "middle";
			ctx.fillText("?", x + size / 2, y + size / 2 + 0.5);
			ctx.restore();
		};

		nodeType.prototype.onMouseDown = function (e, pos, canvas) {
			const r = this.__tgHelpRect;
			if (r) {
				const [x, y, w, h] = r;
				if (pos[0] >= x && pos[0] <= x + w && pos[1] >= y && pos[1] <= y + h) {
					showHelpModal(nodeData);
					return true;
				}
			}
			return onMouseDown?.apply(this, arguments);
		};
	},
});
