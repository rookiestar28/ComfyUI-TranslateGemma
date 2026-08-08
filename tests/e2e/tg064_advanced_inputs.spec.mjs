import { expect, test } from "@playwright/test";
import { mkdir } from "node:fs/promises";
import path from "node:path";

const NODE_NAME = "TranslateGemma";
const VUE_SETTING = "Comfy.VueNodes.Enabled";
const ADVANCED_SETTING = "Comfy.Node.AlwaysShowAdvancedWidgets";
const FRONTEND_VERSION = process.env.COMFYUI_TEST_FRONTEND_VERSION;
const ARTIFACT_DIR = path.resolve("output", "playwright", "tg064");
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
const VISIBLE_WIDGETS = [
	"text",
	"target_language",
	"model_size",
	"device",
	"source_language",
	"chinese_conversion_only",
	"chinese_conversion_direction",
];
const FORBIDDEN_EVIDENCE = [
	"HF_TOKEN",
	"HUGGINGFACE_HUB_TOKEN",
	".planning",
	"reference/docs",
	"COMMAND_LOG",
	"IMPLEMENTATION_RECORD",
];

async function waitForHost(page) {
	await page.waitForFunction(
		() =>
			Boolean(
				globalThis.comfyAPI?.app?.app?.graph &&
				globalThis.comfyAPI?.app?.app?.isGraphReady &&
				globalThis.LiteGraph?.registered_node_types?.TranslateGemma
			),
		null,
		{ timeout: 30_000 }
	);
}

async function getSetting(page, id) {
	return page.evaluate(
		(settingId) => globalThis.comfyAPI.app.app.ui.settings.getSettingValue(settingId),
		id
	);
}

async function setSetting(page, id, value) {
	await page.evaluate(
		async ({ settingId, nextValue }) => {
			const settings = globalThis.comfyAPI.app.app.ui.settings;
			if (typeof settings.setSettingValueAsync === "function") {
				await settings.setSettingValueAsync(settingId, nextValue);
			} else {
				await settings.setSettingValue(settingId, nextValue);
			}
		},
		{ settingId: id, nextValue: value }
	);
}

async function reloadReady(page) {
	await page.reload({ waitUntil: "domcontentloaded" });
	await waitForHost(page);
}

async function installEvidenceGraph(page) {
	const installed = await page.evaluate(() => {
		const app = globalThis.comfyAPI.app.app;
		if (app.graph._nodes.length !== 0) {
			throw new Error("TG064 evidence requires an isolated blank workflow tab.");
		}
		const node = globalThis.LiteGraph.createNode("TranslateGemma");
		if (!node) throw new Error("TranslateGemma node is not registered.");
		node.pos = [320, 80];
		app.graph.add(node);
		app.canvas?.centerOnNode?.(node);
		app.canvas?.setDirty?.(true, true);
		return { id: String(node.id), widgetCount: node.widgets.length };
	});
	await page.waitForFunction(
		(nodeId) =>
			globalThis.comfyAPI.app.app.graph._nodes.some(
				(item) => String(item.id) === nodeId && item.type === "TranslateGemma"
			),
		installed.id
	);
	return installed;
}

async function routePreChangeObjectInfo(page) {
	const handler = async (route) => {
		const response = await route.fetch();
		const body = await response.json();
		const node = body[NODE_NAME];
		if (node) {
			for (const section of ["required", "optional"]) {
				for (const spec of Object.values(node.input?.[section] ?? {})) {
					if (Array.isArray(spec) && spec[1]) delete spec[1].advanced;
				}
			}
		}
		await route.fulfill({ response, json: body });
	};
	await page.route("**/api/object_info*", handler);
	return handler;
}

async function openEphemeralBlankWorkflow(page) {
	const create = page.getByRole("button", { name: "Create a new blank workflow" });
	await expect(create).toBeVisible();
	await create.click();
	await page.waitForFunction(() => globalThis.comfyAPI.app.app.graph._nodes.length === 0);
}

async function inspectNode(page) {
	return page.evaluate(() => {
		const node = globalThis.comfyAPI.app.app.graph._nodes.find(
			(item) => item.type === "TranslateGemma"
		);
		if (!node) throw new Error("Evidence node is unavailable.");
		return {
			advanced: node.widgets.filter((widget) => widget.advanced).map((widget) => widget.name),
			visible: node.widgets
				.filter((widget) => node.isWidgetVisible(widget))
				.map((widget) => widget.name),
			values: node.widgets.map((widget) => widget.value),
			serialized: node.serialize().widgets_values,
			showAdvanced: Boolean(node.showAdvanced),
		};
	});
}

async function assertEvidenceSafe(page) {
	const nodeState = await page.evaluate(() => {
		const node = globalThis.comfyAPI.app.app.graph._nodes.find(
			(item) => item.type === "TranslateGemma"
		);
		return JSON.stringify(node?.serialize() ?? {});
	});
	for (const forbidden of FORBIDDEN_EVIDENCE) expect(nodeState).not.toContain(forbidden);
	expect(JSON.parse(nodeState).widgets_values[0]).toBe("");
}

async function waitForEvidenceRender(page) {
	await page.evaluate(
		() =>
			new Promise((resolve) => {
				let remaining = 8;
				const next = () => {
					globalThis.comfyAPI.app.app.canvas?.setDirty?.(true, true);
					remaining -= 1;
					if (remaining <= 0) resolve();
					else requestAnimationFrame(next);
				};
				requestAnimationFrame(next);
			})
	);
}

async function screenshotEvidenceNode(page, name, vue) {
	await assertEvidenceSafe(page);
	await waitForEvidenceRender(page);
	// The host canvas swaps its loading watermark on a later paint than app readiness.
	await page.waitForTimeout(1_000);
	await page.waitForFunction(() =>
		globalThis.comfyAPI.app.app.graph._nodes.some(
			(item) => item.type === "TranslateGemma"
		)
	);
	if (vue) {
		const nodeId = await page.evaluate(() =>
			String(
				globalThis.comfyAPI.app.app.graph._nodes.find(
					(item) => item.type === "TranslateGemma"
				).id
			)
		);
		const node = page.locator(`[data-testid="node-body-${nodeId}"]`);
		await expect(node).toBeVisible();
		const bodyBox = await node.boundingBox();
		if (!bodyBox) throw new Error("Vue node body has no screenshot bounds.");
		const advancedButton = page.getByTestId("advanced-inputs-button");
		const buttonBox = (await advancedButton.isVisible())
			? await advancedButton.boundingBox()
			: null;
		const x = Math.min(bodyBox.x, buttonBox?.x ?? bodyBox.x);
		const y = Math.min(bodyBox.y, buttonBox?.y ?? bodyBox.y);
		const right = Math.max(
			bodyBox.x + bodyBox.width,
			buttonBox ? buttonBox.x + buttonBox.width : bodyBox.x + bodyBox.width
		);
		const bottom = Math.max(
			bodyBox.y + bodyBox.height,
			buttonBox ? buttonBox.y + buttonBox.height : bodyBox.y + bodyBox.height
		);
		await page.screenshot({
			path: path.join(ARTIFACT_DIR, name),
			clip: { x, y, width: right - x, height: bottom - y },
		});
		return;
	}
	const clip = await page.evaluate(() => {
		const app = globalThis.comfyAPI.app.app;
		const node = app.graph._nodes.find((item) => item.type === "TranslateGemma");
		const canvasRect = app.canvas.canvas.getBoundingClientRect();
		const scale = app.canvas.ds.scale;
		const offset = app.canvas.ds.offset;
		const x = canvasRect.x + (node.pos[0] + offset[0]) * scale - 8;
		const y = canvasRect.y + (node.pos[1] + offset[1]) * scale - 32;
		return {
			x: Math.max(0, x),
			y: Math.max(0, y),
			width: Math.min(canvasRect.width - Math.max(0, x), node.size[0] * scale + 16),
			height: Math.min(canvasRect.height - Math.max(0, y), node.size[1] * scale + 40),
		};
	});
	expect(clip.width).toBeGreaterThan(100);
	expect(clip.height).toBeGreaterThan(100);
	await page.screenshot({ path: path.join(ARTIFACT_DIR, name), clip });
}

async function addAdvancedLink(page) {
	return page.evaluate(() => {
		const app = globalThis.comfyAPI.app.app;
		const target = app.graph._nodes.find((item) => item.type === "TranslateGemma");
		const source = globalThis.LiteGraph.createNode("PrimitiveInt");
		if (!target || !source) throw new Error("Link validation nodes are unavailable.");
		source.pos = [80, 320];
		app.graph.add(source);
		const targetSlot = target.inputs.findIndex(
			(input) => input.widget?.name === "max_new_tokens" || input.name === "max_new_tokens"
		);
		if (targetSlot < 0) throw new Error("max_new_tokens input slot is unavailable.");
		const link = source.connect(0, target, targetSlot);
		if (!link) throw new Error("Advanced widget link was rejected.");
		return { sourceId: String(source.id), targetSlot };
	});
}

async function assertLinkOperable(page) {
	const linkState = await page.evaluate(() => {
		const target = globalThis.comfyAPI.app.app.graph._nodes.find(
			(item) => item.type === "TranslateGemma"
		);
		const input = target?.inputs.find(
			(item) => item.widget?.name === "max_new_tokens" || item.name === "max_new_tokens"
		);
		return { link: input?.link ?? null, widget: input?.widget?.name ?? null };
	});
	expect(linkState.widget).toBe("max_new_tokens");
	expect(linkState.link).not.toBeNull();
}

// The repository Playwright config fixes workers=1, so settings/graph restoration stays ordered.
// Default mode also lets both renderer diagnostics run when one parity assertion fails.
test.describe.configure({ mode: "default" });
test.use({ trace: "off", screenshot: "off" });

for (const renderer of [
	{ name: "Classic", slug: "classic", vue: false },
	{ name: "Vue", slug: "vue", vue: true },
]) {
	test(`host-native advanced input UX works in ${renderer.name}`, async ({ page, request }) => {
		test.setTimeout(180_000);
		const writes = [];
		page.on("request", (outgoing) => {
			if (
				["POST", "PUT", "PATCH", "DELETE"].includes(outgoing.method()) &&
				/(?:\/prompt|\/userdata\/.*workflow)/i.test(outgoing.url())
			) {
				writes.push(new URL(outgoing.url()).pathname);
			}
		});

		const infoResponse = await request.get(`/api/object_info/${NODE_NAME}`);
		expect(infoResponse.ok()).toBeTruthy();
		const input = (await infoResponse.json())[NODE_NAME].input;
		const actualAdvanced = [...Object.entries(input.required), ...Object.entries(input.optional)]
			.filter(([, spec]) => spec[1]?.advanced === true)
			.map(([name]) => name);

		await page.goto("/", { waitUntil: "domcontentloaded" });
		await waitForHost(page);
		expect(await page.evaluate(() => globalThis.__COMFYUI_FRONTEND_VERSION__)).toBe(
			FRONTEND_VERSION
		);
		const originalSettings = {
			[VUE_SETTING]: await getSetting(page, VUE_SETTING),
			[ADVANCED_SETTING]: await getSetting(page, ADVANCED_SETTING),
		};

		let failure;
		try {
			await mkdir(ARTIFACT_DIR, { recursive: true });
			await setSetting(page, VUE_SETTING, renderer.vue);
			await setSetting(page, ADVANCED_SETTING, false);
			const baselineRoute = await routePreChangeObjectInfo(page);
			await reloadReady(page);
			await openEphemeralBlankWorkflow(page);
			await installEvidenceGraph(page);
			await screenshotEvidenceNode(
				page,
				`${renderer.slug}-pre-change-baseline.png`,
				renderer.vue
			);
			await page.unroute("**/api/object_info*", baselineRoute);
			await reloadReady(page);
			await openEphemeralBlankWorkflow(page);
			await installEvidenceGraph(page);

			expect(actualAdvanced).toEqual(ADVANCED);
			const collapsed = await inspectNode(page);
			expect(collapsed.advanced).toEqual(ADVANCED);
			expect(collapsed.visible).toEqual(VISIBLE_WIDGETS);
			expect(collapsed.showAdvanced).toBe(false);
			await screenshotEvidenceNode(page, `${renderer.slug}-collapsed.png`, renderer.vue);

			if (renderer.vue) {
				const nodeId = await page.evaluate(() =>
					String(
						globalThis.comfyAPI.app.app.graph._nodes.find(
							(item) => item.type === "TranslateGemma"
						).id
					)
				);
				const node = page.locator(`[data-testid="node-body-${nodeId}"]`);
				const advancedButton = page.getByTestId("advanced-inputs-button");
				await expect(advancedButton).toBeVisible();
				await expect(advancedButton).toContainText(/show advanced(?: inputs)?/i);
				await advancedButton.click();
				const tokenControl = node.getByLabel("max_new_tokens", { exact: true });
				await expect(tokenControl).toBeVisible();
				await tokenControl.click();
				const tokenInput = tokenControl.locator("input");
				await expect(tokenInput).toBeEditable();
				await tokenInput.fill("640");
				await tokenInput.blur();
			} else {
				await page.evaluate(() => {
					const node = globalThis.comfyAPI.app.app.graph._nodes.find(
						(item) => item.type === "TranslateGemma"
					);
					node.toggleAdvanced();
					const widget = node.widgets.find((item) => item.name === "max_new_tokens");
					widget.value = 640;
					widget.callback?.(640, globalThis.comfyAPI.app.app.canvas, node, [0, 0], null);
				});
			}

			const expanded = await inspectNode(page);
			expect(expanded.showAdvanced).toBe(true);
			expect(expanded.visible).toEqual([...VISIBLE_WIDGETS.slice(0, 4), ...ADVANCED.slice(0, 3), VISIBLE_WIDGETS[4], ...ADVANCED.slice(3, 10), ...VISIBLE_WIDGETS.slice(5), ...ADVANCED.slice(10)]);
			expect(expanded.values[9]).toBe(640);
			await screenshotEvidenceNode(page, `${renderer.slug}-expanded.png`, renderer.vue);

			await addAdvancedLink(page);
			await assertLinkOperable(page);
			await page.evaluate(() => {
				const node = globalThis.comfyAPI.app.app.graph._nodes.find(
					(item) => item.type === "TranslateGemma"
				);
				node.toggleAdvanced();
			});
			await assertLinkOperable(page);

			const afterCollapse = await inspectNode(page);
			expect(afterCollapse.showAdvanced).toBe(false);
			expect(afterCollapse.serialized[9]).toBe(640);
			if (renderer.vue) {
				const linkedVisible = page.getByLabel("max_new_tokens", { exact: true });
				await expect(linkedVisible).toBeVisible();
			}
		} catch (error) {
			failure = error;
		} finally {
			try {
				await page.evaluate(() => globalThis.comfyAPI.app.app.graph.clear());
				for (const [id, value] of Object.entries(originalSettings)) {
					await setSetting(page, id, value);
				}
				for (const [id, value] of Object.entries(originalSettings)) {
					expect(await getSetting(page, id)).toEqual(value);
				}
			} catch (restorationError) {
				failure ??= restorationError;
			} finally {
				if (!page.isClosed()) await page.close();
			}
		}
		if (failure) throw failure;
		expect(writes).toEqual([]);
	});
}
