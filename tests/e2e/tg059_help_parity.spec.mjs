import { expect, test } from "@playwright/test";
import { mkdir } from "node:fs/promises";
import path from "node:path";

const NODE_NAME = "TranslateGemma";
const VUE_SETTING = "Comfy.VueNodes.Enabled";
const NODE_LIBRARY_SETTING = "Comfy.NodeLibrary.NewDesign";
const FRONTEND_VERSION = process.env.COMFYUI_TEST_FRONTEND_VERSION;
const ARTIFACT_DIR = path.resolve("output", "playwright", "tg059");
const FORBIDDEN_SCREENSHOT_TEXT = [
	"HF_TOKEN",
	"HUGGINGFACE_HUB_TOKEN",
	".planning",
	"reference/docs",
	"COMMAND_LOG",
	"IMPLEMENTATION_RECORD",
];

function versionAtLeast(actual, required) {
	const actualParts = actual.split(/[.+-]/, 3).map(Number);
	const requiredParts = required.split(".").map(Number);
	for (let index = 0; index < 3; index += 1) {
		if (actualParts[index] > requiredParts[index]) return true;
		if (actualParts[index] < requiredParts[index]) return false;
	}
	return true;
}

async function waitForComfyApp(page) {
	await page.waitForFunction(
		() => {
			const app = globalThis.comfyAPI?.app?.app;
			// IMPORTANT: graph/settings exist before the async workspace bootstrap finishes.
			return Boolean(
				app?.graph &&
				app?.ui?.settings &&
				app?.isGraphReady &&
				app?.extensionManager?.spinner === false &&
				globalThis.LiteGraph?.registered_node_types?.TranslateGemma
			);
		},
		null,
		{ timeout: 30_000 }
	);
}

async function assertLiveFrontendVersion(page) {
	const liveVersion = await page.evaluate(
		() => globalThis.__COMFYUI_FRONTEND_VERSION__
	);
	expect(liveVersion).toMatch(/^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$/);
	expect(liveVersion).toBe(FRONTEND_VERSION);
	console.log(`TG059_FRONTEND_VERSION=${liveVersion}`);
}

async function getSetting(page, id) {
	return page.evaluate((settingId) => {
		return globalThis.comfyAPI.app.app.ui.settings.getSettingValue(settingId);
	}, id);
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
	await waitForComfyApp(page);
}

async function openEphemeralBlankWorkflow(page) {
	const create = page.getByRole("button", { name: "Create a new blank workflow" });
	await expect(create).toBeVisible();
	await create.click();
	await page.waitForFunction(() => globalThis.comfyAPI.app.app.graph._nodes.length === 0);
}

async function ensureNodeLibraryOpen(page) {
	const search = page.getByTestId("node-library-search").locator("input");
	if (!(await search.isVisible())) {
		await page.waitForFunction(() => {
			const extensionManager = globalThis.comfyAPI?.app?.app?.extensionManager;
			return (
				extensionManager?.sidebarTab?.sidebarTabs?.some(
					(tab) => tab.id === "node-library"
				) &&
				extensionManager?.command?.commands?.some(
					(item) => item.id === "Workspace.ToggleSidebarTab.node-library"
				)
			);
		});
		await page.evaluate(async () => {
			const extensionManager = globalThis.comfyAPI.app.app.extensionManager;
			if (extensionManager.sidebarTab.activeSidebarTabId !== "node-library") {
				await extensionManager.command.execute(
					"Workspace.ToggleSidebarTab.node-library"
				);
			}
		});
		await page.waitForFunction(() => {
			return (
				globalThis.comfyAPI.app.app.extensionManager.sidebarTab
					.activeSidebarTabId === "node-library"
			);
		});
	}
	await expect(search).toBeVisible();
	return search;
}

async function openNativeHelp(page) {
	const search = await ensureNodeLibraryOpen(page);
	await search.fill(NODE_NAME);

	const leaf = page
		.getByTestId("node-tree-leaf")
		.filter({ hasText: NODE_NAME })
		.first();
	await expect(leaf).toBeVisible();
	await leaf.hover();
	await leaf.getByRole("button", { name: /learn more/i }).click();

	const help = page.locator(".node-help-content");
	await expect(help).toBeVisible();
	return help;
}

async function sidebarClip(page) {
	const search = page.getByTestId("node-library-search");
	const clip = await search.evaluate((element) => {
		const panel = element.closest(".side-bar-panel");
		if (!panel) return null;
		const rect = panel.getBoundingClientRect();
		return { x: rect.x, y: rect.y, width: rect.width, height: rect.height };
	});
	if (!clip) throw new Error("Node Library is not inside the host side-bar panel.");
	expect(clip.width).toBeLessThan(page.viewportSize().width / 2);
	return clip;
}

async function applyEvidencePrivacyMatte(page) {
	await page.addStyleTag({
		content: `
			.side-bar-panel {
				background: rgb(18, 18, 20) !important;
				backdrop-filter: none !important;
			}
		`,
	});
	const background = await page
		.locator(".side-bar-panel")
		.first()
		.evaluate((element) => getComputedStyle(element).backgroundColor);
	expect(background).toBe("rgb(18, 18, 20)");
}

async function assertSafeScreenshotSurface(page, clip) {
	const visibleText = await page.evaluate(
		({ x, width }) => {
			const midpoint = x + width / 2;
			const element = document.elementFromPoint(midpoint, 100);
			return element?.closest(".side-bar-panel")?.innerText ?? "";
		},
		clip
	);
	expect(visibleText).not.toBe("");
	for (const forbidden of FORBIDDEN_SCREENSHOT_TEXT) {
		expect(visibleText).not.toContain(forbidden);
	}
}

async function inspectLiveLabels(page) {
	return page.evaluate(() => {
		const app = globalThis.comfyAPI.app.app;
		const activeGraphCount = app.graph._nodes.length;
		const node = globalThis.LiteGraph?.createNode?.("TranslateGemma");
		if (!node) throw new Error("TranslateGemma node type is not registered.");
		if (node.graph || app.graph._nodes.length !== activeGraphCount) {
			throw new Error("Live label inspection must not mutate the active workflow graph.");
		}
		return node.widgets
			.filter((widget) =>
				["max_new_tokens", "max_input_tokens"].includes(widget.name)
			)
			.map((widget) => ({
				name: widget.name,
				label: widget.label ?? widget.options?.label ?? "",
				value: widget.value,
			}));
	});
}

async function preserveSettings(page) {
	const original = {
		[VUE_SETTING]: await getSetting(page, VUE_SETTING),
	};
	if (versionAtLeast(FRONTEND_VERSION, "1.49.0")) {
		original[NODE_LIBRARY_SETTING] = await getSetting(page, NODE_LIBRARY_SETTING);
		await setSetting(page, NODE_LIBRARY_SETTING, false);
	}
	return original;
}

async function restoreSettings(page, original) {
	await waitForComfyApp(page);
	for (const [id, value] of Object.entries(original)) {
		await setSetting(page, id, value);
	}
	await reloadReady(page);
	for (const [id, value] of Object.entries(original)) {
		expect(await getSetting(page, id)).toEqual(value);
	}
}

test.describe.configure({ mode: "serial" });

async function verifyLiveHostContract(request) {
	const nodeResponse = await request.get(`/api/object_info/${NODE_NAME}`);
	expect(nodeResponse.ok()).toBeTruthy();
	const nodeInfo = await nodeResponse.json();
	expect(nodeInfo[NODE_NAME]).toBeTruthy();

	const docsResponse = await request.get(
		`/extensions/ComfyUI-TranslateGemma/docs/${NODE_NAME}.md`
	);
	expect(docsResponse.ok()).toBeTruthy();
	expect(await docsResponse.text()).toContain("# TranslateGemma");

	const extensionResponse = await request.get(
		"/extensions/ComfyUI-TranslateGemma/extensions/translategemma_help.js"
	);
	expect(extensionResponse.ok()).toBeTruthy();
	const extensionSource = await extensionResponse.text();
	expect(extensionSource).toContain("nodeCreated(node)");
	expect(extensionSource).not.toContain("onDrawForeground");
	expect(extensionSource).not.toContain("tg-help-overlay");
}

// Failure artifacts could otherwise capture a host-loaded workflow outside the scoped sidebar.
test.use({ trace: "off", screenshot: "off" });

for (const renderer of [
	{ name: "Classic", artifactName: "classic", vue: false },
	{ name: "Vue", artifactName: "vue", vue: true },
]) {
	test(`native Node Help and labels work in ${renderer.name}`, async ({ page, request }) => {
		test.setTimeout(150_000);
		await verifyLiveHostContract(request);

		const forbiddenWrites = [];
		page.on("request", (outgoing) => {
			if (
				["POST", "PUT", "PATCH", "DELETE"].includes(outgoing.method()) &&
				/(?:\/prompt|\/userdata\/.*workflow)/i.test(outgoing.url())
			) {
				forbiddenWrites.push(new URL(outgoing.url()).pathname);
			}
		});

		await page.goto("/", { waitUntil: "domcontentloaded" });
		await waitForComfyApp(page);
		await assertLiveFrontendVersion(page);
		const original = await preserveSettings(page);

		try {
			await mkdir(ARTIFACT_DIR, { recursive: true });
			await setSetting(page, VUE_SETTING, renderer.vue);
			await reloadReady(page);
			await openEphemeralBlankWorkflow(page);
			expect(await getSetting(page, VUE_SETTING)).toBe(renderer.vue);

			const search = await ensureNodeLibraryOpen(page);
			await search.fill(NODE_NAME);
			await expect(
				page.getByTestId("node-tree-leaf").filter({ hasText: NODE_NAME }).first()
			).toBeVisible();
			await applyEvidencePrivacyMatte(page);

			const clip = await sidebarClip(page);
			await assertSafeScreenshotSurface(page, clip);
			await page.screenshot({
				path: path.join(ARTIFACT_DIR, `${renderer.artifactName}-baseline.png`),
				clip,
			});

			const help = await openNativeHelp(page);
			for (const content of [
				"TranslateGemma",
				"Inputs",
				"Image Translation",
				"Long Text",
			]) {
				await expect(help).toContainText(content);
			}
			await expect(page.locator("#tg-help-overlay")).toHaveCount(0);
			await assertSafeScreenshotSurface(page, clip);
			await page.screenshot({
				path: path.join(ARTIFACT_DIR, `${renderer.artifactName}-help-open.png`),
				clip,
			});

			const labels = await inspectLiveLabels(page);
			expect(labels).toHaveLength(2);
			expect(labels.map(({ name }) => name)).toEqual([
				"max_new_tokens",
				"max_input_tokens",
			]);
			expect(labels.map(({ label }) => label)).toEqual([
				"max_new_tokens (0 = auto)",
				"max_input_tokens (0 = auto)",
			]);
			for (const { label } of labels) {
				expect(label.match(/0 = auto/g)).toHaveLength(1);
			}
		} finally {
			await restoreSettings(page, original);
		}

		expect(forbiddenWrites).toEqual([]);
	});
}

test("malicious backend metadata stays inert in native fallback help", async ({ page }) => {
	test.setTimeout(120_000);
	const attack = '<img data-tg-xss src=x onerror="window.__tgXss=1">';
	const dialogs = [];
	page.on("dialog", async (dialog) => {
		dialogs.push(dialog.type());
		await dialog.dismiss();
	});

	await page.route("**/api/object_info*", async (route) => {
		const response = await route.fetch();
		const body = await response.json();
		const node = body[NODE_NAME];
		if (node) {
			node.description = attack;
			const spec = node.input?.required?.max_new_tokens;
			if (Array.isArray(spec) && spec[1]) spec[1].tooltip = attack;
		}
		await route.fulfill({ response, json: body });
	});
	await page.route(
		`**/extensions/ComfyUI-TranslateGemma/docs/${NODE_NAME}.md`,
		(route) => route.fulfill({ status: 404, body: "not found" })
	);

	await page.goto("/", { waitUntil: "domcontentloaded" });
	await waitForComfyApp(page);
	const original = await preserveSettings(page);

	try {
		await setSetting(page, VUE_SETTING, true);
		await reloadReady(page);
		await openEphemeralBlankWorkflow(page);
		const help = await openNativeHelp(page);
		await expect(help).toContainText("data-tg-xss");
		await expect(help.locator("img[data-tg-xss]")).toHaveCount(0);
		expect(await page.evaluate(() => globalThis.__tgXss)).toBeUndefined();
		expect(dialogs).toEqual([]);
	} finally {
		await restoreSettings(page, original);
	}
});
