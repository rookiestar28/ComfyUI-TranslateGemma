import { expect, test } from "@playwright/test";
import { readFile } from "node:fs/promises";
import path from "node:path";

const NODE_NAME = "TranslateGemma";
const FRONTEND_VERSION = process.env.COMFYUI_TEST_FRONTEND_VERSION;
const WORKFLOW_FILES = [
	"basic_text_translation.json",
	"external_text_override_template.json",
	"chinese_conversion_only.json",
	"image_translation_explicit_source.json",
	"long_text_segmented.json",
];

async function loadWorkflows() {
	return Promise.all(
		WORKFLOW_FILES.map(async (name) => {
			const raw = await readFile(path.resolve("example_workflows", name), "utf8");
			return { name, workflow: JSON.parse(raw) };
		})
	);
}

async function waitForComfyApp(page) {
	await page.waitForFunction(
		() =>
			Boolean(
				globalThis.comfyAPI?.app?.app?.graph &&
				globalThis.LiteGraph?.registered_node_types?.TranslateGemma
			),
		null,
		{ timeout: 30_000 }
	);
}

test("workflow serialization round-trip stays positional on the live legacy host", async ({
	page,
	request,
}) => {
	test.setTimeout(90_000);
	const workflows = await loadWorkflows();
	const forbiddenWrites = [];
	page.on("request", (outgoing) => {
		if (
			["POST", "PUT", "PATCH", "DELETE"].includes(outgoing.method()) &&
			/(?:\/prompt|\/userdata\/.*workflow)/i.test(outgoing.url())
		) {
			forbiddenWrites.push(new URL(outgoing.url()).pathname);
		}
	});

	const objectInfoResponse = await request.get(`/api/object_info/${NODE_NAME}`);
	expect(objectInfoResponse.ok()).toBeTruthy();
	const objectInfo = (await objectInfoResponse.json())[NODE_NAME];
	expect(objectInfo).toBeTruthy();
	const liveInputNames = [
		...Object.keys(objectInfo.input.required),
		...Object.keys(objectInfo.input.optional),
	];

	await page.goto("/", { waitUntil: "domcontentloaded" });
	await waitForComfyApp(page);
	expect(await page.evaluate(() => globalThis.__COMFYUI_FRONTEND_VERSION__)).toBe(
		FRONTEND_VERSION
	);

	const result = await page.evaluate(
		({ fixturePayloads, expectedInputNames }) => {
			const app = globalThis.comfyAPI.app.app;
			const beforeCount = app.graph._nodes.length;
			const fixtures = fixturePayloads.map(({ name, workflow }) => {
				const data = workflow.nodes[0];
				const savedInputNames = data.inputs.map((input) => input.name);
				const node = globalThis.LiteGraph.createNode("TranslateGemma");
				if (!node) throw new Error("TranslateGemma node type is not registered.");
				if (node.graph) throw new Error("Detached validation node unexpectedly has a graph.");
				node.configure(data);
				const widgets = node.widgets.filter((widget) => widget.serialize !== false);
				const serialized = node.serialize();
				return {
					name,
					savedInputNames,
					widgetNames: widgets.map((widget) => widget.name),
					widgetValues: widgets.map((widget) => widget.value),
					positionalValues: data.widgets_values,
					namedKeys: Object.keys(data.widgets_values_named),
					serializedValues: serialized.widgets_values,
				};
			});
			return {
				beforeCount,
				afterCount: app.graph._nodes.length,
				expectedInputNames,
				fixtures,
			};
		},
		{ fixturePayloads: workflows, expectedInputNames: liveInputNames }
	);

	expect(result.afterCount).toBe(result.beforeCount);
	for (const fixture of result.fixtures) {
		expect(fixture.savedInputNames, fixture.name).toEqual(result.expectedInputNames);
		expect(fixture.widgetNames, fixture.name).toEqual(fixture.namedKeys);
		expect(fixture.widgetValues, fixture.name).toEqual(fixture.positionalValues);
		expect(fixture.serializedValues, fixture.name).toEqual(fixture.positionalValues);
	}
	expect(forbiddenWrites).toEqual([]);
});
