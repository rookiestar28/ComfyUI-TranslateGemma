import { defineConfig } from "@playwright/test";

import {
	parseHostUrl,
	requireFrontendVersion,
} from "./tests/frontend/host_url.mjs";

const baseURL = parseHostUrl(process.env.COMFYUI_TEST_URL);
requireFrontendVersion(process.env.COMFYUI_TEST_FRONTEND_VERSION);

export default defineConfig({
	testDir: "./tests/e2e",
	fullyParallel: false,
	workers: 1,
	retries: 0,
	timeout: 60_000,
	expect: { timeout: 15_000 },
	outputDir: "output/playwright/test-results",
	reporter: [
		["line"],
		["html", { outputFolder: "output/playwright/report", open: "never" }],
	],
	use: {
		baseURL,
		browserName: "chromium",
		headless: true,
		viewport: { width: 1440, height: 900 },
		trace: "retain-on-failure",
		screenshot: "only-on-failure",
		video: "off",
	},
});
