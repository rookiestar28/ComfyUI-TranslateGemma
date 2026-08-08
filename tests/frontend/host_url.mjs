const LOOPBACK_NAMES = new Set(["localhost", "127.0.0.1", "::1", "[::1]"]);

export function parseHostUrl(rawUrl) {
	if (!rawUrl) {
		throw new Error(
			"COMFYUI_TEST_URL is required. Start an approved local ComfyUI host that loads this workspace, then set an explicit loopback URL."
		);
	}

	let parsed;
	try {
		parsed = new URL(rawUrl);
	} catch {
		throw new Error("COMFYUI_TEST_URL must be a valid absolute URL.");
	}

	if (!['http:', 'https:'].includes(parsed.protocol)) {
		throw new Error("COMFYUI_TEST_URL must use http or https.");
	}
	if (!LOOPBACK_NAMES.has(parsed.hostname)) {
		throw new Error("COMFYUI_TEST_URL must target localhost or a loopback IP.");
	}
	if (parsed.username || parsed.password) {
		throw new Error("COMFYUI_TEST_URL must not contain credentials.");
	}
	if (parsed.search || parsed.hash) {
		throw new Error("COMFYUI_TEST_URL must not contain a query or fragment.");
	}
	if (parsed.pathname !== "/") {
		throw new Error("COMFYUI_TEST_URL must identify the host root without a path.");
	}

	return parsed.toString().replace(/\/$/, "");
}

export function requireFrontendVersion(rawVersion) {
	if (!rawVersion || !/^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$/.test(rawVersion)) {
		throw new Error(
			"COMFYUI_TEST_FRONTEND_VERSION is required and must be an explicit semantic version used only as a secret-free evidence label."
		);
	}
	return rawVersion;
}
