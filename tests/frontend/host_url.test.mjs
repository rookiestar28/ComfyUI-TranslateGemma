import assert from "node:assert/strict";
import test from "node:test";

import { parseHostUrl, requireFrontendVersion } from "./host_url.mjs";

test("accepts explicit loopback ComfyUI URLs", () => {
	assert.equal(parseHostUrl("http://127.0.0.1:8188/"), "http://127.0.0.1:8188");
	assert.equal(parseHostUrl("http://localhost:8188"), "http://localhost:8188");
	assert.equal(parseHostUrl("https://[::1]:8188/"), "https://[::1]:8188");
});

test("rejects missing, remote, credential-bearing, and state-bearing URLs", () => {
	const credentialUrl = new URL("http://127.0.0.1:8188");
	credentialUrl.username = "test-user";
	for (const value of [
		undefined,
		"not-a-url",
		"file:///tmp/comfy",
		"https://example.com",
		credentialUrl.toString(),
		"http://127.0.0.1:8188/?token=secret",
		"http://127.0.0.1:8188/#private",
		"http://127.0.0.1:8188/private/path",
	]) {
		assert.throws(() => parseHostUrl(value));
	}
});

test("requires a secret-free semantic frontend version label", () => {
	assert.equal(requireFrontendVersion("1.50.2"), "1.50.2");
	assert.equal(requireFrontendVersion("1.50.3-dev.1"), "1.50.3-dev.1");
	for (const value of [undefined, "current", "1.50", "https://private.invalid"]) {
		assert.throws(() => requireFrontendVersion(value));
	}
});
