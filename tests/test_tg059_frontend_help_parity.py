import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTENSION_PATH = ROOT / "web" / "extensions" / "translategemma_help.js"
CONTRACT_PATH = ROOT / "web" / "extensions" / "translategemma_help_contract.js"


class FrontendHelpParityTests(unittest.TestCase):
    def test_extension_uses_renderer_independent_hook_only(self):
        source = EXTENSION_PATH.read_text(encoding="utf-8")

        self.assertIn("nodeCreated(node)", source)
        for legacy_surface in (
            "beforeRegisterNodeDef",
            "computeSize",
            "onDrawForeground",
            "onMouseDown",
            "innerHTML",
            "tg-help-overlay",
        ):
            self.assertNotIn(legacy_surface, source)

    def test_contract_module_is_present_and_imported(self):
        source = EXTENSION_PATH.read_text(encoding="utf-8")

        self.assertTrue(CONTRACT_PATH.exists())
        self.assertIn('from "./translategemma_help_contract.js"', source)
        self.assertIn("applyWidgetLabelFallbacks(node)", source)

    def test_public_help_claim_is_native_and_not_canvas_specific(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        docs = (ROOT / "web" / "docs" / "TranslateGemma.md").read_text(
            encoding="utf-8"
        )

        self.assertIn("native ComfyUI Node Help", readme)
        self.assertNotIn("node UI `?` help modal", readme)
        self.assertIn("Node Library", docs)

    def test_node_schema_contract_is_unchanged_by_frontend_work(self):
        workflow_paths = sorted((ROOT / "example_workflows").glob("*.json"))
        self.assertTrue(workflow_paths)

        for workflow_path in workflow_paths:
            workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
            self.assertEqual(workflow.get("version"), 0.4)

    def test_package_script_requires_contract_and_live_host_lanes(self):
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["test"],
            "npm run test:contract && npm run test:host",
        )
        self.assertEqual(
            package["scripts"]["test:contract"],
            "node --test tests/frontend/*.test.mjs",
        )
        self.assertEqual(package["scripts"]["test:host"], "playwright test")

    def test_live_harness_is_loopback_only_and_never_runs_reference(self):
        harness_sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                ROOT / "playwright.config.mjs",
                ROOT / "tests" / "frontend" / "host_url.mjs",
                ROOT / "tests" / "e2e" / "tg059_help_parity.spec.mjs",
            )
        )

        self.assertIn("LOOPBACK_NAMES", harness_sources)
        self.assertIn("COMFYUI_TEST_URL", harness_sources)
        self.assertIn("COMFYUI_TEST_FRONTEND_VERSION", harness_sources)
        self.assertNotIn("REFERENCE/ComfyUI", harness_sources)
        self.assertNotIn("child_process", harness_sources)
        self.assertNotIn("exec(", harness_sources)
        self.assertIn("forbiddenWrites", harness_sources)
        self.assertIn("restoreSettings", harness_sources)

    def test_generated_browser_evidence_is_ignored(self):
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

        self.assertIn("output/playwright/", gitignore)


if __name__ == "__main__":
    unittest.main()
