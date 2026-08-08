import ast
import importlib.util
import pathlib
import re
import tomllib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
INIT = ROOT / "__init__.py"


class PackageMetadataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        cls.project = cls.metadata["project"]

    def test_runtime_version_has_one_authoritative_project_source(self):
        package_version = self.project["version"]
        init_source = INIT.read_text(encoding="utf-8")
        tree = ast.parse(init_source)
        version_literals = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "__version__" for target in targets):
                value = node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    version_literals.append(value.value)

        self.assertEqual([], version_literals, "runtime must not maintain a second version literal")
        self.assertIn("from ._version import __version__", init_source)
        self.assertRegex(package_version, r"^\d+\.\d+\.\d+$")

    def test_python_range_matches_accepted_matrix(self):
        self.assertEqual(">=3.12,<3.14", self.project.get("requires-python"))

    def test_license_uses_current_spdx_metadata(self):
        self.assertEqual("MIT", self.project.get("license"))
        self.assertEqual(["LICENSE"], self.project.get("license-files"))

    def test_build_backend_and_explicit_root_package_mapping_exist(self):
        build_system = self.metadata.get("build-system", {})
        self.assertEqual("setuptools.build_meta", build_system.get("build-backend"))
        setuptools_config = self.metadata.get("tool", {}).get("setuptools", {})
        self.assertEqual(
            {"comfyui_translategemma": "."},
            setuptools_config.get("package-dir"),
        )
        self.assertIn("comfyui_translategemma", setuptools_config.get("packages", []))

    def test_host_owned_dependencies_and_internal_metadata_are_absent(self):
        dependencies = [item.lower() for item in self.project.get("dependencies", [])]
        self.assertFalse(any(re.match(r"^(torch|torchvision|torchaudio)(?:\W|$)", item) for item in dependencies))
        self.assertFalse(any(item.startswith("comfyui-frontend-package") for item in dependencies))
        public_metadata = PYPROJECT.read_text(encoding="utf-8")
        self.assertNotRegex(public_metadata, r"TG-\d{3}")
        self.assertNotIn(".planning/", public_metadata)
        self.assertNotIn("reference/docs/", public_metadata)

    def test_archive_verifier_is_present_and_deny_lists_internal_paths(self):
        verifier = ROOT / "scripts" / "verify_package_artifacts.py"
        self.assertTrue(verifier.is_file())
        source = verifier.read_text(encoding="utf-8")
        for marker in (".planning/", "reference/", "REFERENCE/", ".sessions/", "tests/"):
            self.assertIn(marker, source)

    def test_archive_verifier_rejects_traversal_and_absolute_members(self):
        verifier = ROOT / "scripts" / "verify_package_artifacts.py"
        spec = importlib.util.spec_from_file_location("tg_package_verifier", verifier)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for unsafe_name in ("../escape", "folder/../../escape", "/absolute", "C:/absolute"):
            with self.subTest(unsafe_name=unsafe_name):
                with self.assertRaises(ValueError):
                    module._validate_names([unsafe_name])


if __name__ == "__main__":
    unittest.main()
