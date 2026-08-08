import ast
import importlib.util
import pathlib
import sys
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_LOADER = ROOT / "utils" / "model_loader.py"
README = ROOT / "README.md"


class _CudaStub:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def current_device():
        return 0

    @staticmethod
    def get_device_capability(device_index=None):
        return (8, 0)


class _TorchStub(types.ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.cuda = _CudaStub()
        self.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
        self.dtype = object
        self.bfloat16 = "bfloat16"
        self.float16 = "float16"
        self.float32 = "float32"


class _RejectBitsAndBytes:
    marker = "C:/private/token-bearing-native-loader-detail"

    def __init__(self, exception_type=ImportError):
        self.exception_type = exception_type

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "bitsandbytes" or fullname.startswith("bitsandbytes."):
            raise self.exception_type(self.marker)
        return None


class BitsAndBytesPolicyTests(unittest.TestCase):
    def setUp(self):
        self.original_modules = {
            name: sys.modules.get(name)
            for name in ("torch", "transformers", "folder_paths", "bitsandbytes")
        }
        sys.modules["torch"] = _TorchStub()
        transformers = types.ModuleType("transformers")
        transformers.__version__ = "test"
        transformers.AutoTokenizer = object()
        sys.modules["transformers"] = transformers
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.models_dir = str(ROOT / ".tmp" / "models")
        sys.modules["folder_paths"] = folder_paths
        sys.modules.pop("bitsandbytes", None)
        self.loaded_names = []

    def tearDown(self):
        for name in self.loaded_names:
            sys.modules.pop(name, None)
        for name, module in self.original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def _load_model_loader(self):
        module_name = f"_tg061_model_loader_{len(self.loaded_names)}"
        spec = importlib.util.spec_from_file_location(module_name, MODEL_LOADER)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        self.loaded_names.append(module_name)
        spec.loader.exec_module(module)
        return module

    def test_optional_dependency_declarations_are_exact_and_base_stays_clean(self):
        optional_requirement = (ROOT / "requirements-quantization.txt").read_text(encoding="utf-8")
        self.assertEqual("bitsandbytes>=0.49.2", optional_requirement.splitlines()[-1])

        import tomllib

        with (ROOT / "pyproject.toml").open("rb") as metadata_file:
            metadata = tomllib.load(metadata_file)
        self.assertEqual(
            ["bitsandbytes>=0.49.2"],
            metadata["project"]["optional-dependencies"]["quantization"],
        )
        base_requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
        self.assertNotIn("bitsandbytes", base_requirements)

    def test_public_guidance_uses_official_sources_and_bounds_project_support(self):
        readme = README.read_text(encoding="utf-8")
        self.assertNotIn("jllllll/bitsandbytes-windows-webui", readme)
        self.assertIn("https://huggingface.co/docs/bitsandbytes/installation", readme)
        self.assertIn("bitsandbytes>=0.49.2", readme)
        for deferred_backend in ("ROCm", "XPU", "HPU", "CPU", "MPS"):
            self.assertIn(deferred_backend, readme)
        self.assertIn("project support", readme.lower())
        self.assertIn("best-effort", readme.lower())
        self.assertIn("does not replace ComfyUI's PyTorch", readme)

    def test_missing_dependency_error_is_fixed_redacted_and_suppresses_context(self):
        finder = _RejectBitsAndBytes()
        sys.meta_path.insert(0, finder)
        try:
            module = self._load_model_loader()
            with self.assertRaises(RuntimeError) as caught:
                module._check_bnb_availability("bnb-8bit", "cuda:0")
        finally:
            sys.meta_path.remove(finder)

        message = str(caught.exception)
        self.assertNotIn(finder.marker, message)
        self.assertNotIn("jllllll/bitsandbytes-windows-webui", message)
        self.assertIn("bitsandbytes>=0.49.2", message)
        self.assertIn("huggingface.co/docs/bitsandbytes/installation", message)
        self.assertIn("quantization=none", message)
        self.assertIsNone(caught.exception.__cause__)
        self.assertTrue(caught.exception.__suppress_context__)

    def test_native_loader_failure_uses_the_same_redacted_recovery_path(self):
        finder = _RejectBitsAndBytes(OSError)
        sys.meta_path.insert(0, finder)
        try:
            module = self._load_model_loader()
            with self.assertRaises(RuntimeError) as caught:
                module._check_bnb_availability("bnb-4bit", "cuda")
        finally:
            sys.meta_path.remove(finder)

        self.assertNotIn(finder.marker, str(caught.exception))
        self.assertIn("requirements-quantization.txt", str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)

    def test_non_cuda_policy_is_actionable_and_does_not_probe_bitsandbytes(self):
        finder = _RejectBitsAndBytes(AssertionError)
        sys.meta_path.insert(0, finder)
        try:
            module = self._load_model_loader()
            with self.assertRaises(RuntimeError) as caught:
                module._check_bnb_availability("bnb-8bit", "cpu")
        finally:
            sys.meta_path.remove(finder)
        self.assertIn("project's CUDA device path", str(caught.exception))
        self.assertIn("quantization=none", str(caught.exception))

    def test_missing_transformers_config_is_redacted_and_suppresses_context(self):
        sys.modules["bitsandbytes"] = types.ModuleType("bitsandbytes")
        module = self._load_model_loader()
        with self.assertRaises(RuntimeError) as caught:
            module._check_bnb_availability("bnb-8bit", "cuda")
        self.assertNotIn("Import error", str(caught.exception))
        self.assertIn("quantization=none", str(caught.exception))
        self.assertIn("huggingface.co/docs/transformers", str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)

    def test_quantization_schema_values_remain_stable(self):
        module = self._load_model_loader()
        self.assertEqual(("none", "bnb-8bit", "bnb-4bit"), module.QUANTIZATION_OPTIONS)
        node_source = (ROOT / "nodes" / "translate_node.py").read_text(encoding="utf-8")
        self.assertIn('"quantization": (list(QUANTIZATION_OPTIONS)', node_source)
        self.assertIn('"default": "none"', node_source)

    def test_runtime_has_no_package_installer_or_subprocess_path(self):
        tree = ast.parse(MODEL_LOADER.read_text(encoding="utf-8"))
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
        self.assertTrue({"subprocess", "pip", "uv"}.isdisjoint(imported_roots))

    def test_official_wheel_verifier_exists_and_never_imports_or_extracts_binaries(self):
        verifier = ROOT / "scripts" / "verify_bitsandbytes_wheel.py"
        self.assertTrue(verifier.is_file())
        source = verifier.read_text(encoding="utf-8")
        self.assertIn("zipfile", source)
        self.assertNotIn("extractall", source)
        self.assertNotIn("import_module", source)
        self.assertIn("0.49.2", source)

        spec = importlib.util.spec_from_file_location("tg_bnb_wheel_verifier", verifier)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for unsafe_name in ("../escape", "/absolute", "C:/absolute"):
            with self.subTest(unsafe_name=unsafe_name):
                with self.assertRaises(ValueError):
                    module._validate_member_names([unsafe_name])


if __name__ == "__main__":
    unittest.main()
