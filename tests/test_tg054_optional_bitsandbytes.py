import importlib.util
import sys
import types
import unittest
from pathlib import Path


class _TorchCudaStub:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def current_device():
        return 0

    @staticmethod
    def get_device_capability(device_index=None):
        return (8, 0)

    @staticmethod
    def empty_cache():
        return None

    @staticmethod
    def ipc_collect():
        return None


class _TorchStub(types.ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.cuda = _TorchCudaStub()
        self.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        )
        self.dtype = object
        self.bfloat16 = "bfloat16"
        self.float16 = "float16"
        self.float32 = "float32"


class _BitsAndBytesFinder:
    def __init__(self, exc_type):
        self.exc_type = exc_type
        self.seen = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "bitsandbytes" or fullname.startswith("bitsandbytes."):
            self.seen = True
            raise self.exc_type("blocked bitsandbytes import for TG-054 test")
        return None


class OptionalBitsAndBytesTests(unittest.TestCase):
    def setUp(self):
        self._orig_modules = {
            name: sys.modules.get(name)
            for name in ("torch", "transformers", "folder_paths", "bitsandbytes")
        }
        self._module_names = []

        transformers_stub = types.ModuleType("transformers")
        transformers_stub.AutoTokenizer = object()
        transformers_stub.__version__ = "test"

        folder_paths_stub = types.ModuleType("folder_paths")
        folder_paths_stub.models_dir = str(Path.cwd() / ".tmp" / "models")

        sys.modules["torch"] = _TorchStub()
        sys.modules["transformers"] = transformers_stub
        sys.modules["folder_paths"] = folder_paths_stub
        sys.modules.pop("bitsandbytes", None)

    def tearDown(self):
        for name in self._module_names:
            sys.modules.pop(name, None)
        for name, module in self._orig_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def _load_model_loader(self):
        module_name = f"_tg054_model_loader_{len(self._module_names)}"
        self._module_names.append(module_name)
        path = Path.cwd() / "utils" / "model_loader.py"
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def test_quantization_none_does_not_import_bitsandbytes(self):
        finder = _BitsAndBytesFinder(AssertionError)
        sys.meta_path.insert(0, finder)
        try:
            module = self._load_model_loader()
            module._check_bnb_availability("none", "cpu")
            self.assertIsNone(module._build_quantization_config("none"))
            self.assertFalse(finder.seen)
        finally:
            sys.meta_path.remove(finder)

    def test_bnb_mode_reports_missing_optional_dependency(self):
        finder = _BitsAndBytesFinder(ImportError)
        sys.meta_path.insert(0, finder)
        try:
            module = self._load_model_loader()
            with self.assertRaises(RuntimeError) as ctx:
                module._check_bnb_availability("bnb-8bit", "cuda")
        finally:
            sys.meta_path.remove(finder)

        message = str(ctx.exception)
        self.assertIn("requires bitsandbytes to be installed", message)
        self.assertIn("requirements-quantization.txt", message)
        self.assertIn("quantization=none", message)
        self.assertTrue(finder.seen)


if __name__ == "__main__":
    unittest.main()
