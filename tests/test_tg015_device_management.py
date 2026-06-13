import importlib.util
import sys
import types
import unittest
from pathlib import Path


class _Device:
    def __init__(self, device_type, index=None):
        self.type = device_type
        self.index = index

    def __str__(self):
        if self.index is None:
            return self.type
        return f"{self.type}:{self.index}"


class _TorchCudaStub:
    def __init__(self, available=True, count=2):
        self._available = available
        self._count = count
        self._current = 0

    def is_available(self):
        return self._available

    def device_count(self):
        return self._count

    def current_device(self):
        return self._current

    def set_device(self, device):
        if isinstance(device, _Device):
            self._current = int(device.index or 0)
        else:
            self._current = int(device)

    def get_device_capability(self, device_index=None):
        return (8, 0)

    def empty_cache(self):
        return None

    def ipc_collect(self):
        return None


class _TorchStub(types.ModuleType):
    def __init__(self, cuda_available=True, cuda_count=2):
        super().__init__("torch")
        self.cuda = _TorchCudaStub(cuda_available, cuda_count)
        self.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        )
        self.dtype = object
        self.bfloat16 = "bfloat16"
        self.float16 = "float16"
        self.float32 = "float32"

    @staticmethod
    def device(device_type, index=None):
        if isinstance(device_type, int):
            return _Device("cuda", device_type)
        if isinstance(device_type, str) and ":" in device_type and index is None:
            base, idx = device_type.split(":", 1)
            return _Device(base, int(idx))
        return _Device(device_type, index)


class DeviceManagementTests(unittest.TestCase):
    def setUp(self):
        self._orig_modules = {
            name: sys.modules.get(name)
            for name in ("torch", "transformers", "folder_paths", "comfy", "comfy.model_management")
        }
        self._module_names = []
        self._install_base_stubs(cuda_available=True, cuda_count=2)

    def tearDown(self):
        for name in self._module_names:
            sys.modules.pop(name, None)
        for name, module in self._orig_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def _install_base_stubs(self, cuda_available=True, cuda_count=2):
        transformers_stub = types.ModuleType("transformers")
        transformers_stub.AutoTokenizer = object()
        transformers_stub.__version__ = "test"

        folder_paths_stub = types.ModuleType("folder_paths")
        folder_paths_stub.models_dir = str(Path.cwd() / ".tmp" / "models")

        sys.modules["torch"] = _TorchStub(cuda_available, cuda_count)
        sys.modules["transformers"] = transformers_stub
        sys.modules["folder_paths"] = folder_paths_stub
        sys.modules.pop("comfy", None)
        sys.modules.pop("comfy.model_management", None)

    def _install_comfy_model_management(self):
        comfy_pkg = types.ModuleType("comfy")
        model_management = types.ModuleType("comfy.model_management")

        model_management.get_gpu_device_options = lambda: ["default", "cpu", "gpu:0", "gpu:1"]
        model_management.get_torch_device = lambda: _Device("cuda", 0)

        def resolve(option):
            return {
                "cpu": _Device("cpu"),
                "gpu:0": _Device("cuda", 0),
                "gpu:1": _Device("cuda", 1),
            }.get(option)

        model_management.resolve_gpu_device_option = resolve

        class _Context:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        model_management.cuda_device_context = lambda device: _Context()
        comfy_pkg.model_management = model_management
        sys.modules["comfy"] = comfy_pkg
        sys.modules["comfy.model_management"] = model_management

    def _load_model_loader(self):
        module_name = f"_tg015_model_loader_{len(self._module_names)}"
        self._module_names.append(module_name)
        path = Path.cwd() / "utils" / "model_loader.py"
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def test_comfy_helper_device_options_and_resolution(self):
        self._install_comfy_model_management()
        module = self._load_model_loader()

        self.assertEqual(module.get_device_options(), ("default", "cpu", "gpu:0", "gpu:1"))

        gpu_selection = module.resolve_device_override("gpu:1")
        self.assertEqual(gpu_selection["device"], "cuda:1")
        self.assertEqual(gpu_selection["effective"], "gpu:1")
        self.assertEqual(gpu_selection["cache_key"], "gpu:1:cuda:1")
        self.assertEqual(gpu_selection["source"], "comfy")

        cpu_selection = module.resolve_device_override("cpu")
        self.assertEqual(cpu_selection["device"], "cpu")
        self.assertEqual(cpu_selection["cache_key"], "cpu:cpu")

    def test_invalid_comfy_override_falls_back_to_default_with_warning(self):
        self._install_comfy_model_management()
        module = self._load_model_loader()

        selection = module.resolve_device_override("gpu:99")

        self.assertEqual(selection["device"], "cuda:0")
        self.assertEqual(selection["effective"], "default")
        self.assertEqual(selection["cache_key"], "default:cuda:0")
        self.assertIn("Unsupported device override", selection["warning"])

    def test_legacy_fallback_device_resolution(self):
        module = self._load_model_loader()

        self.assertEqual(module.get_device_options(), ("default", "cpu", "gpu:0", "gpu:1"))
        self.assertEqual(module.resolve_device_override("gpu:1")["device"], "cuda:1")
        self.assertEqual(module.resolve_device_override("cpu")["device"], "cpu")

        invalid = module.resolve_device_override("xpu:0")
        self.assertEqual(invalid["device"], "cuda")
        self.assertEqual(invalid["effective"], "default")
        self.assertIn("Unsupported device override", invalid["warning"])

    def test_cpu_only_fallback_has_no_gpu_options(self):
        self._install_base_stubs(cuda_available=False, cuda_count=0)
        module = self._load_model_loader()

        self.assertEqual(module.get_device_options(), ("default", "cpu"))
        self.assertEqual(module.get_device(), "cpu")

    def test_model_device_map_contract(self):
        module = self._load_model_loader()

        self.assertEqual(module._build_model_device_map("cuda:0", "default"), "auto")
        self.assertEqual(module._build_model_device_map("cuda:1", "gpu:1"), {"": "cuda:1"})
        self.assertIsNone(module._build_model_device_map("cpu", "cpu"))


if __name__ == "__main__":
    unittest.main()
