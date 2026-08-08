import contextlib
import importlib.util
import io
import sys
import types
import unittest
from contextlib import redirect_stdout
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

    @staticmethod
    def get_device_capability(_device_index=None):
        return (8, 0)

    @staticmethod
    def empty_cache():
        return None

    @staticmethod
    def ipc_collect():
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
            base, raw_index = device_type.split(":", 1)
            return _Device(base, int(raw_index))
        return _Device(device_type, index)


class _SequenceLoader:
    def __init__(self, *outcomes):
        self._outcomes = list(outcomes)
        self.calls = []

    def from_pretrained(self, source, **kwargs):
        self.calls.append((source, dict(kwargs)))
        if not self._outcomes:
            raise AssertionError("unexpected from_pretrained call")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _Model:
    def __init__(self):
        self.to_calls = []
        self.eval_called = False

    def to(self, device):
        self.to_calls.append(device)
        return self

    def eval(self):
        self.eval_called = True
        return self


class PartialDeviceHelperTests(unittest.TestCase):
    def setUp(self):
        self._saved_modules = {
            name: sys.modules.get(name)
            for name in (
                "torch",
                "transformers",
                "folder_paths",
                "comfy",
                "comfy.model_management",
            )
        }
        self._loaded_module_names = []
        self._install_base_stubs(cuda_available=True, cuda_count=2)

    def tearDown(self):
        for name in self._loaded_module_names:
            sys.modules.pop(name, None)
        for name, module in self._saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def _install_base_stubs(self, cuda_available=True, cuda_count=2):
        transformers_stub = types.ModuleType("transformers")
        transformers_stub.AutoTokenizer = object()
        transformers_stub.AutoProcessor = object()
        transformers_stub.AutoModelForImageTextToText = object()
        transformers_stub.__version__ = "test"

        folder_paths_stub = types.ModuleType("folder_paths")
        folder_paths_stub.models_dir = str(Path.cwd() / ".tmp" / "tg057-models")

        sys.modules["torch"] = _TorchStub(cuda_available, cuda_count)
        sys.modules["transformers"] = transformers_stub
        sys.modules["folder_paths"] = folder_paths_stub
        sys.modules.pop("comfy", None)
        sys.modules.pop("comfy.model_management", None)

    def _install_comfy_model_management(self, **helpers):
        comfy_pkg = types.ModuleType("comfy")
        model_management = types.ModuleType("comfy.model_management")
        for name, helper in helpers.items():
            setattr(model_management, name, helper)
        comfy_pkg.model_management = model_management
        sys.modules["comfy"] = comfy_pkg
        sys.modules["comfy.model_management"] = model_management
        return model_management

    def _install_partial_host(self):
        return self._install_comfy_model_management(
            get_torch_device=lambda: _Device("cuda", 0)
        )

    def _load_module(self):
        module_name = f"_tg057_model_loader_{len(self._loaded_module_names)}"
        self._loaded_module_names.append(module_name)
        path = Path.cwd() / "utils" / "model_loader.py"
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def test_desktop_partial_helper_honors_explicit_cpu(self):
        self._install_partial_host()
        module = self._load_module()

        selection = module.resolve_device_override("cpu")

        self.assertEqual(selection["requested"], "cpu")
        self.assertEqual(selection["effective"], "cpu")
        self.assertEqual(selection["device"], "cpu")
        self.assertEqual(selection["cache_key"], "cpu:cpu")
        self.assertIsNone(selection["warning"])

    def test_desktop_partial_helper_honors_valid_gpu_index(self):
        self._install_partial_host()
        module = self._load_module()

        selection = module.resolve_device_override("gpu:1")

        self.assertEqual(selection["requested"], "gpu:1")
        self.assertEqual(selection["effective"], "gpu:1")
        self.assertEqual(selection["device"], "cuda:1")
        self.assertEqual(selection["cache_key"], "gpu:1:cuda:1")
        self.assertIsNone(selection["warning"])

    def test_desktop_partial_helper_keeps_host_default_and_fallback_options(self):
        self._install_partial_host()
        module = self._load_module()

        self.assertEqual(module.get_device_options(), ("default", "cpu", "gpu:0", "gpu:1"))
        selection = module.resolve_device_override("default")
        self.assertEqual(selection["effective"], "default")
        self.assertEqual(selection["device"], "cuda:0")
        self.assertEqual(selection["source"], "comfy")

    def test_modern_resolver_keeps_precedence_over_torch_fallback(self):
        self._install_comfy_model_management(
            get_gpu_device_options=lambda: ["default", "cpu", "gpu:0", "gpu:1"],
            get_torch_device=lambda: _Device("cuda", 0),
            resolve_gpu_device_option=lambda option: (
                _Device("cuda", 7) if option == "gpu:1" else None
            ),
        )
        module = self._load_module()

        selection = module.resolve_device_override("gpu:1")

        self.assertEqual(selection["device"], "cuda:7")
        self.assertEqual(selection["effective"], "gpu:1")
        self.assertEqual(selection["source"], "comfy")

    def test_failing_optional_helpers_fall_back_per_capability(self):
        def fail():
            raise RuntimeError("private helper failure marker")

        def fail_resolver(_option):
            raise RuntimeError("private resolver failure marker")

        self._install_comfy_model_management(
            get_gpu_device_options=fail,
            get_torch_device=fail,
            resolve_gpu_device_option=fail_resolver,
        )
        module = self._load_module()

        self.assertEqual(module.resolve_device_override("cpu")["device"], "cpu")
        gpu_selection = module.resolve_device_override("gpu:1")
        self.assertEqual(gpu_selection["device"], "cuda:1")
        self.assertIsNone(gpu_selection["warning"])
        self.assertEqual(module.resolve_device_override("default")["device"], "cuda")

    def test_cpu_only_runtime_rejects_gpu_without_accelerator_side_effect(self):
        self._install_base_stubs(cuda_available=False, cuda_count=0)
        self._install_partial_host()
        module = self._load_module()

        selection = module.resolve_device_override("gpu:0")

        self.assertEqual(selection["effective"], "default")
        self.assertEqual(selection["device"], "cuda:0")
        self.assertIn("unavailable", selection["warning"].lower())
        self.assertNotIn("gpu:0", selection["warning"])

    def test_invalid_overrides_are_redacted_and_never_selected(self):
        self._install_partial_host()
        module = self._load_module()
        invalid_values = (
            "gpu:-1",
            "gpu:not-a-number",
            "gpu:99",
            "gpu:１２",
            f"gpu:{'9' * 5000}",
            "xpu:private-marker",
        )

        for invalid_value in invalid_values:
            with self.subTest(invalid_value=invalid_value):
                selection = module.resolve_device_override(invalid_value)
                self.assertEqual(selection["effective"], "default")
                self.assertEqual(selection["device"], "cuda:0")
                self.assertIsNotNone(selection["warning"])
                self.assertNotIn(invalid_value, selection["warning"])
                self.assertNotIn("private-marker", selection["warning"])

    def test_invalid_override_payload_is_redacted_from_load_logs(self):
        self._install_partial_host()
        module = self._load_module()
        processor = object()
        model = _Model()
        module.AutoProcessor = _SequenceLoader(processor)
        module.AutoModelForImageTextToText = _SequenceLoader(model)
        module.get_model_path = lambda _repo_id: str(Path.cwd() / ".tmp" / "tg057-cache")
        module.ensure_model_downloaded = lambda _repo_id, _cache_dir, _revision: None
        module._is_snapshot_complete = lambda _cache_dir, debug=False: (False, [])
        module.get_torch_dtype = lambda _device: "float16"
        module._check_bnb_availability = lambda _quantization, _device: None
        module.get_cuda_device_context = lambda _device: contextlib.nullcontext()
        private_marker = "xpu:private-device-payload"
        output = io.StringIO()

        with redirect_stdout(output):
            module.load_model("4B", quantization="none", device_override=private_marker)

        self.assertNotIn(private_marker, output.getvalue())
        self.assertNotIn("private-device-payload", output.getvalue())
        self.assertIn("device_override=unsupported", output.getvalue())

    def test_cuda_context_uses_modern_helper_or_local_fallback(self):
        events = []

        @contextlib.contextmanager
        def modern_context(device):
            events.append(("enter", str(device)))
            try:
                yield
            finally:
                events.append(("exit", str(device)))

        self._install_comfy_model_management(cuda_device_context=modern_context)
        module = self._load_module()
        with module.get_cuda_device_context("cuda:1"):
            events.append(("body", "cuda:1"))
        self.assertEqual(
            events,
            [("enter", "cuda:1"), ("body", "cuda:1"), ("exit", "cuda:1")],
        )

        sys.modules.pop("comfy", None)
        sys.modules.pop("comfy.model_management", None)
        fallback_module = self._load_module()
        self.assertEqual(sys.modules["torch"].cuda.current_device(), 0)
        with fallback_module.get_cuda_device_context("cuda:1"):
            self.assertEqual(sys.modules["torch"].cuda.current_device(), 1)
        self.assertEqual(sys.modules["torch"].cuda.current_device(), 0)

    def test_load_model_consumes_one_resolved_gpu_contract(self):
        self._install_partial_host()
        module = self._load_module()
        processor = object()
        model = _Model()
        processor_loader = _SequenceLoader(processor)
        model_loader = _SequenceLoader(model)
        module.AutoProcessor = processor_loader
        module.AutoModelForImageTextToText = model_loader
        module.get_model_path = lambda _repo_id: str(Path.cwd() / ".tmp" / "tg057-cache")
        module.ensure_model_downloaded = lambda _repo_id, _cache_dir, _revision: None
        module._is_snapshot_complete = lambda _cache_dir, debug=False: (False, [])

        dtype_devices = []
        bnb_devices = []
        context_devices = []
        module.get_torch_dtype = lambda device: dtype_devices.append(device) or "float16"
        module._check_bnb_availability = (
            lambda _quantization, device: bnb_devices.append(device)
        )

        @contextlib.contextmanager
        def capture_context(device):
            context_devices.append(device)
            yield

        module.get_cuda_device_context = capture_context

        with redirect_stdout(io.StringIO()):
            loaded_model, loaded_processor = module.load_model(
                "4B", quantization="none", device_override="gpu:1"
            )

        self.assertIs(loaded_model, model)
        self.assertIs(loaded_processor, processor)
        self.assertEqual(dtype_devices, ["cuda:1"])
        self.assertEqual(bnb_devices, ["cuda:1"])
        self.assertEqual(context_devices, ["cuda:1"])
        self.assertEqual(model_loader.calls[0][1]["device_map"], {"": "cuda:1"})
        self.assertEqual(module._current_model_key[3], "gpu:1:cuda:1")
        self.assertEqual(model.to_calls, [])
        self.assertTrue(model.eval_called)

    def test_load_model_cpu_contract_controls_tensor_placement(self):
        self._install_partial_host()
        module = self._load_module()
        processor = object()
        model = _Model()
        module.AutoProcessor = _SequenceLoader(processor)
        model_loader = _SequenceLoader(model)
        module.AutoModelForImageTextToText = model_loader
        module.get_model_path = lambda _repo_id: str(Path.cwd() / ".tmp" / "tg057-cache")
        module.ensure_model_downloaded = lambda _repo_id, _cache_dir, _revision: None
        module._is_snapshot_complete = lambda _cache_dir, debug=False: (False, [])
        dtype_devices = []
        bnb_devices = []
        context_devices = []
        module.get_torch_dtype = lambda device: dtype_devices.append(device) or "float32"
        module._check_bnb_availability = (
            lambda _quantization, device: bnb_devices.append(device)
        )
        module.get_cuda_device_context = (
            lambda device: context_devices.append(device) or contextlib.nullcontext()
        )

        with redirect_stdout(io.StringIO()):
            loaded_model, loaded_processor = module.load_model(
                "4B", quantization="none", device_override="cpu"
            )

        self.assertIs(loaded_model, model)
        self.assertIs(loaded_processor, processor)
        self.assertEqual(dtype_devices, ["cpu"])
        self.assertEqual(bnb_devices, ["cpu"])
        self.assertEqual(context_devices, ["cpu"])
        self.assertIsNone(model_loader.calls[0][1]["device_map"])
        self.assertEqual(module._current_model_key[3], "cpu:cpu")
        self.assertEqual(model.to_calls, ["cpu"])

    def test_failed_load_cleans_up_without_re_resolving_device(self):
        self._install_partial_host()
        module = self._load_module()
        real_resolver = module.resolve_device_override
        resolve_calls = []
        module.resolve_device_override = (
            lambda option: resolve_calls.append(option) or real_resolver(option)
        )
        module.AutoProcessor = _SequenceLoader(OSError("fixture load failure"))
        module.AutoModelForImageTextToText = _SequenceLoader(_Model())
        module.get_model_path = lambda _repo_id: str(Path.cwd() / ".tmp" / "tg057-cache")
        module.ensure_model_downloaded = lambda _repo_id, _cache_dir, _revision: None
        module._is_snapshot_complete = lambda _cache_dir, debug=False: (False, [])
        module.get_torch_dtype = lambda _device: "float16"
        module._check_bnb_availability = lambda _quantization, _device: None
        cleanup_calls = []
        module.cleanup_torch_memory = lambda: cleanup_calls.append("cleanup")

        with redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                module.load_model("4B", quantization="none", device_override="gpu:1")

        self.assertEqual(resolve_calls, ["gpu:1"])
        self.assertEqual(cleanup_calls, ["cleanup"])


if __name__ == "__main__":
    unittest.main()
