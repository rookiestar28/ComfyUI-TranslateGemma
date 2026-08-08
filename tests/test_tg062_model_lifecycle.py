import ast
import contextlib
import gc
import importlib.util
import io
import sys
import threading
import time
import types
import unittest
import weakref
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_LOADER = ROOT / "utils" / "model_loader.py"
TRANSLATE_NODE = ROOT / "nodes" / "translate_node.py"


class _CudaStub:
    def __init__(self, available=False):
        self.available = available
        self.empty_cache_calls = 0
        self.ipc_collect_calls = 0

    def is_available(self):
        return self.available

    def empty_cache(self):
        self.empty_cache_calls += 1

    def ipc_collect(self):
        self.ipc_collect_calls += 1


class _TorchStub(types.ModuleType):
    def __init__(self, cuda_available=False):
        super().__init__("torch")
        self.cuda = _CudaStub(cuda_available)
        self.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        )
        self.dtype = object
        self.bfloat16 = "bfloat16"
        self.float16 = "float16"
        self.float32 = "float32"

    @staticmethod
    def device(value):
        return value


class _TrackedProcessor:
    pass


class _TrackedModel:
    def __init__(self, label="model"):
        self.label = label
        self.to_calls = []
        self.eval_calls = 0

    def to(self, device):
        self.to_calls.append(device)
        return self

    def eval(self):
        self.eval_calls += 1
        return self


class _FactoryLoader:
    def __init__(self, factory, activity=None, fail=False, captured_ref=None):
        self.factory = factory
        self.activity = activity
        self.fail = fail
        self.captured_ref = captured_ref
        self.calls = 0

    def from_pretrained(self, _source, **_kwargs):
        self.calls += 1
        if self.activity is not None:
            with self.activity["lock"]:
                self.activity["active"] += 1
                self.activity["maximum"] = max(
                    self.activity["maximum"], self.activity["active"]
                )
            time.sleep(0.05)
        try:
            if self.fail:
                raise OSError("private/path/token fixture must be redacted")
            value = self.factory()
            if self.captured_ref is not None:
                self.captured_ref.append(weakref.ref(value))
            return value
        finally:
            if self.activity is not None:
                with self.activity["lock"]:
                    self.activity["active"] -= 1


class ModelLifecycleTests(unittest.TestCase):
    def setUp(self):
        self._saved_modules = {
            name: sys.modules.get(name)
            for name in ("torch", "transformers", "folder_paths", "comfy", "comfy.model_management")
        }
        self._loaded_names = []
        self._install_stubs()

    def tearDown(self):
        for name in self._loaded_names:
            sys.modules.pop(name, None)
        for name, module in self._saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def _install_stubs(self, cuda_available=False):
        sys.modules["torch"] = _TorchStub(cuda_available)
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = object()
        transformers.AutoProcessor = object()
        transformers.AutoModelForImageTextToText = object()
        transformers.__version__ = "test"
        sys.modules["transformers"] = transformers
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.models_dir = str(ROOT / ".tmp" / "tg062-models")
        sys.modules["folder_paths"] = folder_paths
        sys.modules.pop("comfy", None)
        sys.modules.pop("comfy.model_management", None)

    def _load_module(self):
        name = f"_tg062_model_loader_{len(self._loaded_names)}"
        spec = importlib.util.spec_from_file_location(name, MODEL_LOADER)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        self._loaded_names.append(name)
        spec.loader.exec_module(module)
        return module

    def _configure_loaders(self, module, processor_loader, model_loader):
        module.AutoProcessor = processor_loader
        module.AutoModelForImageTextToText = model_loader
        module.resolve_device_override = lambda option: {
            "device": "cpu",
            "cache_key": "cpu:cpu",
            "effective": "cpu",
            "requested": option,
            "warning": None,
        }
        module.get_model_path = lambda _repo_id: str(ROOT / ".tmp" / "tg062-cache")
        module.ensure_model_downloaded = lambda *_args: None
        module._is_snapshot_complete = lambda *_args, **_kwargs: (False, [])
        module.get_torch_dtype = lambda _device: "float32"
        module._check_bnb_availability = lambda *_args: None
        module.get_cuda_device_context = lambda _device: contextlib.nullcontext()

    def test_same_key_concurrent_misses_have_one_owner_and_one_load(self):
        module = self._load_module()
        activity = {"lock": threading.Lock(), "active": 0, "maximum": 0}
        processors = _FactoryLoader(_TrackedProcessor, activity=activity)
        models = _FactoryLoader(_TrackedModel, activity=activity)
        self._configure_loaders(module, processors, models)
        start = threading.Barrier(3)
        results = []

        def worker():
            start.wait()
            with redirect_stdout(io.StringIO()):
                results.append(module.load_model("4B", device_override="cpu"))

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        start.wait()
        for thread in threads:
            thread.join(timeout=3)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(activity["maximum"], 1)
        self.assertEqual(processors.calls, 1)
        self.assertEqual(models.calls, 1)
        self.assertIs(results[0][0], results[1][0])
        self.assertIs(results[0][1], results[1][1])

    def test_first_load_cached_repeat_and_switch_release_previous_pair(self):
        module = self._load_module()
        model_refs = []
        processor_refs = []
        processors = _FactoryLoader(_TrackedProcessor, captured_ref=processor_refs)
        models = _FactoryLoader(_TrackedModel, captured_ref=model_refs)
        self._configure_loaders(module, processors, models)

        with redirect_stdout(io.StringIO()):
            first = module.load_model("4B", device_override="cpu")
            repeated = module.load_model("4B", device_override="cpu")
        self.assertIs(first[0], repeated[0])
        self.assertIs(first[1], repeated[1])
        self.assertEqual((processors.calls, models.calls), (1, 1))

        del first, repeated
        with redirect_stdout(io.StringIO()):
            replacement = module.load_model("12B", device_override="cpu")
        gc.collect()

        self.assertIsNone(model_refs[0]())
        self.assertIsNone(processor_refs[0]())
        self.assertIs(replacement[0], model_refs[1]())
        self.assertIs(replacement[1], processor_refs[1]())
        self.assertEqual((processors.calls, models.calls), (2, 2))
        self.assertEqual(module.get_current_model_size(), "12B")

    def test_different_key_requests_never_overlap_and_last_owner_is_coherent(self):
        module = self._load_module()
        activity = {"lock": threading.Lock(), "active": 0, "maximum": 0}
        processors = _FactoryLoader(_TrackedProcessor, activity=activity)
        models = _FactoryLoader(_TrackedModel, activity=activity)
        self._configure_loaders(module, processors, models)
        start = threading.Barrier(3)
        errors = []

        def worker(size):
            try:
                start.wait()
                with redirect_stdout(io.StringIO()):
                    module.load_model(size, device_override="cpu")
            except Exception as error:  # pragma: no cover - surfaced below
                errors.append(error)

        threads = [threading.Thread(target=worker, args=(size,)) for size in ("4B", "12B")]
        for thread in threads:
            thread.start()
        start.wait()
        for thread in threads:
            thread.join(timeout=3)

        self.assertEqual(errors, [])
        self.assertEqual(activity["maximum"], 1)
        self.assertEqual(processors.calls, 2)
        self.assertEqual(models.calls, 2)
        self.assertIn(module.get_current_model_size(), {"4B", "12B"})
        self.assertTrue(module.is_model_loaded())

    def test_explicit_unload_releases_model_and_processor_references(self):
        module = self._load_module()
        model = _TrackedModel()
        processor = _TrackedProcessor()
        model_ref = weakref.ref(model)
        processor_ref = weakref.ref(processor)
        module._current_model = (model, processor)
        module._current_model_key = ("4B", "none", None, "cpu:cpu")
        del model, processor

        with redirect_stdout(io.StringIO()):
            module.unload_current_model()
        gc.collect()

        self.assertIsNone(model_ref())
        self.assertIsNone(processor_ref())
        self.assertFalse(module.is_model_loaded())
        self.assertIsNone(module.get_current_model_size())
        self.assertIsNone(module.get_current_quantization())

    def test_failed_switch_drops_partial_processor_before_cleanup(self):
        module = self._load_module()
        old_model = _TrackedModel("old")
        old_processor = _TrackedProcessor()
        old_refs = (weakref.ref(old_model), weakref.ref(old_processor))
        module._current_model = (old_model, old_processor)
        module._current_model_key = ("4B", "none", None, "cpu:cpu")
        del old_model, old_processor

        partial_refs = []
        processors = _FactoryLoader(_TrackedProcessor, captured_ref=partial_refs)
        models = _FactoryLoader(_TrackedModel, fail=True)
        self._configure_loaders(module, processors, models)
        cleanup_observations = []

        def observe_cleanup():
            gc.collect()
            cleanup_observations.append(
                partial_refs[-1]() is None if partial_refs else all(ref() is None for ref in old_refs)
            )

        module.cleanup_torch_memory = observe_cleanup
        with redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "TranslateGemma"):
                module.load_model("12B", device_override="cpu")

        self.assertTrue(all(ref() is None for ref in old_refs))
        self.assertEqual(cleanup_observations[-1], True)
        self.assertIsNone(module._current_model)
        self.assertIsNone(module._current_model_key)

    def test_cleanup_covers_cpu_and_available_cuda_without_hardware_claim(self):
        cpu_module = self._load_module()
        cpu_module.cleanup_torch_memory()
        cpu_cuda = sys.modules["torch"].cuda
        self.assertEqual((cpu_cuda.empty_cache_calls, cpu_cuda.ipc_collect_calls), (0, 0))

        self._install_stubs(cuda_available=True)
        cuda_module = self._load_module()
        cuda_module.cleanup_torch_memory()
        cuda_stub = sys.modules["torch"].cuda
        self.assertEqual((cuda_stub.empty_cache_calls, cuda_stub.ipc_collect_calls), (1, 1))

    def test_inference_paths_hold_reentrant_lifecycle_and_evict_on_failure(self):
        tree = ast.parse(TRANSLATE_NODE.read_text(encoding="utf-8"))
        methods = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name in ("_translate_one", "_translate_image"):
            method = methods[name]
            decorators = {
                ast.unparse(decorator) for decorator in method.decorator_list
            }
            self.assertIn("serialized_model_access", decorators)
            unload_calls = [
                call
                for call in ast.walk(method)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "unload_current_model"
            ]
            self.assertGreaterEqual(len(unload_calls), 2)

        module = self._load_module()
        events = []

        @module.serialized_model_access
        def outer():
            events.append("outer")
            return inner()

        @module.serialized_model_access
        def inner():
            events.append("inner")
            return "done"

        self.assertEqual(outer(), "done")
        self.assertEqual(events, ["outer", "inner"])


if __name__ == "__main__":
    unittest.main()
