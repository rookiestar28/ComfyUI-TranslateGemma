import contextlib
import importlib.util
import io
import os
import sys
import traceback
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path


_POLICY_ENV = (
    "TRANSLATEGEMMA_ALLOW_REMOTE_CODE",
    "TRANSLATEGEMMA_REMOTE_CODE_ALLOWLIST",
    "TRANSLATEGEMMA_ALLOW_UNSAFE_WEIGHTS",
    "TRANSLATEGEMMA_REVISION",
    "TRANSLATEGEMMA_DOWNLOAD_RETRIES",
    "HF_ENDPOINT",
    "HUGGINGFACE_HUB_ENDPOINT",
    "HTTP_PROXY",
    "HTTPS_PROXY",
)


class _TorchCudaStub:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def device_count():
        return 0

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

    @staticmethod
    def device(device_type, index=None):
        return types.SimpleNamespace(type=device_type, index=index)


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


class RemoteCodePolicyTests(unittest.TestCase):
    def setUp(self):
        self._saved_env = {name: os.environ.get(name) for name in _POLICY_ENV}
        for name in _POLICY_ENV:
            os.environ.pop(name, None)

        self._saved_modules = {
            name: sys.modules.get(name)
            for name in (
                "torch",
                "transformers",
                "folder_paths",
                "huggingface_hub",
                "comfy",
                "comfy.model_management",
            )
        }
        self._loaded_module_names = []

        transformers_stub = types.ModuleType("transformers")
        transformers_stub.AutoTokenizer = object()
        transformers_stub.AutoProcessor = object()
        transformers_stub.AutoModelForImageTextToText = object()
        transformers_stub.__version__ = "test"

        folder_paths_stub = types.ModuleType("folder_paths")
        folder_paths_stub.models_dir = str(Path.cwd() / ".tmp" / "tg056-models")

        sys.modules["torch"] = _TorchStub()
        sys.modules["transformers"] = transformers_stub
        sys.modules["folder_paths"] = folder_paths_stub
        sys.modules.pop("comfy", None)
        sys.modules.pop("comfy.model_management", None)

    def tearDown(self):
        for name in self._loaded_module_names:
            sys.modules.pop(name, None)
        for name, module in self._saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        for name, value in self._saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def _load_module(self):
        module_name = f"_tg056_model_loader_{len(self._loaded_module_names)}"
        self._loaded_module_names.append(module_name)
        path = Path.cwd() / "utils" / "model_loader.py"
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        module.resolve_device_override = lambda _option: {
            "device": "cpu",
            "cache_key": "cpu:cpu",
            "effective": "cpu",
            "requested": "cpu",
            "warning": None,
        }
        module.get_model_path = lambda _repo_id: str(
            Path.cwd() / ".tmp" / "tg056-cache"
        )
        module.get_torch_dtype = lambda _device: "float32"
        module._check_bnb_availability = lambda _quantization, _device: None
        module._tg056_real_ensure_model_downloaded = module.ensure_model_downloaded
        module.ensure_model_downloaded = lambda _repo_id, _cache_dir, _revision: None
        module._is_snapshot_complete = lambda _cache_dir, debug=False: (False, [])
        module.get_cuda_device_context = lambda _device: contextlib.nullcontext()
        return module

    @staticmethod
    def _remote_code_required_error():
        return ValueError(
            "Loading this model requires custom repository code; set trust_remote_code=True."
        )

    def test_missing_remote_code_opt_in_denies_explicit_requirement(self):
        module = self._load_module()
        processor_loader = _SequenceLoader(self._remote_code_required_error(), object())
        module.AutoProcessor = processor_loader
        module.AutoModelForImageTextToText = _SequenceLoader(_Model())

        with self.assertRaises(RuntimeError):
            module.load_model("4B", device_override="cpu")

        self.assertEqual(len(processor_loader.calls), 1)
        self.assertFalse(processor_loader.calls[0][1]["trust_remote_code"])

    def test_non_remote_processor_errors_never_retry_with_remote_code(self):
        os.environ["TRANSLATEGEMMA_ALLOW_REMOTE_CODE"] = "1"
        cases = (
            OSError("network timeout"),
            OSError("401 unauthorized"),
            OSError("disk full"),
            ImportError("transformers version mismatch"),
            RuntimeError("model configuration is invalid"),
        )
        for load_error in cases:
            with self.subTest(error=str(load_error)):
                module = self._load_module()
                processor_loader = _SequenceLoader(load_error, object())
                module.AutoProcessor = processor_loader
                module.AutoModelForImageTextToText = _SequenceLoader(_Model())

                with self.assertRaises(RuntimeError):
                    module.load_model("4B", device_override="cpu")

                self.assertEqual(len(processor_loader.calls), 1)
                self.assertFalse(processor_loader.calls[0][1]["trust_remote_code"])

    def test_malformed_remote_code_policy_fails_closed(self):
        os.environ["TRANSLATEGEMMA_ALLOW_REMOTE_CODE"] = "sometimes"
        module = self._load_module()
        processor_loader = _SequenceLoader(self._remote_code_required_error(), object())
        module.AutoProcessor = processor_loader
        module.AutoModelForImageTextToText = _SequenceLoader(_Model())

        with self.assertRaises(RuntimeError):
            module.load_model("4B", device_override="cpu")

        self.assertEqual(len(processor_loader.calls), 1)

    def test_explicit_remote_code_requirement_obeys_opt_in_and_allowlist(self):
        os.environ["TRANSLATEGEMMA_ALLOW_REMOTE_CODE"] = "1"
        os.environ["TRANSLATEGEMMA_REMOTE_CODE_ALLOWLIST"] = (
            "google/translategemma-4b-it,example/other"
        )
        module = self._load_module()
        processor = object()
        processor_loader = _SequenceLoader(self._remote_code_required_error(), processor)
        model = _Model()
        module.AutoProcessor = processor_loader
        module.AutoModelForImageTextToText = _SequenceLoader(model)

        loaded_model, loaded_processor = module.load_model(
            "4B", revision="0123456789abcdef", device_override="cpu"
        )

        self.assertIs(loaded_model, model)
        self.assertIs(loaded_processor, processor)
        self.assertEqual(
            [call[1]["trust_remote_code"] for call in processor_loader.calls],
            [False, True],
        )
        self.assertTrue(
            all(call[1]["revision"] == "0123456789abcdef" for call in processor_loader.calls)
        )

    def test_allowlist_mismatch_denies_remote_code_retry(self):
        os.environ["TRANSLATEGEMMA_ALLOW_REMOTE_CODE"] = "1"
        os.environ["TRANSLATEGEMMA_REMOTE_CODE_ALLOWLIST"] = "example/other"
        module = self._load_module()
        processor_loader = _SequenceLoader(self._remote_code_required_error(), object())
        module.AutoProcessor = processor_loader
        module.AutoModelForImageTextToText = _SequenceLoader(_Model())

        with self.assertRaises(RuntimeError):
            module.load_model("4B", device_override="cpu")

        self.assertEqual(len(processor_loader.calls), 1)

    def test_safetensors_failure_does_not_enable_remote_code_or_pickle_by_default(self):
        os.environ["TRANSLATEGEMMA_ALLOW_REMOTE_CODE"] = "1"
        module = self._load_module()
        module.AutoProcessor = _SequenceLoader(object())
        model_loader = _SequenceLoader(
            OSError("No safetensors weights were found for this model"),
            _Model(),
        )
        module.AutoModelForImageTextToText = model_loader

        with self.assertRaises(RuntimeError):
            module.load_model("4B", device_override="cpu")

        self.assertEqual(len(model_loader.calls), 1)
        self.assertFalse(model_loader.calls[0][1]["trust_remote_code"])
        self.assertTrue(model_loader.calls[0][1]["use_safetensors"])

    def test_ordinary_model_error_never_retries_with_remote_code(self):
        os.environ["TRANSLATEGEMMA_ALLOW_REMOTE_CODE"] = "1"
        module = self._load_module()
        module.AutoProcessor = _SequenceLoader(object())
        model_loader = _SequenceLoader(RuntimeError("model configuration is invalid"), _Model())
        module.AutoModelForImageTextToText = model_loader

        with self.assertRaises(RuntimeError):
            module.load_model("4B", device_override="cpu")

        self.assertEqual(len(model_loader.calls), 1)
        self.assertFalse(model_loader.calls[0][1]["trust_remote_code"])

    def test_model_remote_code_retry_preserves_safetensors_requirement(self):
        os.environ["TRANSLATEGEMMA_ALLOW_REMOTE_CODE"] = "1"
        module = self._load_module()
        module.AutoProcessor = _SequenceLoader(object())
        model = _Model()
        model_loader = _SequenceLoader(self._remote_code_required_error(), model)
        module.AutoModelForImageTextToText = model_loader

        loaded_model, _ = module.load_model(
            "4B", revision="0123456789abcdef", device_override="cpu"
        )

        self.assertIs(loaded_model, model)
        self.assertEqual(len(model_loader.calls), 2)
        self.assertTrue(model_loader.calls[1][1]["trust_remote_code"])
        self.assertIn("use_safetensors", model_loader.calls[1][1])
        self.assertTrue(model_loader.calls[1][1]["use_safetensors"])
        self.assertEqual(model_loader.calls[1][1]["revision"], "0123456789abcdef")

    def test_unsafe_weight_opt_in_retries_without_enabling_remote_code(self):
        os.environ["TRANSLATEGEMMA_ALLOW_REMOTE_CODE"] = "0"
        os.environ["TRANSLATEGEMMA_ALLOW_UNSAFE_WEIGHTS"] = "1"
        module = self._load_module()
        processor = object()
        module.AutoProcessor = _SequenceLoader(processor)
        model = _Model()
        model_loader = _SequenceLoader(
            OSError("No safetensors weights were found for this model"),
            model,
        )
        module.AutoModelForImageTextToText = model_loader

        try:
            loaded_model, loaded_processor = module.load_model("4B", device_override="cpu")
        except Exception as exc:  # pragma: no cover - RED path on the pre-fix implementation
            self.fail(f"independent unsafe-weight retry was not available: {type(exc).__name__}")

        self.assertIs(loaded_model, model)
        self.assertIs(loaded_processor, processor)
        self.assertEqual(len(model_loader.calls), 2)
        self.assertFalse(model_loader.calls[1][1]["trust_remote_code"])
        self.assertFalse(model_loader.calls[1][1]["use_safetensors"])

    def test_remote_code_without_revision_emits_pinning_guidance(self):
        os.environ["TRANSLATEGEMMA_ALLOW_REMOTE_CODE"] = "1"
        module = self._load_module()
        module.AutoProcessor = _SequenceLoader(self._remote_code_required_error(), object())
        module.AutoModelForImageTextToText = _SequenceLoader(_Model())
        output = io.StringIO()

        with redirect_stdout(output):
            module.load_model("4B", device_override="cpu")

        normalized = output.getvalue().lower()
        self.assertIn("pin", normalized)
        self.assertIn("revision", normalized)

    def test_revision_is_propagated_but_not_echoed_to_logs(self):
        os.environ["TRANSLATEGEMMA_ALLOW_REMOTE_CODE"] = "1"
        module = self._load_module()
        revision_marker = "private-revision-fixture"
        processor_loader = _SequenceLoader(self._remote_code_required_error(), object())
        module.AutoProcessor = processor_loader
        module.AutoModelForImageTextToText = _SequenceLoader(_Model())
        output = io.StringIO()

        with redirect_stdout(output):
            module.load_model("4B", revision=revision_marker, device_override="cpu")

        self.assertNotIn(revision_marker, output.getvalue())
        self.assertTrue(
            all(call[1]["revision"] == revision_marker for call in processor_loader.calls)
        )

    def test_raw_sensitive_exception_content_is_not_exposed(self):
        os.environ["TRANSLATEGEMMA_ALLOW_REMOTE_CODE"] = "1"
        module = self._load_module()
        sensitive_marker = "hf_private_fixture_token"
        private_cache_path = str(Path.cwd() / ".tmp" / "sensitive-cache")
        processor_loader = _SequenceLoader(
            OSError(f"network failure token={sensitive_marker} path={private_cache_path}"),
            OSError(f"second failure token={sensitive_marker} path={private_cache_path}"),
        )
        module.AutoProcessor = processor_loader
        module.AutoModelForImageTextToText = _SequenceLoader(_Model())
        output = io.StringIO()

        with redirect_stdout(output):
            try:
                module.load_model("4B", device_override="cpu")
            except RuntimeError as raised:
                raised_message = str(raised)
                formatted_traceback = traceback.format_exc()
            else:  # pragma: no cover - assertion guard
                self.fail("expected a safe model-load failure")

        self.assertNotIn(sensitive_marker, raised_message)
        self.assertNotIn(sensitive_marker, formatted_traceback)
        self.assertNotIn(private_cache_path, raised_message)
        self.assertNotIn(private_cache_path, formatted_traceback)
        self.assertNotIn(private_cache_path, output.getvalue())
        self.assertEqual(len(processor_loader.calls), 1)

    def test_download_failure_redacts_raw_error_path_and_endpoint(self):
        os.environ["TRANSLATEGEMMA_DOWNLOAD_RETRIES"] = "1"
        private_endpoint = "https://private.invalid.example"
        os.environ["HF_ENDPOINT"] = private_endpoint
        module = self._load_module()
        sensitive_marker = "hf_private_download_fixture"
        local_dir = str(Path.cwd() / ".tmp" / "private-download-cache")

        hub_stub = types.ModuleType("huggingface_hub")

        def fail_download(**_kwargs):
            raise OSError(
                f"timeout token={sensitive_marker} path={local_dir} endpoint={private_endpoint}"
            )

        hub_stub.snapshot_download = fail_download
        sys.modules["huggingface_hub"] = hub_stub
        module._is_snapshot_complete = lambda _path, debug=False: (False, ["config.json"])
        output = io.StringIO()

        with redirect_stdout(output):
            try:
                module._tg056_real_ensure_model_downloaded(
                    "google/translategemma-4b-it", local_dir, debug=True
                )
            except RuntimeError as raised:
                raised_message = str(raised)
                formatted_traceback = traceback.format_exc()
            else:  # pragma: no cover - assertion guard
                self.fail("expected a safe download failure")

        combined = output.getvalue() + raised_message + formatted_traceback
        self.assertNotIn(sensitive_marker, combined)
        self.assertNotIn(local_dir, combined)
        self.assertNotIn(private_endpoint, combined)
        self.assertIn("network", combined.lower())

    def test_download_success_logs_do_not_expose_local_cache_path(self):
        os.environ["TRANSLATEGEMMA_DOWNLOAD_RETRIES"] = "1"
        module = self._load_module()
        local_dir = str(Path.cwd() / ".tmp" / "private-download-cache")
        completeness = iter(((False, ["config.json"]), (True, [])))
        module._is_snapshot_complete = lambda _path, debug=False: next(completeness)

        hub_stub = types.ModuleType("huggingface_hub")
        hub_stub.snapshot_download = lambda **_kwargs: None
        sys.modules["huggingface_hub"] = hub_stub
        output = io.StringIO()

        with redirect_stdout(output):
            result = module._tg056_real_ensure_model_downloaded(
                "google/translategemma-4b-it", local_dir, debug=True
            )

        self.assertTrue(result)
        self.assertNotIn(local_dir, output.getvalue())


if __name__ == "__main__":
    unittest.main()
