import ast
import contextlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODEL_LOADER = ROOT / "utils" / "model_loader.py"
TRANSLATE_NODE = ROOT / "nodes" / "translate_node.py"
REPO_4B = "google/translategemma-4b-it"
REPO_12B = "google/translategemma-12b-it"
REPO_27B = "google/translategemma-27b-it"


class _CudaStub:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def empty_cache():
        return None

    @staticmethod
    def ipc_collect():
        return None


class _TorchStub(types.ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.cuda = _CudaStub()
        self.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        )
        self.dtype = object
        self.bfloat16 = "bfloat16"
        self.float16 = "float16"
        self.float32 = "float32"

    @staticmethod
    def device(value, index=None):
        return value if index is None else f"{value}:{index}"


class _SequenceLoader:
    def __init__(self, *outcomes):
        self._outcomes = list(outcomes)
        self.calls = []

    def from_pretrained(self, source, **kwargs):
        self.calls.append((source, dict(kwargs)))
        if not self._outcomes:
            raise AssertionError("unexpected from_pretrained call")
        return self._outcomes.pop(0)


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


class ModelPathTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory(prefix="translategemma-model-paths-")
        self.addCleanup(self._temp.cleanup)
        self.temp_root = Path(self._temp.name)
        self.instance_models = self.temp_root / "instance" / "models"
        self.registered_paths = []
        self._loaded_module_names = []
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

        transformers_stub = types.ModuleType("transformers")
        transformers_stub.AutoTokenizer = object()
        transformers_stub.AutoProcessor = object()
        transformers_stub.AutoModelForImageTextToText = object()
        transformers_stub.__version__ = "test"

        self.folder_paths_stub = types.ModuleType("folder_paths")
        self.folder_paths_stub.models_dir = str(self.instance_models)
        self.folder_paths_stub.get_folder_paths = self._get_folder_paths

        sys.modules["torch"] = _TorchStub()
        sys.modules["transformers"] = transformers_stub
        sys.modules["folder_paths"] = self.folder_paths_stub
        sys.modules.pop("huggingface_hub", None)
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

    def _get_folder_paths(self, category):
        if category != "LLM":
            raise KeyError(category)
        return list(self.registered_paths)

    def _load_module(self):
        module_name = f"_model_path_test_loader_{len(self._loaded_module_names)}"
        self._loaded_module_names.append(module_name)
        spec = importlib.util.spec_from_file_location(module_name, MODEL_LOADER)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _load_node_cache_context(get_model_path, redact_path, *, verbose=False):
        tree = ast.parse(TRANSLATE_NODE.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_get_model_cache_context"
        )
        namespace = {
            "get_model_path": get_model_path,
            "is_verbose_debug": lambda: verbose,
            "redact_path": redact_path,
        }
        exec(
            compile(
                ast.Module(body=[function], type_ignores=[]),
                str(TRANSLATE_NODE),
                "exec",
            ),
            namespace,
        )
        return namespace["_get_model_cache_context"]

    @staticmethod
    def _repo_name(repo_id):
        return repo_id.rsplit("/", 1)[-1]

    def _snapshot_dir(self, llm_root, repo_id):
        return Path(llm_root) / "TranslateGemma" / self._repo_name(repo_id)

    def _write_complete_snapshot(self, llm_root, repo_id, *, weight="model.safetensors"):
        snapshot = self._snapshot_dir(llm_root, repo_id)
        snapshot.mkdir(parents=True, exist_ok=True)
        for filename in (
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "processor_config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            weight,
        ):
            (snapshot / filename).write_text("{}", encoding="utf-8")
        return snapshot

    def _write_metadata_only_snapshot(self, llm_root, repo_id):
        snapshot = self._snapshot_dir(llm_root, repo_id)
        snapshot.mkdir(parents=True, exist_ok=True)
        for filename in (
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "processor_config.json",
            "preprocessor_config.json",
            "tokenizer.json",
        ):
            (snapshot / filename).write_text("{}", encoding="utf-8")
        return snapshot

    def test_complete_requested_snapshot_in_second_registered_root_is_reused_read_only(self):
        empty_default = self.temp_root / "default-llm"
        shared = self.temp_root / "shared-llm"
        self.registered_paths[:] = [str(empty_default), str(shared)]
        expected = self._write_complete_snapshot(shared, REPO_4B)
        module = self._load_module()

        selected = Path(module.get_model_path(REPO_4B))

        self.assertEqual(selected, expected)
        self.assertFalse(empty_default.exists())
        self.assertFalse(self.instance_models.exists())

    def test_other_model_size_in_first_root_does_not_hide_requested_snapshot(self):
        for requested_repo in (REPO_12B, REPO_27B):
            with self.subTest(requested_repo=requested_repo):
                first = self.temp_root / self._repo_name(requested_repo) / "first-llm"
                second = self.temp_root / self._repo_name(requested_repo) / "second-llm"
                self.registered_paths[:] = [str(first), str(second)]
                self._write_complete_snapshot(first, REPO_4B)
                expected = self._write_complete_snapshot(second, requested_repo)
                module = self._load_module()

                self.assertEqual(Path(module.get_model_path(requested_repo)), expected)

    def test_first_complete_requested_snapshot_wins_in_registered_order(self):
        first = self.temp_root / "first-llm"
        second = self.temp_root / "second-llm"
        self.registered_paths[:] = [str(first), str(second)]
        expected = self._write_complete_snapshot(first, REPO_4B)
        self._write_complete_snapshot(second, REPO_4B)
        before = list(self.registered_paths)
        module = self._load_module()

        self.assertEqual(Path(module.get_model_path(REPO_4B)), expected)
        self.assertEqual(self.registered_paths, before)

    def test_incomplete_or_distractor_first_root_does_not_hide_complete_later_root(self):
        first = self.temp_root / "first-llm"
        second = self.temp_root / "second-llm"
        self.registered_paths[:] = [str(first), str(second)]
        partial = self._write_metadata_only_snapshot(first, REPO_4B)
        (partial / "unrelated.gguf").write_text("not supported", encoding="utf-8")
        cache_dir = partial / ".cache"
        cache_dir.mkdir()
        (cache_dir / "model.safetensors").write_text("partial", encoding="utf-8")
        expected = self._write_complete_snapshot(second, REPO_4B)
        module = self._load_module()

        self.assertEqual(Path(module.get_model_path(REPO_4B)), expected)

    def test_malformed_or_unsafe_weight_indexes_are_incomplete_not_exceptions(self):
        bad_indexes = (
            [],
            {},
            {"weight_map": {}},
            {"weight_map": {"layer": "../outside.safetensors"}},
            {"weight_map": {"layer": str(self.temp_root / "outside.safetensors")}},
            {"weight_map": {"layer": ""}},
        )
        for case_number, index_data in enumerate(bad_indexes):
            with self.subTest(case=case_number):
                first = self.temp_root / f"bad-{case_number}"
                second = self.temp_root / f"good-{case_number}"
                self.registered_paths[:] = [str(first), str(second)]
                partial = self._write_metadata_only_snapshot(first, REPO_4B)
                (partial / "model.safetensors.index.json").write_text(
                    json.dumps(index_data), encoding="utf-8"
                )
                expected = self._write_complete_snapshot(second, REPO_4B)
                module = self._load_module()

                self.assertEqual(Path(module.get_model_path(REPO_4B)), expected)

    def test_valid_sharded_snapshot_is_complete_and_missing_or_directory_shards_are_not(self):
        module = self._load_module()
        for index_name, shard_names in (
            (
                "model.safetensors.index.json",
                ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"),
            ),
            (
                "pytorch_model.bin.index.json",
                ("pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin"),
            ),
        ):
            with self.subTest(index=index_name):
                root = self.temp_root / index_name.replace(".", "-")
                snapshot = self._write_metadata_only_snapshot(root, REPO_4B)
                weight_map = {
                    "layer.one": shard_names[0],
                    "layer.two": shard_names[1],
                }
                (snapshot / index_name).write_text(
                    json.dumps({"weight_map": weight_map}), encoding="utf-8"
                )
                for shard_name in shard_names:
                    (snapshot / shard_name).write_text("weights", encoding="utf-8")

                self.assertEqual(module._is_snapshot_complete(str(snapshot)), (True, []))

                (snapshot / shard_names[1]).unlink()
                complete, missing = module._is_snapshot_complete(str(snapshot))
                self.assertFalse(complete)
                self.assertIn(f"shard: {shard_names[1]}", missing)

                (snapshot / shard_names[1]).mkdir()
                complete, missing = module._is_snapshot_complete(str(snapshot))
                self.assertFalse(complete)
                self.assertIn(f"shard: {shard_names[1]}", missing)

    def test_no_complete_snapshot_chooses_first_registered_destination_without_creating_it(self):
        default_root = self.temp_root / "default-llm"
        secondary = self.temp_root / "secondary-llm"
        self.registered_paths[:] = [str(default_root), str(secondary)]
        self._write_metadata_only_snapshot(secondary, REPO_4B)
        module = self._load_module()

        selected = Path(module.get_model_path(REPO_4B))

        self.assertEqual(selected, self._snapshot_dir(default_root, REPO_4B))
        self.assertFalse(default_root.exists())

    def test_missing_registered_category_and_api_preserve_canonical_and_legacy_fallbacks(self):
        module = self._load_module()
        self.folder_paths_stub.get_folder_paths = mock.Mock(side_effect=KeyError("LLM"))
        canonical = self.instance_models / "LLM" / "TranslateGemma" / self._repo_name(REPO_4B)
        self.assertEqual(Path(module.get_model_path(REPO_4B)), canonical)
        self.assertFalse(canonical.exists())

        legacy_base = self.instance_models / "translate_gemma"
        legacy_base.mkdir(parents=True)
        module = self._load_module()
        del self.folder_paths_stub.get_folder_paths
        legacy = legacy_base / self._repo_name(REPO_4B)
        self.assertEqual(Path(module.get_model_path(REPO_4B)), legacy)
        self.assertFalse(legacy.exists())

    def test_unexpected_registry_failure_is_safe_and_not_silently_ignored(self):
        private_marker = "private-registry-path-marker"
        self.folder_paths_stub.get_folder_paths = mock.Mock(
            side_effect=OSError(private_marker)
        )
        module = self._load_module()

        with self.assertRaises(RuntimeError) as raised:
            module.get_model_path(REPO_4B)

        self.assertNotIn(private_marker, str(raised.exception))
        self.assertIn("registered LLM", str(raised.exception))
        self.assertFalse(self.instance_models.exists())

    @unittest.skipUnless(os.name == "nt", "Windows path deduplication contract")
    def test_windows_case_variant_registered_roots_are_scanned_once(self):
        shared = self.temp_root / "SharedRoot"
        self.registered_paths[:] = [str(shared), str(shared).upper()]
        module = self._load_module()
        inspected = []
        real_check = module._is_snapshot_complete

        def recording_check(path, debug=False):
            inspected.append(os.path.normcase(os.path.abspath(path)))
            return real_check(path, debug=debug)

        module._is_snapshot_complete = recording_check
        selected = Path(module.get_model_path(REPO_4B))
        shared_snapshot = self._snapshot_dir(shared, REPO_4B)

        self.assertEqual(selected, shared_snapshot)
        normalized_shared = os.path.normcase(os.path.abspath(shared_snapshot))
        self.assertEqual(inspected.count(normalized_shared), 1)

    def test_download_stage_creates_only_selected_destination(self):
        default_root = self.temp_root / "default-llm"
        secondary = self.temp_root / "secondary-llm"
        self.registered_paths[:] = [str(default_root), str(secondary)]
        self._write_metadata_only_snapshot(secondary, REPO_4B)
        module = self._load_module()
        target = Path(module.get_model_path(REPO_4B))
        checks = 0

        def completeness(path, debug=False):
            nonlocal checks
            checks += 1
            return (checks > 1, [] if checks > 1 else ["config.json"])

        hub_stub = types.ModuleType("huggingface_hub")

        def snapshot_download(**kwargs):
            self.assertEqual(Path(kwargs["local_dir"]), target)
            self.assertTrue(target.is_dir())

        hub_stub.snapshot_download = snapshot_download
        sys.modules["huggingface_hub"] = hub_stub
        module._is_snapshot_complete = completeness

        self.assertTrue(module.ensure_model_downloaded(REPO_4B, str(target)))
        self.assertTrue(target.is_dir())
        self.assertTrue(self._snapshot_dir(secondary, REPO_4B).is_dir())

    def test_denied_destination_creation_does_not_attempt_download_or_fallback(self):
        default_root = self.temp_root / "default-llm"
        secondary = self.temp_root / "secondary-llm"
        self.registered_paths[:] = [str(default_root), str(secondary)]
        module = self._load_module()
        target = Path(module.get_model_path(REPO_4B))
        hub_calls = []
        hub_stub = types.ModuleType("huggingface_hub")
        hub_stub.snapshot_download = lambda **kwargs: hub_calls.append(kwargs)
        sys.modules["huggingface_hub"] = hub_stub

        with mock.patch.object(module.os, "makedirs", side_effect=PermissionError("denied")):
            with self.assertRaises(RuntimeError) as raised:
                module.ensure_model_downloaded(REPO_4B, str(target))

        self.assertIn("download destination", str(raised.exception).lower())
        self.assertEqual(hub_calls, [])
        self.assertFalse(secondary.exists())

    def test_load_model_uses_same_complete_shared_snapshot_for_processor_and_model(self):
        default_root = self.temp_root / "default-llm"
        shared = self.temp_root / "shared-llm"
        self.registered_paths[:] = [str(default_root), str(shared)]
        expected = self._write_complete_snapshot(shared, REPO_4B)
        module = self._load_module()
        processor = object()
        model = _Model()
        processor_loader = _SequenceLoader(processor)
        model_loader = _SequenceLoader(model)
        module.AutoProcessor = processor_loader
        module.AutoModelForImageTextToText = model_loader
        module.resolve_device_override = lambda _option: {
            "device": "cpu",
            "cache_key": "cpu:cpu",
            "effective": "cpu",
            "requested": "cpu",
            "warning": None,
        }
        module.get_torch_dtype = lambda _device: "float32"
        module._check_bnb_availability = lambda _quantization, _device: None
        module.get_cuda_device_context = lambda _device: contextlib.nullcontext()

        with mock.patch.object(
            module.os,
            "makedirs",
            side_effect=AssertionError("complete shared snapshot must not trigger mkdir"),
        ):
            loaded_model, loaded_processor = module.load_model("4B", device_override="cpu")

        self.assertIs(loaded_model, model)
        self.assertIs(loaded_processor, processor)
        self.assertEqual(processor_loader.calls[0][0], str(expected))
        self.assertEqual(model_loader.calls[0][0], str(expected))
        for _source, kwargs in processor_loader.calls + model_loader.calls:
            self.assertEqual(kwargs["cache_dir"], str(expected))
            self.assertTrue(kwargs["local_files_only"])
        self.assertFalse(default_root.exists())

    def test_node_cache_context_is_redacted_and_diagnostic_failure_safe(self):
        private_path = str(self.temp_root / "private-share" / self._repo_name(REPO_4B))
        normal_resolution_calls = []
        cache_context = self._load_node_cache_context(
            lambda repo_id: normal_resolution_calls.append(repo_id) or private_path,
            lambda path: os.path.basename(path),
        )
        self.assertEqual(cache_context(REPO_4B), self._repo_name(REPO_4B))
        self.assertNotIn(str(self.temp_root), cache_context(REPO_4B))
        self.assertEqual(normal_resolution_calls, [])

        verbose_context = self._load_node_cache_context(
            lambda _repo_id: private_path,
            lambda path: path,
            verbose=True,
        )
        self.assertEqual(verbose_context(REPO_4B), private_path)

        private_marker = "private-secondary-diagnostic-marker"

        def fail_resolution(_repo_id):
            raise OSError(private_marker)

        safe_context = self._load_node_cache_context(
            fail_resolution,
            lambda path: path,
            verbose=True,
        )
        self.assertEqual(safe_context(REPO_4B), "unavailable")
        self.assertNotIn(private_marker, safe_context(REPO_4B))
        self.assertEqual(safe_context("unknown"), "unknown")


if __name__ == "__main__":
    unittest.main()
