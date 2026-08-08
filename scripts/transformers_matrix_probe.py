#!/usr/bin/env python3
"""Offline real-package worker for the TG-058 Transformers matrix.

The worker emits exactly one JSON object and never exposes raw exception text.
API sources:
- https://huggingface.co/docs/transformers/v5.0.0/en/model_doc/auto
- https://huggingface.co/docs/transformers/v5.0.0/en/main_classes/processors
- https://huggingface.co/docs/transformers/v5.0.0/en/main_classes/model
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import types
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable


RESULT_SCHEMA_VERSION = 1
REQUIRED_SURFACES = (
    "imports",
    "auto_processor",
    "auto_image_text_model",
    "chat_template",
    "load_kwargs",
    "safetensors_local_load",
    "generation",
    "bnb_config_without_bnb",
    "project_model_loader",
)
_EXPECTED_LANES = {
    "py312-tf4.57.3": ("3.12", "4.57.3"),
    "py312-tf5.0.0": ("3.12", "5.0.0"),
    "py312-tf5.14.1": ("3.12", "5.14.1"),
    "py313-tf4.57.3": ("3.13", "4.57.3"),
    "py313-tf5.0.0": ("3.13", "5.0.0"),
    "py313-tf5.14.1": ("3.13", "5.14.1"),
}


class WorkerValidationError(ValueError):
    """Raised before any generated matrix path is mutated."""


def failure_category(error: BaseException) -> str:
    if isinstance(error, ImportError):
        return "IMPORT_FAILURE"
    if isinstance(error, AssertionError):
        return "CONTRACT_FAILURE"
    if isinstance(error, OSError):
        return "IO_FAILURE"
    return "PROBE_FAILURE"


def _register(register: Callable[..., Any], *args: Any) -> None:
    try:
        register(*args, exist_ok=True)
    except TypeError:
        register(*args)


def _probe_imports(context: dict[str, Any]) -> None:
    import torch
    import transformers
    from transformers import (
        AutoConfig,
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
        GenerationMixin,
        PretrainedConfig,
        PreTrainedModel,
        PreTrainedTokenizerFast,
        ProcessorMixin,
    )

    assert transformers.__version__ == context["expected_transformers"]
    assert torch.__version__.startswith(context["expected_torch"])
    assert f"{sys.version_info.major}.{sys.version_info.minor}" == context["expected_python"]
    context.update(
        {
            "torch": torch,
            "transformers_module": transformers,
            "AutoConfig": AutoConfig,
            "AutoModelForImageTextToText": AutoModelForImageTextToText,
            "AutoProcessor": AutoProcessor,
            "BitsAndBytesConfig": BitsAndBytesConfig,
            "GenerationMixin": GenerationMixin,
            "PretrainedConfig": PretrainedConfig,
            "PreTrainedModel": PreTrainedModel,
            "PreTrainedTokenizerFast": PreTrainedTokenizerFast,
            "ProcessorMixin": ProcessorMixin,
        }
    )


def _build_registered_types(context: dict[str, Any]) -> None:
    if "MatrixConfig" in context:
        return
    torch = context["torch"]
    PretrainedConfig = context["PretrainedConfig"]
    PreTrainedModel = context["PreTrainedModel"]
    GenerationMixin = context["GenerationMixin"]
    ProcessorMixin = context["ProcessorMixin"]

    class MatrixConfig(PretrainedConfig):
        model_type = "tg058_matrix"

        def __init__(
            self,
            vocab_size: int = 8,
            hidden_size: int = 4,
            **kwargs: Any,
        ) -> None:
            kwargs.setdefault("bos_token_id", 1)
            kwargs.setdefault("eos_token_id", 2)
            kwargs.setdefault("pad_token_id", 0)
            kwargs.setdefault("is_decoder", True)
            kwargs.setdefault("tie_word_embeddings", False)
            super().__init__(**kwargs)
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size

    class MatrixProcessor(ProcessorMixin):
        attributes = ["tokenizer"]
        tokenizer_class = "AutoTokenizer"

        def __init__(self, tokenizer=None, chat_template: str | None = None, **kwargs: Any) -> None:
            super().__init__(tokenizer)
            self.chat_template = chat_template

    class MatrixModel(PreTrainedModel, GenerationMixin):
        config_class = MatrixConfig
        main_input_name = "input_ids"

        def __init__(self, config: MatrixConfig) -> None:
            super().__init__(config)
            self.embedding = torch.nn.Embedding(config.vocab_size, config.hidden_size)
            self.projection = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            if hasattr(self, "post_init"):
                self.post_init()

        def get_input_embeddings(self):
            return self.embedding

        def set_input_embeddings(self, value):
            self.embedding = value

        def get_output_embeddings(self):
            return self.projection

        def set_output_embeddings(self, value):
            self.projection = value

        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {"input_ids": input_ids}

        def forward(self, input_ids=None, **kwargs):
            from transformers.modeling_outputs import CausalLMOutput

            hidden = self.embedding(input_ids)
            return CausalLMOutput(logits=self.projection(hidden))

    _register(context["AutoConfig"].register, MatrixConfig.model_type, MatrixConfig)
    _register(context["AutoProcessor"].register, MatrixConfig, MatrixProcessor)
    _register(context["AutoModelForImageTextToText"].register, MatrixConfig, MatrixModel)
    context.update(
        {
            "MatrixConfig": MatrixConfig,
            "MatrixProcessor": MatrixProcessor,
            "MatrixModel": MatrixModel,
        }
    )


def _probe_auto_processor(context: dict[str, Any]) -> None:
    _build_registered_types(context)
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    processor_dir = context["work_dir"] / "processor"
    processor_dir.mkdir(parents=True, exist_ok=True)
    config = context["MatrixConfig"]()
    config.save_pretrained(processor_dir)
    backend = Tokenizer(
        WordLevel(
            vocab={
                "<pad>": 0,
                "<bos>": 1,
                "<eos>": 2,
                "<unk>": 3,
                "matrix-fixture": 4,
            },
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = context["PreTrainedTokenizerFast"](
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    context["MatrixProcessor"](tokenizer=tokenizer).save_pretrained(processor_dir)
    processor = context["AutoProcessor"].from_pretrained(
        processor_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    assert isinstance(processor, context["MatrixProcessor"])
    context["processor"] = processor


def _probe_auto_model(context: dict[str, Any]) -> None:
    _build_registered_types(context)
    model = context["AutoModelForImageTextToText"].from_config(
        context["MatrixConfig"]()
    )
    assert isinstance(model, context["MatrixModel"])
    assert model.can_generate()
    context["model"] = model


def _probe_chat_template(context: dict[str, Any]) -> None:
    processor = context["processor"]
    template = (
        "{% for message in messages %}{{ message['role'] }}:"
        "{% for item in message['content'] %}"
        "{% if item['type'] == 'text' %}{{ item['text'] }}{% endif %}"
        "{% endfor %};{% endfor %}"
        "{% if add_generation_prompt %}assistant:{% endif %}"
    )
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "matrix-fixture"}]}
    ]
    rendered = processor.apply_chat_template(
        conversation,
        chat_template=template,
        add_generation_prompt=True,
        tokenize=False,
    )
    assert rendered == "user:matrix-fixture;assistant:"


def _probe_load_kwargs(context: dict[str, Any]) -> None:
    torch = context["torch"]
    model = context["model"]
    model_dir = context["work_dir"] / "model"
    model.save_pretrained(model_dir, safe_serialization=True)
    context["model_dir"] = model_dir
    context["load_kwargs"] = {
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
        "device_map": None,
        "torch_dtype": torch.float32,
    }


def _probe_local_load(context: dict[str, Any]) -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        loaded = context["AutoModelForImageTextToText"].from_pretrained(
            context["model_dir"],
            **context["load_kwargs"],
        )
    warning_text = " ".join(str(item.message).casefold() for item in captured)
    assert "torch_dtype" not in warning_text or "deprecated" not in warning_text
    assert isinstance(loaded, context["MatrixModel"])
    assert next(loaded.parameters()).dtype == context["torch"].float32
    context["loaded_model"] = loaded


def _probe_generation(context: dict[str, Any]) -> None:
    torch = context["torch"]
    model = context["loaded_model"]
    input_ids = torch.tensor([[1]], dtype=torch.long)
    generated = model.generate(
        input_ids=input_ids,
        max_new_tokens=1,
        do_sample=False,
        use_cache=False,
    )
    assert tuple(generated.shape) == (1, 2)


def _probe_bnb_config(context: dict[str, Any]) -> None:
    assert importlib.util.find_spec("bitsandbytes") is None
    config = context["BitsAndBytesConfig"](load_in_8bit=True)
    assert config.load_in_8bit is True


def _probe_project_model_loader(context: dict[str, Any]) -> None:
    repo_root = context["repo_root"]
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = str(context["work_dir"] / "models")
    previous = sys.modules.get("folder_paths")
    sys.modules["folder_paths"] = folder_paths
    module_name = f"_tg058_model_loader_{context['lane'].replace('.', '_')}"
    try:
        spec = importlib.util.spec_from_file_location(
            module_name, repo_root / "utils" / "model_loader.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        assert module.AutoProcessor is not None
        assert module.AutoModelForImageTextToText is not None
        assert module._build_model_device_map("cpu", "cpu") is None
    finally:
        sys.modules.pop(module_name, None)
        if previous is None:
            sys.modules.pop("folder_paths", None)
        else:
            sys.modules["folder_paths"] = previous


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--expected-python", required=True)
    parser.add_argument("--expected-transformers", required=True)
    parser.add_argument("--expected-torch", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--work-dir", required=True)
    return parser.parse_args(argv)


def resolve_work_dir(repo_root: Path, lane: str, requested: str | os.PathLike[str]) -> Path:
    actual_repo_root = Path(__file__).resolve().parents[1]
    repo_root = Path(repo_root).resolve()
    if repo_root != actual_repo_root:
        raise WorkerValidationError("REPOSITORY_ROOT_MISMATCH")
    if lane not in _EXPECTED_LANES:
        raise WorkerValidationError("LANE_INVALID")
    expected = (repo_root / ".tmp" / "tg058" / "work" / lane).resolve()
    resolved = Path(requested).resolve()
    if resolved != expected:
        raise WorkerValidationError("WORK_DIRECTORY_MISMATCH")
    return resolved


def _validate_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    expected_versions = _EXPECTED_LANES.get(args.lane)
    if expected_versions != (args.expected_python, args.expected_transformers):
        raise WorkerValidationError("LANE_VERSION_MISMATCH")
    if args.expected_torch != "2.10.0":
        raise WorkerValidationError("TORCH_VERSION_MISMATCH")
    repo_root = Path(args.repo_root).resolve()
    return repo_root, resolve_work_dir(repo_root, args.lane, args.work_dir)


def _input_failure_payload(args: argparse.Namespace) -> dict[str, Any]:
    surfaces = OrderedDict((surface, "SKIP") for surface in REQUIRED_SURFACES)
    surfaces[REQUIRED_SURFACES[0]] = "FAIL"
    expected = _EXPECTED_LANES.get(args.lane)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "lane": args.lane if expected is not None else "invalid-lane",
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "transformers": expected[1] if expected is not None else "invalid",
        "torch": "2.10.0",
        "command": "python scripts/transformers_matrix_probe.py",
        "result": "FAIL",
        "surfaces": surfaces,
        "error_category": "WORKER_INPUT_INVALID",
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        repo_root, work_dir = _validate_inputs(args)
    except WorkerValidationError:
        print(json.dumps(_input_failure_payload(args), separators=(",", ":")))
        return 1
    if work_dir.exists():
        # The lane directory is generated and validated above; remove only its synthetic probe data.
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    context: dict[str, Any] = {
        "lane": args.lane,
        "expected_python": args.expected_python,
        "expected_transformers": args.expected_transformers,
        "expected_torch": args.expected_torch,
        "repo_root": repo_root,
        "work_dir": work_dir,
    }
    probes: OrderedDict[str, Callable[[dict[str, Any]], None]] = OrderedDict(
        (
            ("imports", _probe_imports),
            ("auto_processor", _probe_auto_processor),
            ("auto_image_text_model", _probe_auto_model),
            ("chat_template", _probe_chat_template),
            ("load_kwargs", _probe_load_kwargs),
            ("safetensors_local_load", _probe_local_load),
            ("generation", _probe_generation),
            ("bnb_config_without_bnb", _probe_bnb_config),
            ("project_model_loader", _probe_project_model_loader),
        )
    )
    surfaces = OrderedDict((surface, "SKIP") for surface in REQUIRED_SURFACES)
    error_category = None
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        for surface, probe in probes.items():
            try:
                probe(context)
                surfaces[surface] = "PASS"
            except BaseException as error:
                surfaces[surface] = "FAIL"
                error_category = f"{surface.upper()}_{failure_category(error)}"
                break
    passed = error_category is None and all(value == "PASS" for value in surfaces.values())
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "lane": args.lane,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "transformers": getattr(context.get("transformers_module"), "__version__", args.expected_transformers),
        "torch": getattr(context.get("torch"), "__version__", args.expected_torch),
        "command": "python scripts/transformers_matrix_probe.py",
        "result": "PASS" if passed else "FAIL",
        "surfaces": surfaces,
        "error_category": error_category,
    }
    print(json.dumps(payload, separators=(",", ":")))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
