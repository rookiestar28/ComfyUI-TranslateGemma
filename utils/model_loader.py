"""
Model loader for TranslateGemma with single-model cache and robust memory management.

TG-004: Rework in-memory cache to avoid VRAM/RAM blowups
TG-005: Structured disk layout
TG-006: trust_remote_code risk reduction
TG-016: Explicit snapshot_download with Windows-friendly settings
TG-017: Prefer safetensors

Reference: 260117-BUNDLE-B_PLAN.md
"""

import os
import gc
import threading
from contextlib import contextmanager, nullcontext
from functools import wraps
from typing import Any, Callable, NoReturn, Optional, ParamSpec, TypeVar
import torch
from transformers import AutoTokenizer
try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
except Exception:  # pragma: no cover
    AutoProcessor = None
    AutoModelForImageTextToText = None
import folder_paths

# Model repository mapping
MODEL_REPOS = {
    "4B": "google/translategemma-4b-it",
    "12B": "google/translategemma-12b-it",
    "27B": "google/translategemma-27b-it",
}

# TG-029: Thread-safety lock for cache operations
_MODEL_CACHE_LOCK = threading.RLock()

_P = ParamSpec("_P")
_R = TypeVar("_R")


def serialized_model_access(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Serialize a complete model-cache ownership interval."""

    @wraps(function)
    def synchronized(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        # CRITICAL: keep load through inference/unload under this RLock; narrowing
        # the interval permits duplicate residency or unload during active inference.
        with _MODEL_CACHE_LOCK:
            return function(*args, **kwargs)

    return synchronized

# Single-model cache (TG-004: only one model in memory at a time)
# TG-014: Cache key now includes quantization and revision for correct reload
_current_model: Optional[tuple] = None  # (model, processor_or_tokenizer)
_current_model_key: Optional[tuple[str, str, Optional[str], str]] = None  # (model_size, quantization, revision, device_key)

# TG-014: Valid quantization options
QUANTIZATION_OPTIONS = ("none", "bnb-8bit", "bnb-4bit")

# TG-015: Host-aligned device override options.
DEVICE_OPTION_DEFAULT = "default"
DEVICE_OPTION_CPU = "cpu"
DEVICE_OPTION_PREFIX_GPU = "gpu:"
_FALLBACK_DEVICE_OPTIONS = (DEVICE_OPTION_DEFAULT, DEVICE_OPTION_CPU)

_TRUE_POLICY_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_POLICY_VALUES = frozenset({"", "0", "false", "no", "off"})


class _SafeModelLoadError(RuntimeError):
    """A user-facing load failure whose message is safe to expose."""


def _read_explicit_opt_in(name: str) -> tuple[bool, str]:
    """Read a boolean security policy without echoing its potentially sensitive value."""
    raw_value = os.environ.get(name)
    normalized = "" if raw_value is None else raw_value.strip().lower()
    if normalized in _TRUE_POLICY_VALUES:
        return True, f"{name}=enabled"
    if normalized in _FALSE_POLICY_VALUES:
        return False, f"{name}=disabled"
    return False, f"{name}=invalid (expected 0/1, false/true, no/yes, or off/on)"


def _is_remote_code_allowed(repo_id: Optional[str] = None) -> tuple[bool, str]:
    """
    Check if remote code execution is allowed based on policy (TG-026).
    
    Environment variables:
    - TRANSLATEGEMMA_ALLOW_REMOTE_CODE: explicit boolean opt-in (default: deny)
    - TRANSLATEGEMMA_REMOTE_CODE_ALLOWLIST: comma-separated repo IDs (optional)
    
    Precedence:
    1) Missing, false, or malformed opt-in → deny
    2) If allowlist is set → require an exact repo_id match
    3) Otherwise → allow because the explicit opt-in is present
    
    Returns:
        Tuple of (is_allowed, reason_string)
    """
    allowed, reason = _read_explicit_opt_in("TRANSLATEGEMMA_ALLOW_REMOTE_CODE")
    if not allowed:
        return False, reason

    allowlist_env = os.environ.get("TRANSLATEGEMMA_REMOTE_CODE_ALLOWLIST", "").strip()
    if allowlist_env:
        allowed_repos = {entry.strip() for entry in allowlist_env.split(",") if entry.strip()}
        if repo_id is None or repo_id not in allowed_repos:
            return False, "repository is not in TRANSLATEGEMMA_REMOTE_CODE_ALLOWLIST"
        return True, "explicit opt-in and exact allowlist match"

    return True, "explicit opt-in"


def _is_unsafe_weights_allowed() -> tuple[bool, str]:
    """Return the independent opt-in state for non-safetensors model weights."""
    return _read_explicit_opt_in("TRANSLATEGEMMA_ALLOW_UNSAFE_WEIGHTS")


def _is_remote_code_required_error(error: Exception) -> bool:
    """Classify only errors that explicitly request repository code execution."""
    message = str(error).casefold()
    if "trust_remote_code" not in message:
        return False
    return any(
        marker in message
        for marker in (
            "requires custom",
            "requires you to execute",
            "custom repository code",
            "contains custom code",
            "must be executed",
            "must execute",
            "execute the configuration file",
        )
    )


def _is_safetensors_unavailable_error(error: Exception) -> bool:
    """Classify absence of safetensors without treating corruption as format absence."""
    message = str(error).casefold()
    if "safetensor" not in message:
        return False
    return any(
        marker in message
        for marker in (
            "no safetensor",
            "safetensors not available",
            "safetensors weights were not found",
            "does not appear to have a file named",
            "does not contain safetensor",
        )
    )


def _raise_safe_stage_failure(stage: str, error: Exception) -> NoReturn:
    if _detect_auth_error(error):
        raise _SafeModelLoadError(
            f"TranslateGemma {stage} loading failed due to authentication or repository access. "
            "Verify model access and Hugging Face authentication."
        ) from None
    raise _SafeModelLoadError(
        f"TranslateGemma {stage} loading failed. Check network access, disk space, "
        "model files, and Transformers compatibility."
    ) from None


def _load_pretrained_with_security_policy(
    loader: Any,
    load_source: str,
    load_kwargs: dict[str, Any],
    *,
    repo_id: str,
    revision: Optional[str],
    stage: str,
    allow_weight_fallback: bool,
) -> Any:
    """Load with explicit, independently authorized permission transitions."""
    kwargs = dict(load_kwargs)
    attempted_states = set()

    while True:
        state = (
            bool(kwargs.get("trust_remote_code", False)),
            kwargs.get("use_safetensors"),
        )
        if state in attempted_states:
            raise _SafeModelLoadError(
                f"TranslateGemma {stage} loading stopped after a repeated security-policy state."
            )
        attempted_states.add(state)

        try:
            return loader.from_pretrained(load_source, **kwargs)
        except Exception as load_error:
            # CRITICAL: generic load failures must never authorize executable repository code.
            if (
                not kwargs.get("trust_remote_code", False)
                and _is_remote_code_required_error(load_error)
            ):
                remote_allowed, policy_reason = _is_remote_code_allowed(repo_id)
                if not remote_allowed:
                    raise _SafeModelLoadError(
                        f"TranslateGemma {stage} loading requires repository custom code, but "
                        f"remote code is denied ({policy_reason}). Set "
                        "TRANSLATEGEMMA_ALLOW_REMOTE_CODE=1 and optionally configure the exact "
                        "repository allowlist."
                    ) from None
                kwargs["trust_remote_code"] = True
                print(
                    f"[TranslateGemma] Loading {stage} with trust_remote_code=True "
                    f"(policy: {policy_reason})"
                )
                if not revision:
                    print(
                        "[TranslateGemma] SECURITY: pin a commit revision when remote code is enabled."
                    )
                continue

            # CRITICAL: non-safetensors weights require a separate opt-in from remote code.
            if (
                allow_weight_fallback
                and kwargs.get("use_safetensors") is True
                and _is_safetensors_unavailable_error(load_error)
            ):
                unsafe_allowed, policy_reason = _is_unsafe_weights_allowed()
                if not unsafe_allowed:
                    raise _SafeModelLoadError(
                        "TranslateGemma model loading found no compatible safetensors weights, "
                        f"but non-safetensors weights are denied ({policy_reason}). Set "
                        "TRANSLATEGEMMA_ALLOW_UNSAFE_WEIGHTS=1 only after reviewing the model source."
                    ) from None
                kwargs["use_safetensors"] = False
                print(
                    "[TranslateGemma] SECURITY: loading non-safetensors weights after explicit "
                    f"opt-in ({policy_reason})."
                )
                continue

            _raise_safe_stage_failure(stage, load_error)


def get_model_dir() -> str:
    """
    Return the default base directory for TranslateGemma models without creating it.

    Registered ComfyUI ``LLM`` roots are already ordered by the host, including
    any ``is_default`` promotion. When none are registered, retain the existing
    canonical/legacy destination rule beneath ``folder_paths.models_dir``.

    Returns:
        Base directory path for model storage
    """
    registered_roots = _get_registered_llm_roots()
    if registered_roots:
        return os.path.join(registered_roots[0], "TranslateGemma")
    return _get_unregistered_default_model_dir()


def _normalize_model_base(path: str) -> tuple[str, str]:
    """Return an absolute base path and its platform-aware deduplication key."""
    normalized = os.path.abspath(os.path.normpath(path))
    return normalized, os.path.normcase(normalized)


def _get_registered_llm_roots() -> list[str]:
    """Read ComfyUI's ordered ``LLM`` roots without mutating host registration."""
    get_folder_paths = getattr(folder_paths, "get_folder_paths", None)
    if not callable(get_folder_paths):
        return []

    try:
        configured_roots = get_folder_paths("LLM")
    except KeyError:
        return []
    except Exception:
        raise RuntimeError(
            "Unable to read ComfyUI registered LLM model paths. Check the host's "
            "extra model path configuration."
        ) from None

    if not isinstance(configured_roots, (list, tuple)):
        raise RuntimeError("ComfyUI registered LLM model paths must be an ordered list.")

    roots = []
    seen = set()
    for configured_root in configured_roots:
        if not isinstance(configured_root, str) or not configured_root.strip():
            raise RuntimeError("ComfyUI registered LLM model paths contain an invalid entry.")
        normalized, key = _normalize_model_base(configured_root)
        if key not in seen:
            seen.add(key)
            roots.append(normalized)
    return roots


def _get_unregistered_default_model_dir() -> str:
    models_dir, _ = _normalize_model_base(folder_paths.models_dir)
    llm_root = os.path.join(models_dir, "LLM")
    canonical = os.path.join(llm_root, "TranslateGemma")
    legacy = os.path.join(models_dir, "translate_gemma")
    if os.path.exists(llm_root) or not os.path.exists(legacy):
        return canonical
    return legacy


def _get_model_base_candidates(registered_roots: list[str]) -> list[str]:
    """Build ordered read candidates while preserving first occurrence."""
    models_dir, _ = _normalize_model_base(folder_paths.models_dir)
    raw_candidates = [
        *(os.path.join(root, "TranslateGemma") for root in registered_roots),
        os.path.join(models_dir, "LLM", "TranslateGemma"),
        os.path.join(models_dir, "translate_gemma"),
    ]
    candidates = []
    seen = set()
    for candidate in raw_candidates:
        normalized, key = _normalize_model_base(candidate)
        if key not in seen:
            seen.add(key)
            candidates.append(normalized)
    return candidates


def get_model_path(repo_id: str) -> str:
    """
    Resolve the local path for a specific model repo without creating directories.
    
    Args:
        repo_id: HuggingFace repo ID (e.g., "google/translategemma-4b-it")
        
    Returns:
        Directory path for this specific model
    """
    repo_name = repo_id.rsplit("/", 1)[-1]
    if not repo_name or repo_name in {".", ".."} or os.path.basename(repo_name) != repo_name:
        raise ValueError("Invalid model repository identifier.")

    registered_roots = _get_registered_llm_roots()
    base_candidates = _get_model_base_candidates(registered_roots)

    # IMPORTANT: search every root for the requested complete snapshot before
    # choosing a download destination; eager selection can pick another model
    # size or write a duplicate model onto the wrong disk.
    for base_dir in base_candidates:
        candidate = os.path.join(base_dir, repo_name)
        is_complete, _ = _is_snapshot_complete(candidate, debug=False)
        if is_complete:
            return candidate

    default_base = (
        os.path.join(registered_roots[0], "TranslateGemma")
        if registered_roots
        else _get_unregistered_default_model_dir()
    )
    return os.path.join(default_base, repo_name)


def _get_comfy_model_management():
    """Return ComfyUI model_management when available."""
    try:
        import comfy.model_management as model_management

        return model_management
    except Exception:
        return None


def _normalize_device_override(device_override: str | None) -> str:
    option = DEVICE_OPTION_DEFAULT if device_override is None else str(device_override).strip()
    if option in {"", "auto"}:
        return DEVICE_OPTION_DEFAULT
    return option


def _device_to_string(device) -> str:
    if device is None:
        return _legacy_default_device()
    return str(device)


def _legacy_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _legacy_device_count() -> int:
    try:
        return int(torch.cuda.device_count())
    except Exception:
        return 1 if torch.cuda.is_available() else 0


def get_device_options() -> tuple[str, ...]:
    """Return host-aligned device override options for the node widget (TG-015)."""
    model_management = _get_comfy_model_management()
    option_helper = (
        getattr(model_management, "get_gpu_device_options", None)
        if model_management is not None
        else None
    )
    if callable(option_helper):
        try:
            options = list(option_helper())
            if DEVICE_OPTION_DEFAULT not in options:
                options.insert(0, DEVICE_OPTION_DEFAULT)
            if DEVICE_OPTION_CPU not in options:
                options.insert(1, DEVICE_OPTION_CPU)
            return tuple(str(option) for option in options)
        except Exception:
            pass

    options = [DEVICE_OPTION_DEFAULT, DEVICE_OPTION_CPU]
    if torch.cuda.is_available():
        count = _legacy_device_count()
        if count > 1:
            options.extend(f"{DEVICE_OPTION_PREFIX_GPU}{i}" for i in range(count))
    return tuple(options)


def _resolve_default_device(model_management) -> tuple[str, str]:
    default_helper = (
        getattr(model_management, "get_torch_device", None)
        if model_management is not None
        else None
    )
    if callable(default_helper):
        try:
            return _device_to_string(default_helper()), "comfy"
        except Exception:
            pass
    return _legacy_default_device(), "legacy"


def _device_override_warning(requested: str) -> str:
    if requested.startswith(DEVICE_OPTION_PREFIX_GPU):
        return "Unsupported device override: requested GPU device is unavailable; falling back to default."
    return "Unsupported device override; falling back to default."


def _parse_gpu_override_index(requested: str) -> int | None:
    if not requested.startswith(DEVICE_OPTION_PREFIX_GPU):
        return None
    raw_index = requested[len(DEVICE_OPTION_PREFIX_GPU):]
    if not raw_index.isascii() or not raw_index.isdecimal():
        return None
    try:
        return int(raw_index)
    except ValueError:
        return None


def _safe_device_option_label(requested: str) -> str:
    if requested in {DEVICE_OPTION_DEFAULT, DEVICE_OPTION_CPU}:
        return requested
    if requested.startswith(DEVICE_OPTION_PREFIX_GPU):
        gpu_index = _parse_gpu_override_index(requested)
        if gpu_index is not None:
            return f"{DEVICE_OPTION_PREFIX_GPU}{gpu_index}"
        return "gpu:invalid"
    return "unsupported"


def resolve_device_override(device_override: str | None = DEVICE_OPTION_DEFAULT) -> dict[str, str | None]:
    """
    Resolve a node device override to a concrete torch device string (TG-015).

    Returns a small dict to keep the public surface stable while still exposing
    enough context for cache keys, logs, tests, and error messages.
    """
    requested = _normalize_device_override(device_override)
    model_management = _get_comfy_model_management()

    if requested == DEVICE_OPTION_DEFAULT:
        device, source = _resolve_default_device(model_management)
        return {
            "requested": requested,
            "effective": DEVICE_OPTION_DEFAULT,
            "device": device,
            "cache_key": f"{DEVICE_OPTION_DEFAULT}:{device}",
            "source": source,
            "warning": None,
        }

    gpu_index = _parse_gpu_override_index(requested)

    is_supported_shape = requested == DEVICE_OPTION_CPU or gpu_index is not None
    resolver = (
        getattr(model_management, "resolve_gpu_device_option", None)
        if model_management is not None
        else None
    )
    if is_supported_shape and callable(resolver):
        try:
            resolved = resolver(requested)
        except Exception:
            resolved = None
        if resolved is not None:
            device = _device_to_string(resolved)
            return {
                "requested": requested,
                "effective": requested,
                "device": device,
                "cache_key": f"{requested}:{device}",
                "source": "comfy",
                "warning": None,
            }

    if requested == DEVICE_OPTION_CPU:
        device = "cpu"
        return {
            "requested": requested,
            "effective": requested,
            "device": device,
            "cache_key": f"{requested}:{device}",
            "source": "legacy",
            "warning": None,
        }

    # IMPORTANT: partial ComfyUI hosts may expose only get_torch_device(); keep
    # explicit GPU overrides independent from every optional host helper.
    if gpu_index is not None and torch.cuda.is_available():
        if gpu_index < _legacy_device_count():
            device = f"cuda:{gpu_index}"
            return {
                "requested": requested,
                "effective": requested,
                "device": device,
                "cache_key": f"{requested}:{device}",
                "source": "legacy",
                "warning": None,
            }

    default_device, source = _resolve_default_device(model_management)
    return {
        "requested": requested,
        "effective": DEVICE_OPTION_DEFAULT,
        "device": default_device,
        "cache_key": f"{DEVICE_OPTION_DEFAULT}:{default_device}",
        "source": source,
        "warning": _device_override_warning(requested),
    }


def get_device(device_override: str | None = DEVICE_OPTION_DEFAULT) -> str:
    """Determine the active device, optionally honoring a host-aligned override."""
    return str(resolve_device_override(device_override)["device"])


@contextmanager
def _legacy_cuda_device_context(device: str):
    previous = None
    if device.startswith("cuda:") and torch.cuda.is_available():
        try:
            requested = _parse_cuda_device_index(device)
            previous = torch.cuda.current_device()
            if previous != requested:
                torch.cuda.set_device(requested)
            else:
                previous = None
        except Exception:
            previous = None
    try:
        yield
    finally:
        if previous is not None:
            try:
                torch.cuda.set_device(previous)
            except Exception:
                pass


def get_cuda_device_context(device: str):
    """Return ComfyUI's CUDA device context when available, otherwise a local fallback."""
    if not str(device).startswith("cuda"):
        return nullcontext()

    model_management = _get_comfy_model_management()
    context_helper = (
        getattr(model_management, "cuda_device_context", None)
        if model_management is not None
        else None
    )
    if callable(context_helper):
        try:
            return context_helper(torch.device(device))
        except Exception:
            pass
    return _legacy_cuda_device_context(str(device))


def _build_model_device_map(device: str, effective_device_option: str):
    if not str(device).startswith("cuda"):
        return None
    if effective_device_option == DEVICE_OPTION_DEFAULT:
        return "auto"
    return {"": str(device)}


def cuda_supports_bf16(device_index: int = None) -> bool:
    """
    Check if CUDA device supports BF16 (bfloat16).
    
    Args:
        device_index: CUDA device index to check. If None, uses current device.
    
    Detection strategy:
    - Uses torch.cuda.get_device_capability(device_index) for multi-GPU accuracy
    - Ampere (SM 8.0) and newer support BF16
    
    Returns:
        True if BF16 is supported, False otherwise
    """
    if not torch.cuda.is_available():
        return False
    
    try:
        if device_index is None:
            device_index = torch.cuda.current_device()
        major, _ = torch.cuda.get_device_capability(device_index)
        return major >= 8
    except Exception:
        return False


def _parse_cuda_device_index(device: str) -> int:
    """Parse CUDA device index from device string."""
    if ":" in device:
        try:
            return int(device.split(":")[1])
        except (ValueError, IndexError):
            return 0
    return torch.cuda.current_device() if torch.cuda.is_available() else 0


def get_torch_dtype(device: str) -> torch.dtype:
    """
    Determine the optimal torch dtype for the given device.
    
    Returns:
        - cuda: bfloat16 if supported, else float16
        - mps/cpu: float32 (conservative, stable)
    """
    if device.startswith("cuda"):
        device_index = _parse_cuda_device_index(device)
        if cuda_supports_bf16(device_index):
            return torch.bfloat16
        else:
            return torch.float16
    else:
        return torch.float32


def cleanup_torch_memory():
    """
    Clean up GPU/CPU memory after model unload (TG-004).
    
    Performs:
    - Python garbage collection
    - CUDA cache clearing (if available)
    - CUDA IPC collection (if available)
    """
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def _weights_present(local_dir: str) -> bool:
    """
    Check if model weights are present in local_dir (TG-016).
    
    Looks for *.safetensors (preferred) or *.bin files.
    
    Note: TG-027 adds `_is_snapshot_complete()` for stricter validation.
    This function is kept for backward compat but deprecated.
    """
    if not os.path.isdir(local_dir):
        return False
    
    for root, _, files in os.walk(local_dir):
        for f in files:
            if f.endswith(".safetensors") or f.endswith(".bin"):
                return True
    return False


def _weight_index_missing_entries(local_dir: str, index_filename: str) -> list[str]:
    """Validate an atomic local shard index without exposing filesystem paths."""
    import json

    index_path = os.path.join(local_dir, index_filename)
    try:
        with open(index_path, "r", encoding="utf-8") as index_file:
            index_data = json.load(index_file)
    except (json.JSONDecodeError, OSError) as error:
        return [f"{index_filename} (parse error: {type(error).__name__})"]

    if not isinstance(index_data, dict):
        return [f"{index_filename} (expected an object)"]
    weight_map = index_data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return [f"{index_filename} (weight_map must be a non-empty object)"]

    local_root = os.path.abspath(local_dir)
    shard_files = set()
    for tensor_name, shard in weight_map.items():
        if (
            not isinstance(tensor_name, str)
            or not tensor_name
            or not isinstance(shard, str)
            or not shard
        ):
            return [f"{index_filename} (invalid weight_map entry)"]
        normalized_shard = os.path.normpath(shard)
        if (
            os.path.isabs(shard)
            or normalized_shard in {".", ".."}
            or normalized_shard.startswith(".." + os.sep)
        ):
            return [f"{index_filename} (invalid shard path)"]
        shard_path = os.path.abspath(os.path.join(local_root, normalized_shard))
        try:
            if os.path.commonpath((local_root, shard_path)) != local_root:
                return [f"{index_filename} (invalid shard path)"]
        except ValueError:
            return [f"{index_filename} (invalid shard path)"]
        shard_files.add((shard, shard_path))

    return [
        f"shard: {name}"
        for name, path in sorted(shard_files)
        if not os.path.isfile(path)
    ]


def _is_snapshot_complete(local_dir: str, debug: bool = False) -> tuple[bool, list[str]]:
    """
    Check if model snapshot in local_dir is complete (TG-027).
    
    Performs stricter validation than `_weights_present()`:
    1. Required metadata: config.json + tokenizer/processor config
    2. Weights completeness:
       - If sharded (model.safetensors.index.json): verify all shards exist
       - Else: verify single weight file exists
    
    Args:
        local_dir: Model directory to check
        debug: Print detailed missing file info
        
    Returns:
        Tuple of (is_complete, missing_files_list)
    """
    if not os.path.isdir(local_dir):
        return False, ["directory does not exist"]
    
    missing = []
    
    # 1. Check required metadata (stricter for manual/offline snapshots)
    # NOTE: We intentionally require more than "just config + weights" so that
    # users who manually copy a partial folder get an actionable "incomplete
    # snapshot" message before runtime load fails with a vague network/cache error.
    required_metadata = [
        "config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "processor_config.json",
        "preprocessor_config.json",
    ]

    for f in required_metadata:
        if not os.path.isfile(os.path.join(local_dir, f)):
            missing.append(f)

    # Tokenizer vocabulary/model artifact (name varies by tokenizer implementation).
    tokenizer_artifacts_any = [
        "tokenizer.json",
        "tokenizer.model",
        "spiece.model",
        "sentencepiece.bpe.model",
    ]
    tokenizer_artifact_found = any(
        os.path.isfile(os.path.join(local_dir, f)) for f in tokenizer_artifacts_any
    )
    if not tokenizer_artifact_found:
        missing.append("tokenizer artifact (any of: " + ", ".join(tokenizer_artifacts_any) + ")")
    
    # 2. Check weights completeness
    index_file = os.path.join(local_dir, "model.safetensors.index.json")
    
    if os.path.isfile(index_file):
        missing.extend(
            _weight_index_missing_entries(local_dir, "model.safetensors.index.json")
        )
    else:
        # Single-file weights
        single_weight_files = ["model.safetensors", "pytorch_model.bin"]
        weight_found = any(
            os.path.isfile(os.path.join(local_dir, f)) for f in single_weight_files
        )
        if not weight_found:
            # Also check for sharded bin files
            bin_index = os.path.join(local_dir, "pytorch_model.bin.index.json")
            if os.path.isfile(bin_index):
                missing.extend(
                    _weight_index_missing_entries(local_dir, "pytorch_model.bin.index.json")
                )
            else:
                missing.append("weights (expected model.safetensors or pytorch_model.bin)")
    
    is_complete = len(missing) == 0
    
    if debug and missing:
        capped = missing[:5]
        suffix = f" (+{len(missing)-5} more)" if len(missing) > 5 else ""
        print(f"[TranslateGemma] Incomplete snapshot, missing: {', '.join(capped)}{suffix}")
    
    return is_complete, missing


def ensure_model_downloaded(
    repo_id: str,
    local_dir: str,
    revision: Optional[str] = None,
    debug: bool = False,
) -> bool:
    """
    Ensure model is downloaded to local_dir using huggingface_hub (TG-016/TG-027).
    
    TG-027: Uses `_is_snapshot_complete()` for stricter validation and supports
    resumable downloads for incomplete snapshots.
    
    Args:
        repo_id: HuggingFace repo ID
        local_dir: Target directory for download
        revision: Optional revision (commit hash or tag)
        debug: Print detailed diagnostics
        
    Returns:
        True if download was performed or skipped (snapshot complete)
        
    Raises:
        RuntimeError: If download fails (with actionable auth instructions if applicable)
    """
    # TG-027: Use stricter completeness check
    is_complete, missing = _is_snapshot_complete(local_dir, debug=debug)
    
    if is_complete:
        print("[TranslateGemma] Snapshot complete in configured model cache; skipping download")
        return True
    
    # Snapshot incomplete or missing
    initial_missing = list(missing)

    if os.path.isdir(local_dir) and missing:
        print(f"[TranslateGemma] Incomplete snapshot detected, will resume download...")
        capped = missing[:5]
        suffix = f" (+{len(missing)-5} more)" if len(missing) > 5 else ""
        print(f"[TranslateGemma] Missing files/components: {', '.join(capped)}{suffix}")
        print(
            "[TranslateGemma] Manual/offline snapshots must include tokenizer + processor metadata "
            "(e.g. tokenizer_config.json, special_tokens_map.json, processor_config.json, "
            "preprocessor_config.json, and a tokenizer model/json file)."
        )

    try:
        os.makedirs(local_dir, exist_ok=True)
    except OSError:
        raise RuntimeError(
            "Unable to create the configured model download destination. Configure a writable "
            "default LLM root in ComfyUI and retry. No alternate model root was selected."
        ) from None

    # Try huggingface_hub.snapshot_download
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "[TranslateGemma] WARNING: huggingface_hub not available, "
            "falling back to transformers implicit download"
        )
        return False

    def _classify_download_error(exc: Exception) -> tuple[str, str]:
        """
        Best-effort error classification with actionable hints.

        Returns:
            (kind, hint)
            kind in {"auth", "network", "disk", "unknown"}
        """
        error_str = str(exc)
        lower = error_str.lower()

        # Auth / gated / forbidden
        auth_keywords = [
            "401",
            "403",
            "gated",
            "forbidden",
            "unauthorized",
            "access denied",
            "requires authentication",
            "invalid token",
        ]
        if any(kw in lower for kw in auth_keywords):
            return (
                "auth",
                "This model may be gated or require authentication. "
                "Accept the license on Hugging Face and set `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`), then retry.",
            )

        # Disk / permissions
        disk_keywords = [
            "no space left on device",
            "disk full",
            "not enough space",
            "permission denied",
            "access is denied",
            "read-only file system",
            "readonly file system",
        ]
        if any(kw in lower for kw in disk_keywords):
            return (
                "disk",
                "Check free disk space and write permissions for the model cache directory, then retry. "
                "Tip: try deleting the incomplete folder and re-downloading.",
            )

        # Network / connectivity (common behind firewalls / blocked regions)
        network_keywords = [
            "timed out",
            "timeout",
            "connection error",
            "connection aborted",
            "connection reset",
            "connection refused",
            "max retries exceeded",
            "name or service not known",
            "temporary failure in name resolution",
            "dns",
            "proxy",
            "ssl",
            "tls",
            "certificate verify failed",
            "remote end closed connection",
        ]
        if any(kw in lower for kw in network_keywords):
            return (
                "network",
                "Network issue while contacting Hugging Face. This can happen behind a firewall, unstable network, "
                "or in regions where `huggingface.co` is blocked (e.g. some China networks). "
                "Options:\n"
                "- Configure a proxy: set `HTTP_PROXY` / `HTTPS_PROXY`\n"
                "- Use a mirror endpoint (community): set `HF_ENDPOINT` (or `HUGGINGFACE_HUB_ENDPOINT`) to a mirror URL\n"
                "- Offline: download the model on another machine and copy files into the cache dir, then restart ComfyUI",
            )

        return ("unknown", "Retry later and check the full stack trace for details.")

    def _download_attempt(attempt: int, max_attempts: int) -> None:
        if max_attempts > 1:
            print(f"[TranslateGemma] Download attempt {attempt}/{max_attempts}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Windows-friendly
            revision=revision,
            resume_download=True,  # TG-027: Resume partial downloads
        )
    
    print(f"[TranslateGemma] Downloading {repo_id} to configured model cache...")
    if revision:
        print("[TranslateGemma] Using configured revision")
    
    # TG-052: Clearer download diagnostics + small retry loop
    retries_env = os.environ.get("TRANSLATEGEMMA_DOWNLOAD_RETRIES", "").strip()
    try:
        max_attempts = max(1, int(retries_env)) if retries_env else 2
    except Exception:
        max_attempts = 2

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            _download_attempt(attempt, max_attempts)
            last_exc = None
            break
        except Exception as e:
            last_exc = e
            kind, hint = _classify_download_error(e)
            if debug:
                print(f"[TranslateGemma] Download error classified as: {kind}")
            if attempt < max_attempts:
                # Basic backoff (1s, 2s, 4s...)
                backoff = 2 ** (attempt - 1)
                print(
                    f"[TranslateGemma] Download failed (kind={kind}, "
                    f"error_type={type(e).__name__})"
                )
                print(f"[TranslateGemma] Retrying in {backoff}s... ({hint})")
                try:
                    import time

                    time.sleep(backoff)
                except Exception:
                    pass
            else:
                break

    if last_exc is None:
        print("[TranslateGemma] Download complete in configured model cache")
        
        # TG-027: Post-download verification
        is_complete_after, missing_after = _is_snapshot_complete(local_dir, debug=debug)
        if not is_complete_after:
            capped = missing_after[:5]
            raise RuntimeError(
                f"Download completed but snapshot still incomplete. "
                f"Missing: {', '.join(capped)}. "
                "Try deleting the incomplete model cache folder and retrying."
            )
        
        return True

    assert last_exc is not None
    kind, hint = _classify_download_error(last_exc)

    if initial_missing:
        capped = initial_missing[:5]
        suffix = f" (+{len(initial_missing)-5} more)" if len(initial_missing) > 5 else ""
        print(
            "[TranslateGemma] Local snapshot is still incomplete and online recovery failed. "
            f"Missing: {', '.join(capped)}{suffix}"
        )

    # Keep the legacy auth flow message (more explicit)
    if kind == "auth":
        raise RuntimeError(
            f"Failed to download model '{repo_id}' - access denied.\n\n"
            "To resolve:\n"
            "1. Visit the model page on Hugging Face and accept the license terms\n"
            "2. Authenticate with Hugging Face (one of):\n"
            "   - Run: hf auth login\n"
            "   - Or set: HF_TOKEN=your_token_here (or HUGGINGFACE_HUB_TOKEN)\n"
            "3. Restart ComfyUI"
        ) from None

    # Generic / network / disk
    endpoint = os.environ.get("HF_ENDPOINT") or os.environ.get("HUGGINGFACE_HUB_ENDPOINT") or ""
    proxy_http = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy") or ""
    proxy_https = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or ""

    extra_lines = []
    if endpoint:
        extra_lines.append("- Custom HF endpoint configured")
    if proxy_http or proxy_https:
        extra_lines.append("- Proxy configured via HTTP(S)_PROXY")

    extra_context = ("\n\n[Env]\n" + "\n".join(extra_lines)) if extra_lines else ""

    local_snapshot_context = ""
    if initial_missing:
        local_snapshot_context = (
            "\n\n[Local snapshot check]\n"
            f"Incomplete local snapshot detected (missing: {', '.join(initial_missing[:8])}"
            + (f", +{len(initial_missing)-8} more" if len(initial_missing) > 8 else "")
            + ")."
        )

    raise RuntimeError(
        f"Failed to download model '{repo_id}'.\n\n"
        f"[Cause]\n{kind} download failure ({type(last_exc).__name__})\n\n"
        f"[Hint]\n{hint}\n\n"
        "[Cache dir]\n[configured model cache]"
        f"{local_snapshot_context}"
        f"{extra_context}"
    ) from None


def unload_current_model():
    """
    Unload the currently loaded model and free memory (TG-004).

    This ensures only one model is in memory at a time.
    TG-029: Thread-safe with _MODEL_CACHE_LOCK.
    TG-014: Now uses _current_model_key tuple instead of _current_model_size.
    """
    global _current_model, _current_model_key

    with _MODEL_CACHE_LOCK:
        if _current_model is not None:
            model, processor_or_tokenizer = _current_model
            model_key = _current_model_key

            # Clear references
            _current_model = None
            _current_model_key = None
            del model
            del processor_or_tokenizer

            # Clean up memory
            cleanup_torch_memory()

            if model_key:
                model_size = model_key[0]
                quantization = model_key[1] if len(model_key) > 1 else "none"
                device_key = model_key[3] if len(model_key) > 3 else "unknown"
                print(
                    f"[TranslateGemma] {model_size} model "
                    f"(quantization={quantization}, device={device_key}) unloaded, memory cleaned"
                )
            else:
                print("[TranslateGemma] Model unloaded, memory cleaned")


def _detect_auth_error(error: Exception) -> bool:
    """
    Detect if error is related to Hugging Face authentication / gating.

    Note: Avoid overly broad "token" matching to prevent false positives like "tokenizer".
    """
    error_str = str(error).lower()
    if any(keyword in error_str for keyword in [
        "401",
        "403",
        "gated",
        "gated repo",
        "requires authentication",
        "forbidden",
        "unauthorized",
        "access denied",
        "authentication required",
        "not authenticated",
    ]):
        return True

    # huggingface_hub sometimes reports gated/private repos as "not a valid model identifier"
    # and suggests authentication. Only treat this as auth-related when that phrase appears.
    return (
        "is not a local folder and is not a valid model identifier" in error_str
        and ("hf auth login" in error_str or "token=" in error_str or "use_auth_token" in error_str)
    )


def _build_load_context(
    model_size: str,
    repo_id: str,
    device: str,
    dtype,
    revision: str = None,
    device_override: str = DEVICE_OPTION_DEFAULT,
    effective_device_option: str = DEVICE_OPTION_DEFAULT,
) -> str:
    """
    Build context string for error messages (TG-008).
    
    Provides key information for debugging load/inference failures.
    """
    lines = [
        f"model_size: {model_size}",
        f"repo_id: {repo_id}",
        f"device: {device}",
        f"dtype: {dtype}",
        f"device_override: {device_override}",
        f"effective_device_option: {effective_device_option}",
    ]
    if revision:
        lines.append("revision: configured")
    return " | ".join(lines)

def _raise_auth_error(repo_id: str):
    """Raise user-friendly error for gated model access issues (TG-004)."""
    raise _SafeModelLoadError(
        f"Failed to access model '{repo_id}' - this may be a gated model.\n\n"
        "To resolve:\n"
        "1. Visit the model page on Hugging Face and accept the license terms\n"
        "2. Authenticate with Hugging Face (one of):\n"
        "   - Run: hf auth login\n"
        "   - Or set: HF_TOKEN=your_token_here (or HUGGINGFACE_HUB_TOKEN)\n"
        "3. Restart ComfyUI"
    ) from None


def _get_bnb_compute_dtype() -> torch.dtype:
    """
    Get compute dtype for BitsAndBytes 4-bit quantization (TG-014).

    Priority:
    1. TRANSLATEGEMMA_BNB_4BIT_COMPUTE_DTYPE env var (bf16 or fp16)
    2. Auto-detect: BF16 if CUDA supports it, else FP16
    """
    env_dtype = os.environ.get("TRANSLATEGEMMA_BNB_4BIT_COMPUTE_DTYPE", "").strip().lower()
    if env_dtype == "bf16":
        return torch.bfloat16
    elif env_dtype == "fp16":
        return torch.float16
    # Auto-detect
    if cuda_supports_bf16():
        return torch.bfloat16
    return torch.float16


def _get_bnb_double_quant() -> bool:
    """
    Get double quantization setting for BitsAndBytes 4-bit (TG-014).

    Default: True (enabled)
    Env override: TRANSLATEGEMMA_BNB_4BIT_DOUBLE_QUANT=0 to disable
    """
    env_val = os.environ.get("TRANSLATEGEMMA_BNB_4BIT_DOUBLE_QUANT", "").strip()
    if env_val == "0":
        return False
    return True


def _check_bnb_availability(quantization: str, device: str) -> None:
    """
    Check if bitsandbytes quantization is available (TG-014).

    Raises RuntimeError with actionable message if:
    - Device is not CUDA
    - bitsandbytes is not installed
    - BitsAndBytesConfig is not available in transformers
    """
    if quantization == "none":
        return

    # This project keeps a narrower policy than upstream BnB until each host backend is tested.
    if not device.startswith("cuda"):
        raise RuntimeError(
            f"TranslateGemma bitsandbytes mode (quantization={quantization}) is currently enabled "
            "only for this project's CUDA device path.\n\n"
            "To resolve:\n"
            "- Select a CUDA device supported by the active ComfyUI environment, OR\n"
            "- Set quantization=none to use the host's normal model-loading path\n"
            "Official upstream backend availability does not imply project-level support."
        )

    # IMPORTANT: keep this import inside the BnB-only branch and redact every
    # native-loader failure; these errors often expose local library paths.
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        raise RuntimeError(
            f"bitsandbytes quantization (quantization={quantization}) requires bitsandbytes to be installed.\n\n"
            "To resolve:\n"
            "- In ComfyUI's Python environment, run: python -m pip install -r requirements-quantization.txt\n"
            "- Or run: python -m pip install \"bitsandbytes>=0.49.2\"\n"
            "- Official instructions: https://huggingface.co/docs/bitsandbytes/installation\n"
            "- Or set quantization=none to use the host's normal model-loading path\n\n"
            "Do not replace ComfyUI's PyTorch build as part of this optional install."
        ) from None

    # 3. Check BitsAndBytesConfig in transformers
    try:
        from transformers import BitsAndBytesConfig  # noqa: F401
    except ImportError:
        raise RuntimeError(
            f"bitsandbytes quantization (quantization={quantization}) requires BitsAndBytesConfig from transformers.\n"
            "To resolve in ComfyUI's Python environment:\n"
            "- Reinstall this node's declared requirements without replacing the host's PyTorch build\n"
            "- Or set quantization=none to use the host's normal model-loading path\n"
            "- Integration guide: https://huggingface.co/docs/transformers/main/quantization/bitsandbytes"
        ) from None


def _build_quantization_config(quantization: str):
    """
    Build BitsAndBytesConfig for the requested quantization mode (TG-014).

    Args:
        quantization: One of "none", "bnb-8bit", "bnb-4bit"

    Returns:
        BitsAndBytesConfig or None (for quantization="none")
    """
    if quantization == "none":
        return None

    from transformers import BitsAndBytesConfig

    if quantization == "bnb-8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "bnb-4bit":
        compute_dtype = _get_bnb_compute_dtype()
        use_double_quant = _get_bnb_double_quant()
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        raise ValueError(f"Invalid quantization mode: {quantization}. Choose from: {QUANTIZATION_OPTIONS}")


@serialized_model_access
def load_model(
    model_size: str,
    revision: Optional[str] = None,
    quantization: str = "none",
    device_override: str = DEVICE_OPTION_DEFAULT,
) -> tuple:
    """
    Load TranslateGemma model and tokenizer with single-model cache (TG-004).

    Args:
        model_size: One of "4B", "12B", "27B"
        revision: Optional HuggingFace revision/commit hash for reproducibility (TG-006)
        quantization: Quantization mode (TG-014): "none", "bnb-8bit", or "bnb-4bit"
        device_override: Device override (TG-015): "default", "cpu", or "gpu:N"

    Returns:
        Tuple of (model, processor)
    """
    global _current_model, _current_model_key

    # Validate quantization option
    if quantization not in QUANTIZATION_OPTIONS:
        raise ValueError(f"Invalid quantization mode: {quantization}. Choose from: {QUANTIZATION_OPTIONS}")

    device_selection = resolve_device_override(device_override)
    resolved_device = str(device_selection["device"])
    resolved_device_key = str(device_selection["cache_key"])
    effective_device_option = str(device_selection["effective"])
    requested_device_option = str(device_selection["requested"])
    requested_device_label = _safe_device_option_label(requested_device_option)

    # Check env var for revision override
    env_revision = os.environ.get("TRANSLATEGEMMA_REVISION")
    if env_revision and not revision:
        revision = env_revision

    # TG-015: Include device in the cache key so CPU/GPU override switches cannot reuse
    # a model loaded on the wrong device.
    model_key = (model_size, quantization, revision, resolved_device_key)

    # TG-029: Thread-safe cache check and switch
    with _MODEL_CACHE_LOCK:
        # Return cached model if same key is requested (TG-014: includes quantization)
        if _current_model is not None and _current_model_key == model_key:
            print(
                f"[TranslateGemma] Using cached {model_size} model "
                f"(quantization={quantization}, device={resolved_device})"
            )
            return _current_model

        # Unload current model before loading new one (single-model cache)
        if _current_model is not None:
            old_size = _current_model_key[0] if _current_model_key else "unknown"
            old_quant = _current_model_key[1] if _current_model_key and len(_current_model_key) > 1 else "none"
            old_device = _current_model_key[3] if _current_model_key and len(_current_model_key) > 3 else "unknown"
            print(
                f"[TranslateGemma] Switching from {old_size} "
                f"(quantization={old_quant}, device={old_device}) to {model_size} "
                f"(quantization={quantization}, device={resolved_device_key})"
            )
        unload_current_model()

    repo_id = MODEL_REPOS.get(model_size)
    if not repo_id:
        raise ValueError(f"Invalid model size: {model_size}. Choose from: {list(MODEL_REPOS.keys())}")

    device = resolved_device
    cache_dir = get_model_path(repo_id)  # TG-005: isolated per repo
    dtype = get_torch_dtype(device)

    print(f"[TranslateGemma] Loading {model_size} model from {repo_id}...")
    if device_selection.get("warning"):
        print(f"[TranslateGemma] WARNING: {device_selection['warning']}")
    print(
        f"[TranslateGemma] Device: {device}, dtype: {dtype}, quantization: {quantization}, "
        f"device_override={requested_device_label}, effective_device={effective_device_option}"
    )
    print("[TranslateGemma] Cache dir: [configured model cache]")
    if revision:
        print("[TranslateGemma] Revision: [configured]")

    # TG-014: Check bitsandbytes availability before loading
    _check_bnb_availability(quantization, device)

    processor = None
    model = None
    try:
        if AutoProcessor is None or AutoModelForImageTextToText is None:
            raise RuntimeError(
                "TranslateGemma (Gemma 3) requires a newer transformers build that provides "
                "`AutoProcessor` and `AutoModelForImageTextToText` (recommended: transformers>=4.57)."
            )

        # TG-016: Explicit download before loading (Windows-friendly, skip if present)
        ensure_model_downloaded(repo_id, cache_dir, revision)

        # BUGFIX (manual/offline snapshot loading):
        # `cache_dir` in from_pretrained(...) is only a cache root hint; it does NOT
        # mean "load directly from this folder". If we pass `repo_id`, transformers
        # may still resolve through HF endpoints (e.g. HF_ENDPOINT mirrors), which
        # breaks users who manually copied a complete snapshot for offline use.
        #
        # When the local snapshot folder is complete, load from the local path itself
        # and force local_files_only=True so no network lookup is attempted.
        snapshot_complete, _ = _is_snapshot_complete(cache_dir, debug=False)
        load_source = cache_dir if snapshot_complete else repo_id
        local_files_only = bool(snapshot_complete)
        if snapshot_complete:
            print("[TranslateGemma] Using complete local snapshot (local_files_only=True)")
        else:
            print(f"[TranslateGemma] Local snapshot incomplete; loading via repo id: {repo_id}")

        processor = _load_pretrained_with_security_policy(
            AutoProcessor,
            load_source,
            {
                "cache_dir": cache_dir,
                "revision": revision,
                "local_files_only": local_files_only,
                "trust_remote_code": False,
            },
            repo_id=repo_id,
            revision=revision,
            stage="processor",
            allow_weight_fallback=False,
        )

        # TG-014: Build quantization config
        quantization_config = _build_quantization_config(quantization)
        if quantization_config:
            compute_dtype = _get_bnb_compute_dtype()
            print(f"[TranslateGemma] Using BitsAndBytes quantization: {quantization}, compute_dtype={compute_dtype}")

        # TG-017 + TG-026: Prefer safetensors, apply remote-code policy
        # TG-014: Add quantization_config when using bitsandbytes
        model_kwargs = dict(
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=local_files_only,
            torch_dtype=dtype,
            device_map=_build_model_device_map(device, effective_device_option),
            trust_remote_code=False,
            use_safetensors=True,
        )
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        with get_cuda_device_context(device):
            model = _load_pretrained_with_security_policy(
                AutoModelForImageTextToText,
                load_source,
                model_kwargs,
                repo_id=repo_id,
                revision=revision,
                stage="model",
                allow_weight_fallback=True,
            )

        # TG-014: Do not call model.to(device) for quantized models (already device-mapped)
        if not device.startswith("cuda") and quantization == "none":
            model = model.to(device)

        # TG-007: Set model to eval mode for inference
        model.eval()

        # TG-014: Store full model key including quantization. The lifecycle
        # decorator retains exclusive ownership through this publication.
        _current_model = (model, processor)
        _current_model_key = model_key

        print(
            f"[TranslateGemma] {model_size} model loaded successfully on {device} "
            f"(eval mode, quantization={quantization}, device_override={requested_device_label})"
        )
        return model, processor

    except Exception as e:
        # Drop partial objects before allocator cleanup so collection is effective.
        model = None
        processor = None
        cleanup_torch_memory()

        if isinstance(e, _SafeModelLoadError):
            raise

        # TG-004: Detect gated/auth errors and provide actionable guidance
        if _detect_auth_error(e):
            _raise_auth_error(repo_id)

        # TG-008: Include context in error message
        context = _build_load_context(
            model_size,
            repo_id,
            device,
            dtype,
            revision,
            device_override=requested_device_label,
            effective_device_option=effective_device_option,
        )
        context += f" | quantization: {quantization}"
        raise _SafeModelLoadError(
            "Failed to load TranslateGemma. Check model access, local storage, dependency "
            f"compatibility, and device configuration.\n[Context] {context}"
        ) from None


def unload_model(model_size: str = None):
    """
    Unload model from memory (backward compatible API).
    
    Args:
        model_size: Ignored for single-model cache; unloads current model
    """
    unload_current_model()


def get_available_models() -> list[str]:
    """Return list of available model sizes."""
    return list(MODEL_REPOS.keys())


def get_current_model_size() -> Optional[str]:
    """Return the currently loaded model size, or None if no model is loaded."""
    with _MODEL_CACHE_LOCK:
        if _current_model_key is None:
            return None
        return _current_model_key[0]


def get_current_quantization() -> Optional[str]:
    """Return the currently loaded model's quantization mode, or None if no model is loaded (TG-014)."""
    with _MODEL_CACHE_LOCK:
        if _current_model_key is None:
            return None
        return _current_model_key[1] if len(_current_model_key) > 1 else "none"


def is_model_loaded() -> bool:
    """Check if a model is currently loaded."""
    with _MODEL_CACHE_LOCK:
        return _current_model is not None
