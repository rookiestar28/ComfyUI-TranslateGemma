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
from typing import Optional
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

# Single-model cache (TG-004: only one model in memory at a time)
_current_model: Optional[tuple] = None  # (model, processor_or_tokenizer)
_current_model_size: Optional[str] = None


def _is_remote_code_allowed(repo_id: str = None) -> tuple[bool, str]:
    """
    Check if remote code execution is allowed based on policy (TG-026).
    
    Environment variables:
    - TRANSLATEGEMMA_ALLOW_REMOTE_CODE: "1" (default) to allow, "0" to deny
    - TRANSLATEGEMMA_REMOTE_CODE_ALLOWLIST: comma-separated repo IDs (optional)
    
    Precedence:
    1) ALLOW_REMOTE_CODE=0 → deny always
    2) If allowlist set → allow only if repo_id in allowlist
    3) Else → allow (default)
    
    Returns:
        Tuple of (is_allowed, reason_string)
    """
    allow_env = os.environ.get("TRANSLATEGEMMA_ALLOW_REMOTE_CODE", "1").strip()
    
    # Explicit deny
    if allow_env == "0":
        return False, "TRANSLATEGEMMA_ALLOW_REMOTE_CODE=0"
    
    # Check allowlist if set
    allowlist_env = os.environ.get("TRANSLATEGEMMA_REMOTE_CODE_ALLOWLIST", "").strip()
    if allowlist_env:
        allowed_repos = [r.strip() for r in allowlist_env.split(",") if r.strip()]
        if repo_id and repo_id not in allowed_repos:
            return False, f"repo_id '{repo_id}' not in TRANSLATEGEMMA_REMOTE_CODE_ALLOWLIST"
    
    return True, "default policy"


def get_model_dir() -> str:
    """
    Get the base directory for TranslateGemma models.
    
    Uses structured path: ComfyUI/models/LLM/TranslateGemma/ if LLM exists,
    otherwise ComfyUI/models/translate_gemma/
    
    Returns:
        Base directory path for model storage
    """
    models_dir = folder_paths.models_dir
    
    # Try LLM folder first (preferred)
    llm_dir = os.path.join(models_dir, "LLM", "TranslateGemma")
    if os.path.exists(os.path.join(models_dir, "LLM")) or not os.path.exists(os.path.join(models_dir, "translate_gemma")):
        os.makedirs(llm_dir, exist_ok=True)
        return llm_dir
    
    # Fallback to translate_gemma
    translate_dir = os.path.join(models_dir, "translate_gemma")
    os.makedirs(translate_dir, exist_ok=True)
    return translate_dir


def get_model_path(repo_id: str) -> str:
    """
    Get the local cache path for a specific model repo (TG-005: isolated per repo).
    
    Args:
        repo_id: HuggingFace repo ID (e.g., "google/translategemma-4b-it")
        
    Returns:
        Directory path for this specific model
    """
    base_dir = get_model_dir()
    # Convert repo_id to safe directory name (e.g., google/translategemma-4b-it -> translategemma-4b-it)
    repo_name = repo_id.split("/")[-1]
    model_dir = os.path.join(base_dir, repo_name)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def get_device() -> str:
    """Determine the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    import json
    
    if not os.path.isdir(local_dir):
        return False, ["directory does not exist"]
    
    missing = []
    
    # 1. Check required metadata
    required_metadata = ["config.json"]
    tokenizer_files = [
        "tokenizer_config.json",
        "processor_config.json", 
        "preprocessor_config.json",
    ]
    
    for f in required_metadata:
        if not os.path.exists(os.path.join(local_dir, f)):
            missing.append(f)
    
    # At least one tokenizer/processor config should exist
    tokenizer_found = any(
        os.path.exists(os.path.join(local_dir, f)) for f in tokenizer_files
    )
    if not tokenizer_found:
        missing.append("tokenizer/processor config (any of: " + ", ".join(tokenizer_files) + ")")
    
    # 2. Check weights completeness
    index_file = os.path.join(local_dir, "model.safetensors.index.json")
    
    if os.path.exists(index_file):
        # Sharded weights: parse index and verify all shards
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            
            weight_map = index_data.get("weight_map", {})
            shard_files = set(weight_map.values())
            
            for shard in shard_files:
                shard_path = os.path.join(local_dir, shard)
                if not os.path.exists(shard_path):
                    missing.append(f"shard: {shard}")
            
            if debug and shard_files:
                print(f"[TranslateGemma] Checking {len(shard_files)} shards from index")
                
        except (json.JSONDecodeError, KeyError, IOError) as e:
            missing.append(f"model.safetensors.index.json (parse error: {e})")
    else:
        # Single-file weights
        single_weight_files = ["model.safetensors", "pytorch_model.bin"]
        weight_found = any(
            os.path.exists(os.path.join(local_dir, f)) for f in single_weight_files
        )
        if not weight_found:
            # Also check for sharded bin files
            bin_index = os.path.join(local_dir, "pytorch_model.bin.index.json")
            if os.path.exists(bin_index):
                try:
                    with open(bin_index, "r", encoding="utf-8") as f:
                        index_data = json.load(f)
                    weight_map = index_data.get("weight_map", {})
                    shard_files = set(weight_map.values())
                    for shard in shard_files:
                        if not os.path.exists(os.path.join(local_dir, shard)):
                            missing.append(f"shard: {shard}")
                except Exception:
                    missing.append("weights (no valid weight files found)")
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
        print(f"[TranslateGemma] Snapshot complete in {local_dir}, skipping download")
        return True
    
    # Snapshot incomplete or missing
    if os.path.isdir(local_dir) and missing:
        print(f"[TranslateGemma] Incomplete snapshot detected, will resume download...")
        if debug:
            capped = missing[:3]
            suffix = f" (+{len(missing)-3} more)" if len(missing) > 3 else ""
            print(f"[TranslateGemma] Missing: {', '.join(capped)}{suffix}")
    
    # Try huggingface_hub.snapshot_download
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "[TranslateGemma] WARNING: huggingface_hub not available, "
            "falling back to transformers implicit download"
        )
        return False
    
    print(f"[TranslateGemma] Downloading {repo_id} to {local_dir}...")
    if revision:
        print(f"[TranslateGemma] Using revision: {revision}")
    
    try:
        # TG-027: Enable resume_download for interrupted downloads
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Windows-friendly
            revision=revision,
            resume_download=True,  # TG-027: Resume partial downloads
        )
        print(f"[TranslateGemma] Download complete: {local_dir}")
        
        # TG-027: Post-download verification
        is_complete_after, missing_after = _is_snapshot_complete(local_dir, debug=debug)
        if not is_complete_after:
            capped = missing_after[:5]
            raise RuntimeError(
                f"Download completed but snapshot still incomplete. "
                f"Missing: {', '.join(capped)}. "
                f"Try deleting '{local_dir}' and retrying."
            )
        
        return True
    except Exception as e:
        # Check for auth/gating errors
        error_str = str(e).lower()
        auth_keywords = ["401", "403", "gated", "forbidden", "unauthorized", "access denied"]
        if any(kw in error_str for kw in auth_keywords):
            raise RuntimeError(
                f"Failed to download model '{repo_id}' - access denied.\n\n"
                "To resolve:\n"
                "1. Visit the model page on Hugging Face and accept the license terms\n"
                "2. Authenticate with Hugging Face (one of):\n"
                "   - Run: hf auth login\n"
                "   - Or set: HF_TOKEN=your_token_here (or HUGGINGFACE_HUB_TOKEN)\n"
                "3. Restart ComfyUI\n\n"
                f"Original error: {e}"
            ) from e
        raise RuntimeError(f"Failed to download model '{repo_id}': {e}") from e


def unload_current_model():
    """
    Unload the currently loaded model and free memory (TG-004).
    
    This ensures only one model is in memory at a time.
    TG-029: Thread-safe with _MODEL_CACHE_LOCK.
    """
    global _current_model, _current_model_size
    
    with _MODEL_CACHE_LOCK:
        if _current_model is not None:
            model, processor_or_tokenizer = _current_model
            model_size = _current_model_size
            
            # Clear references
            _current_model = None
            _current_model_size = None
            del model
            del processor_or_tokenizer
            
            # Clean up memory
            cleanup_torch_memory()
            
            print(f"[TranslateGemma] {model_size} model unloaded, memory cleaned")


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
    cache_dir: str,
    revision: str = None,
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
        f"cache_dir: {cache_dir}",
    ]
    if revision:
        lines.append(f"revision: {revision}")
    return " | ".join(lines)

def _raise_auth_error(repo_id: str, original_error: Exception):
    """Raise user-friendly error for gated model access issues (TG-004)."""
    raise RuntimeError(
        f"Failed to access model '{repo_id}' - this may be a gated model.\n\n"
        "To resolve:\n"
        "1. Visit the model page on Hugging Face and accept the license terms\n"
        "2. Authenticate with Hugging Face (one of):\n"
        "   - Run: hf auth login\n"
        "   - Or set: HF_TOKEN=your_token_here (or HUGGINGFACE_HUB_TOKEN)\n"
        "3. Restart ComfyUI\n\n"
        f"Original error: {original_error}"
    )


def load_model(
    model_size: str,
    revision: Optional[str] = None,
) -> tuple:
    """
    Load TranslateGemma model and tokenizer with single-model cache (TG-004).
    
    Args:
        model_size: One of "4B", "12B", "27B"
        revision: Optional HuggingFace revision/commit hash for reproducibility (TG-006)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    global _current_model, _current_model_size
    
    # Check env var for revision override
    env_revision = os.environ.get("TRANSLATEGEMMA_REVISION")
    if env_revision and not revision:
        revision = env_revision
    
    # TG-029: Thread-safe cache check and switch
    with _MODEL_CACHE_LOCK:
        # Return cached model if same size is requested
        if _current_model is not None and _current_model_size == model_size:
            print(f"[TranslateGemma] Using cached {model_size} model")
            return _current_model
        
        # Unload current model before loading new one (single-model cache)
        if _current_model is not None:
            print(f"[TranslateGemma] Switching from {_current_model_size} to {model_size}")
        unload_current_model()
    
    repo_id = MODEL_REPOS.get(model_size)
    if not repo_id:
        raise ValueError(f"Invalid model size: {model_size}. Choose from: {list(MODEL_REPOS.keys())}")
    
    device = get_device()
    cache_dir = get_model_path(repo_id)  # TG-005: isolated per repo
    dtype = get_torch_dtype(device)
    
    print(f"[TranslateGemma] Loading {model_size} model from {repo_id}...")
    print(f"[TranslateGemma] Device: {device}, dtype: {dtype}")
    print(f"[TranslateGemma] Cache dir: {cache_dir}")
    if revision:
        print(f"[TranslateGemma] Revision: {revision}")

    try:
        if AutoProcessor is None or AutoModelForImageTextToText is None:
            raise RuntimeError(
                "TranslateGemma (Gemma 3) requires a newer transformers build that provides "
                "`AutoProcessor` and `AutoModelForImageTextToText` (recommended: transformers>=4.57)."
            )

        # TG-016: Explicit download before loading (Windows-friendly, skip if present)
        ensure_model_downloaded(repo_id, cache_dir, revision)

        # TG-006 + TG-026: Try without trust_remote_code first, fall back if policy allows
        remote_code_allowed, policy_reason = _is_remote_code_allowed(repo_id)
        
        try:
            processor = AutoProcessor.from_pretrained(
                repo_id,
                cache_dir=cache_dir,
                revision=revision,
                trust_remote_code=False,
            )
        except Exception as load_err:
            if not remote_code_allowed:
                raise RuntimeError(
                    f"Loading '{repo_id}' requires remote code, but policy denies it ({policy_reason}). "
                    f"Set TRANSLATEGEMMA_ALLOW_REMOTE_CODE=1 to allow, or add the repo to allowlist."
                ) from load_err
            print(f"[TranslateGemma] Loading processor with trust_remote_code=True (policy: {policy_reason})")
            processor = AutoProcessor.from_pretrained(
                repo_id,
                cache_dir=cache_dir,
                revision=revision,
                trust_remote_code=True,
            )

        # TG-017 + TG-026: Prefer safetensors, apply remote-code policy
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                repo_id,
                cache_dir=cache_dir,
                revision=revision,
                torch_dtype=dtype,
                device_map="auto" if device.startswith("cuda") else None,
                trust_remote_code=False,
                use_safetensors=True,
            )
        except Exception as e:
            if "safetensors" in str(e).lower():
                print("[TranslateGemma] safetensors not available, falling back to .bin")
            
            if not remote_code_allowed:
                raise RuntimeError(
                    f"Loading '{repo_id}' model weights requires remote code, but policy denies it ({policy_reason}). "
                    f"Set TRANSLATEGEMMA_ALLOW_REMOTE_CODE=1 to allow."
                ) from e
            
            print(f"[TranslateGemma] Loading model with trust_remote_code=True (policy: {policy_reason})")
            model = AutoModelForImageTextToText.from_pretrained(
                repo_id,
                cache_dir=cache_dir,
                revision=revision,
                torch_dtype=dtype,
                device_map="auto" if device.startswith("cuda") else None,
                trust_remote_code=True,
            )

        if not device.startswith("cuda"):
            model = model.to(device)

        # TG-007: Set model to eval mode for inference
        model.eval()

        # TG-029: Thread-safe cache write
        with _MODEL_CACHE_LOCK:
            # Cache the loaded model (single-model cache)
            _current_model = (model, processor)
            _current_model_size = model_size

        print(f"[TranslateGemma] {model_size} model loaded successfully on {device} (eval mode)")
        return model, processor

    except Exception as e:
        # Clean up any partially loaded resources
        cleanup_torch_memory()
        
        # TG-004: Detect gated/auth errors and provide actionable guidance
        if _detect_auth_error(e):
            _raise_auth_error(repo_id, e)
        
        # TG-008: Include context in error message
        context = _build_load_context(model_size, repo_id, device, dtype, cache_dir, revision)
        raise RuntimeError(f"Failed to load TranslateGemma: {e}\n[Context] {context}") from e


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
    return _current_model_size


def is_model_loaded() -> bool:
    """Check if a model is currently loaded."""
    return _current_model is not None
