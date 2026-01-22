import torch
import os
import tempfile
import time
import threading
from typing import Any
from ..utils.language_utils import get_language_names, get_language_code
from ..utils.language_detect import detect_source_lang_code
from ..utils.image_preprocess import (
    preprocess_for_translategemma,
    save_preprocessed_image,
    cleanup_temp_image,
)
from ..utils.chinese_postedit import postedit_traditional_chinese
from ..utils.chinese_convert import convert_chinese_variants, is_chinese_variant_code, infer_chinese_variant
from ..utils.model_loader import (
    load_model, get_available_models, unload_current_model, 
    cleanup_torch_memory, get_device, get_torch_dtype, get_model_path, MODEL_REPOS
)
from ..utils.prompt_builder import (
    PromptMode,
    render_prompt,
    probe_tokenizer_capabilities,
    build_structured_messages,
)
from ..utils.context_utils import compute_effective_limits, suggest_max_new_tokens, get_context_limit
from ..utils.template_guard import (
    assert_structured_generation_prompt,
    check_structured_generation_prompt,
    StructuredTruncationError,
)
from ..utils.runtime_constants import (
    VISION_TARGET_SIZE,
    GENERATION_HEADROOM_TOKENS,
    IMAGE_EXTRACTION_MAX_TOKENS,
    IMAGE_MIN_INPUT_TOKENS,
    REPEAT_GUARD_WINDOW,
    REPEAT_GUARD_MIN_GEN,
    STRUCTURED_TRUNCATION_MAX_ITERATIONS,
    STRUCTURED_TRUNCATION_ITERATION_BUFFER,
    TOKEN_COUNT_UNLIMITED,
    get_generation_heuristic_cap,
    get_repeat_guard_params,
)
from ..utils.debug_utils import (
    is_verbose_debug,
    redact_path,
    redact_text,
    StopReason,
    GenerationMeta,
    infer_stop_reason,
    log_stop_reason,
    log_runtime_capabilities_once,
    log_truncation_event,
    log_image_input_mode,
)
from ..utils.truncation_compat import (
    tokenizer_truncation_side,
    truncate_text_tokens_to_fit,
    TruncationMeta,
    log_truncation_if_occurred,
)


# TG-028: Debug privacy controls
# TG-041: Redaction functions moved to utils/debug_utils.py for consistency
_TOKENIZER_MUTATION_LOCK = threading.RLock()

def _clamp_int(value: int, min_val: int, max_val: int) -> int:
    return max(min_val, min(max_val, int(value)))


def _get_env_int_optional(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _suggest_auto_max_new_tokens(tokenizer, input_len: int) -> int:
    """
    Auto sizing for max_new_tokens=0.

    Uses the translation heuristic in utils/context_utils.py, but caps the
    suggestion to the remaining context budget (context_limit - input_len).

    Optional hard cap: TRANSLATEGEMMA_AUTO_MAX_NEW_TOKENS_MAX
    """
    context_limit = int(get_context_limit(tokenizer))
    available = max(int(context_limit) - int(input_len), 0)

    env_cap = _get_env_int_optional("TRANSLATEGEMMA_AUTO_MAX_NEW_TOKENS_MAX")
    if env_cap is not None:
        env_cap = _clamp_int(env_cap, 64, 8192)
        available = min(int(available), int(env_cap))

    return int(suggest_max_new_tokens(int(input_len), max_suggested=int(available)))


# TG-003 (Critical bug fix / Case T3):
# Do not truncate the full structured chat template produced by
# `processor.apply_chat_template(...)`. The official template (see
# `REFERENCE/translategemma4b/chat_template.jinja`) relies on tail markers like
# `<end_of_turn>` and the generation prompt `<start_of_turn>model\n`.
# If truncation removes those markers, the model may continue the user text
# (often in the source language), causing target-language drift.
#
# TG-042: Text truncation helper moved to utils/truncation_compat.py
# for encapsulation and reuse. See truncate_text_tokens_to_fit().


def _get_generation_eos_token_ids(tokenizer, debug: bool = False, include_end_of_turn: bool = True):
    """
    Best-effort EOS token ids for TranslateGemma chat templates.

    TranslateGemma often uses `<end_of_turn>` as the turn terminator. Some stacks
    may not emit the tokenizer's `eos_token_id` reliably for short completions,
    so we include `<end_of_turn>` as an additional stop token when available.
    """
    eos_ids: list[int] = []

    base_eos = getattr(tokenizer, "eos_token_id", None)
    if base_eos is not None:
        if isinstance(base_eos, (list, tuple)):
            eos_ids.extend([int(x) for x in base_eos if x is not None])
        else:
            eos_ids.append(int(base_eos))

    # Gemma / TranslateGemma chat templates typically use this marker.
    if include_end_of_turn:
        try:
            end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
            unk_id = getattr(tokenizer, "unk_token_id", None)
            if end_of_turn_id is not None and (unk_id is None or int(end_of_turn_id) != int(unk_id)):
                eos_ids.append(int(end_of_turn_id))
        except Exception:
            pass

    eos_ids = sorted(set(eos_ids))
    if debug and eos_ids:
        print(f"[TranslateGemma] Generation stop token ids: {eos_ids}")
    return eos_ids or None


def _build_degeneracy_stopping_criteria(prompt_len: int, debug: bool = False):
    """
    Add a lightweight guard to stop pathological repetitions (e.g., '리리리리...').

    This does not replace correct EOS handling; it is a safety net when the model
    fails to emit EOS/end-of-turn markers and starts looping.

    TG-040: Parameters are now configurable via environment variables:
    - TRANSLATEGEMMA_REPEAT_GUARD_WINDOW: detection window size (default 12)
    - TRANSLATEGEMMA_REPEAT_GUARD_MIN_GEN: min tokens before activation (default 32)
    """
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
    except Exception:
        return None

    # TG-040: Get configurable repeat guard parameters
    window, min_gen = get_repeat_guard_params()

    class _RepeatTokenStop(StoppingCriteria):
        def __init__(self, prompt_len: int, window: int, min_gen: int):
            self.prompt_len = int(prompt_len)
            self.window = int(window)
            self.min_gen = int(min_gen)
            self.triggered = False  # TG-041: Track if guard fired

        def __call__(self, input_ids, scores, **kwargs):
            if input_ids is None or input_ids.numel() == 0:
                return False
            ids = input_ids[0] if input_ids.dim() == 2 else input_ids
            gen_len = int(ids.shape[0]) - self.prompt_len
            if gen_len < self.min_gen:
                return False
            if int(ids.shape[0]) < self.window:
                return False
            tail = ids[-self.window :]
            if bool((tail == tail[0]).all().item()):
                self.triggered = True
                return True
            return False

    if debug:
        print(f"[TranslateGemma] Degeneracy stop enabled (repeat-token guard, window={window}, min_gen={min_gen})")
    return StoppingCriteriaList([_RepeatTokenStop(prompt_len=prompt_len, window=window, min_gen=min_gen)])


def _build_end_of_turn_stopping_criteria(tokenizer, prompt_len: int, min_gen: int, debug: bool = False):
    """
    Stop on <end_of_turn>, but only after at least `min_gen` tokens are generated.

    This is used as a fallback when the model emits <end_of_turn> immediately
    (resulting in empty decoded outputs).
    """
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
    except Exception:
        return None

    try:
        end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if end_of_turn_id is None or (unk_id is not None and int(end_of_turn_id) == int(unk_id)):
            return None
        end_of_turn_id = int(end_of_turn_id)
    except Exception:
        return None

    class _EndOfTurnMinGenStop(StoppingCriteria):
        def __init__(self, prompt_len: int, min_gen: int, end_of_turn_id: int):
            self.prompt_len = int(prompt_len)
            self.min_gen = int(min_gen)
            self.end_of_turn_id = int(end_of_turn_id)

        def __call__(self, input_ids, scores, **kwargs):
            if input_ids is None or input_ids.numel() == 0:
                return False
            ids = input_ids[0] if input_ids.dim() == 2 else input_ids
            gen_len = int(ids.shape[0]) - self.prompt_len
            if gen_len < self.min_gen:
                return False
            return bool(int(ids[-1].item()) == self.end_of_turn_id)

    if debug:
        print(f"[TranslateGemma] End-of-turn stop enabled (min_gen={int(min_gen)})")
    return StoppingCriteriaList(
        [_EndOfTurnMinGenStop(prompt_len=int(prompt_len), min_gen=int(min_gen), end_of_turn_id=end_of_turn_id)]
    )


class TranslateGemmaNode:
    """
    ComfyUI node for translating text using TranslateGemma models.
    Supports 4B, 12B, and 27B model sizes with 55 languages.
    
    Prompt Modes:
    - auto (default): Try structured format first, fall back to plain on failure
    - structured: Force structured chat-template format (fail loudly if unsupported)
    - plain: Force plain instruction format (no chat template)
    
    Token Limits (TG-003):
    - max_new_tokens: Control output length (0 = auto)
    - max_input_tokens: Control input truncation
    - truncate_input: Enable/disable input truncation
    - strict_context_limit: Clamp output to fit context window
    
    Memory Management (TG-004):
    - keep_model_loaded: Keep model in memory between runs (default True)
    - Single-model cache: Only one model in memory at a time
    """
    
    CATEGORY = "text/translation"
    FUNCTION = "translate"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translated_text",)

    # Avoid ComfyUI per-item slicing (`slice_dict`) so we can safely normalize
    # edge-cases like connected inputs producing `[]` (empty batch) instead of
    # crashing the whole workflow before this node runs.
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    
    @classmethod
    def INPUT_TYPES(cls):
        languages = get_language_names()
        model_sizes = get_available_models()
        prompt_modes = PromptMode.get_values()
        
        return {
            "required": {
                "text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Enter text to translate...",
                    "tooltip": "Built-in text input. Ignored when `external_text` is connected. Empty/whitespace returns empty output.",
                }),
                "target_language": (languages, {
                    "default": "English",
                    "tooltip": "Translation target language. Does not affect `chinese_conversion_only=true` (direction is controlled by `chinese_conversion_direction`).",
                }),
                "model_size": (model_sizes, {
                    "default": "4B",
                    "tooltip": "Model size: 4B (fastest) / 12B / 27B trade-off (speed vs quality vs VRAM). Gated HF repos require accepting Gemma terms + authentication (`hf auth login` or HF token env var).",
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "forceInput": True,
                    "tooltip": "If connected, uses multimodal path to translate text from the image. Requires explicit `source_language` (Auto Detect is not supported for images).",
                }),
                "image_enhance": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Mild contrast/sharpening to help small text visibility; may introduce artifacts on some images.",
                }),
                "image_resize_mode": (["letterbox", "processor", "stretch"], {
                    "default": "letterbox",
                    "tooltip": "letterbox: preserve aspect ratio (pad to 896×896, recommended). processor: official resize (may stretch). stretch: force 896×896 (may distort).",
                }),
                "image_two_pass": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Extract text from image first (source→source), then translate extracted text (more accurate, slower).",
                }),
                "source_language": (["Auto Detect"] + languages, {
                    "default": "Auto Detect",
                    "tooltip": "Auto Detect is supported for text only. Images require explicit source language. If you get wrong-language behavior, set this explicitly.",
                }),
                "external_text": ("STRING", {
                    "forceInput": True,
                    "tooltip": "When connected, overrides `text` even if empty. Intended for chaining from other nodes.",
                }),
                "prompt_mode": (prompt_modes, {
                    "default": "auto",
                    "tooltip": "auto: structured first, fallback to plain. structured: fail loudly if chat template unavailable. plain: instruction-only (no chat template).",
                }),
                "max_new_tokens": ("INT", {
                    "default": 512,
                    "min": 0,
                    "max": 8192,
                    "step": 16,
                    # NOTE: Some ComfyUI builds may ignore per-input labels and only show the key name.
                    # Tooltips still work on hover in supported UIs.
                    "label": "max_new_tokens (0 = auto)",
                    "display_name": "max_new_tokens (0 = auto)",
                    "tooltip": "0 = Auto. Maximum output tokens. Higher values allow longer outputs but increase latency; output is also clamped by the model context window.",
                }),
                "max_input_tokens": ("INT", {
                    "default": 2048,
                    "min": 0,
                    "max": 8192,
                    "step": 64,
                    "label": "max_input_tokens (0 = auto)",
                    "display_name": "max_input_tokens (0 = auto)",
                    "tooltip": "0 = Auto. Input truncation limit (reserves room for output). Too low can break multimodal inputs/templates. Recommended 2048+ for long documents.",
                }),
                "truncate_input": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Truncate input if it exceeds max_input_tokens. Disable may cause OOM on long texts.",
                }),
                "strict_context_limit": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clamp output so input+output stays within model context window.",
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model in memory between runs for faster repeated use; may keep VRAM allocated.",
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable debug logging. Sensitive data is redacted by default; set TRANSLATEGEMMA_VERBOSE_DEBUG=1 for full details.",
                }),
                "chinese_conversion_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "OpenCC conversion only (Simplified↔Traditional) without loading the model. Text-only; image not supported.",
                }),
                "chinese_conversion_direction": (["auto_flip", "to_traditional", "to_simplified"], {
                    "default": "auto_flip",
                    "tooltip": "auto_flip: detect input variant and convert to opposite. to_traditional: force Simplified→Traditional. to_simplified: force Traditional→Simplified. Returns error if input is ambiguous with auto_flip.",
                }),
            },
        }

    @staticmethod
    def _slice_or_default(values, i: int, default=None):
        if isinstance(values, (list, tuple)):
            if len(values) == 0:
                return default
            return values[i] if i < len(values) else values[-1]
        return values

    @staticmethod
    def _list_max_len(*values) -> int:
        lengths = [len(v) for v in values if isinstance(v, (list, tuple))]
        return max(lengths) if lengths else 0
    
    def translate(
        self,
        text: Any,
        target_language: Any,
        model_size: Any,
        image=None,
        image_enhance: Any = False,
        image_resize_mode: Any = "letterbox",
        image_two_pass: Any = True,
        source_language: Any = "Auto Detect",
        external_text: Any = None,
        prompt_mode: Any = "auto",
        max_new_tokens: Any = 512,
        max_input_tokens: Any = 2048,
        truncate_input: Any = True,
        strict_context_limit: Any = True,
        keep_model_loaded: Any = True,
        debug: Any = False,
        chinese_conversion_only: Any = False,
        chinese_conversion_direction: Any = "auto_flip",
    ) -> tuple[list[str]]:
        batch = self._list_max_len(
            text,
            target_language,
            model_size,
            image,
            image_enhance,
            image_resize_mode,
            image_two_pass,
            source_language,
            external_text,
            prompt_mode,
            max_new_tokens,
            max_input_tokens,
            truncate_input,
            strict_context_limit,
            keep_model_loaded,
            debug,
            chinese_conversion_only,
            chinese_conversion_direction,
        )
        batch = max(int(batch), 1)

        results: list[str] = []
        for i in range(batch):
            one_text = self._slice_or_default(text, i, "")
            one_text = "" if one_text is None else str(one_text)

            one_target_language = self._slice_or_default(target_language, i, "English")
            one_target_language = "English" if one_target_language is None else str(one_target_language)

            one_model_size = self._slice_or_default(model_size, i, "4B")
            one_model_size = "4B" if one_model_size is None else str(one_model_size)

            one_image = self._slice_or_default(image, i, None)

            raw_image_enhance = self._slice_or_default(image_enhance, i, False)
            one_image_enhance = bool(raw_image_enhance) if raw_image_enhance is not None else False

            raw_image_resize_mode = self._slice_or_default(image_resize_mode, i, "letterbox")
            one_image_resize_mode = "letterbox" if raw_image_resize_mode is None else str(raw_image_resize_mode)

            raw_image_two_pass = self._slice_or_default(image_two_pass, i, True)
            one_image_two_pass = bool(raw_image_two_pass) if raw_image_two_pass is not None else True

            raw_source_language = self._slice_or_default(source_language, i, "Auto Detect")
            one_source_language = "Auto Detect" if raw_source_language is None else str(raw_source_language)

            # Keep None when the input is not connected; normalize connected empty lists to "".
            one_external_text = self._slice_or_default(external_text, i, "")
            if one_external_text is not None:
                one_external_text = str(one_external_text)

            raw_prompt_mode = self._slice_or_default(prompt_mode, i, "auto")
            one_prompt_mode = "auto" if raw_prompt_mode is None else str(raw_prompt_mode)

            raw_max_new_tokens = self._slice_or_default(max_new_tokens, i, 512)
            one_max_new_tokens = int(raw_max_new_tokens) if raw_max_new_tokens is not None else 512

            raw_max_input_tokens = self._slice_or_default(max_input_tokens, i, 2048)
            one_max_input_tokens = int(raw_max_input_tokens) if raw_max_input_tokens is not None else 2048

            raw_truncate_input = self._slice_or_default(truncate_input, i, True)
            one_truncate_input = bool(raw_truncate_input) if raw_truncate_input is not None else True

            raw_strict_context_limit = self._slice_or_default(strict_context_limit, i, True)
            one_strict_context_limit = bool(raw_strict_context_limit) if raw_strict_context_limit is not None else True

            raw_keep_model_loaded = self._slice_or_default(keep_model_loaded, i, True)
            one_keep_model_loaded = bool(raw_keep_model_loaded) if raw_keep_model_loaded is not None else True

            raw_debug = self._slice_or_default(debug, i, False)
            one_debug = bool(raw_debug) if raw_debug is not None else False

            raw_chinese_conversion_only = self._slice_or_default(chinese_conversion_only, i, False)
            one_chinese_conversion_only = bool(raw_chinese_conversion_only) if raw_chinese_conversion_only is not None else False

            raw_chinese_conversion_direction = self._slice_or_default(
                chinese_conversion_direction, i, "auto_flip"
            )
            one_chinese_conversion_direction = (
                "auto_flip" if raw_chinese_conversion_direction is None else str(raw_chinese_conversion_direction)
            )

            translated_text = self._translate_one(
                text=one_text,
                target_language=one_target_language,
                model_size=one_model_size,
                image=one_image,
                image_enhance=one_image_enhance,
                image_resize_mode=one_image_resize_mode,
                image_two_pass=one_image_two_pass,
                source_language=one_source_language,
                external_text=one_external_text,
                prompt_mode=one_prompt_mode,
                max_new_tokens=one_max_new_tokens,
                max_input_tokens=one_max_input_tokens,
                truncate_input=one_truncate_input,
                strict_context_limit=one_strict_context_limit,
                keep_model_loaded=one_keep_model_loaded,
                debug=one_debug,
                chinese_conversion_only=one_chinese_conversion_only,
                chinese_conversion_direction=one_chinese_conversion_direction,
            )[0]
            results.append(translated_text)

        return (results,)

    def _translate_one(
        self,
        text: str,
        target_language: str,
        model_size: str,
        image=None,
        image_enhance: bool = False,
        image_resize_mode: str = "letterbox",
        image_two_pass: bool = True,
        source_language: str = "Auto Detect",
        external_text: str | None = None,
        prompt_mode: str = "auto",
        max_new_tokens: int = 512,
        max_input_tokens: int = 2048,
        truncate_input: bool = True,
        strict_context_limit: bool = True,
        keep_model_loaded: bool = True,
        debug: bool = False,
        chinese_conversion_only: bool = False,
        chinese_conversion_direction: str = "auto_flip",
        _retry_plain_on_empty: bool = True,
        _eot_relaxed_retry: bool = False,
    ) -> tuple[str]:
        """
        Translate text to target language using TranslateGemma.
        
        Args:
            text: Text from the built-in input field
            target_language: Target language name
            model_size: Model size (4B, 12B, 27B)
            image: Optional image input (TG-025)
            image_enhance: Apply enhancement for small text in images (TG-025)
            image_resize_mode: Image preprocessing mode (TG-025)
            image_two_pass: Two-pass image translation (extract → translate) (TG-025)
            source_language: Source language name or "Auto Detect"
            external_text: Optional external text input (overrides built-in text)
            prompt_mode: Prompt construction mode (auto/structured/plain)
            max_new_tokens: Maximum output tokens (TG-003)
            max_input_tokens: Maximum input tokens before truncation (TG-003)
            truncate_input: Whether to truncate long inputs (TG-003)
            strict_context_limit: Clamp output to fit context (TG-003)
            debug: Enable debug logging
            chinese_conversion_only: Use OpenCC for Chinese conversion only (TG-038)
            chinese_conversion_direction: Conversion direction for TG-038 (auto_flip/to_traditional/to_simplified)

        Returns:
            Tuple containing translated text
        """
        # TG-038: Fast-path for Chinese conversion-only mode (no model load)
        if chinese_conversion_only:
            return self._convert_chinese_only(
                text=text,
                external_text=external_text,
                image=image,
                conversion_direction=chinese_conversion_direction,
                debug=debug,
            )

        if image is not None:
            return self._translate_image(
                image=image,
                target_language=target_language,
                model_size=model_size,
                source_language=source_language,
                image_enhance=image_enhance,
                image_resize_mode=image_resize_mode,
                image_two_pass=image_two_pass,
                max_new_tokens=max_new_tokens,
                max_input_tokens=max_input_tokens,
                truncate_input=truncate_input,
                strict_context_limit=strict_context_limit,
                keep_model_loaded=keep_model_loaded,
                debug=debug,
            )

        # TG-009: Use external text if connected (is not None), otherwise use built-in text
        input_text = external_text if external_text is not None else text
        
        if not input_text or not input_text.strip():
            return ("",)
        
        target_code = get_language_code(target_language)
        # TranslateGemma's official chat template requires an explicit source_lang_code
        # (the model card does not define "auto"). For Auto Detect, do local detection.
        if source_language != "Auto Detect":
            source_code = get_language_code(source_language)
        else:
            source_code = detect_source_lang_code(input_text)
            if debug:
                print(f"[TranslateGemma] Auto-detected source_lang_code={source_code}")

        # Heuristic warning for Chinese script variants: users often set the wrong source variant
        # (e.g. choose Traditional but provide Simplified input), which can lead to "no-op" output
        # or script mismatch. Only warn when the user explicitly selected a Chinese variant.
        if source_language != "Auto Detect" and source_code in {"zh", "zh_Hant"}:
            detected_variant = detect_source_lang_code(input_text, fallback=source_code)
            if detected_variant in {"zh", "zh_Hant"} and detected_variant != source_code:
                print(
                    "[TranslateGemma] WARNING: source_language may not match the input script. "
                    f"selected_source_lang_code={source_code}, detected_source_lang_code={detected_variant}. "
                    "If you want Simplified↔Traditional conversion, set source_language=Auto Detect (recommended) "
                    "or select the correct Chinese variant explicitly."
                )
        
        # Load model and processor/tokenizer
        t_load_start = time.time()
        model, processor = load_model(model_size)
        t_load_s = time.time() - t_load_start
        tokenizer = getattr(processor, "tokenizer", processor)

        # Resolve max_input_tokens=0 (Auto): reserve room for output within context.
        if int(max_input_tokens) == 0:
            context_limit = get_context_limit(tokenizer)
            reserved_for_output = (
                int(max_new_tokens) if int(max_new_tokens) > 0 else 1024
            )
            reserved_for_output = min(int(reserved_for_output), int(context_limit) - 1)
            resolved_max_input_tokens = max(int(context_limit) - int(reserved_for_output) - 1, 64)
            if debug:
                print(
                    f"[TranslateGemma] max_input_tokens=0 (Auto) -> "
                    f"resolved_max_input_tokens={resolved_max_input_tokens} "
                    f"(context_limit={context_limit}, reserved_for_output={reserved_for_output})"
                )
            max_input_tokens = int(resolved_max_input_tokens)

        # Pick a safe input device (for device_map models, use the first parameter device).
        try:
            input_device = next(model.parameters()).device
        except StopIteration:
            input_device = getattr(model, "device", "cpu")

        # Preferred compute dtype for floating inputs (pixel values, embeddings, etc).
        # Keep consistent with utils/model_loader.py selection.
        device = get_device()
        dtype = get_torch_dtype(device)

        # TG-041: Log runtime capabilities once per process
        runtime_caps = log_runtime_capabilities_once(processor, tokenizer, debug=debug)

        if debug:
            cuda_available = torch.cuda.is_available()
            cuda_name = None
            if cuda_available:
                try:
                    cuda_name = torch.cuda.get_device_name(torch.cuda.current_device())
                except Exception:
                    cuda_name = "unknown"

            hf_device_map = getattr(model, "hf_device_map", None)
            print(
                "[TranslateGemma] Runtime info: "
                f"torch={getattr(torch, '__version__', 'unknown')}, "
                f"cuda_available={cuda_available}, cuda_device={cuda_name}, "
                f"model_input_device={input_device}, "
                f"hf_device_map={'present' if hf_device_map else 'none'}, "
                f"dtype={dtype}, load_time_s={t_load_s:.2f}"
            )
            print(
                "[TranslateGemma] Language codes: "
                f"source_lang_code={source_code}, target_lang_code={target_code}"
            )
        
        # Prompt mode handling
        try:
            mode = PromptMode(prompt_mode or "auto")
        except ValueError:
            print(f"[TranslateGemma] Invalid prompt_mode '{prompt_mode}', falling back to 'auto'")
            mode = PromptMode.AUTO

        # Prefer the official structured processor.apply_chat_template path for text translation.
        # Ref: REFERENCE/translategemma4b/README.md ("With direct initialization")
        used_path = "unknown"
        inputs = None
        raw_input_len = None
        actual_input_len = None
        try:
            if mode != PromptMode.PLAIN and hasattr(processor, "apply_chat_template"):
                kwargs_base = dict(
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )

                # IMPORTANT: Avoid truncating the whole templated prompt. If truncation cuts off the
                # end-of-turn + generation prompt tokens, the model may continue the user's text
                # (often in the source language), causing target-language drift (Case T3).
                # Instead, truncate only the user-provided text so the template wrapper remains intact.
                with _TOKENIZER_MUTATION_LOCK:
                    overhead_messages = build_structured_messages(
                        input_text="",
                        target_lang_code=target_code,
                        source_lang_code=source_code,
                    )
                    overhead_inputs = processor.apply_chat_template(overhead_messages, **kwargs_base)
                    overhead_len = int(overhead_inputs["input_ids"].shape[1])
                    available_for_text = int(max_input_tokens) - overhead_len

                    # TG-040: Use TOKEN_COUNT_UNLIMITED sentinel for "no limit" probing
                    _, raw_text_tokens, _ = truncate_text_tokens_to_fit(
                        tokenizer=tokenizer,
                        text=input_text,
                        max_tokens=TOKEN_COUNT_UNLIMITED,
                    )
                    raw_input_len = overhead_len + int(raw_text_tokens)

                    if truncate_input and int(max_input_tokens) <= overhead_len:
                        raise RuntimeError(
                            "max_input_tokens is too low for the official structured chat template. "
                            f"(max_input_tokens={max_input_tokens}, template_overhead_tokens={overhead_len}) "
                            "Increase max_input_tokens or use prompt_mode=plain."
                        )

                    truncated_text = input_text
                    if truncate_input and raw_input_len > int(max_input_tokens) and available_for_text > 0:
                        truncated_text, _, _ = truncate_text_tokens_to_fit(
                            tokenizer=tokenizer,
                            text=input_text,
                            max_tokens=available_for_text,
                        )

                    # TG-042: Structured truncation debug logging moved to after guard check
                    # to include guard_outcome in the unified log format

                    # TG-040: Use named constants for truncation loop parameters
                    for _ in range(STRUCTURED_TRUNCATION_MAX_ITERATIONS):
                        messages = build_structured_messages(
                            input_text=truncated_text,
                            target_lang_code=target_code,
                            source_lang_code=source_code,
                        )
                        inputs = processor.apply_chat_template(messages, **kwargs_base)
                        actual_input_len = int(inputs["input_ids"].shape[1])
                        if actual_input_len <= int(max_input_tokens) or not truncate_input:
                            break
                        overflow = actual_input_len - int(max_input_tokens)
                        # TG-040: Use named constant for iteration buffer
                        available_for_text = max(available_for_text - overflow - STRUCTURED_TRUNCATION_ITERATION_BUFFER, 0)
                        if available_for_text <= 0:
                            break
                        truncated_text, _, _ = truncate_text_tokens_to_fit(
                            tokenizer=tokenizer,
                            text=input_text,
                            max_tokens=available_for_text,
                        )
                used_path = "processor.apply_chat_template(structured)"

                # TG-034: Validate structured generation prompt wrapper
                # Only run guard when truncation occurred (raw > actual)
                guard_outcome = None
                if truncate_input and raw_input_len and actual_input_len and raw_input_len > actual_input_len:
                    input_ids = inputs["input_ids"] if inputs else None
                    wrapper_ok = check_structured_generation_prompt(tokenizer, input_ids) if input_ids is not None else False
                    if not wrapper_ok:
                        if mode == PromptMode.STRUCTURED:
                            # Strict mode: fail loudly
                            guard_outcome = "error"
                            raise StructuredTruncationError(
                                "Structured truncation removed the required generation wrapper. "
                                "Increase max_input_tokens, disable truncation, or use prompt_mode=plain."
                            )
                        else:
                            # Auto mode: warn and fallback to plain
                            guard_outcome = "fallback"
                            print(
                                "[TranslateGemma] WARNING (TG-034): Structured truncation removed "
                                "required template markers. Falling back to plain prompt."
                            )
                            inputs = None  # Trigger fallback
                            used_path = "unknown"
                    else:
                        guard_outcome = "pass"

                    # TG-042: Log structured truncation with unified format
                    log_truncation_event(
                        path_used="structured",
                        raw_input_tokens=raw_input_len,
                        actual_input_tokens=actual_input_len,
                        max_input_tokens=max_input_tokens,
                        overhead_tokens=overhead_len,
                        guard_outcome=guard_outcome,
                        debug=True,  # Always log truncation events (quality-impacting)
                    )
                    print(
                        "[TranslateGemma] WARNING: Input was truncated to fit max_input_tokens. "
                        "Translation may be incomplete."
                    )
        except StructuredTruncationError:
            raise  # Re-raise guard errors
        except Exception as e:
            if mode == PromptMode.STRUCTURED:
                raise RuntimeError(
                    "Structured mode requested but processor.apply_chat_template failed."
                ) from e
            inputs = None

        if inputs is None:
            # Fallback to prompt-string rendering (older/unsupported stacks).
            if debug and mode != PromptMode.PLAIN:
                capabilities = probe_tokenizer_capabilities(tokenizer, debug=True)
                print(f"[TranslateGemma] Tokenizer capabilities: {capabilities}")

            prompt = render_prompt(
                tokenizer=tokenizer,
                input_text=input_text,
                target_lang_code=target_code,
                source_lang_code=source_code,
                mode=mode,
                debug=debug,
            )

            if debug:
                print(f"[TranslateGemma] Prompt rendered via fallback, mode={mode.value}, length={len(prompt)}")
                # TG-028: Redact prompt content unless verbose debug enabled
                if is_verbose_debug():
                    preview = prompt[:50] + "..." + prompt[-50:] if len(prompt) > 100 else prompt
                    print(f"[TranslateGemma] Prompt preview: {preview}")

            if truncate_input:
                with _TOKENIZER_MUTATION_LOCK:
                    old_truncation_side = getattr(tokenizer, "truncation_side", None)
                    if old_truncation_side and old_truncation_side != "right":
                        if debug:
                            print(
                                "[TranslateGemma] Forcing tokenizer.truncation_side='right' "
                                "(to preserve instruction tokens under truncation)"
                            )
                        tokenizer.truncation_side = "right"
                    try:
                        inputs = tokenizer(
                            prompt,
                            return_tensors="pt",
                            truncation=True,
                            max_length=max_input_tokens,
                        )
                        actual_input_len = int(inputs["input_ids"].shape[1])
                        raw_input_len = actual_input_len

                        if actual_input_len == int(max_input_tokens):
                            probe_max = int(max_input_tokens) + 1024
                            probe_result = tokenizer(
                                prompt,
                                return_tensors="pt",
                                truncation=True,
                                max_length=probe_max,
                            )
                            raw_input_len = int(probe_result["input_ids"].shape[1])
                    finally:
                        if old_truncation_side and old_truncation_side != tokenizer.truncation_side:
                            tokenizer.truncation_side = old_truncation_side
            else:
                inputs = tokenizer(prompt, return_tensors="pt")
                actual_input_len = int(inputs["input_ids"].shape[1])
                raw_input_len = actual_input_len

            used_path = "tokenizer(prompt)"

        actual_input_len = actual_input_len or int(inputs["input_ids"].shape[1])
        raw_input_len = raw_input_len or actual_input_len
        if debug:
            print(
                f"[TranslateGemma] Text path: {used_path}, "
                f"raw_input_tokens={raw_input_len}, actual_input_tokens={actual_input_len}"
            )

        # TG-042: Use unified truncation logging format
        if truncate_input and raw_input_len and actual_input_len and int(raw_input_len) > int(actual_input_len):
            # Structured path is already logged above (with overhead + guard outcome).
            # Only log here for plain fallback tokenization.
            if used_path == "tokenizer(prompt)":
                log_truncation_event(
                    path_used="plain",
                    raw_input_tokens=raw_input_len,
                    actual_input_tokens=actual_input_len,
                    max_input_tokens=max_input_tokens,
                    overhead_tokens=None,
                    guard_outcome=None,
                    debug=True,  # Always log truncation events (quality-impacting)
                )
                print(
                    "[TranslateGemma] WARNING: Input was truncated to fit max_input_tokens. "
                    "Translation may be incomplete."
                )

        requested_max_new_tokens = max_new_tokens
        if int(max_new_tokens) == 0:
            requested_max_new_tokens = _suggest_auto_max_new_tokens(tokenizer, int(actual_input_len))
            if debug:
                print(
                    f"[TranslateGemma] max_new_tokens=0 (Auto) -> "
                    f"requested_max_new_tokens={requested_max_new_tokens}"
                )

        # Compute effective limits (TG-003)
        limits = compute_effective_limits(
            tokenizer=tokenizer,
            input_len=int(actual_input_len),
            max_new_tokens=int(requested_max_new_tokens),
            max_input_tokens=max_input_tokens,
            strict_context_limit=strict_context_limit,
            truncate_input=truncate_input,
            debug=debug,
        )

        # TG-040: Extra guard for translation tasks: short inputs rarely need huge generation windows.
        # This helps reduce repetitive continuations when the model does not emit EOS early.
        # Use get_generation_heuristic_cap() which supports env var overrides for long-form translation.
        heuristic_cap = int(get_generation_heuristic_cap(actual_input_len))
        before_heuristic = int(limits["effective_max_new_tokens"])
        limits["effective_max_new_tokens"] = min(before_heuristic, heuristic_cap)
        if debug:
            if int(limits["effective_max_new_tokens"]) < before_heuristic:
                print(
                    f"[TranslateGemma] Generation heuristic cap applied: "
                    f"cap={heuristic_cap}, effective_max_new_tokens={before_heuristic}->{limits['effective_max_new_tokens']}"
                )
            else:
                print(
                    f"[TranslateGemma] Generation heuristic cap: "
                    f"cap={heuristic_cap}, effective_max_new_tokens={limits['effective_max_new_tokens']}"
                )
        
        # Check if we have room for generation
        if limits["effective_max_new_tokens"] == 0:
            warning_msg = (
                f"[TranslateGemma] Input too long ({raw_input_len} tokens) - "
                f"no room for generation within context limit ({limits['context_limit']})"
            )
            print(warning_msg)
            return (f"[Error: Input too long - {raw_input_len} tokens exceeds context limit]",)
        
        # Move tensors to device (do not cast int tensors to dtype).
        moved_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    moved_inputs[k] = v.to(input_device, dtype=dtype)
                else:
                    moved_inputs[k] = v.to(input_device)
            else:
                moved_inputs[k] = v
        inputs = moved_inputs
        
        # TG-007: Defensive model.eval() (idempotent) + inference_mode for best practices
        model.eval()
        
        # Build context for error messages (TG-008)
        # Include repo_id/device/dtype/cache_dir as per TG-008 requirements
        repo_id = MODEL_REPOS.get(model_size) or "unknown"
        cache_dir = get_model_path(repo_id) if repo_id != "unknown" else "unknown"
        inference_context = (
            f"model_size: {model_size} | repo_id: {repo_id} | "
            f"device: {device} | dtype: {dtype} | cache_dir: {cache_dir} | "
            f"user_max_new_tokens: {max_new_tokens} | "
            f"requested_max_new_tokens: {requested_max_new_tokens} | "
            f"effective_max_new_tokens: {limits['effective_max_new_tokens']} | "
            f"raw_input_tokens: {raw_input_len} | actual_input_tokens: {actual_input_len} | "
            f"truncate: {truncate_input} | strict_limit: {strict_context_limit}"
        )
        
        # Generate translation with effective max_new_tokens (TG-003)
        # Use try/except/finally for error handling and guaranteed cleanup (TG-004/008/018)
        try:
            with torch.inference_mode():  # TG-007: Stronger than no_grad()
                gen_kwargs = dict(
                    **inputs,
                    max_new_tokens=limits["effective_max_new_tokens"],
                    do_sample=False,
                    use_cache=True,  # TG-007: Explicit for performance
                )
                # If we previously observed "empty output" due to immediate <end_of_turn>,
                # retry once while *not* stopping on <end_of_turn> as an EOS token. Instead,
                # stop on <end_of_turn> only after a small minimum generation length.
                eos_ids = _get_generation_eos_token_ids(
                    tokenizer,
                    debug=debug,
                    include_end_of_turn=(not _eot_relaxed_retry),
                )
                if eos_ids:
                    gen_kwargs["eos_token_id"] = eos_ids if len(eos_ids) > 1 else eos_ids[0]
                    gen_kwargs["pad_token_id"] = int(getattr(tokenizer, "eos_token_id", eos_ids[0]) or eos_ids[0])

                stopping_criteria = _build_degeneracy_stopping_criteria(
                    prompt_len=int(inputs["input_ids"].shape[1]),
                    debug=debug,
                )
                if _eot_relaxed_retry:
                    eot_criteria = _build_end_of_turn_stopping_criteria(
                        tokenizer=tokenizer,
                        prompt_len=int(inputs["input_ids"].shape[1]),
                        min_gen=8,
                        debug=debug,
                    )
                    if eot_criteria is not None:
                        if stopping_criteria is None:
                            stopping_criteria = eot_criteria
                        else:
                            stopping_criteria.extend(eot_criteria)
                if stopping_criteria is not None:
                    gen_kwargs["stopping_criteria"] = stopping_criteria

                t_gen_start = time.time()
                outputs = model.generate(**gen_kwargs)
                t_gen_elapsed = time.time() - t_gen_start

                # TG-041: Infer and log stop reason
                prompt_len = int(inputs["input_ids"].shape[1])
                end_of_turn_id = None
                try:
                    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
                    unk_id = getattr(tokenizer, "unk_token_id", None)
                    if unk_id is not None and end_of_turn_id == unk_id:
                        end_of_turn_id = None
                except Exception:
                    pass

                gen_meta = infer_stop_reason(
                    output_ids=outputs[0] if outputs is not None else None,
                    prompt_len=prompt_len,
                    max_new_tokens=limits["effective_max_new_tokens"],
                    eos_token_ids=eos_ids or [],
                    end_of_turn_id=end_of_turn_id,
                    stopping_criteria=stopping_criteria,
                )

                if debug:
                    print(f"[TranslateGemma] Generation time_s={t_gen_elapsed:.2f}")
                log_stop_reason(gen_meta, debug=debug)

            # Decode output
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            if hasattr(processor, "decode"):
                translated_text = processor.decode(generated_ids, skip_special_tokens=True)
            else:
                translated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            translated_text = postedit_traditional_chinese(
                translated_text.strip(), target_lang_code=target_code, debug=debug
            )
            translated_text = translated_text.strip()

            # TG-047: Guard against empty model outputs (e.g., immediate <end_of_turn>).
            # This can happen for some language pairs/prompts and will silently break downstream nodes.
            if not translated_text:
                # Only auto-retry in auto mode; structured mode should remain strict.
                if mode == PromptMode.AUTO and _retry_plain_on_empty:
                    if not _eot_relaxed_retry:
                        print(
                            "[TranslateGemma] WARNING: Model returned empty output "
                            f"(stop_reason={gen_meta.stop_reason.value}, generated_tokens={gen_meta.generated_tokens}). "
                            "Retrying with relaxed <end_of_turn> stopping."
                        )
                        return self._translate_one(
                            text=text,
                            target_language=target_language,
                            model_size=model_size,
                            image=image,
                            image_enhance=image_enhance,
                            image_resize_mode=image_resize_mode,
                            image_two_pass=image_two_pass,
                            source_language=source_language,
                            external_text=external_text,
                            prompt_mode=PromptMode.AUTO.value,
                            max_new_tokens=max_new_tokens,
                            max_input_tokens=max_input_tokens,
                            truncate_input=truncate_input,
                            strict_context_limit=strict_context_limit,
                            keep_model_loaded=keep_model_loaded,
                            debug=debug,
                            chinese_conversion_only=chinese_conversion_only,
                            chinese_conversion_direction=chinese_conversion_direction,
                            _retry_plain_on_empty=True,
                            _eot_relaxed_retry=True,
                        )
                    print(
                        "[TranslateGemma] WARNING: Model returned empty output "
                        f"(stop_reason={gen_meta.stop_reason.value}, generated_tokens={gen_meta.generated_tokens}). "
                        "Retrying with prompt_mode=plain."
                    )
                    return self._translate_one(
                        text=text,
                        target_language=target_language,
                        model_size=model_size,
                        image=image,
                        image_enhance=image_enhance,
                        image_resize_mode=image_resize_mode,
                        image_two_pass=image_two_pass,
                        source_language=source_language,
                        external_text=external_text,
                        prompt_mode=PromptMode.PLAIN.value,
                        max_new_tokens=max_new_tokens,
                        max_input_tokens=max_input_tokens,
                        truncate_input=truncate_input,
                        strict_context_limit=strict_context_limit,
                        keep_model_loaded=keep_model_loaded,
                        debug=debug,
                        chinese_conversion_only=chinese_conversion_only,
                        chinese_conversion_direction=chinese_conversion_direction,
                        _retry_plain_on_empty=False,
                        _eot_relaxed_retry=False,
                    )
                if mode == PromptMode.STRUCTURED:
                    return (
                        "[Error: Model returned empty output in structured mode. "
                        "Try prompt_mode=plain or prompt_mode=auto.]",
                    )
                # Plain mode (or already retried): return an explicit error string to avoid silent downstream failures.
                return (
                    "[Error: Model returned empty output. "
                    "Try increasing max_new_tokens or switching prompt_mode.]",
                )

            return (translated_text,)
        except Exception as e:
            # TG-008: Include context in inference errors
            # TG-018: Cleanup on error (even if keep_model_loaded=True)
            cleanup_torch_memory()
            raise RuntimeError(f"Translation failed: {e}\n[Context] {inference_context}") from e
        finally:
            # TG-004: Unload model if keep_model_loaded is False
            if not keep_model_loaded:
                unload_current_model()

    def _translate_image(
        self,
        image: Any,
        target_language: str,
        model_size: str,
        source_language: str,
        image_enhance: bool,
        image_resize_mode: str,
        image_two_pass: bool,
        max_new_tokens: int,
        max_input_tokens: int,
        truncate_input: bool,
        strict_context_limit: bool,
        keep_model_loaded: bool,
        debug: bool,
    ) -> tuple[str]:
        """
        Translate text found in an image using TranslateGemma multimodal chat template.

        TG-025: Uses 896×896 preprocessing for optimal vision encoder grounding.
        TG-025: Optional two-pass mode: extract (source→source) then translate extracted text (source→target).
        
        Ref: `REFERENCE/translategemma4b/README.md` ("Text Extraction and Translation")
        """
        # The official template requires an explicit source_lang_code; OCR-free auto-detect
        # is not supported. Require the user to select a source language for images.
        if source_language == "Auto Detect":
            return (
                "[Error: Image translation requires an explicit source_language. "
                "Auto Detect is not supported by the official TranslateGemma chat template.]",
            )

        t_load_start = time.time()
        model, processor = load_model(model_size)
        t_load_s = time.time() - t_load_start
        tokenizer = getattr(processor, "tokenizer", processor)

        try:
            input_device = next(model.parameters()).device
        except StopIteration:
            input_device = getattr(model, "device", "cpu")

        if debug:
            cuda_available = torch.cuda.is_available()
            cuda_name = None
            if cuda_available:
                try:
                    cuda_name = torch.cuda.get_device_name(torch.cuda.current_device())
                except Exception:
                    cuda_name = "unknown"
            print(
                "[TranslateGemma] Image runtime info: "
                f"cuda_available={cuda_available}, cuda_device={cuda_name}, "
                f"model_input_device={input_device}, load_time_s={t_load_s:.2f}"
            )

        target_code = get_language_code(target_language)
        source_code = get_language_code(source_language)

        resize_mode = (image_resize_mode or "letterbox").strip().lower()
        if resize_mode not in {"letterbox", "processor", "stretch"}:
            print(f"[TranslateGemma] WARNING: Invalid image_resize_mode='{image_resize_mode}', using 'letterbox'")
            resize_mode = "letterbox"

        # Optional two-pass strategy to reduce hallucinations:
        # Pass 1: "extract" by translating image from source → source.
        # Pass 2: translate extracted text via the normal text path.
        target_code_for_image = target_code
        if image_two_pass and source_code != target_code:
            target_code_for_image = source_code
            if debug:
                print(
                    "[TranslateGemma] Image two-pass enabled: "
                    f"extract_target_lang_code={target_code_for_image}, final_target_lang_code={target_code}"
                )

        # TG-040: Important: For image translation, visual tokens are inserted by the processor.
        # If `max_input_tokens` is too small and truncation is enabled, the image tokens
        # can be truncated away, causing hallucinated/unrelated outputs.
        # TranslateGemma model card notes ~2K total input context for image translation.
        effective_max_input_tokens = max_input_tokens
        if int(max_input_tokens) == 0:
            context_limit = get_context_limit(tokenizer)
            reserved_for_output = (
                int(max_new_tokens) if int(max_new_tokens) > 0 else 1024
            )
            reserved_for_output = min(int(reserved_for_output), int(context_limit) - 1)
            # For images, keep at least IMAGE_MIN_INPUT_TOKENS if possible, but never exceed context.
            auto_value = max(int(context_limit) - int(reserved_for_output) - 1, IMAGE_MIN_INPUT_TOKENS)
            effective_max_input_tokens = min(int(auto_value), int(context_limit) - 1)
            if debug:
                print(
                    f"[TranslateGemma] (image) max_input_tokens=0 (Auto) -> "
                    f"effective_max_input_tokens={effective_max_input_tokens} "
                    f"(context_limit={context_limit}, reserved_for_output={reserved_for_output})"
                )
        if truncate_input and int(effective_max_input_tokens) < IMAGE_MIN_INPUT_TOKENS:
            effective_max_input_tokens = IMAGE_MIN_INPUT_TOKENS
            # Only warn when the user explicitly set a low value (not when using Auto=0).
            if int(max_input_tokens) != 0:
                print(
                    "[TranslateGemma] WARNING: max_input_tokens is too low for image translation "
                    f"({max_input_tokens}). Overriding to {effective_max_input_tokens} to avoid truncating visual tokens."
                )

        # TG-025/TG-040: Convert ComfyUI image to PIL, then preprocess to vision target size
        pil_image = self._comfy_image_to_pil(image)
        original_size = pil_image.size

        preprocessed_image = preprocess_for_translategemma(
            pil_image=pil_image,
            target_size=VISION_TARGET_SIZE,
            enhance=image_enhance,
            debug=debug,
            resize_mode=resize_mode,
        )

        repo_id = MODEL_REPOS.get(model_size) or "unknown"
        device = get_device()
        dtype = get_torch_dtype(device)
        cache_dir = get_model_path(repo_id) if repo_id != "unknown" else "unknown"
        inference_context = (
            f"mode: image | model_size: {model_size} | repo_id: {repo_id} | "
            f"device: {device} | dtype: {dtype} | cache_dir: {cache_dir} | "
            f"user_max_new_tokens: {max_new_tokens} | max_input_tokens: {max_input_tokens} | "
            f"effective_max_input_tokens: {effective_max_input_tokens} | "
            f"truncate: {truncate_input} | strict_limit: {strict_context_limit} | "
            f"source_lang_code: {source_code} | target_lang_code: {target_code} | "
            f"image_target_lang_code: {target_code_for_image} | "
            f"image_enhance: {image_enhance} | image_two_pass: {image_two_pass}"
        )

        tmp_path = None

        try:
            # TG-030: Check multimodal capability before attempting
            if not hasattr(processor, "apply_chat_template"):
                raise RuntimeError(
                    f"Image translation requires processor.apply_chat_template, which is missing.\n\n"
                    f"This usually means your transformers version is too old.\n"
                    f"Recommended: transformers>=4.57\n\n"
                    f"[Context] model_size={model_size}, repo_id={repo_id}, cache_dir={redact_path(cache_dir)}"
                )
            
            kwargs = dict(
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            if truncate_input:
                kwargs["truncation"] = True
                kwargs["max_length"] = int(effective_max_input_tokens)

            inputs = None
            used_image_path = None

            # Best-effort: try an in-memory image path first to avoid temp-file I/O.
            try:
                import inspect

                try:
                    sig = inspect.signature(processor.apply_chat_template)
                except Exception:
                    sig = None

                if sig is not None and "images" in sig.parameters:
                    messages_mem = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source_lang_code": source_code,
                                    "target_lang_code": target_code_for_image,
                                }
                            ],
                        }
                    ]
                    try:
                        inputs = processor.apply_chat_template(messages_mem, images=[preprocessed_image], **kwargs)
                        used_image_path = "in-memory(images=...)"
                    except Exception:
                        inputs = None

                if inputs is None:
                    messages_mem_inline = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source_lang_code": source_code,
                                    "target_lang_code": target_code_for_image,
                                    "image": preprocessed_image,
                                }
                            ],
                        }
                    ]
                    try:
                        inputs = processor.apply_chat_template(messages_mem_inline, **kwargs)
                        used_image_path = "in-message(image=...)"
                    except Exception:
                        inputs = None
            except Exception:
                inputs = None

            if inputs is None:
                # TG-025: url-only path (matches official examples; slower due to disk I/O)
                tmp_path = save_preprocessed_image(
                    pil_image=preprocessed_image,
                    keep_file=debug,  # Keep file for inspection if debug mode
                    debug=debug,
                )

                messages_url = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source_lang_code": source_code,
                                "target_lang_code": target_code_for_image,
                                "url": tmp_path,
                            }
                        ],
                    }
                ]

                if debug:
                    print(f"[TranslateGemma] Image path: {redact_path(tmp_path)} (url-only structured)")

                # TG-030: Wrap apply_chat_template with targeted error handling
                try:
                    inputs = processor.apply_chat_template(messages_url, **kwargs)
                    used_image_path = "url-only(file)"
                except Exception as template_err:
                    raise RuntimeError(
                        f"processor.apply_chat_template failed for image translation.\n\n"
                        f"Error: {type(template_err).__name__}: {template_err}\n\n"
                        f"Possible causes:\n"
                        f"- Invalid language codes (source='{source_code}', target='{target_code_for_image}')\n"
                        f"- Incompatible transformers/processor version\n"
                        f"- Corrupted image or temp file\n\n"
                        f"To debug, enable: debug=true + TRANSLATEGEMMA_VERBOSE_DEBUG=1\n"
                        f"[Context] model_size={model_size}, repo_id={repo_id}"
                    ) from template_err
            elif debug:
                print(f"[TranslateGemma] Image input mode: {used_image_path}")

            # Move tensors to the model device; cast only floating tensors to dtype (e.g. pixel values).
            moved_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    if v.is_floating_point():
                        moved_inputs[k] = v.to(input_device, dtype=dtype)
                    else:
                        moved_inputs[k] = v.to(input_device)
                else:
                    moved_inputs[k] = v
            inputs = moved_inputs

            input_len = int(inputs["input_ids"].shape[1])

            requested_max_new_tokens = max_new_tokens
            if int(max_new_tokens) == 0:
                requested_max_new_tokens = _suggest_auto_max_new_tokens(tokenizer, int(input_len))
                if debug:
                    print(
                        f"[TranslateGemma] (image) max_new_tokens=0 (Auto) -> "
                        f"requested_max_new_tokens={requested_max_new_tokens}"
                    )
            # TG-040: Extraction (source→source) should be short; cap to reduce rambling.
            if image_two_pass and source_code != target_code:
                requested_max_new_tokens = min(int(requested_max_new_tokens), IMAGE_EXTRACTION_MAX_TOKENS)

            limits = compute_effective_limits(
                tokenizer=tokenizer,
                input_len=input_len,
                max_new_tokens=int(requested_max_new_tokens),
                max_input_tokens=int(effective_max_input_tokens),
                strict_context_limit=strict_context_limit,
                truncate_input=truncate_input,
                debug=debug,
            )
            # TG-040: Use configurable heuristic cap for generation limit
            heuristic_cap = get_generation_heuristic_cap(input_len)
            limits["effective_max_new_tokens"] = min(limits["effective_max_new_tokens"], heuristic_cap)
            if limits["effective_max_new_tokens"] == 0:
                return (f"[Error: Input too long - {input_len} tokens exceeds context limit]",)

            model.eval()
            with torch.inference_mode():
                gen_kwargs = dict(
                    **inputs,
                    max_new_tokens=limits["effective_max_new_tokens"],
                    do_sample=False,
                    use_cache=True,
                )
                eos_ids = _get_generation_eos_token_ids(tokenizer, debug=debug)
                if eos_ids:
                    gen_kwargs["eos_token_id"] = eos_ids if len(eos_ids) > 1 else eos_ids[0]
                    gen_kwargs["pad_token_id"] = int(getattr(tokenizer, "eos_token_id", eos_ids[0]) or eos_ids[0])

                stopping_criteria = _build_degeneracy_stopping_criteria(
                    prompt_len=int(input_len),
                    debug=debug,
                )
                if stopping_criteria is not None:
                    gen_kwargs["stopping_criteria"] = stopping_criteria

                t_gen_start = time.time()
                outputs = model.generate(**gen_kwargs)
                t_gen_elapsed = time.time() - t_gen_start

                # TG-041: Infer and log stop reason for image path
                end_of_turn_id = None
                try:
                    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
                    unk_id = getattr(tokenizer, "unk_token_id", None)
                    if unk_id is not None and end_of_turn_id == unk_id:
                        end_of_turn_id = None
                except Exception:
                    pass

                gen_meta = infer_stop_reason(
                    output_ids=outputs[0] if outputs is not None else None,
                    prompt_len=input_len,
                    max_new_tokens=limits["effective_max_new_tokens"],
                    eos_token_ids=eos_ids or [],
                    end_of_turn_id=end_of_turn_id,
                    stopping_criteria=stopping_criteria,
                )

                if debug:
                    print(f"[TranslateGemma] Image generation time_s={t_gen_elapsed:.2f}")
                log_stop_reason(gen_meta, debug=debug)

            generated_ids = outputs[0][input_len:]
            if hasattr(processor, "decode"):
                translated_text = processor.decode(generated_ids, skip_special_tokens=True)
            else:
                translated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            image_result = translated_text.strip()
            if not image_result:
                return ("",)

            if image_two_pass and source_code != target_code:
                if debug:
                    # TG-028: Redact extracted text unless verbose debug enabled
                    preview = redact_text(image_result) if not is_verbose_debug() else (
                        image_result if len(image_result) <= 200 else image_result[:200] + "..."
                    )
                    print(f"[TranslateGemma] Extracted text (pass 1): {preview}")

                # Pass 2: translate extracted text to the user-selected target.
                return self._translate_one(
                    text=image_result,
                    target_language=target_language,
                    model_size=model_size,
                    image=None,
                    image_enhance=False,
                    image_resize_mode="letterbox",
                    image_two_pass=False,
                    source_language=source_language,
                    external_text=None,
                    prompt_mode="auto",
                    max_new_tokens=max_new_tokens,
                    max_input_tokens=max_input_tokens,
                    truncate_input=truncate_input,
                    strict_context_limit=strict_context_limit,
                    keep_model_loaded=keep_model_loaded,
                    debug=debug,
                    chinese_conversion_only=False,
                    chinese_conversion_direction="auto_flip",
                )

            # Single-pass output: enforce Traditional script when targeting zh_Hant.
            image_result = postedit_traditional_chinese(
                image_result, target_lang_code=target_code, debug=debug
            )
            return (image_result,)
        except Exception as e:
            cleanup_torch_memory()
            hint = ""
            if source_language == "Auto Detect":
                hint = (
                    "Note: Image translation may require an explicit `source_language` "
                    "(the official template requires source_lang_code)."
                )
            raise RuntimeError(
                f"Image translation failed: {e}\n{hint}\n[Context] {inference_context}"
            ) from e
        finally:
            # TG-025: Clean up temp image (respect debug mode)
            if tmp_path:
                cleanup_temp_image(tmp_path, keep_for_debug=debug)
            if not keep_model_loaded:
                unload_current_model()

    def _convert_chinese_only(
        self,
        text: str,
        external_text: str | None,
        image,
        conversion_direction: str,
        debug: bool,
    ) -> tuple[str]:
        """
        TG-038: Fast-path Chinese variant conversion using OpenCC (no model load).

        This method performs deterministic Simplified ↔ Traditional conversion
        without loading any translation model. It uses OpenCC's phrase-level
        dictionaries followed by character-level fallback.

        Args:
            text: Text from the built-in input field
            external_text: Optional external text input
            image: Image input (must be None for conversion-only)
            conversion_direction: Direction selector (auto_flip/to_traditional/to_simplified)
            debug: Enable debug logging

        Returns:
            Tuple containing converted text
        """
        # Reject image input for conversion-only mode
        if image is not None:
            return (
                "[Error: chinese_conversion_only mode is text-only. "
                "Disconnect the image input to use conversion-only mode, "
                "or disable chinese_conversion_only for image translation.]",
            )

        # TG-009: Use external text if connected (is not None), otherwise use built-in text
        input_text = external_text if external_text is not None else text

        if not input_text or not input_text.strip():
            return ("",)

        # Determine target_lang_code based on conversion_direction
        target_code: str | None = None

        if conversion_direction == "to_traditional":
            target_code = "zh_Hant"
            if debug:
                print("[TranslateGemma] Conversion-only: forced direction → Traditional (s2t)")

        elif conversion_direction == "to_simplified":
            target_code = "zh"
            if debug:
                print("[TranslateGemma] Conversion-only: forced direction → Simplified (t2s)")

        elif conversion_direction == "auto_flip":
            # Auto-detect input variant and flip to opposite
            try:
                inferred_variant = infer_chinese_variant(input_text, debug=debug)
            except ImportError as e:
                return (f"[Error: {e}]",)

            if inferred_variant is None:
                # Ambiguous input
                return (
                    "[Error: Input variant ambiguous. Cannot determine if text is "
                    "Simplified or Traditional Chinese.\n"
                    "Select chinese_conversion_direction='to_traditional' or 'to_simplified' "
                    "to force direction.]",
                )

            # Flip: if input is Simplified → output Traditional, and vice versa
            if inferred_variant == "zh":
                target_code = "zh_Hant"
                if debug:
                    print(
                        "[TranslateGemma] Conversion-only auto_flip: "
                        "detected Simplified → converting to Traditional"
                    )
            else:  # zh_Hant
                target_code = "zh"
                if debug:
                    print(
                        "[TranslateGemma] Conversion-only auto_flip: "
                        "detected Traditional → converting to Simplified"
                    )
        else:
            return (
                f"[Error: Invalid chinese_conversion_direction='{conversion_direction}'. "
                f"Expected 'auto_flip', 'to_traditional', or 'to_simplified'.]",
            )

        # Perform conversion
        try:
            converted = convert_chinese_variants(
                text=input_text,
                target_lang_code=target_code,
                debug=debug,
            )
            return (converted.strip(),)
        except ImportError as e:
            # OpenCC not installed
            return (f"[Error: {e}]",)
        except ValueError as e:
            # Invalid target language
            return (f"[Error: {e}]",)
        except Exception as e:
            return (f"[Error: Chinese conversion failed: {e}]",)

    @staticmethod
    def _comfy_image_to_pil(image: Any):
        import numpy as np
        from PIL import Image

        if isinstance(image, torch.Tensor):
            arr = image.detach().cpu().numpy()
        else:
            arr = np.asarray(image)

        # ComfyUI IMAGE is typically [B,H,W,C]
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3:
            raise ValueError(f"Unsupported image shape: {arr.shape}")

        if arr.shape[-1] == 4:
            arr = arr[..., :3]

        # Normalize to uint8 RGB
        if arr.dtype != np.uint8:
            max_val = float(arr.max()) if arr.size else 1.0
            if max_val <= 1.5:
                arr = (arr.clip(0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                arr = arr.clip(0.0, 255.0).astype(np.uint8)

        return Image.fromarray(arr, mode="RGB")


# Node class mappings for this module
NODE_CLASS_MAPPINGS = {
    "TranslateGemma": TranslateGemmaNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TranslateGemma": "TranslateGemma",
}
