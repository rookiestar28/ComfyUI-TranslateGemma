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
from ..utils.context_utils import compute_effective_limits, suggest_max_new_tokens


# TG-028: Debug privacy controls
_TOKENIZER_MUTATION_LOCK = threading.RLock()


def _is_verbose_debug() -> bool:
    """Check if verbose debug mode is enabled (TG-028)."""
    return os.environ.get("TRANSLATEGEMMA_VERBOSE_DEBUG", "").strip() == "1"


def _redact_path(path: str) -> str:
    """Redact full path to basename only for non-verbose debug (TG-028)."""
    if _is_verbose_debug():
        return path
    return os.path.basename(path) if path else path


def _redact_text(text: str, max_len: int = 50) -> str:
    """Redact text content for non-verbose debug (TG-028)."""
    if _is_verbose_debug():
        return text
    if not text:
        return "(empty)"
    return f"[{len(text)} chars]"


# TG-003 (Critical bug fix / Case T3):
# Do not truncate the full structured chat template produced by
# `processor.apply_chat_template(...)`. The official template (see
# `REFERENCE/translategemma4b/chat_template.jinja`) relies on tail markers like
# `<end_of_turn>` and the generation prompt `<start_of_turn>model\n`.
# If truncation removes those markers, the model may continue the user text
# (often in the source language), causing target-language drift.
def _truncate_text_tokens_to_fit(
    tokenizer,
    text: str,
    max_tokens: int,
) -> tuple[str, int, bool]:
    text_ids = tokenizer(text, add_special_tokens=False).get("input_ids", [])
    if len(text_ids) <= int(max_tokens):
        return text, len(text_ids), False
    truncated_ids = text_ids[: int(max_tokens)]
    truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
    return truncated_text, len(text_ids), True


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
                }),
                "target_language": (languages, {
                    "default": "English",
                }),
                "model_size": (model_sizes, {
                    "default": "4B",
                    "tooltip": "Model size. If the repo is gated/private: accept terms on HuggingFace and authenticate via `hf auth login` (or HF_TOKEN env var).",
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "forceInput": True,
                    "tooltip": "Optional image input. If connected, the node will translate text found in the image.",
                }),
                "image_enhance": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "(TG-025) Apply mild contrast/sharpening to improve small text visibility in images",
                }),
                "image_resize_mode": (["letterbox", "processor", "stretch"], {
                    "default": "letterbox",
                    "tooltip": "(TG-025) Image preprocessing: letterbox (preserve aspect), processor (official resize to 896×896, may stretch), stretch (force 896×896, may distort).",
                }),
                "image_two_pass": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "(TG-025) Two-pass image translation: extract text from image first, then translate extracted text (more accurate, slower).",
                }),
                "source_language": (["Auto Detect"] + languages, {
                    "default": "Auto Detect",
                }),
                "external_text": ("STRING", {
                    "forceInput": True,
                    "tooltip": "(TG-009) External text input. When connected, overrides the built-in text field (even if empty).",
                }),
                "prompt_mode": (prompt_modes, {
                    "default": "auto",
                    "tooltip": "Prompt format: auto (try structured, fallback to plain), structured (force chat-template), plain (instruction only)",
                }),
                "max_new_tokens": ("INT", {
                    "default": 512,
                    "min": 0,
                    "max": 2048,
                    "step": 16,
                    "tooltip": "Maximum number of tokens to generate (output length). Set to 0 for Auto.",
                }),
                "max_input_tokens": ("INT", {
                    "default": 2048,
                    "min": 64,
                    "max": 8192,
                    "step": 64,
                    "tooltip": "(TG-003) Maximum input tokens. Long inputs will be truncated (may reduce translation completeness). Low values like 512 are for stress-testing; recommended 2048+ for long documents.",
                }),
                "truncate_input": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Truncate input if it exceeds max_input_tokens (disable may cause OOM on long texts)",
                }),
                "strict_context_limit": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clamp max_new_tokens so input + output doesn't exceed context window",
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model in memory after inference (disable to free VRAM between runs)",
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable debug logging for prompt construction and tokenizer probing",
                }),
            },
        }
    
    def translate(
        self,
        text: str,
        target_language: str,
        model_size: str,
        image=None,
        image_enhance: bool = False,
        image_resize_mode: str = "letterbox",
        image_two_pass: bool = True,
        source_language: str = "Auto Detect",
        external_text: str = None,
        prompt_mode: str = "auto",
        max_new_tokens: int = 512,
        max_input_tokens: int = 2048,
        truncate_input: bool = True,
        strict_context_limit: bool = True,
        keep_model_loaded: bool = True,
        debug: bool = False,
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
            
        Returns:
            Tuple containing translated text
        """
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
        
        # Load model and processor/tokenizer
        t_load_start = time.time()
        model, processor = load_model(model_size)
        t_load_s = time.time() - t_load_start
        tokenizer = getattr(processor, "tokenizer", processor)

        # Pick a safe input device (for device_map models, use the first parameter device).
        try:
            input_device = next(model.parameters()).device
        except StopIteration:
            input_device = getattr(model, "device", "cpu")

        # Preferred compute dtype for floating inputs (pixel values, embeddings, etc).
        # Keep consistent with utils/model_loader.py selection.
        device = get_device()
        dtype = get_torch_dtype(device)

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

                    _, raw_text_tokens, _ = _truncate_text_tokens_to_fit(
                        tokenizer=tokenizer,
                        text=input_text,
                        max_tokens=10**9,
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
                        truncated_text, _, _ = _truncate_text_tokens_to_fit(
                            tokenizer=tokenizer,
                            text=input_text,
                            max_tokens=available_for_text,
                        )

                    if debug and truncate_input and raw_input_len > int(max_input_tokens):
                        print(
                            "[TranslateGemma] Structured truncation: "
                            f"overhead_tokens={overhead_len}, "
                            f"available_for_text_tokens={max(available_for_text, 0)}, "
                            f"raw_input_tokens={raw_input_len}, max_input_tokens={max_input_tokens}"
                        )

                    for _ in range(8):
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
                        available_for_text = max(available_for_text - overflow - 8, 0)
                        if available_for_text <= 0:
                            break
                        truncated_text, _, _ = _truncate_text_tokens_to_fit(
                            tokenizer=tokenizer,
                            text=input_text,
                            max_tokens=available_for_text,
                        )
                used_path = "processor.apply_chat_template(structured)"
        except Exception as e:
            if mode == PromptMode.STRUCTURED:
                raise RuntimeError(
                    "Structured mode requested but processor.apply_chat_template failed."
                ) from e
            inputs = None

        if inputs is None:
            # Fallback to prompt-string rendering (older/unsupported stacks).
            if debug:
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
                if _is_verbose_debug():
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

        if truncate_input and raw_input_len and actual_input_len and int(raw_input_len) > int(actual_input_len):
            print(
                f"[TranslateGemma] Input truncated from {raw_input_len} to {actual_input_len} tokens "
                f"(max_input_tokens={max_input_tokens}, template-safe)"
            )

        requested_max_new_tokens = max_new_tokens
        if int(max_new_tokens) == 0:
            requested_max_new_tokens = suggest_max_new_tokens(actual_input_len)
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

        # Extra guard for translation tasks: short inputs rarely need huge generation windows.
        # This helps reduce repetitive continuations when the model does not emit EOS early.
        heuristic_cap = int(actual_input_len * 2 + 64)
        limits["effective_max_new_tokens"] = min(limits["effective_max_new_tokens"], heuristic_cap)
        
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
                eos_token_id = getattr(tokenizer, "eos_token_id", None)
                if eos_token_id is not None:
                    gen_kwargs["eos_token_id"] = eos_token_id
                    gen_kwargs["pad_token_id"] = eos_token_id

                t_gen_start = time.time()
                outputs = model.generate(**gen_kwargs)
                if debug:
                    print(f"[TranslateGemma] Generation time_s={time.time() - t_gen_start:.2f}")
            
            # Decode output
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            if hasattr(processor, "decode"):
                translated_text = processor.decode(generated_ids, skip_special_tokens=True)
            else:
                translated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            return (translated_text.strip(),)
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

        # Important: For image translation, visual tokens are inserted by the processor.
        # If `max_input_tokens` is too small and truncation is enabled, the image tokens
        # can be truncated away, causing hallucinated/unrelated outputs.
        # TranslateGemma model card notes ~2K total input context for image translation.
        effective_max_input_tokens = max_input_tokens
        if truncate_input and int(max_input_tokens) < 2048:
            effective_max_input_tokens = 2048
            print(
                "[TranslateGemma] WARNING: max_input_tokens is too low for image translation "
                f"({max_input_tokens}). Overriding to {effective_max_input_tokens} to avoid truncating visual tokens."
            )

        # TG-025: Convert ComfyUI image to PIL, then preprocess to 896×896
        pil_image = self._comfy_image_to_pil(image)
        original_size = pil_image.size
        
        preprocessed_image = preprocess_for_translategemma(
            pil_image=pil_image,
            target_size=896,
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

        # TG-025: Always use url-only path per official examples
        tmp_path = save_preprocessed_image(
            pil_image=preprocessed_image,
            keep_file=debug,  # Keep file for inspection if debug mode
            debug=debug,
        )
        
        try:
            # TG-030: Check multimodal capability before attempting
            if not hasattr(processor, "apply_chat_template"):
                raise RuntimeError(
                    f"Image translation requires processor.apply_chat_template, which is missing.\n\n"
                    f"This usually means your transformers version is too old.\n"
                    f"Recommended: transformers>=4.57\n\n"
                    f"[Context] model_size={model_size}, repo_id={repo_id}, cache_dir={_redact_path(cache_dir)}"
                )
            
            kwargs = dict(
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            if truncate_input:
                kwargs.update(dict(truncation=True, max_length=effective_max_input_tokens))

            # TG-025: Use url-only path (matches official examples)
            messages = [
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
                print(f"[TranslateGemma] Image path: {_redact_path(tmp_path)} (url-only structured)")
            
            # TG-030: Wrap apply_chat_template with targeted error handling
            try:
                inputs = processor.apply_chat_template(messages, **kwargs)
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
                requested_max_new_tokens = suggest_max_new_tokens(input_len)
                if debug:
                    print(
                        f"[TranslateGemma] (image) max_new_tokens=0 (Auto) -> "
                        f"requested_max_new_tokens={requested_max_new_tokens}"
                    )
            # Extraction (source→source) should be short; cap to reduce rambling.
            if image_two_pass and source_code != target_code:
                requested_max_new_tokens = min(int(requested_max_new_tokens), 256)

            limits = compute_effective_limits(
                tokenizer=tokenizer,
                input_len=input_len,
                max_new_tokens=int(requested_max_new_tokens),
                max_input_tokens=int(effective_max_input_tokens),
                strict_context_limit=strict_context_limit,
                truncate_input=truncate_input,
                debug=debug,
            )
            heuristic_cap = int(input_len * 2 + 64)
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
                eos_token_id = getattr(tokenizer, "eos_token_id", None)
                if eos_token_id is not None:
                    gen_kwargs["eos_token_id"] = eos_token_id
                    gen_kwargs["pad_token_id"] = eos_token_id

                t_gen_start = time.time()
                outputs = model.generate(**gen_kwargs)
                if debug:
                    print(f"[TranslateGemma] Image generation time_s={time.time() - t_gen_start:.2f}")

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
                    preview = _redact_text(image_result) if not _is_verbose_debug() else (
                        image_result if len(image_result) <= 200 else image_result[:200] + "..."
                    )
                    print(f"[TranslateGemma] Extracted text (pass 1): {preview}")

                # Pass 2: translate extracted text to the user-selected target.
                return self.translate(
                    text=image_result,
                    target_language=target_language,
                    model_size=model_size,
                    image=None,
                    image_enhance=False,
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
            cleanup_temp_image(tmp_path, keep_for_debug=debug)
            if not keep_model_loaded:
                unload_current_model()

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
