"""
Prompt builder for TranslateGemma translation node.

Provides centralized prompt construction with support for:
- Structured chat-template format (preferred)
- Plain instruction format (fallback)
- Auto mode: tries structured first, falls back to plain on failure

Reference: TG-001 Implementation Plan
"""

from enum import Enum
from typing import Any, Optional
import logging

# Import full language mapping from language_utils (55 languages)
from .language_utils import LANGUAGE_MAP

logger = logging.getLogger(__name__)


class PromptMode(str, Enum):
    """Prompt construction mode."""
    AUTO = "auto"
    STRUCTURED = "structured"
    PLAIN = "plain"

    @classmethod
    def get_values(cls) -> list[str]:
        return [item.value for item in cls]


def _get_language_name(lang_code: str) -> str:
    """Get human-readable language name from code, using full 55-language mapping."""
    return LANGUAGE_MAP.get(lang_code, lang_code)


def build_structured_messages(
    input_text: str,
    target_lang_code: str,
    source_lang_code: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Build structured messages using TranslateGemma's expected format.
    
    The structured format uses a typed content array with translation metadata.
    
    Args:
        input_text: Text to translate
        target_lang_code: Target language ISO code
        source_lang_code: Optional source language ISO code (None for auto-detect)
        
    Returns:
        List of message dictionaries for chat template
    """
    # TranslateGemma's chat template requires source_lang_code and target_lang_code.
    # The model card does not define an "auto" source code; callers should pass an
    # explicit ISO code. Default to "en" to keep probes and fallbacks safe.
    content_item = {
        "type": "text",
        "text": input_text,
        "source_lang_code": source_lang_code or "en",
        "target_lang_code": target_lang_code,
    }

    return [{"role": "user", "content": [content_item]}]


def build_simple_messages(
    input_text: str,
    target_lang_code: str,
    source_lang_code: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Build simple string-content messages as a fallback format.
    
    Some tokenizers may only support string content instead of typed arrays.
    
    Args:
        input_text: Text to translate
        target_lang_code: Target language ISO code
        source_lang_code: Optional source language ISO code
        
    Returns:
        List of message dictionaries with string content
    """
    target_name = _get_language_name(target_lang_code)

    # Keep the simple prompt style close to the official template wording to
    # reduce target-language drift when we must fall back.
    if source_lang_code:
        source_name = _get_language_name(source_lang_code)
        instruction = (
            f"You are a professional {source_name} ({source_lang_code}) to {target_name} ({target_lang_code}) translator.\n"
            f"Produce only the {target_name} translation, without any additional explanations or commentary.\n\n"
            f"Please translate the following {source_name} text into {target_name}:\n\n{input_text}"
        )
    else:
        instruction = (
            f"Produce only the {target_name} translation, without any additional explanations or commentary.\n\n"
            f"Please translate the following text into {target_name}:\n\n{input_text}"
        )
    
    return [{"role": "user", "content": instruction}]


def build_plain_prompt(
    input_text: str,
    target_lang_code: str,
    source_lang_code: Optional[str] = None,
) -> str:
    """
    Build a plain instruction prompt without chat template.
    
    This is the ultimate fallback when chat_template is not available.
    
    Args:
        input_text: Text to translate
        target_lang_code: Target language ISO code
        source_lang_code: Optional source language ISO code
        
    Returns:
        Plain instruction string
    """
    target_name = _get_language_name(target_lang_code)
    
    if source_lang_code:
        source_name = _get_language_name(source_lang_code)
        return (
            f"You are a professional {source_name} ({source_lang_code}) to {target_name} ({target_lang_code}) translator. "
            f"Your goal is to accurately convey the meaning and nuances of the original {source_name} text while adhering to "
            f"{target_name} grammar, vocabulary, and cultural sensitivities.\n\n"
            f"Produce only the {target_name} translation, without any additional explanations or commentary. "
            f"Please translate the following {source_name} text into {target_name}:\n\n\n"
            f"{input_text.strip()}"
        )
    else:
        return (
            f"Produce only the {target_name} translation, without any additional explanations or commentary. "
            f"Please translate the following text into {target_name}:\n\n\n"
            f"{input_text.strip()}"
        )


def _has_chat_template(tokenizer) -> bool:
    """Check if tokenizer has a valid chat_template."""
    return hasattr(tokenizer, "chat_template") and tokenizer.chat_template


def _try_render(tokenizer, messages: list[dict], description: str, debug: bool) -> Optional[str]:
    """
    Attempt to render messages using chat template.
    
    Args:
        tokenizer: HuggingFace tokenizer
        messages: Message list to render
        description: Description for logging
        debug: Whether to log debug information
        
    Returns:
        Rendered prompt string, or None if rendering fails
    """
    try:
        result = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if debug:
            logger.info(f"[TranslateGemma] {description} succeeded, length={len(result)}")
        return result
    except Exception as e:
        if debug:
            logger.warning(f"[TranslateGemma] {description} failed: {type(e).__name__}: {e}")
        return None


def probe_tokenizer_capabilities(tokenizer, debug: bool = False) -> dict[str, bool]:
    """
    Probe tokenizer to determine which message formats are supported.
    
    Only runs detailed probing when debug is enabled.
    
    Args:
        tokenizer: HuggingFace tokenizer
        debug: Whether to run full capability probe
        
    Returns:
        Dictionary of capability flags
    """
    capabilities = {
        "has_chat_template": _has_chat_template(tokenizer),
        "structured_content": False,
        "simple_content": False,
    }
    
    if not capabilities["has_chat_template"]:
        return capabilities
    
    if not debug:
        # Quick check only - assume structured works if chat_template exists
        # NOTE: These are ASSUMED values, not verified. Only use for informational
        # purposes when debug=False. For decision-making, call with debug=True.
        capabilities["structured_content"] = True  # assumed, not verified
        capabilities["simple_content"] = True  # assumed, not verified
        return capabilities
    
    # Full probe with test messages
    test_text = "Hello"
    test_target = "zh"
    
    # Test structured format
    structured_msgs = build_structured_messages(test_text, test_target, source_lang_code="en")
    capabilities["structured_content"] = _try_render(
        tokenizer, structured_msgs, "Structured format probe", debug
    ) is not None
    
    # Test simple format
    simple_msgs = build_simple_messages(test_text, test_target)
    capabilities["simple_content"] = _try_render(
        tokenizer, simple_msgs, "Simple format probe", debug
    ) is not None
    
    if debug:
        logger.info(f"[TranslateGemma] Tokenizer capabilities: {capabilities}")
    
    return capabilities


def render_prompt(
    tokenizer,
    input_text: str,
    target_lang_code: str,
    source_lang_code: Optional[str] = None,
    mode: PromptMode = PromptMode.AUTO,
    debug: bool = False,
) -> str:
    """
    Render the translation prompt using the specified mode.
    
    Args:
        tokenizer: HuggingFace tokenizer
        input_text: Text to translate
        target_lang_code: Target language ISO code
        source_lang_code: Optional source language ISO code
        mode: Prompt construction mode (auto/structured/plain)
        debug: Whether to enable debug logging
        
    Returns:
        Rendered prompt string
        
    Raises:
        RuntimeError: If structured mode is forced but fails
    """
    if debug:
        # Log input metadata (not full text to protect privacy)
        logger.info(
            f"[TranslateGemma] render_prompt: mode={mode.value}, "
            f"text_len={len(input_text)}, target={target_lang_code}, "
            f"source={source_lang_code or 'auto'}"
        )
    
    if mode == PromptMode.PLAIN:
        # Force plain mode - bypass chat template entirely
        prompt = build_plain_prompt(input_text, target_lang_code, source_lang_code)
        if debug:
            logger.info(f"[TranslateGemma] Using plain prompt, length={len(prompt)}")
        return prompt
    
    # Check tokenizer capabilities
    has_template = _has_chat_template(tokenizer)
    
    if not has_template:
        if mode == PromptMode.STRUCTURED:
            raise RuntimeError(
                "Structured mode requested but tokenizer has no chat_template. "
                "Use 'auto' or 'plain' mode instead."
            )
        if debug:
            logger.warning("[TranslateGemma] No chat_template, falling back to plain")
        return build_plain_prompt(input_text, target_lang_code, source_lang_code)
    
    # Try structured format first
    structured_msgs = build_structured_messages(input_text, target_lang_code, source_lang_code)
    result = _try_render(tokenizer, structured_msgs, "Structured render", debug)
    
    if result is not None:
        return result
    
    if mode == PromptMode.STRUCTURED:
        raise RuntimeError(
            "Structured mode requested but apply_chat_template failed. "
            "The tokenizer may not support typed content arrays."
        )
    
    # Auto mode: try simple string content
    if debug:
        logger.info("[TranslateGemma] Structured failed, trying simple format")
    
    simple_msgs = build_simple_messages(input_text, target_lang_code, source_lang_code)
    result = _try_render(tokenizer, simple_msgs, "Simple render", debug)
    
    if result is not None:
        return result
    
    # Ultimate fallback
    if debug:
        logger.warning("[TranslateGemma] All chat template attempts failed, using plain")
    
    return build_plain_prompt(input_text, target_lang_code, source_lang_code)


def render_auto(
    tokenizer,
    input_text: str,
    target_lang_code: str,
    source_lang_code: Optional[str] = None,
    debug: bool = False,
) -> str:
    """
    Convenience function for auto-mode prompt rendering.
    
    Tries structured format first, falls back to plain on failure.
    
    Args:
        tokenizer: HuggingFace tokenizer
        input_text: Text to translate
        target_lang_code: Target language ISO code
        source_lang_code: Optional source language ISO code
        debug: Whether to enable debug logging
        
    Returns:
        Rendered prompt string
    """
    return render_prompt(
        tokenizer=tokenizer,
        input_text=input_text,
        target_lang_code=target_lang_code,
        source_lang_code=source_lang_code,
        mode=PromptMode.AUTO,
        debug=debug,
    )
