"""
TG-034: Structured truncation guard utility.

Validates that the TranslateGemma chat template wrapper is preserved after
structured prompt construction. If the wrapper is missing (due to truncation
or other issues), the model may produce target-language drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    import torch


# Required markers in the structured template tail
# These must appear at the end of the prompt for proper model behavior.
REQUIRED_TAIL_MARKER = "<start_of_turn>model"
OPTIONAL_PRIOR_MARKER = "<end_of_turn>"


class StructuredTruncationError(RuntimeError):
    """
    Raised when structured truncation removes required template markers.
    
    This indicates the generated prompt is missing the chat template wrapper
    and would likely cause target-language drift.
    """
    pass


def assert_structured_generation_prompt(
    tokenizer: "PreTrainedTokenizerBase",
    input_ids: "torch.Tensor",
    *,
    context: str = "",
    tail_tokens: int = 32,
) -> None:
    """
    Assert that the structured prompt still contains the required generation wrapper.
    
    Decodes the tail of `input_ids` and checks for `<start_of_turn>model`.
    If missing, raises StructuredTruncationError with actionable guidance.
    
    Args:
        tokenizer: The tokenizer used to decode.
        input_ids: The final input_ids tensor (shape: [1, seq_len] or [seq_len]).
        context: Optional context string for error messages.
        tail_tokens: Number of tokens from the tail to decode for checking.
    
    Raises:
        StructuredTruncationError: If the required wrapper is missing.
    """
    # Flatten if batched
    if input_ids.dim() == 2:
        ids = input_ids[0]
    else:
        ids = input_ids
    
    # Decode the tail
    tail_ids = ids[-tail_tokens:] if len(ids) > tail_tokens else ids
    tail_text = tokenizer.decode(tail_ids, skip_special_tokens=False)
    
    if REQUIRED_TAIL_MARKER not in tail_text:
        raise StructuredTruncationError(
            f"Structured prompt is missing the required generation wrapper '{REQUIRED_TAIL_MARKER}'. "
            "This can cause target-language drift (Case T3). "
            "Mitigation options:\n"
            "  1. Increase `max_input_tokens`\n"
            "  2. Set `truncate_input=false`\n"
            "  3. Use `prompt_mode=plain`\n"
            f"{f'[Context] {context}' if context else ''}"
        )


def check_structured_generation_prompt(
    tokenizer: "PreTrainedTokenizerBase",
    input_ids: "torch.Tensor",
    *,
    tail_tokens: int = 32,
) -> bool:
    """
    Check (non-throwing) whether the structured prompt contains the required wrapper.
    
    Returns:
        True if the wrapper is present, False otherwise.
    """
    try:
        if input_ids.dim() == 2:
            ids = input_ids[0]
        else:
            ids = input_ids
        
        tail_ids = ids[-tail_tokens:] if len(ids) > tail_tokens else ids
        tail_text = tokenizer.decode(tail_ids, skip_special_tokens=False)
        
        return REQUIRED_TAIL_MARKER in tail_text
    except Exception:
        return False
