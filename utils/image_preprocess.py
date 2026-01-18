"""
Image preprocessing for TranslateGemma multimodal translation (TG-025).

TranslateGemma expects 896×896 input for optimal vision encoder grounding.
This module provides deterministic preprocessing to meet these requirements.
"""

from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from typing import Optional
import os
import tempfile


# TranslateGemma optimal vision input size (from model card)
TRANSLATEGEMMA_VISION_SIZE = 896
DEFAULT_IMAGE_RESIZE_MODE = "letterbox"  # TG-032: aligned with node UI default


# TG-037: Enhancement configuration
def _get_enhance_params() -> dict:
    """Get enhancement parameters from environment or defaults."""
    mode = os.environ.get("TRANSLATEGEMMA_IMAGE_ENHANCE_MODE", "gentle").strip().lower()
    contrast = float(os.environ.get("TRANSLATEGEMMA_IMAGE_ENHANCE_CONTRAST", "1.10"))
    sharpness = float(os.environ.get("TRANSLATEGEMMA_IMAGE_ENHANCE_SHARPNESS", "1.10"))
    return {
        "mode": mode if mode in {"gentle", "legacy"} else "gentle",
        "contrast": max(0.5, min(2.0, contrast)),  # clamp to safe range
        "sharpness": max(0.5, min(2.0, sharpness)),
    }


def _apply_enhancement_gentle(img: Image.Image, contrast: float = 1.10, sharpness: float = 1.10) -> Image.Image:
    """
    TG-037: Gentle enhancement for small text visibility.
    
    Uses PIL.ImageEnhance for mild, controllable adjustments:
    - Contrast: slight increase (default 1.10)
    - Sharpness: slight increase (default 1.10)
    
    This avoids the aggressive autocontrast/unsharp mask that can degrade quality.
    """
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(sharpness)
    return img


def _apply_enhancement_legacy(img: Image.Image) -> Image.Image:
    """
    Legacy enhancement (pre-TG-037) for comparison.
    
    Transforms:
    - Autocontrast with low cutoff
    - Unsharp mask (aggressive)
    
    WARNING: This may degrade translation quality on some images.
    """
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=50, threshold=2))
    return img


def _apply_enhancement(img: Image.Image, debug: bool = False) -> Image.Image:
    """
    Apply enhancement based on configured mode (TG-037).
    
    Modes:
    - gentle (default): PIL.ImageEnhance with mild contrast/sharpness
    - legacy: autocontrast + unsharp mask (pre-TG-037 behavior)
    """
    params = _get_enhance_params()
    mode = params["mode"]
    
    if mode == "legacy":
        if debug:
            print(f"[TranslateGemma] Enhancement mode=legacy (autocontrast+unsharp)")
        return _apply_enhancement_legacy(img)
    else:
        contrast = params["contrast"]
        sharpness = params["sharpness"]
        if debug:
            print(f"[TranslateGemma] Enhancement mode=gentle (contrast={contrast}, sharpness={sharpness})")
        return _apply_enhancement_gentle(img, contrast=contrast, sharpness=sharpness)


def preprocess_for_translategemma(
    pil_image: Image.Image,
    target_size: int = TRANSLATEGEMMA_VISION_SIZE,
    enhance: bool = False,
    debug: bool = False,
    resize_mode: str = DEFAULT_IMAGE_RESIZE_MODE,
) -> Image.Image:
    """
    Preprocess a PIL image for TranslateGemma's vision encoder (TG-025).
    
    Strategy:
    1. Convert to RGB (drop alpha)
    2. Resizing based on resize_mode
    3. Optional enhancement AFTER resize (TG-037: post-resize for consistency)
    
    Args:
        pil_image: Input PIL Image
        target_size: Target square size (default 896 per model card)
        enhance: Apply mild contrast/sharpening for small text (default False)
        debug: Print preprocessing info
        resize_mode: "letterbox", "stretch", or "processor"
        
    Returns:
        Preprocessed PIL Image (RGB, target_size×target_size or original if processor)
    """
    original_size = pil_image.size
    
    # 1. Ensure RGB (drop alpha if present)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # 2. Resizing strategy (TG-037: do resize BEFORE enhancement)
    resize_mode = (resize_mode or DEFAULT_IMAGE_RESIZE_MODE).strip().lower()
    if resize_mode not in {"processor", "letterbox", "stretch"}:
        raise ValueError(f"Invalid resize_mode: {resize_mode}")

    if resize_mode == "processor":
        # For processor mode with enhance=true, pre-resize to 896 for deterministic enhancement
        if enhance:
            # Pre-resize to target size so enhancement is consistent
            width, height = pil_image.size
            max_dim = max(width, height)
            square_img = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
            offset_x = (max_dim - width) // 2
            offset_y = (max_dim - height) // 2
            square_img.paste(pil_image, (offset_x, offset_y))
            resized_img = square_img.resize((target_size, target_size), Image.LANCZOS)
        else:
            # No enhancement: just pass through
            resized_img = pil_image
    elif resize_mode == "stretch":
        resized_img = pil_image.resize((target_size, target_size), Image.BILINEAR)
    else:  # letterbox
        width, height = pil_image.size
        max_dim = max(width, height)
        square_img = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        offset_x = (max_dim - width) // 2
        offset_y = (max_dim - height) // 2
        square_img.paste(pil_image, (offset_x, offset_y))
        resized_img = square_img.resize((target_size, target_size), Image.LANCZOS)

    # 3. Optional enhancement AFTER resize (TG-037: post-resize for consistency)
    if enhance:
        resized_img = _apply_enhancement(resized_img, debug=debug)

    if debug:
        params = _get_enhance_params() if enhance else {}
        mode_str = params.get("mode", "N/A") if enhance else "N/A"
        if resize_mode == "processor" and not enhance:
            print(
                f"[TranslateGemma] Image preprocessing: "
                f"{original_size[0]}×{original_size[1]} (processor will resize to {target_size}×{target_size}), "
                f"enhance={enhance}"
            )
        else:
            stage = "post-resize" if enhance else "N/A"
            print(
                f"[TranslateGemma] Image preprocessing: "
                f"{original_size[0]}×{original_size[1]} → {target_size}×{target_size} "
                f"({resize_mode}, enhance={enhance}, enhance_mode={mode_str}, stage={stage})"
            )
    
    return resized_img


def save_preprocessed_image(
    pil_image: Image.Image,
    keep_file: bool = False,
    debug: bool = False,
) -> str:
    """
    Save preprocessed image to a temporary PNG file (TG-025).
    
    Args:
        pil_image: Preprocessed PIL Image
        keep_file: If True, don't auto-delete (for debug inspection)
        debug: Print file path
        
    Returns:
        Path to the saved PNG file
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    
    pil_image.save(tmp_path, format="PNG")
    
    if debug:
        print(f"[TranslateGemma] Preprocessed image saved: {tmp_path}")
        if keep_file:
            print(f"[TranslateGemma] Debug mode: keeping file for inspection")
    
    return tmp_path


def cleanup_temp_image(tmp_path: str, keep_for_debug: bool = False):
    """
    Clean up temporary image file unless debug mode requests keeping it.
    """
    if keep_for_debug:
        return
    
    try:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
    except Exception:
        pass  # Best effort cleanup
