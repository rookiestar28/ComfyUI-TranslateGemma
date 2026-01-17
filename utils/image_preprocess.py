"""
Image preprocessing for TranslateGemma multimodal translation (TG-025).

TranslateGemma expects 896×896 input for optimal vision encoder grounding.
This module provides deterministic preprocessing to meet these requirements.
"""

from PIL import Image, ImageOps, ImageFilter
from typing import Optional
import os
import tempfile


# TranslateGemma optimal vision input size (from model card)
TRANSLATEGEMMA_VISION_SIZE = 896
DEFAULT_IMAGE_RESIZE_MODE = "processor"  # rely on Gemma3ImageProcessor resize


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
    2. Optional pixel-only enhancement for small text
    3. Resizing is handled based on resize_mode:
       - "processor" (default): do not resize here; let Gemma3Processor resize to 896×896
       - "letterbox": pad to square then resize to target_size×target_size
       - "stretch": directly resize to target_size×target_size (aspect ratio is not preserved)
    
    Args:
        pil_image: Input PIL Image
        target_size: Target square size (default 896 per model card)
        enhance: Apply mild contrast/sharpening for small text (default False)
        debug: Print preprocessing info
        
    Returns:
        Preprocessed PIL Image (RGB, target_size×target_size)
    """
    original_size = pil_image.size
    
    # 1. Ensure RGB (drop alpha if present)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    
    # 2. Optional enhancement for small text (TG-025 Part B)
    if enhance:
        pil_image = _apply_enhancement(pil_image)
        if debug:
            print(f"[TranslateGemma] Image enhancement applied")

    # 3. Resizing strategy
    resize_mode = (resize_mode or DEFAULT_IMAGE_RESIZE_MODE).strip().lower()
    if resize_mode not in {"processor", "letterbox", "stretch"}:
        raise ValueError(f"Invalid resize_mode: {resize_mode}")

    if resize_mode == "processor":
        # Let the Gemma3ImageProcessor perform the official resize to 896×896.
        resized_img = pil_image
    elif resize_mode == "stretch":
        # Match the default processor behavior (direct resize) but keep it explicit/inspectable.
        resized_img = pil_image.resize((target_size, target_size), Image.BILINEAR)
    else:
        # Letterbox to square (preserve aspect ratio, pad with white), then resize.
        width, height = pil_image.size
        max_dim = max(width, height)

        square_img = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        offset_x = (max_dim - width) // 2
        offset_y = (max_dim - height) // 2
        square_img.paste(pil_image, (offset_x, offset_y))

        resized_img = square_img.resize((target_size, target_size), Image.LANCZOS)

    if debug:
        if resize_mode == "processor":
            print(
                f"[TranslateGemma] Image preprocessing: "
                f"{original_size[0]}×{original_size[1]} (processor will resize to {target_size}×{target_size}), "
                f"enhance={enhance}"
            )
        else:
            print(
                f"[TranslateGemma] Image preprocessing: "
                f"{original_size[0]}×{original_size[1]} → {target_size}×{target_size} "
                f"({resize_mode}, enhance={enhance})"
            )
    
    return resized_img


def _apply_enhancement(img: Image.Image) -> Image.Image:
    """
    Apply mild enhancement to improve small text visibility (TG-025 Part B).
    
    Transforms:
    - Autocontrast with low cutoff (preserve most pixels)
    - Mild unsharp mask (enhance edges without artifacts)
    """
    # Autocontrast: normalize histogram (low cutoff to avoid clipping)
    # Pillow documents cutoff as an integer percentage. Keep it conservative.
    img = ImageOps.autocontrast(img, cutoff=1)
    
    # Unsharp mask: mild sharpening for text edges
    img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=50, threshold=2))
    
    return img


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
    # Create temp file (delete=False so we can pass the path to processor)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    
    # Save as PNG (lossless)
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
