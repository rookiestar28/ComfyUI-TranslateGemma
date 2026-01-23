"""
ComfyUI-Translate: TranslateGemma Translation Node

A ComfyUI custom node for translating text using Google's open-source
TranslateGemma models. Supports 4B, 12B, and 27B model sizes with
automatic model downloading and 55 language support.
"""

import os

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__version__ = "1.1.9"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# ComfyUI registration
WEB_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), "web")
