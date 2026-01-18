# ComfyUI-TranslateGemma

<p align="center">
  <img src="assets/TranslateGemma.png" alt="TranslateGemma" />
</p>

A ComfyUI integration for TranslateGemma — Google's new open source translation model family built on Gemma 3. It supports 55 languages, multimodal image-to-text translation, and efficient inference from mobile (4B), and local (12B) to cloud (27B).

 [TranslateGemma: A new suite of open translation models](https://blog.google/innovation-and-ai/technology/developers-tools/translategemma/)

## Features

- Text translation across 55 languages
- Model size selection: 4B / 12B / 27B
- First-run auto download via Hugging Face (requires accepting Gemma terms)
- Flexible inputs: built-in text box + external string input
- Optional image input: translate text found in images (multimodal)

## Installation

### Option A: ComfyUI-Manager

1) Open ComfyUI-Manager.
2) Search for `TranslateGemma`.
3) Install and restart ComfyUI.

### Option B: Manual

1) Clone into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/rookiestar28/ComfyUI-TranslateGemma.git
```

1) Install dependencies:

```bash
cd ComfyUI-TranslateGemma
pip install -r requirements.txt
```

1) Restart ComfyUI.

## Hugging Face Access (Gated Models)

TranslateGemma repos are gated under the Gemma terms.

1) Visit the model page and accept the license terms:

- `google/translategemma-4b-it`
- `google/translategemma-12b-it`
- `google/translategemma-27b-it`

1) Authenticate (recommended):

```bash
hf auth login
```

Alternatively, set one of these environment variables for the ComfyUI process:

- `HF_TOKEN`
- `HUGGINGFACE_HUB_TOKEN`

1) Restart ComfyUI after changing authentication.

## Model Storage Location

Models are stored under ComfyUI's models directory in a per-repo folder:

- `ComfyUI/models/LLM/TranslateGemma/translategemma-4b-it/`
- `ComfyUI/models/LLM/TranslateGemma/translategemma-12b-it/`
- `ComfyUI/models/LLM/TranslateGemma/translategemma-27b-it/`

## Node: TranslateGemma

Category: `text/translation`

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `text` | STRING | Built-in text input (multiline) |
| `external_text` | STRING | Optional external input; when connected, **overrides** `text` (even if empty) |
| `image` | IMAGE | Optional image input; when connected, translates text found in the image |
| `image_enhance` | BOOLEAN | Apply mild contrast/sharpening to improve small text visibility in images (default: `false`) |
| `image_resize_mode` | COMBO | Image preprocessing mode: `letterbox` / `processor` / `stretch` (default: `letterbox`) |
| `image_two_pass` | BOOLEAN | Two-pass image translation: extract text from image first, then translate extracted text (default: `true`) |
| `target_language` | COMBO | Target language |
| `source_language` | COMBO | Source language (default: Auto Detect) |
| `model_size` | COMBO | 4B / 12B / 27B |
| `prompt_mode` | COMBO | `auto` / `structured` / `plain` |
| `max_new_tokens` | INT | Max output tokens (default: 512). Set to `0` for Auto |
| `max_input_tokens` | INT | Max input tokens (default: 2048) |
| `truncate_input` | BOOLEAN | Truncate input when it exceeds `max_input_tokens` |
| `strict_context_limit` | BOOLEAN | Clamp output so input + output fits the model context |
| `keep_model_loaded` | BOOLEAN | Keep the model in memory between runs |
| `debug` | BOOLEAN | Enable debug logging |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `translated_text` | STRING | Translated text |

## Usage Notes

### Text: Auto Detect

TranslateGemma's official chat template requires an explicit `source_lang_code`.
When `source_language=Auto Detect`, this node performs a best-effort local detection for text inputs.
If you see wrong-language behavior, pick the `source_language` explicitly.

### Image Translation Requires Source Language

For images, `source_language=Auto Detect` is not supported (no OCR pre-pass). Select the correct `source_language`.

### Image Preprocessing (896×896)

For image translation, the node supports multiple preprocessing modes via `image_resize_mode`:

- `letterbox` (default): preserve aspect ratio (no stretching) by padding, then resize
- `processor`: rely on the official Gemma3 image processor resize to **896×896** (may stretch)
- `stretch`: force resize to **896×896** (may distort)

If small text is missed, try enabling `image_enhance=true` to apply mild pixel-only enhancement.

When `debug=true`, the node prints the path of the preprocessed temporary PNG and keeps it for inspection.

Note: For image translation, `max_input_tokens` values that are too small can truncate the model’s visual tokens and cause unrelated outputs. The node enforces a safe minimum when truncation is enabled.

### Notes on Chinese Variants

For better Traditional Chinese output consistency, the node maps:

- Chinese (Simplified) -> `zh`
- Chinese (Traditional) -> `zh-Hant`

When `source_language=Auto Detect`, the node will try to distinguish Simplified vs Traditional Chinese:
- Region hints (when available): `zh_TW/zh_HK/zh_MO` -> `zh_Hant`, `zh_CN/zh_SG/zh_MY` -> `zh`
- Character-variant heuristic: counts common simplified/traditional characters and picks `zh_Hant` only when the signal is strong

If the text is too short or ambiguous, Auto Detect may still resolve to `zh`. For guaranteed behavior, select the desired `source_language` explicitly.

Tip: If your input is Simplified Chinese but you want Traditional output, set `source_language=Auto Detect` (or `Chinese (Simplified)`) and `target_language=Chinese (Traditional)`.

If you still see mixed Simplified/Traditional output when targeting Traditional Chinese, you can enable a best-effort post-edit conversion using OpenCC:
- Install: `pip install opencc-python-reimplemented`
- Default behavior: when `target_language=Chinese (Traditional)` the node will convert Simplified → Traditional if OpenCC is available
- Disable: set `TRANSLATEGEMMA_TRADITIONAL_POSTEDIT=0`

### Language Code Normalization

The node accepts both `_` and `-` variants for language codes (e.g., `zh_Hant` and `zh-Hant`). Internally, codes are normalized to match the official TranslateGemma template format.

If an unsupported language is passed, the node prints a warning and defaults to English. Set `TRANSLATEGEMMA_STRICT_LANG=1` to raise an error instead.

## Default Settings (TG-032)

The following are the authoritative default values for node inputs:

| Setting | Default | Notes |
|---------|---------|-------|
| `model_size` | `4B` | Smallest, fastest |
| `max_new_tokens` | `512` | Use `0` for auto-sizing |
| `max_input_tokens` | `2048` | Input truncation limit |
| `keep_model_loaded` | `true` | Avoids reload overhead |
| `truncate_input` | `true` | Prevents OOM on long texts |
| `debug` | `false` | Enable for diagnostics |
| `image_resize_mode` | `letterbox` | Preserves aspect ratio |
| `image_enhance` | `false` | Enables contrast/sharpening |
| `image_two_pass` | `true` | Extract then translate |

## Performance Tips

- Leave `keep_model_loaded=true` for repeated use (avoids reload time).
- Use the 4B model if you are unsure about hardware limits.
- First run is slower due to download and weight initialization.

## VRAM Notes (Native Models)

- 4B model: ~12 GB
- 12B model: ~27 GB
- 27B model: ~56 GB

## Security / Reproducibility Notes

### Remote Code Policy (TG-026)

- The loader attempts `trust_remote_code=False` first and only falls back to `True` if required.
- Set `TRANSLATEGEMMA_ALLOW_REMOTE_CODE=0` to deny remote code entirely (fails if code is needed).
- Set `TRANSLATEGEMMA_REMOTE_CODE_ALLOWLIST=google/translategemma-4b-it,google/translategemma-12b-it` to allow only specific repos.

### Revision Pinning

- You can pin a specific revision for reproducibility via `TRANSLATEGEMMA_REVISION=<commit-or-tag>`.

### Debug Privacy (TG-028)

- By default, `debug=true` redacts sensitive data (user text content, full filesystem paths).
- Set `TRANSLATEGEMMA_VERBOSE_DEBUG=1` to enable full diagnostics (for troubleshooting).

### Download Recovery

- If a download is interrupted, the loader auto-resumes on next run.
- If corruption persists, delete the model folder under `ComfyUI/models/LLM/TranslateGemma/` and retry.

## License

This repository is licensed under the MIT License (see `LICENSE`). TranslateGemma model weights are governed by Google's Gemma Terms of Use.
