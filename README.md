# ComfyUI-TranslateGemma

A ComfyUI custom node that translates text (including prompts) using Google's open-weight **TranslateGemma** models.

## Features

- Text translation across 55 languages
- Model size selection: 4B / 12B / 27B
- First-run auto download via Hugging Face (requires accepting Gemma terms)
- Flexible inputs: built-in text box + external string input
- Optional image input: translate text found in images (multimodal)

## Installation

1) Clone into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/rookiestar28/ComfyUI-TranslateGemma.git
```

2) Install dependencies:

```bash
cd ComfyUI-TranslateGemma
pip install -r requirements.txt
```

3) Restart ComfyUI.

## Hugging Face Access (Gated Models)

TranslateGemma repos are gated under the Gemma terms.

1) Visit the model page and accept the license terms:
- `google/translategemma-4b-it`
- `google/translategemma-12b-it`
- `google/translategemma-27b-it`

2) Authenticate (recommended):

```bash
hf auth login
```

Alternatively, set one of these environment variables for the ComfyUI process:
- `HF_TOKEN`
- `HUGGINGFACE_HUB_TOKEN`

3) Restart ComfyUI after changing authentication.

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
| `external_text` | STRING | Optional external input; overrides `text` when connected |
| `image` | IMAGE | Optional image input; when connected, translates text found in the image |
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

### Notes on Chinese Variants

For better Traditional Chinese output consistency, the node maps:

- Chinese (Simplified) -> `zh`
- Chinese (Traditional) -> `zh-Hant`

## Performance Tips

- Leave `keep_model_loaded=true` for repeated use (avoids reload time).
- Use the 4B model if you are unsure about hardware limits.
- First run is slower due to download and weight initialization.

## Security / Reproducibility Notes

- The loader attempts `trust_remote_code=False` first and only falls back to `True` if required by the repo.
- You can pin a specific revision for reproducibility via `TRANSLATEGEMMA_REVISION=<commit-or-tag>`.

## License

This repository provides a ComfyUI integration. TranslateGemma models are governed by the Gemma Terms of Use.
