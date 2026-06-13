# TranslateGemma

Translate text with Google's TranslateGemma instruction models inside ComfyUI. The node supports text translation, image text translation through the multimodal model path, Chinese Simplified/Traditional conversion-only mode, and long-text strategies.

## Inputs

- `text`: Built-in text input. Ignored when `external_text` is connected.
- `target_language`: Translation target language.
- `model_size`: `4B`, `12B`, or `27B`. Larger models can improve quality but need more memory.
- `device`: `default` follows ComfyUI's active device. `cpu` forces CPU. `gpu:N` selects the Nth host GPU option when ComfyUI exposes one.
- `source_language`: Use `Auto Detect` for text. Image translation requires an explicit source language.
- `external_text`: Optional connected text input. When connected, it overrides `text`.
- `prompt_mode`: `auto` tries the structured TranslateGemma chat template first and falls back to plain prompting. `structured` fails loudly if the template is unavailable. `plain` uses instruction text only.
- `max_new_tokens`: Maximum output tokens. `0` enables automatic sizing.
- `max_input_tokens`: Input token budget. `0` enables automatic sizing that reserves room for output.
- `truncate_input`: Truncate long inputs to fit the selected input token budget.
- `strict_context_limit`: Clamp output so input plus output stays within the model context window.
- `keep_model_loaded`: Keep the model in memory for repeated runs.
- `debug`: Print diagnostics with sensitive text redacted by default.
- `quantization`: `none` is the default and does not require BitsAndBytes. `bnb-8bit` and `bnb-4bit` require a CUDA GPU and optional `bitsandbytes` installation.

## Image Translation

Connect an image and set `source_language` explicitly. Auto Detect is text-only because the official TranslateGemma image template requires a source language code.

Useful image options:

- `image_resize_mode`: `letterbox` preserves aspect ratio and is recommended. `processor` uses the processor default. `stretch` forces the target size.
- `image_enhance`: applies mild contrast/sharpening after resize.
- `image_two_pass`: first extracts text in the source language, then translates that extracted text to the target language.

## Chinese Conversion-Only Mode

Enable `chinese_conversion_only` to run OpenCC Simplified/Traditional conversion without loading a translation model.

Directions:

- `auto_flip`: detect Simplified vs Traditional and convert to the opposite variant.
- `to_traditional`: force Simplified to Traditional.
- `to_simplified`: force Traditional to Simplified.

## Long Text

`long_text_strategy` controls long input behavior:

- `disable`: single model call.
- `auto-continue`: best-effort continuation if the model stops early on long text.
- `segmented`: translate paragraph-by-paragraph and preserve blank-line separators.

## Optional Quantization

Base installation does not install BitsAndBytes. Keep `quantization=none` unless you have a compatible CUDA environment.

For BnB modes, install the optional dependency into the same Python environment that runs ComfyUI:

```bash
pip install -r requirements-quantization.txt
```

If BitsAndBytes installation or CUDA setup fails, use `quantization=none` or a smaller model.

## Common Workflows

- Basic text translation: fill `text`, choose `target_language`, keep `device=default`, and run.
- Chained translation: connect another node to `external_text`.
- Image text translation: connect an image and set `source_language` explicitly.
- Chinese variant conversion: enable `chinese_conversion_only` and choose the conversion direction.
