# TranslateGemma Example Workflows

These workflows are minimal templates that depend only on the `TranslateGemma` node. Import them from ComfyUI's workflow menu, then adjust language/model settings for your environment.

The JSON files keep both legacy positional widget values and the additive named widget map written by frontend 1.49.2+. Their text fields are intentionally empty so distributed templates never contain prompt or user content; enter your own text after importing.

## Files

- `basic_text_translation.json`: simple English to Japanese text translation.
- `external_text_override_template.json`: shows the node configured for upstream `external_text` use. Connect any upstream `STRING` output to `external_text`; connected text overrides the built-in `text` field.
- `chinese_conversion_only.json`: Simplified/Traditional conversion without loading a translation model.
- `image_translation_explicit_source.json`: image translation template with an explicit source language. Connect an upstream `IMAGE` output to the `image` input before running.
- `long_text_segmented.json`: paragraph-by-paragraph long text translation.

## Notes

- The first model run may download gated Hugging Face model files. Accept the model license and authenticate with Hugging Face first when required.
- Keep `device=default` unless you need to force `cpu` or a specific `gpu:N` option.
- Keep `quantization=none` unless you installed optional BitsAndBytes dependencies into the ComfyUI Python environment.
