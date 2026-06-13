# TranslateGemma Example Workflows

These workflows are minimal templates that depend only on the `TranslateGemma` node. Import them from ComfyUI's workflow menu, then adjust language/model settings for your environment.

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
