# Prompt Injection Detection Study

Prompt injection detection algorithms study.

## Usage

```sh
uv run huggingface-cli login  # login with your token
uv run huggingface-cli download deepset/prompt-injections --local-dir data --repo-type dataset
uv run hy app.hy
```

Generate Python script with `uv run hy2py app.hy`.

## References

* [BERT-based detection](https://github.com/sinanw/llm-security-prompt-injection)
