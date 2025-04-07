# Prompt Injection Detection Study

Prompt injection detection algorithms study.

## Usage

```sh
uv install
uv run huggingface-cli login  # login with your token
uv run huggingface-cli download --repo-type dataset deepset/prompt-injections --local-dir data/deepset
uv run huggingface-cli download --repo-type dataset reshabhs/SPML_Chatbot_Prompt_Injection --local-dir data/SPML
uv run huggingface-cli download --repo-type dataset xTRam1/safe-guard-prompt-injection --local-dir data/safe-guard
mkdir -p models
uv run hy app.hy
```

Generate corresponding Python script with `uv run hy2py app.hy`.
Generate human-friendly Python script with LLM (kimi, gremini, etc).

## References

* [BERT-based detection](https://github.com/sinanw/llm-security-prompt-injection)
