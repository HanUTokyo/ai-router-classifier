# Contributing

## Development setup

Python 3.11 or newer is required.

```bash
bash start.sh setup
bash start.sh test
```

The unit test suite uses mocked Ollama transports and does not download models.
For local integration testing, install Ollama and pull `qwen2.5:0.5b`.

## Pull requests

- Keep the classifier limited to `code`, `reason`, and `chat` in v1.
- Add tests for behavior changes and preserve existing API contracts.
- Do not use model predictions as gold labels.
- Do not edit active hard rules directly. Follow
  [the rule lifecycle](docs/RULE_LIFECYCLE.md).
- Changes to the model, prompt, or hard rules require a new, unexposed locked
  test before calibration can be renewed.

Run these checks before opening a pull request:

```bash
bash start.sh test
git diff --check
docker compose config --quiet
docker build -t ai-router-classifier:test .
```
