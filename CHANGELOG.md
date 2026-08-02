# Changelog

All notable changes to this project are documented in this file.

## 1.0.0 - 2026-08-02

### Added

- Calibrated `code`, `reason`, and `chat` routing with hard-rule short circuits.
- Structured small-model classification through a local Ollama backend.
- Explicit weak-rule and default fallback behavior.
- OpenAI Chat Completions and Ollama Chat compatibility endpoints.
- Human review, frozen dataset manifests, benchmark gates, and rule lifecycle.
- Docker Compose demo, GitHub Actions, bilingual README, and public API docs.

### Changed

- The default profile now uses only `qwen2.5:0.5b` for a lightweight demo.
- The specialized multi-model gateway is available as `config/router.full.yaml`.
- Evaluation reports now distinguish fallback rate from general degradation.

### Removed

- Historical runtime, monitoring, and third-party integration experiments from
  the public release tree. They remain available in Git history.
