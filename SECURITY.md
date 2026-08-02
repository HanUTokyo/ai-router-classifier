# Security Policy

## Reporting a vulnerability

Please use GitHub's private security advisory flow instead of opening a public
issue. Include the affected version, reproduction steps, impact, and any known
mitigations.

## Deployment notes

- The service binds to loopback by default.
- A non-loopback bind requires `AI_ROUTER_API_KEY`.
- The Compose key `local-demo-key` is only for the loopback-bound demo. Replace
  it before any shared or remote deployment.
- Keep Ollama on a private network and do not expose port 11434 publicly.
- Runtime feedback, review queues, rule receipts, and `.env` files are ignored
  by Git and must not be committed.
