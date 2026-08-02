FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AI_ROUTER_CONFIG=/app/config/router.yaml \
    AI_ROUTER_BIND_HOST=0.0.0.0

WORKDIR /app

RUN groupadd --system router \
    && useradd --system --gid router --home-dir /app router

COPY pyproject.toml README.md ./
COPY router ./router
COPY config ./config
COPY data ./data

RUN python -m pip install --no-cache-dir . \
    && chown -R router:router /app

USER router

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=6 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2)"

CMD ["python", "-m", "router", "--config", "config/router.yaml", "serve"]
