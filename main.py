"""Compatibility entrypoint for `uvicorn main:app`."""

from router.app import create_app

app = create_app()

__all__ = ["app"]
