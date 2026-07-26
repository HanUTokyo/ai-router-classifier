"""Compatibility entrypoint for older launch configurations."""

from router.app import create_app

app = create_app()

__all__ = ["app"]
