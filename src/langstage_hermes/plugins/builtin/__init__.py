"""Bundled plug-ins shipped with ``langstage-hermes``.

Each subpackage here is a self-contained plug-in (manifest + ``__init__.py``
with a ``register(ctx)`` entry point). The plug-in loader discovers them by
walking this directory; nothing here is auto-imported.
"""
