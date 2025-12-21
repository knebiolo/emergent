"""Viewer V3 package: experimental refactor scaffold.

This package provides a pure `mesh_builder` implementation and a
compatibility shim `viewer` that preserves the public API while the
internal implementation is migrated to a modern OpenGL renderer.
"""

__all__ = ["mesh_builder", "viewer"]
