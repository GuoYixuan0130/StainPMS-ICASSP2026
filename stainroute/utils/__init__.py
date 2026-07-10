"""Small deterministic utilities shared by StainRoute tooling."""

from .manifest import canonical_json_sha256, sha256_file

__all__ = ["canonical_json_sha256", "sha256_file"]
