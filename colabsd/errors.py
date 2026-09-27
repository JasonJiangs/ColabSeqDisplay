"""User-facing errors.

Every message must say what went wrong *and* what to do about it: these surface
inside a Colab notebook in front of someone who is not reading the source.
"""

from __future__ import annotations


class ColabSDError(Exception):
    """Base class for every failure a notebook user can cause or fix."""


class DataError(ColabSDError):
    """The variant table or library specification is inconsistent."""


class StructureError(ColabSDError):
    """A structure or 3Di input could not be turned into a usable 3Di string."""


class BackboneError(ColabSDError):
    """A backbone could not be loaded, or is not usable in this environment."""


class ConfigError(ColabSDError):
    """A best-config registry entry is missing or malformed."""


class BundleError(ColabSDError):
    """A model bundle could not be written or read back."""
