"""Runtime version derived from the Comfy Registry project metadata."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path


_SEMANTIC_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


def _read_project_version() -> str:
    metadata_path = Path(__file__).with_name("pyproject.toml")
    try:
        with metadata_path.open("rb") as metadata_file:
            version = tomllib.load(metadata_file)["project"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError("TranslateGemma package version metadata is unavailable") from exc

    if not isinstance(version, str) or not _SEMANTIC_VERSION.fullmatch(version):
        raise RuntimeError("TranslateGemma package version must use X.Y.Z semantic form")
    return version


# CRITICAL: keep pyproject.toml as the only editable version source; duplicate literals drifted before.
__version__ = _read_project_version()
