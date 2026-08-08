"""Inspect an official BitsAndBytes wheel without installing or importing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath


EXPECTED_NAME = "bitsandbytes"
EXPECTED_VERSION = "0.49.2"
EXPECTED_PYTHON = ">=3.10"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as wheel_file:
        for chunk in iter(lambda: wheel_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_member_names(names: list[str]) -> None:
    for raw_name in names:
        normalized = raw_name.replace("\\", "/")
        member = PurePosixPath(normalized)
        first = member.parts[0] if member.parts else ""
        if member.is_absolute() or ".." in member.parts or ":" in first or "\x00" in normalized:
            raise ValueError(f"unsafe wheel member path: {raw_name}")


def verify(wheel_dir: Path) -> dict:
    wheels = sorted(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("wheel directory must contain exactly one wheel")
    wheel = wheels[0]
    if not re.fullmatch(r"bitsandbytes-0\.49\.2-py3-none-win_amd64\.whl", wheel.name):
        raise ValueError("wheel is not the expected official Windows x86-64 artifact")

    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        _validate_member_names(names)
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise ValueError("wheel must contain exactly one METADATA file")
        metadata_text = archive.read(metadata_names[0]).decode("utf-8")
        metadata = Parser().parsestr(metadata_text)
        if metadata.get("Name", "").lower() != EXPECTED_NAME:
            raise ValueError("wheel package name mismatch")
        if metadata.get("Version") != EXPECTED_VERSION:
            raise ValueError("wheel package version mismatch")
        if metadata.get("Requires-Python") != EXPECTED_PYTHON:
            raise ValueError("wheel Python requirement mismatch")
        if "bitsandbytes/__init__.py" not in names:
            raise ValueError("wheel is missing the package initializer")
        native_libraries = [name for name in names if name.lower().endswith(".dll")]
        if not native_libraries:
            raise ValueError("Windows wheel contains no native libraries")

    return {
        "artifact": wheel.name,
        "bytes": wheel.stat().st_size,
        "file_count": len(names),
        "native_library_count": len(native_libraries),
        "name": EXPECTED_NAME,
        "requires_python": EXPECTED_PYTHON,
        "sha256": _sha256(wheel),
        "version": EXPECTED_VERSION,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel_dir", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.wheel_dir.resolve())
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"BITSANDBYTES_WHEEL_FAILURE: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "PASS", "wheel": result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
