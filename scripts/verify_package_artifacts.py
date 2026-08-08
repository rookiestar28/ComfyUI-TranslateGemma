"""Inspect TG package artifacts without installing them into the host environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import re
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_NAME = "comfyui-translategemma"
EXPECTED_PYTHON = ">=3.12,<3.14"
FORBIDDEN_PATH_MARKERS = (
    ".planning/",
    "reference/",
    "REFERENCE/",
    ".sessions/",
    "tests/",
    "scripts/",
    "output/",
    ".tmp/",
    "node_modules/",
    "playwright-report/",
    "test-results/",
)
FORBIDDEN_BASENAME_PATTERNS = (
    re.compile(r"^ROADMAP(?:\.|$)", re.IGNORECASE),
    re.compile(r"COMMAND_LOG", re.IGNORECASE),
    re.compile(r"^\.env(?:\.|$)", re.IGNORECASE),
)
INTERNAL_METADATA_PATTERNS = (
    re.compile(r"TG-[0-9]{3}"),
    re.compile(r"(?:^|[ /])\.planning/", re.IGNORECASE),
    re.compile(r"(?:^|[ /])reference/docs/", re.IGNORECASE),
    re.compile(r"(?:^|[ /])REFERENCE/"),
)


def _project_metadata() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as metadata_file:
        return tomllib.load(metadata_file)["project"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strip_sdist_root(names: list[str]) -> list[str]:
    roots = {PurePosixPath(name).parts[0] for name in names if PurePosixPath(name).parts}
    if len(roots) != 1:
        raise ValueError("sdist must contain exactly one top-level directory")
    root = next(iter(roots))
    return [name[len(root) + 1 :] if name.startswith(f"{root}/") else "" for name in names]


def _validate_names(names: list[str]) -> None:
    for raw_name in names:
        name = raw_name.replace("\\", "/")
        member_path = PurePosixPath(name)
        first_part = member_path.parts[0] if member_path.parts else ""
        if (
            member_path.is_absolute()
            or ".." in member_path.parts
            or ":" in first_part
            or "\x00" in name
        ):
            raise ValueError(f"unsafe package member path: {raw_name}")
        name = name.lstrip("./")
        if not name:
            continue
        if any(marker in f"{name}/" for marker in FORBIDDEN_PATH_MARKERS):
            raise ValueError(f"forbidden package path: {name}")
        if any(pattern.search(PurePosixPath(name).name) for pattern in FORBIDDEN_BASENAME_PATTERNS):
            raise ValueError(f"forbidden package file: {name}")


def _validate_metadata(text: str, expected_version: str) -> None:
    metadata = Parser().parsestr(text)
    if metadata.get("Name", "").lower().replace("_", "-") != EXPECTED_NAME:
        raise ValueError("artifact Name metadata mismatch")
    if metadata.get("Version") != expected_version:
        raise ValueError("artifact Version metadata mismatch")
    expected_python_parts = {part.strip() for part in EXPECTED_PYTHON.split(",")}
    actual_python_parts = {part.strip() for part in metadata.get("Requires-Python", "").split(",")}
    if actual_python_parts != expected_python_parts:
        raise ValueError("artifact Requires-Python metadata mismatch")
    for dependency in metadata.get_all("Requires-Dist", []):
        normalized = re.split(r"[ (<>=!~;\[]", dependency.lower(), maxsplit=1)[0]
        if normalized in {"torch", "torchvision", "torchaudio", "comfyui-frontend-package"}:
            raise ValueError(f"host-owned dependency is forbidden: {normalized}")
    if any(pattern.search(text) for pattern in INTERNAL_METADATA_PATTERNS):
        raise ValueError("public package metadata exposes an internal identifier or path")


def _require(names: list[str], required: tuple[str, ...], artifact_kind: str) -> None:
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"{artifact_kind} missing required files: {', '.join(missing)}")


def _inspect_sdist(path: Path, expected_version: str) -> dict:
    with tarfile.open(path, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        relative_names = _strip_sdist_root([member.name for member in members])
        _validate_names(relative_names)
        _require(
            relative_names,
            (
                "pyproject.toml",
                "PKG-INFO",
                "__init__.py",
                "_version.py",
                "README.md",
                "LICENSE",
                "requirements.txt",
                "requirements-quantization.txt",
                "nodes/translate_node.py",
                "utils/model_loader.py",
                "web/docs/TranslateGemma.md",
                "web/extensions/translategemma_help.js",
                "example_workflows/README.md",
                "example_workflows/basic_text_translation.json",
                "example_workflows/chinese_conversion_only.json",
                "example_workflows/external_text_override_template.json",
                "example_workflows/image_translation_explicit_source.json",
                "example_workflows/long_text_segmented.json",
            ),
            "sdist",
        )
        metadata_member = next(member for member in members if member.name.endswith("/PKG-INFO"))
        metadata_stream = archive.extractfile(metadata_member)
        if metadata_stream is None:
            raise ValueError("sdist PKG-INFO is unreadable")
        _validate_metadata(metadata_stream.read().decode("utf-8"), expected_version)
    return {"kind": "sdist", "file_count": len(relative_names)}


def _inspect_wheel(path: Path, expected_version: str) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        _validate_names(names)
        package = "comfyui_translategemma"
        _require(
            names,
            (
                f"{package}/__init__.py",
                f"{package}/_version.py",
                f"{package}/pyproject.toml",
                f"{package}/README.md",
                f"{package}/LICENSE",
                f"{package}/requirements.txt",
                f"{package}/requirements-quantization.txt",
                f"{package}/nodes/translate_node.py",
                f"{package}/utils/model_loader.py",
                f"{package}/web/docs/TranslateGemma.md",
                f"{package}/web/extensions/translategemma_help.js",
                f"{package}/example_workflows/README.md",
                f"{package}/example_workflows/basic_text_translation.json",
                f"{package}/example_workflows/chinese_conversion_only.json",
                f"{package}/example_workflows/external_text_override_template.json",
                f"{package}/example_workflows/image_translation_explicit_source.json",
                f"{package}/example_workflows/long_text_segmented.json",
            ),
            "wheel",
        )
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise ValueError("wheel must contain exactly one METADATA file")
        _validate_metadata(archive.read(metadata_names[0]).decode("utf-8"), expected_version)

        verify_parent = ROOT / ".tmp"
        verify_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="tg060-wheel-readback-", dir=verify_parent) as temp_dir:
            verify_root = Path(temp_dir)
            archive.extractall(verify_root)
            distributions = list(importlib.metadata.distributions(path=[str(verify_root)]))
            matching = [
                dist
                for dist in distributions
                if dist.metadata.get("Name", "").lower().replace("_", "-") == EXPECTED_NAME
            ]
            if len(matching) != 1 or matching[0].version != expected_version:
                raise ValueError("isolated wheel distribution metadata readback mismatch")
            version_path = verify_root / package / "_version.py"
            spec = importlib.util.spec_from_file_location("_tg_package_version_readback", version_path)
            if spec is None or spec.loader is None:
                raise ValueError("isolated version module is not importable")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if module.__version__ != expected_version:
                raise ValueError("isolated runtime version readback mismatch")
    return {"kind": "wheel", "file_count": len(names)}


def verify(dist_dir: Path) -> list[dict]:
    project = _project_metadata()
    expected_version = project["version"]
    if project.get("requires-python") != EXPECTED_PYTHON:
        raise ValueError("source Requires-Python does not match accepted matrix")
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    wheels = sorted(dist_dir.glob("*.whl"))
    if len(sdists) != 1 or len(wheels) != 1:
        raise ValueError("dist must contain exactly one sdist and one wheel")
    results = []
    for artifact, inspector in ((sdists[0], _inspect_sdist), (wheels[0], _inspect_wheel)):
        result = inspector(artifact, expected_version)
        result.update({"artifact": artifact.name, "sha256": _sha256(artifact)})
        results.append(result)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist_dir", type=Path)
    args = parser.parse_args()
    try:
        results = verify(args.dist_dir.resolve())
    except (OSError, ValueError, KeyError, tarfile.TarError, zipfile.BadZipFile) as exc:
        print(f"PACKAGE_ARTIFACT_FAILURE: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "PASS", "artifacts": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
