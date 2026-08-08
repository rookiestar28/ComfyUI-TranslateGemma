#!/usr/bin/env python3
"""Prepare and verify the repo-local TG-058 Transformers compatibility matrix.

Sources:
- https://pypi.org/project/transformers/
- https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md
- https://huggingface.co/docs/transformers/v5.0.0/en/main_classes/model
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_RELATIVE_ROOT = Path(".tmp") / "tg058"
PYTHON_VERSIONS = ("3.12", "3.13")
TRANSFORMERS_VERSIONS = ("4.57.3", "5.0.0", "5.14.1")
TORCH_VERSION = "2.10.0"
RESULT_SCHEMA_VERSION = 1
REQUIRED_SURFACES = (
    "imports",
    "auto_processor",
    "auto_image_text_model",
    "chat_template",
    "load_kwargs",
    "safetensors_local_load",
    "generation",
    "bnb_config_without_bnb",
    "project_model_loader",
)
_ALLOWED_RESULT_KEYS = {
    "schema_version",
    "lane",
    "python",
    "transformers",
    "torch",
    "command",
    "result",
    "surfaces",
    "error_category",
}
_PATH_ENV_KEYS = (
    "CONDA_PKGS_DIRS",
    "PIP_CACHE_DIR",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "TORCH_HOME",
    "XDG_CACHE_HOME",
    "TMP",
    "TEMP",
)
_SAFE_INHERITED_ENV_KEYS = (
    "COMSPEC",
    "LANG",
    "NUMBER_OF_PROCESSORS",
    "OS",
    "PATH",
    "PATHEXT",
    "PROCESSOR_ARCHITECTURE",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "WINDIR",
)
_SAFE_CATEGORY = re.compile(r"^[A-Z][A-Z0-9_]*$")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"[A-Za-z]:[\\/]")


class MatrixValidationError(ValueError):
    """Raised when a lane or output violates the frozen matrix contract."""


class MatrixSetupError(RuntimeError):
    """A categorized setup failure safe to expose without its raw cause."""

    def __init__(self, category: str):
        super().__init__(category)
        self.category = category


@dataclass(frozen=True)
class MatrixRow:
    python_version: str
    transformers_version: str

    @property
    def lane_id(self) -> str:
        compact_python = self.python_version.replace(".", "")
        return f"py{compact_python}-tf{self.transformers_version}"


def matrix_rows() -> tuple[MatrixRow, ...]:
    return tuple(
        MatrixRow(python_version, transformers_version)
        for python_version in PYTHON_VERSIONS
        for transformers_version in TRANSFORMERS_VERSIONS
    )


def _matrix_root(repo_root: Path) -> Path:
    return (Path(repo_root).resolve() / MATRIX_RELATIVE_ROOT).resolve()


def resolve_output_path(repo_root: Path, requested: str | os.PathLike[str]) -> Path:
    repo_root = Path(repo_root).resolve()
    root = _matrix_root(repo_root)
    requested_path = Path(requested)
    if not requested_path.is_absolute():
        requested_path = repo_root / requested_path
    resolved = requested_path.resolve()
    if resolved == root or not resolved.is_relative_to(root) or resolved.suffix.lower() != ".json":
        raise MatrixValidationError("OUTPUT_PATH_OUTSIDE_MATRIX_ROOT")
    return resolved


def build_matrix_environment(repo_root: Path, *, probe_offline: bool) -> dict[str, str]:
    repo_root = Path(repo_root).resolve()
    root = _matrix_root(repo_root)
    environment = {
        key: os.environ[key]
        for key in _SAFE_INHERITED_ENV_KEYS
        if key in os.environ
    }
    environment.update(
        {
            "CONDA_PKGS_DIRS": str(root / "cache" / "conda-pkgs"),
            "CONDA_ENVS_PATH": str(root / "conda-envs"),
            "PIP_CACHE_DIR": str(root / "cache" / "pip"),
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "HF_HOME": str(root / "cache" / "huggingface"),
            "HUGGINGFACE_HUB_CACHE": str(root / "cache" / "huggingface" / "hub"),
            "TRANSFORMERS_CACHE": str(root / "cache" / "transformers"),
            "TORCH_HOME": str(root / "cache" / "torch"),
            "XDG_CACHE_HOME": str(root / "cache" / "xdg"),
            "TMP": str(root / "temp"),
            "TEMP": str(root / "temp"),
            "HOME": str(root / "home"),
            "USERPROFILE": str(root / "home"),
            "APPDATA": str(root / "home" / "AppData" / "Roaming"),
            "LOCALAPPDATA": str(root / "home" / "AppData" / "Local"),
            "CONDARC": str(root / "config" / "condarc"),
            "PYTHONUTF8": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "USERNAME": "tg058",
            "USER": "tg058",
            "LOGNAME": "tg058",
        }
    )
    if probe_offline:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    return environment


def _ensure_matrix_directories(repo_root: Path) -> None:
    root = _matrix_root(repo_root)
    for relative in (
        Path("cache/conda-pkgs"),
        Path("cache/pip"),
        Path("cache/huggingface/hub"),
        Path("cache/transformers"),
        Path("cache/torch"),
        Path("cache/xdg"),
        Path("conda-envs"),
        Path("bases"),
        Path("lanes"),
        Path("work"),
        Path("temp"),
        Path("home/AppData/Roaming"),
        Path("home/AppData/Local"),
        Path("config"),
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)


def _python_executable(prefix: Path, *, venv: bool) -> Path:
    if os.name == "nt":
        return prefix / ("Scripts/python.exe" if venv else "python.exe")
    return prefix / ("bin/python")


def _run_setup(command: list[str], environment: dict[str, str], category: str) -> None:
    try:
        completed = subprocess.run(
            command,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            shell=False,
        )
    except OSError as error:
        raise MatrixSetupError(category) from None
    if completed.returncode != 0:
        raise MatrixSetupError(category)


def _installed_version(python: Path, distribution: str, environment: dict[str, str]) -> str | None:
    command = [
        str(python),
        "-c",
        (
            "import importlib.metadata as m; "
            f"print(m.version({distribution!r}))"
        ),
    ]
    try:
        completed = subprocess.run(
            command,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            check=False,
            shell=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _prepare_base(repo_root: Path, python_version: str) -> Path:
    root = _matrix_root(repo_root)
    environment = build_matrix_environment(repo_root, probe_offline=False)
    conda = shutil.which("conda", path=environment.get("PATH"))
    if not conda:
        raise MatrixSetupError("CONDA_UNAVAILABLE")
    prefix = root / "bases" / f"py{python_version.replace('.', '')}"
    python = _python_executable(prefix, venv=False)

    if not python.is_file():
        print(f"PREPARE py{python_version.replace('.', '')}-base")
        _run_setup(
            [
                conda,
                "create",
                "--yes",
                "--override-channels",
                "--channel",
                "conda-forge",
                "--prefix",
                str(prefix),
                f"python={python_version}",
                "pip",
            ],
            environment,
            "PYTHON_BASE_SETUP_FAILURE",
        )
    if not python.is_file():
        raise MatrixSetupError("PYTHON_BASE_MISSING")

    torch_version = _installed_version(python, "torch", environment)
    if torch_version is None or not torch_version.startswith(TORCH_VERSION):
        print(f"PREPARE py{python_version.replace('.', '')}-torch")
        _run_setup(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--only-binary=:all:",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
                f"torch=={TORCH_VERSION}",
            ],
            environment,
            "TORCH_SETUP_FAILURE",
        )
    torch_version = _installed_version(python, "torch", environment)
    if torch_version is None or not torch_version.startswith(TORCH_VERSION):
        raise MatrixSetupError("TORCH_VERSION_MISMATCH")
    return prefix


def _prepare_lane(repo_root: Path, row: MatrixRow, base_prefix: Path) -> Path:
    root = _matrix_root(repo_root)
    environment = build_matrix_environment(repo_root, probe_offline=False)
    lane_prefix = root / "lanes" / row.lane_id
    base_python = _python_executable(base_prefix, venv=False)
    lane_python = _python_executable(lane_prefix, venv=True)

    if not lane_python.is_file():
        print(f"PREPARE {row.lane_id}-venv")
        _run_setup(
            [
                str(base_python),
                "-m",
                "venv",
                "--system-site-packages",
                str(lane_prefix),
            ],
            environment,
            "LANE_VENV_SETUP_FAILURE",
        )
    if not lane_python.is_file():
        raise MatrixSetupError("LANE_PYTHON_MISSING")

    installed = _installed_version(lane_python, "transformers", environment)
    if installed != row.transformers_version:
        print(f"PREPARE {row.lane_id}-packages")
        _run_setup(
            [
                str(lane_python),
                "-m",
                "pip",
                "install",
                "--only-binary=:all:",
                "--upgrade",
                f"transformers=={row.transformers_version}",
                "jinja2",
            ],
            environment,
            "TRANSFORMERS_SETUP_FAILURE",
        )
    installed = _installed_version(lane_python, "transformers", environment)
    if installed != row.transformers_version:
        raise MatrixSetupError("TRANSFORMERS_VERSION_MISMATCH")
    return lane_prefix


def prepare_matrix(repo_root: Path) -> None:
    _ensure_matrix_directories(repo_root)
    bases = {
        version: _prepare_base(repo_root, version)
        for version in PYTHON_VERSIONS
    }
    for row in matrix_rows():
        _prepare_lane(repo_root, row, bases[row.python_version])


def _contains_private_content(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            _contains_private_content(key) or _contains_private_content(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_private_content(item) for item in value)
    if not isinstance(value, str):
        return False
    lowered = value.casefold()
    forbidden = (
        "traceback",
        "token=",
        "authorization",
        "http://",
        "https://",
        "private-",
        "private_",
        "userprofile",
    )
    return (
        any(marker in lowered for marker in forbidden)
        or bool(_WINDOWS_ABSOLUTE_PATH.search(value))
        or value.startswith("/")
    )


def validate_worker_result(row: MatrixRow, payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != _ALLOWED_RESULT_KEYS:
        raise MatrixValidationError("WORKER_SCHEMA_MISMATCH")
    if _contains_private_content(payload):
        raise MatrixValidationError("WORKER_PRIVACY_VIOLATION")
    if payload["schema_version"] != RESULT_SCHEMA_VERSION:
        raise MatrixValidationError("WORKER_SCHEMA_VERSION_MISMATCH")
    if payload["lane"] != row.lane_id:
        raise MatrixValidationError("WORKER_LANE_MISMATCH")
    if not isinstance(payload["python"], str) or not payload["python"].startswith(
        f"{row.python_version}."
    ):
        raise MatrixValidationError("WORKER_PYTHON_VERSION_MISMATCH")
    if payload["transformers"] != row.transformers_version:
        raise MatrixValidationError("WORKER_TRANSFORMERS_VERSION_MISMATCH")
    if not isinstance(payload["torch"], str) or not payload["torch"].startswith(TORCH_VERSION):
        raise MatrixValidationError("WORKER_TORCH_VERSION_MISMATCH")
    if payload["command"] != "python scripts/transformers_matrix_probe.py":
        raise MatrixValidationError("WORKER_COMMAND_MISMATCH")
    surfaces = payload["surfaces"]
    if not isinstance(surfaces, dict) or tuple(surfaces) != REQUIRED_SURFACES:
        raise MatrixValidationError("WORKER_SURFACE_SCHEMA_MISMATCH")
    if any(value not in {"PASS", "FAIL", "SKIP"} for value in surfaces.values()):
        raise MatrixValidationError("WORKER_SURFACE_RESULT_INVALID")
    if payload["result"] == "PASS":
        if any(value != "PASS" for value in surfaces.values()) or payload["error_category"] is not None:
            raise MatrixValidationError("WORKER_FALSE_PASS")
    elif payload["result"] == "FAIL":
        category = payload["error_category"]
        if not isinstance(category, str) or not _SAFE_CATEGORY.fullmatch(category):
            raise MatrixValidationError("WORKER_FAILURE_CATEGORY_INVALID")
        if all(value == "PASS" for value in surfaces.values()):
            raise MatrixValidationError("WORKER_FAILURE_WITHOUT_SURFACE")
    else:
        raise MatrixValidationError("WORKER_RESULT_INVALID")
    return payload


def validate_aggregate(rows: Iterable[MatrixRow], payloads: list[dict[str, Any]]) -> None:
    expected = tuple(row.lane_id for row in rows)
    actual = tuple(payload.get("lane") for payload in payloads)
    if len(actual) != len(expected) or set(actual) != set(expected) or len(set(actual)) != len(actual):
        raise MatrixValidationError("AGGREGATE_ROW_MISMATCH")


def format_matrix_table(payloads: list[dict[str, Any]]) -> str:
    lines = [
        "| Lane | Python | Transformers | PyTorch | Surfaces | Result |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for payload in payloads:
        passed = sum(value == "PASS" for value in payload["surfaces"].values())
        lines.append(
            f"| {payload['lane']} | {payload['python']} | {payload['transformers']} | "
            f"{payload['torch']} | {passed}/{len(REQUIRED_SURFACES)} | {payload['result']} |"
        )
    return "\n".join(lines)


def write_aggregate(repo_root: Path, output: Path, aggregate: dict[str, Any]) -> None:
    output = resolve_output_path(repo_root, output)
    if _contains_private_content(aggregate):
        raise MatrixValidationError("AGGREGATE_PRIVACY_VIOLATION")
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=output.parent,
        prefix=".tg058-",
        suffix=".json.tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(serialized)
    os.replace(temp_path, output)


def _failure_payload(row: MatrixRow, category: str) -> dict[str, Any]:
    surfaces = {surface: "SKIP" for surface in REQUIRED_SURFACES}
    surfaces[REQUIRED_SURFACES[0]] = "FAIL"
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "lane": row.lane_id,
        "python": f"{row.python_version}.0",
        "transformers": row.transformers_version,
        "torch": TORCH_VERSION,
        "command": "python scripts/transformers_matrix_probe.py",
        "result": "FAIL",
        "surfaces": surfaces,
        "error_category": category,
    }


def run_lane(repo_root: Path, row: MatrixRow) -> dict[str, Any]:
    root = _matrix_root(repo_root)
    lane_python = _python_executable(root / "lanes" / row.lane_id, venv=True)
    if not lane_python.is_file():
        return _failure_payload(row, "LANE_MISSING")
    work_dir = root / "work" / row.lane_id
    work_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(lane_python),
        str(Path(repo_root).resolve() / "scripts" / "transformers_matrix_probe.py"),
        "--lane",
        row.lane_id,
        "--expected-python",
        row.python_version,
        "--expected-transformers",
        row.transformers_version,
        "--expected-torch",
        TORCH_VERSION,
        "--repo-root",
        str(Path(repo_root).resolve()),
        "--work-dir",
        str(work_dir),
    ]
    environment = build_matrix_environment(repo_root, probe_offline=True)
    try:
        completed = subprocess.run(
            command,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            check=False,
            shell=False,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        return _failure_payload(row, "WORKER_TIMEOUT")
    except OSError:
        return _failure_payload(row, "WORKER_START_FAILURE")
    if completed.returncode not in {0, 1}:
        return _failure_payload(row, "WORKER_PROCESS_FAILURE")
    try:
        payload = json.loads(completed.stdout.strip())
        return validate_worker_result(row, payload)
    except (json.JSONDecodeError, MatrixValidationError):
        return _failure_payload(row, "WORKER_OUTPUT_INVALID")


def verify_matrix(repo_root: Path) -> tuple[list[dict[str, Any]], bool]:
    rows = matrix_rows()
    payloads = [run_lane(repo_root, row) for row in rows]
    validate_aggregate(rows, payloads)
    return payloads, all(payload["result"] == "PASS" for payload in payloads)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Create/update the fixed repo-local matrix before verification.",
    )
    parser.add_argument(
        "--output",
        default=str(MATRIX_RELATIVE_ROOT / "results.json"),
        help="Aggregate JSON path; must remain under .tmp/tg058/.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        output = resolve_output_path(REPO_ROOT, args.output)
        _ensure_matrix_directories(REPO_ROOT)
        if args.prepare:
            prepare_matrix(REPO_ROOT)
        payloads, passed = verify_matrix(REPO_ROOT)
        aggregate = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "matrix": {
                "python": list(PYTHON_VERSIONS),
                "transformers": list(TRANSFORMERS_VERSIONS),
                "torch": TORCH_VERSION,
            },
            "rows": payloads,
            "result": "PASS" if passed else "FAIL",
        }
        write_aggregate(REPO_ROOT, output, aggregate)
        print(format_matrix_table(payloads))
        return 0 if passed else 1
    except MatrixSetupError as error:
        print(f"TG058_MATRIX_SETUP_FAILURE:{error.category}", file=sys.stderr)
        return 2
    except MatrixValidationError as error:
        category = str(error) if _SAFE_CATEGORY.fullmatch(str(error)) else "MATRIX_VALIDATION_FAILURE"
        print(f"TG058_MATRIX_VALIDATION_FAILURE:{category}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
