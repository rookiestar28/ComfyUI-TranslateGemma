#!/usr/bin/env python3
"""
Lightweight smoke check: ensure internal relative-import targets exist.

This intentionally avoids importing the full node (which may require heavyweight
deps like torch) and instead statically verifies that files referenced by
relative imports are present in the repository.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _iter_python_files(*relative_dirs: str) -> list[Path]:
    files: list[Path] = []
    for rel in relative_dirs:
        base = REPO_ROOT / rel
        if not base.exists():
            continue
        files.extend(sorted(p for p in base.rglob("*.py") if p.is_file()))
    return files


def _expected_path_for_import(module: str) -> Path | None:
    """
    Map a relative-import module like '..utils.foo' to a repo path like utils/foo.py.
    """
    if not module:
        return None
    module = module.lstrip(".")
    if not module.startswith("utils."):
        return None
    remainder = module.removeprefix("utils.")
    if not remainder:
        return REPO_ROOT / "utils" / "__init__.py"
    return REPO_ROOT / "utils" / (remainder.replace(".", "/") + ".py")


def main() -> int:
    missing: list[str] = []
    checked: int = 0

    for py_file in _iter_python_files("nodes"):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except Exception as e:
            missing.append(f"{py_file}: failed to parse: {type(e).__name__}: {e}")
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if not node.module:
                continue
            if not node.level or node.level < 1:
                continue

            # Example: from ..utils.language_utils import ...
            dots = "." * node.level
            module = f"{dots}{node.module}"
            target = _expected_path_for_import(module)
            if target is None:
                continue

            checked += 1
            if not target.exists():
                missing.append(f"{py_file}: import '{module}' missing file '{target.relative_to(REPO_ROOT)}'")

    if missing:
        sys.stderr.write("Internal import check failed:\n")
        for line in missing:
            sys.stderr.write(f"- {line}\n")
        return 2

    print(f"OK: checked {checked} internal utils imports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

