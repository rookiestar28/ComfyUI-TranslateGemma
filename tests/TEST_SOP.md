# Test SOP

This document defines the source-of-truth local verification workflow for this repository. Use it before pushing and before marking implementation work as accepted.

## Scope

- Repository hygiene checks through `pre-commit`.
- Backend/unit tests through Python `unittest`.
- Frontend/E2E checks only when a Node or Playwright harness exists.
- Static/type validation when a changed surface has a typed or generated contract.

## Current Repo Facts

- The repository currently has no `package.json`, npm script, or Playwright harness.
- The repository currently has no one-command full-test wrapper script.
- The canonical unit-test command is `python -m unittest discover -s tests -p "test_*.py"`.
- For E2E-sensitive changes, use targeted static/unit coverage as the replacement lane until a real harness exists.

## Required Reading Order

1. `tests/TEST_SOP.md`
2. `tests/E2E_TESTING_NOTICE.md`
3. `tests/E2E_TESTING_SOP.md`

## Acceptance Rule

A change is not accepted until required checks pass and evidence is recorded.

Required shared gate:

1. `pre-commit run detect-secrets --all-files`
2. `pre-commit run --all-files --show-diff-on-failure`
3. `python -m unittest discover -s tests -p "test_*.py"`
4. Targeted static/unit validation for the changed surface.
5. E2E validation only when a real harness exists; otherwise record the non-applicability and replacement lane.

If this repo has no frontend/E2E harness for the changed surface, record the non-applicability and identify the replacement smoke, unit, or integration lane that catches the same user-facing risk.

## Problem-First Test Design Rule

All test scripts, test harnesses, and validation flows must be designed first to reproduce real failures and catch bugs early.

The purpose of testing is to expose defects, regressions, drift, and broken assumptions before users hit them. Tests must not be designed merely to produce a green validation result, satisfy a checklist, or prove that a happy path still passes.

Every bugfix or high-risk change must start from the question: which test would have caught this before release? If the existing gate missed the bug, update the targeted test or SOP flow so the same class of bug fails deterministically next time.

## Bugfix / Hotfix Rule

Bugfix and hotfix work must follow `Reproduce -> Pin -> Sweep`.

Acceptance evidence must include:

1. Pre-fix reproduction evidence.
2. Post-fix targeted regression evidence.
3. Final full-gate evidence.

A green full gate alone is not sufficient bugfix evidence unless the record also shows how the specific failure was reproduced and pinned.

## Documentation-Only Exception

If all touched files are documentation/planning text only and no code, tests, scripts, config, generated artifacts, dependency manifests, or runtime behavior changed, full test execution is optional.

Once executable or runtime-affecting files change, this exception does not apply.

## Environment Guardrails

- Keep the Python interpreter consistent across all commands.
- Prefer a project-local virtual environment: `.venv` on Windows and `.venv-wsl` on WSL/Linux when the repo supports dual-OS validation.
- Do not mix global and venv-installed `pre-commit` accidentally.
- Node.js must be 18+ before running frontend/E2E tests.
- On Windows, prefer repo-local `PRE_COMMIT_HOME` to avoid cache lock issues.
- On WSL, if `python` is missing but `python3` exists, create a local shim before running Playwright or harness commands.
- If pre-commit modifies files, review/stage/commit those changes and rerun hooks until clean.

## Evidence Recording

Validation notes must include date/time, OS/environment, exact command, and pass/fail result for each required stage. If a gate is intentionally skipped as non-applicable, record why and name the replacement validation lane.

## Manual Full Gate

```powershell
$env:PRE_COMMIT_HOME = (Join-Path (Get-Location).ProviderPath '.tmp\pre-commit')
.\.venv\Scripts\pre-commit.exe run detect-secrets --all-files
.\.venv\Scripts\pre-commit.exe run --all-files --show-diff-on-failure
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
node -v
if (Test-Path package.json) { npm test } else { "NO_PACKAGE_JSON_E2E_NON_APPLICABLE" }
git diff --check
git diff --cached --check
```
