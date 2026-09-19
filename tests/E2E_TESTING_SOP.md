# E2E Testing SOP

<!-- CURRENT-TEST-GOVERNANCE:START -->
## Current Governance Scope

A change limited to pure text/documentation files, a version-field-only `pyproject.toml` update, or
both does not enter this E2E workflow and requires no planning, roadmap item, record/log,
independent review, documentation test contract, browser installation, or full gate. Behavior-
bearing metadata changes do not qualify. For non-exempt work, applicable E2E runs through the
authoritative Windows Full Gate. Hosted CI repetitions are optional diagnostics and are not
acceptance prerequisites or pushed-commit evidence. Explicit item-scoped live/supported-host checks
remain separate when required.
<!-- CURRENT-TEST-GOVERNANCE:END -->

This SOP defines the E2E workflow policy for this repository.

Current repo fact: `package.json` and a Playwright harness exist. The harness validates the frontend contract against an explicitly supplied, already-running local ComfyUI host. Missing live-host evidence blocks frontend acceptance.

## Problem-First Test Design Rule

E2E scripts and mocked harness flows must be designed to reproduce failures and catch bugs early. The goal is not to make the harness pass; the goal is to make the harness fail when a real user-facing contract breaks.

When adding or reviewing E2E coverage, prefer assertions that prove final user-visible behavior, request routing, payload shape, state synchronization, and failure feedback. Avoid pass-only checks that only prove the page loaded or a mocked happy path returned.

## Requirements

- Node.js 18+.
- npm 9+ when the repo uses npm.
- Python command available (`python` or a local shim to `python3`) when the harness serves files through Python.
- Playwright Chromium installed with `npx playwright install chromium` when Playwright is used.

## Required Host Contract

- The user or maintainer explicitly supplies the already-running host URL.
- `COMFYUI_TEST_URL` must be a loopback HTTP(S) URL without credentials, query, or fragment.
- `COMFYUI_TEST_FRONTEND_VERSION` must be the host's actual semantic frontend version.
- The host must load this workspace's TranslateGemma custom node.
- Do not run, build, install, or serve a `REFERENCE/` checkout.
- The harness preserves and restores renderer settings and must not queue prompts or save workflows.

## Windows (PowerShell)

```powershell
node -v
npm -v
python --version

$env:COMFYUI_TEST_URL = 'http://127.0.0.1:8188'
$env:COMFYUI_TEST_FRONTEND_VERSION = '<actual-version>'
npm ci
npx playwright install chromium
npm test
```

## WSL2 (bash)

```bash
source ~/.nvm/nvm.sh
nvm use 18
node -v
python3 --version

mkdir -p .tmp/bin
ln -sf "$(command -v python3)" .tmp/bin/python

export COMFYUI_TEST_URL='http://127.0.0.1:8188'
export COMFYUI_TEST_FRONTEND_VERSION='<actual-version>'
npm ci
npx playwright install chromium

mkdir -p .tmp/playwright
TMPDIR=.tmp/playwright TMP=.tmp/playwright TEMP=.tmp/playwright \
  PATH=".tmp/bin:$PATH" npm test
```

## Troubleshooting

- `python: command not found` on WSL: create `.tmp/bin/python` as a shim to `python3`.
- Port bind failure: use the repo-documented E2E port override or stop the conflicting process.
- Browser missing: run `npx playwright install chromium`.
- Dependency drift: remove `node_modules` and rerun `npm install`.

## Missing Host

If the approved local host is not available, `npm run test:contract` may still be used as an intermediate diagnostic lane. It is not a substitute for `npm test`, and the item remains `IMPLEMENTED_NOT_ACCEPTED` until live E2E and visual evidence pass.
