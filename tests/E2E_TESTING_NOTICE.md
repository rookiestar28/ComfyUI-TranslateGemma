# E2E Testing Notice

All E2E execution must follow `tests/E2E_TESTING_SOP.md`.

Gate order and full acceptance requirements remain defined in `tests/TEST_SOP.md`.

This repository has a host-backed Playwright harness. `npm test` requires `COMFYUI_TEST_URL` and `COMFYUI_TEST_FRONTEND_VERSION`, and the URL must identify an explicitly supplied, already-running loopback ComfyUI host that loads this workspace's custom node.

## E2E Test Design Rule

- E2E tests must be designed to reproduce real user-visible failures and catch bugs early, not merely to pass validation.
- Do not add pass-only E2E checks that cannot fail for the bug class under review.
- For every user-reported or high-risk frontend regression, identify which E2E assertion would have caught it before release, then add or update that assertion.

## Host Boundary

- Do not auto-discover, start, stop, install, or build ComfyUI.
- Never execute a checkout under `REFERENCE/`.
- The harness may make the reversible `Comfy.VueNodes.Enabled` setting change and must restore/read back the original value.
- The harness must not enqueue prompts, execute nodes, save workflows, or expose user content in artifacts.

## Exception

Strictly documentation-only changes do not require entering the E2E workflow. Once code, tests, scripts, config, or runtime files change, this exception does not apply.

For transaction-sensitive features, acceptance evidence must include at least one action-level assertion of final outcome, not route-load evidence only.
