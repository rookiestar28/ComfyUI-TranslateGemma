# E2E Testing Notice

All E2E execution must follow `tests/E2E_TESTING_SOP.md`.

Gate order and full acceptance requirements remain defined in `tests/TEST_SOP.md`.

This repository currently has no `package.json`, npm script, or Playwright harness. Until one is added, E2E is non-applicable and must be replaced by targeted static/unit validation for the changed user-facing contract.

## E2E Test Design Rule

- E2E tests must be designed to reproduce real user-visible failures and catch bugs early, not merely to pass validation.
- Do not add pass-only E2E checks that cannot fail for the bug class under review.
- For every user-reported or high-risk frontend regression, identify which E2E assertion would have caught it before release, then add or update that assertion.

## Exception

Strictly documentation-only changes do not require entering the E2E workflow. Once code, tests, scripts, config, or runtime files change, this exception does not apply.

For transaction-sensitive features, acceptance evidence must include at least one action-level assertion of final outcome, not route-load evidence only.
