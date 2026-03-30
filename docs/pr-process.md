# PR Process

This repository has separate expectations for `dev` and `main` pull requests.

## Dev PRs

- Raise feature and integration PRs into `dev`.
- Before opening the PR, run:
  ```bash
  python -m pytest -q
  ```
- Keep the PR focused. Do not mix unrelated refactors with the intended change.
- If CI fails, fix the branch and push again to the same PR branch.

## Production PRs

- Only raise PRs into `main` after the change is already validated on `dev` or otherwise confirmed ready for production.
- Before opening the PR, run:
  ```bash
  python -m pytest -q
  ```
- Make sure the GitHub Actions checks pass before merge.
- Treat `main` as production-facing: no experimental or partially validated changes.

## Recommended Flow

1. Branch from the latest target branch.
2. Make only the intended change.
3. Run `python -m pytest -q`.
4. Push the branch.
5. Open the PR to `dev` or `main` as appropriate.
6. Wait for GitHub Actions to pass before merge.

## Related Docs

- [LOCAL_SETUP.md](../LOCAL_SETUP.md)
- [CLAUDE.md](../CLAUDE.md)
