# Alumnx Vector DB CI/CD Rollout Plan

## Objective

Implement CI/CD for `alumnx-vector-db` in a way that is operationally safe and aligned with the latest requirement:

- unit tests must run automatically
- production merges should be protected by passing checks
- production deployment should happen only after CI succeeds
- development automation can continue, but should not become a hard merge blocker unless the team opts in later

## Current Repository Controls

- `.github/workflows/ci.yml`
  Runs the Python unit test suite with `uv run pytest -q` on every push and on pull requests targeting `dev` or `main`.
- `.github/workflows/deploy-dev.yml`
  Deploys only after a successful `CI` run for the `dev` branch.
- `.github/workflows/deploy-prod.yml`
  Deploys only after a successful `CI` run for the `main` branch.

This means deployment is already gated on test success. Merge protection still depends on GitHub branch protection settings and is not enforced by workflow files alone.

## EC2 Service Context

The EC2 inventory shows two vector database deployments that matter for this rollout:

- `Edge Prod`, service `alumnx-vector-db`, port `8001`, type `Dev`, endpoint `http://13.126.130.56:8001/docs`
- `Agriculture`, service `alumnx-vector-db`, port `8012`, type `Prod`, endpoint `http://13.205.59.184:8012/docs`

Neighbor services that can be affected by retrieval or downstream integration testing:

- `Core Prod` -> `alumnx-prod-api` on port `8000`
- `Core Prod` -> `alumni-search` on port `8002`

Operationally, the production CI/CD decision should be centered on the `Agriculture` host running `alumnx-vector-db` on port `8012`.

## Recommended Rollout

1. Keep `CI` as the required unit-test workflow for the repository.
2. Make `main` the only protected branch for mandatory passing checks right now.
3. Require `CI / Unit Tests` before merge into `main`.
4. Keep `dev` deployment automated, but do not require the status check for merge until the team is ready.
5. Keep production deployment chained from successful CI by using the existing `workflow_run` design.

## GitHub Configuration Required

The following GitHub settings should be applied in the repository after this PR is merged:

1. Open repository `Settings -> Branches`.
2. Add or update the protection rule for `main`.
3. Enable `Require a pull request before merging`.
4. Enable `Require status checks to pass before merging`.
5. Mark `CI / Unit Tests` as a required status check.
6. Optionally enable `Require branches to be up to date before merging`.

Do not make the same rule mandatory on `dev` yet unless the team explicitly approves it.

## Production Validation Plan

1. Merge the PR into `main`.
2. Confirm the `CI` workflow passes on the merge commit.
3. Confirm `Deploy Prod` starts only after `CI` succeeds.
4. Verify the production service health at `http://13.205.59.184:8012/docs`.
5. Validate at least one ingest flow and one retrieval flow against the production API.
6. Check PM2 process status and logs on the production EC2 instance if deployment fails.

## Future Hardening

- Add a lightweight `/health` smoke test step after deploy.
- Add branch-specific environment documentation for `DEV_EC2_HOST` and `PROD_EC2_HOST`.
- Add rollback instructions for the production PM2 process.
- Add integration tests only after stable test fixtures exist for downstream services.

## Test Plan

- Run `uv run pytest -q`
- Review `.github/workflows/ci.yml`, `.github/workflows/deploy-dev.yml`, and `.github/workflows/deploy-prod.yml`
- Confirm docs now match the live workflow design and production rollout path
