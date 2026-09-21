# Contributing to SEAM

Thanks for your interest! This project is a research codebase for studying
memory collapse and contamination in self-evolving LLM agents. Contributions of
reproducible experiments, honest analyses, and tested code are very welcome.

## Ground rules

- **Honesty over results.** Do not weaken a result, drop null rows, or reframe
  a failure as a success to "improve" the story. The repository already carries
  a documented history of post-hoc claim changes (see `docs/audit-history.md`);
  we treat reproducibility as a feature, not an inconvenience.
- **No unverified claims.** README, `docs/`, and any generated `reports/` must
  match the implemented behavior. If you add a metric or experiment, make sure
  it is actually emitted and exercised by a test.
- **Tests, not simulations, in CI.** Unit and integration tests are mocked so
  they run without Ollama or a GPU. Do not add a test that requires a live model.

## Development setup

```bash
python -m venv venv
source venv/bin/activate        # or .\venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

The repository also ships an `environment.yml` (conda) and a `Dockerfile`.

## Quality gates

Run all four before submitting:

```bash
python -m ruff check src scripts tests
python -m ruff format --check src scripts tests
python -m mypy src
python -m pytest -q
```

CI (`.github/workflows/ci.yml`) runs the same four commands on every push and
pull request. A pull request that fails a gate will not be merged.

## Where things live

- Core package: `src/seam/` (agents, envs, memory, sharing, poisoning,
  orchestration, metrics, analysis, logging, utils)
- Entry points: `scripts/`
- Tests: `tests/`
- Documentation: `docs/` — keep claims there aligned with code and artifacts
- Manuscript: `reports/manuscript/`

## Reading order before you dig in

1. `docs/README.md` — documentation index and conventions
2. `docs/overview.md` — research problem and repository map
3. `docs/architecture.md` — components and episode data flow

## Making changes

1. Fork and branch: `git checkout -b topic/your-change`.
2. Make the change; add or update tests in `tests/`.
3. Run the four quality gates above.
4. If your change affects claims, update the relevant `docs/` or `reports/`
   text in the same pull request.
5. Open a pull request against `master` and describe what changed and why.

## Commit message style

Use a short imperative summary with a conventional prefix:

```
feat: add per-round regret metric
fix: strip format boilerplate before Self-BLEU scoring
docs: clarify contamination measurement
```

Body paragraphs (when needed) explain the *why*, not the what.

## Data and artifacts

`runs/`, `figures/`, and `coverage.xml` are generated and gitignored. Do not
commit run artifacts; commit the code, configs, tests, and documentation that
(should) regenerate them from a pinned commit.