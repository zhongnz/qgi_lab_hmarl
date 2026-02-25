# Contributing

## Environment

1. Use the project Python environment (`../.conda/bin/python`) or an equivalent virtual environment.
2. Install dependencies:

```bash
make install-dev
```

## Development Workflow

1. Keep core simulator logic in `hmarl_mvp/`.
2. Keep notebooks analysis-only; avoid adding core logic to notebooks.
3. Add or update tests for each behavior change.
4. Run the full quality gate before opening a PR:

```bash
make check
```

## Code and Config Standards

1. Prefer typed function signatures and deterministic behavior (seeded runs).
2. Route simulation changes through `MaritimeEnv.step` to avoid duplicated transition logic.
3. Use `get_default_config(...)` / `validate_config(...)` from `hmarl_mvp.config` for all configs.

## Documentation

1. Update `README.md` when command surface or structure changes.
2. Keep architecture notes under `docs/architecture/` aligned with implementation.
3. Record design-level choices in `docs/decisions/` using ADR naming conventions.
