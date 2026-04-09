# Development Guide

## Prerequisites
- Python 3.12
- uv
- PostgreSQL
- Neo4j
- Ollama

## Setup
```bash
uv sync --all-groups --extra dev --python 3.12
```

Optional extras:

```bash
# MCP server support
uv sync --all-groups --extra dev --extra mcp --python 3.12

# Streamlit UI support
uv sync --all-groups --extra dev --extra streamlit --python 3.12

# Documentation tooling
uv sync --all-groups --extra dev --extra docs --python 3.12
```

## Unit Tests (PR Gate)
```bash
uv run --extra dev --python 3.12 pytest -m "not integration and not e2e" -q
```

## Integration / E2E Tests
```bash
# Full local suite
uv run --extra dev --python 3.12 pytest -q

# Integration only
uv run --extra dev --python 3.12 pytest -m integration -q

# End-to-end only
uv run --extra dev --python 3.12 pytest -m e2e -q
```

## Coverage
```bash
uv run --extra dev --python 3.12 pytest -m "not integration and not e2e" --cov --cov-report=term-missing
```

## Test Layout
```text
tests/
├── conftest.py
├── unit/
├── integration/
└── e2e/
```

Markers are applied by directory via `tests/conftest.py`:
- `tests/unit/` → `@pytest.mark.unit`
- `tests/integration/` → `@pytest.mark.integration`
- `tests/e2e/` → `@pytest.mark.e2e`

## CI Behavior
- Pull requests should gate on `uv run --extra dev --python 3.12 pytest -m "not integration and not e2e" -q`
- Integration and e2e tests require local services and are intended for targeted local verification
- Streamlit UI tests are skipped unless the `streamlit` extra is installed

## Known Issues
- `streamlit` is intentionally optional because it pulls heavy transitive build dependencies
- Integration and e2e tests require PostgreSQL, Neo4j, and Ollama to be available locally
- Some tests emit existing deprecation/runtime warnings; the PR-gated suite currently passes despite those warnings
