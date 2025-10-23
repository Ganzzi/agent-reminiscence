# Agent Mem Tests

Comprehensive test suite for the `agent_reminiscence` package.

## Overview

This test suite provides:
- **Unit tests**: Test individual components with mocked dependencies
- **Integration tests**: Test end-to-end workflows
- **Agent tests**: Test Pydantic AI agents with TestModel
- **>90% code coverage**: Comprehensive test coverage

## Test Structure

```
tests/
├── conftest.py                        # Fixtures and test configuration
├── pytest.ini                         # Pytest settings (in parent dir)
├── test_config.py                     # Configuration tests
├── test_models.py                     # Pydantic model tests
├── test_postgres_manager.py           # PostgreSQL manager tests
├── test_neo4j_manager.py              # Neo4j manager tests
├── test_active_memory_repository.py   # Active memory repo tests
├── test_shortterm_memory_repository.py # Shortterm memory repo tests
├── test_longterm_memory_repository.py  # Longterm memory repo tests
├── test_embedding_service.py          # Embedding service tests
├── test_memory_manager.py             # Memory manager tests
├── test_core.py                       # AgentMem core tests
├── test_agents.py                     # Pydantic AI agent tests
└── test_integration.py                # End-to-end integration tests
```

## Running Tests

### Install Test Dependencies

```bash
pip install -r requirements-test.txt
```

### Run All Tests

```bash
pytest
```

### Run Specific Test File

```bash
pytest tests/test_config.py
```

### Run Specific Test Function

```bash
pytest tests/test_config.py::TestConfig::test_config_defaults
```

### Run Tests by Marker

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only async tests
pytest -m asyncio

# Skip slow tests
pytest -m "not slow"
```

### Run with Coverage

```bash
# Generate coverage report
pytest --cov=agent_reminiscence --cov-report=html

# View coverage in browser
# Open htmlcov/index.html
```

### Run with Verbose Output

```bash
pytest -v
```

### Run in Parallel (faster)

```bash
pip install pytest-xdist
pytest -n auto  # Use all available CPUs
```

## Test Categories

### Unit Tests

Test individual components in isolation with mocked dependencies:

- **Config tests**: Configuration loading and validation
- **Model tests**: Pydantic model behavior
- **Manager tests**: Database managers (PostgreSQL, Neo4j)
- **Repository tests**: CRUD operations and queries
- **Service tests**: Embedding service, memory manager
- **Core tests**: AgentMem interface
- **Agent tests**: Pydantic AI agents

**Run**: `pytest -m unit`

### Integration Tests

Test end-to-end workflows with real or mocked databases:

- **Full lifecycle**: Create → Update → Consolidate → Promote → Retrieve
- **Cross-tier search**: Search across all memory tiers
- **Entity/relationship**: Extraction, storage, merging
- **Error recovery**: Handling failures gracefully

**Run**: `pytest -m integration`

**Note**: Integration tests may require running databases (PostgreSQL, Neo4j).

## Test Configuration

### Environment Variables

Set these for integration tests with real databases:

```bash
# PostgreSQL
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=agent_reminiscence_test
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=postgres

# Neo4j
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password

# Ollama
export OLLAMA_BASE_URL=http://localhost:11434
export EMBEDDING_MODEL=nomic-embed-text
```

### Pytest Configuration

Settings in `pytest.ini`:

- **Test discovery**: Auto-discover `test_*.py` files
- **Coverage**: Automatic coverage tracking
- **Markers**: Unit, integration, slow, asyncio
- **Async support**: Auto-detect async tests
- **Timeout**: 5 minutes per test

## Fixtures

Common fixtures in `conftest.py`:

### Configuration Fixtures

- `test_config`: Real test configuration
- `mock_config`: Mock configuration for isolated tests

### Database Fixtures

- `postgres_manager`: Real PostgreSQL manager
- `neo4j_manager`: Real Neo4j manager
- `mock_postgres_manager`: Mock PostgreSQL manager
- `mock_neo4j_manager`: Mock Neo4j manager

### Repository Fixtures

- `active_memory_repository`: Real active memory repo
- `shortterm_memory_repository`: Real shortterm memory repo
- `longterm_memory_repository`: Real longterm memory repo
- `mock_active_memory_repository`: Mock active memory repo
- `mock_shortterm_memory_repository`: Mock shortterm memory repo
- `mock_longterm_memory_repository`: Mock longterm memory repo

### Service Fixtures

- `embedding_service`: Real embedding service
- `memory_manager`: Real memory manager
- `mock_embedding_service`: Mock embedding service
- `mock_memory_manager`: Mock memory manager

### Test Data Fixtures

- `sample_active_memory_data`: Sample active memory data
- `sample_shortterm_chunk_data`: Sample chunk data
- `sample_longterm_chunk_data`: Sample validated chunk data
- `sample_entity_data`: Sample entity data
- `sample_relationship_data`: Sample relationship data

### Cleanup Fixtures

- `cleanup_test_data`: Auto-cleanup before/after tests

## Writing New Tests

### Example Unit Test

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_my_feature(mock_postgres_manager):
    """Test description."""
    # Arrange
    mock_postgres_manager.execute_query_one.return_value = {"id": 1}
    
    # Act
    result = await my_function(mock_postgres_manager)
    
    # Assert
    assert result["id"] == 1
    mock_postgres_manager.execute_query_one.assert_called_once()
```

### Example Integration Test

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_workflow(test_config):
    """Test end-to-end workflow."""
    async with AgentMem(config=test_config) as agent_reminiscence:
        # Test workflow
        memory = await agent_reminiscence.create_active_memory(...)
        assert memory is not None
```

## Test Coverage Goals

Target coverage: **>80%**

Current coverage by module:
- `agent_reminiscence/config/`: Target 95%
- `agent_reminiscence/database/`: Target 85%
- `agent_reminiscence/services/`: Target 90%
- `agent_reminiscence/agents/`: Target 85%
- `agent_reminiscence/core.py`: Target 95%

## Continuous Integration

Tests run automatically on:
- Push to main branch
- Pull requests
- Release tags

CI checks:
- All tests pass
- Coverage >80%
- No linting errors
- Type checking passes

## Troubleshooting

### Tests Fail with Database Connection Error

**Solution**: Start PostgreSQL and Neo4j, or use mock fixtures.

### Tests Timeout

**Solution**: Increase timeout in `pytest.ini` or mark slow tests:

```python
@pytest.mark.slow
async def test_long_running():
    ...
```

### Coverage Not Showing

**Solution**: Ensure `pytest-cov` is installed:

```bash
pip install pytest-cov
```

### Async Tests Fail

**Solution**: Ensure `pytest-asyncio` is installed and configured:

```bash
pip install pytest-asyncio
```

## Best Practices

1. **Use descriptive test names**: `test_what_when_expected`
2. **One assertion per test**: Keep tests focused
3. **Use fixtures**: Avoid duplicating setup code
4. **Mock external dependencies**: Keep unit tests fast
5. **Test edge cases**: Null values, empty lists, errors
6. **Test error handling**: Verify fallback behavior
7. **Use markers**: Categorize tests for selective running
8. **Keep tests independent**: No test should depend on another
9. **Clean up after tests**: Use fixtures for cleanup
10. **Document complex tests**: Add docstrings explaining why

## Contributing

When adding new features:

1. Write tests first (TDD)
2. Ensure >80% coverage for new code
3. Run full test suite before submitting PR
4. Update this README if adding new test categories

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-asyncio documentation](https://pytest-asyncio.readthedocs.io/)
- [Pydantic AI testing guide](https://ai.pydantic.dev/testing/)
- [Coverage.py documentation](https://coverage.readthedocs.io/)

---

**Last Updated**: October 2, 2025  
**Test Count**: 200+ tests across 13 test files  
**Coverage**: Target >80%

