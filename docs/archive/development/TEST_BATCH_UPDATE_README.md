# Test Suite for Batch Update Features (Phases 6-10)

This directory contains comprehensive tests for the new batch update and enhanced consolidation features implemented in Phases 6-10.

## Test Files

### 1. `test_batch_update_features.py`
Comprehensive unit tests for core batch update functionality.

**Test Classes:**
- `TestBatchUpdate` - Batch section update operations
- `TestSectionTracking` - Section ID tracking through workflow
- `TestUpdateCountTracking` - Update count increment/reset
- `TestMetadataTracking` - Entity metadata.updates array
- `TestConsolidationWorkflow` - Enhanced consolidation logic
- `TestConcurrencyControl` - Consolidation locking mechanism
- `TestBackwardCompatibility` - Old API still works
- `TestResetOperations` - Section/chunk cleanup
- `TestEndToEndWorkflow` - Complete workflow integration

**Coverage:**
- ✅ Batch update in active_memory repository
- ✅ Section_id tracking in shortterm chunks
- ✅ Update_count increment/reset
- ✅ Consolidation threshold calculation
- ✅ Section filtering (only updated sections)
- ✅ Promotion threshold check
- ✅ Metadata.updates array population
- ✅ Concurrency control with asyncio.Lock
- ✅ Backward compatibility

### 2. `test_mcp_batch_update.py`
Tests for MCP server batch update handler.

**Test Classes:**
- `TestMCPBatchUpdate` - MCP server batch update handler
- `TestMCPIntegration` - Full integration tests

**Coverage:**
- ✅ Successful batch update via MCP
- ✅ Input validation
- ✅ Error handling
- ✅ Consolidation info in response
- ✅ Single atomic transaction

## Running Tests

### Run All New Tests
```bash
pytest tests/test_batch_update_features.py tests/test_mcp_batch_update.py -v
```

### Run Specific Test Class
```bash
pytest tests/test_batch_update_features.py::TestBatchUpdate -v
```

### Run Specific Test
```bash
pytest tests/test_batch_update_features.py::TestBatchUpdate::test_update_sections_batch -v
```

### Run with Coverage
```bash
pytest tests/test_batch_update_features.py --cov=agent_mem --cov-report=html
```

### Run Integration Tests Only
```bash
pytest tests/test_batch_update_features.py -m integration -v
```

### Run All Tests (Including Existing)
```bash
pytest tests/ -v
```

## Test Configuration

Tests use the `test_config` fixture from `conftest.py`. Make sure your test environment has:

- PostgreSQL test database
- Neo4j test instance (optional for some tests)
- Ollama for embeddings (optional, mocked in unit tests)

## Mock Strategy

Most tests use mocks to avoid requiring a full database setup:

- `MagicMock` for database managers
- `AsyncMock` for async methods
- `patch` for method interception

## Test Categories

### Unit Tests (Fast)
- No database required
- Uses mocks extensively
- Run frequently during development

```bash
pytest tests/test_batch_update_features.py -v
```

### Integration Tests (Slow)
- Requires database
- Tests real interactions
- Run before commits

```bash
pytest tests/ -m integration -v
```

## Key Test Scenarios

### 1. Batch Update Flow
```python
sections = [
    {"section_id": "progress", "new_content": "Updated"},
    {"section_id": "notes", "new_content": "Updated"}
]
memory = await repo.update_sections(memory_id=1, section_updates=sections)
```

### 2. Threshold Calculation
```python
threshold = avg_section_update_count * num_sections
total_count = sum(section["update_count"] for section in sections.values())
should_consolidate = total_count >= threshold
```

### 3. Section Tracking
```python
chunk = await repo.create_chunk(
    ...,
    section_id="progress"  # NEW
)
chunks = await repo.get_chunks_by_section_id(memory_id, "progress")
```

### 4. Metadata Updates
```python
entity.metadata["updates"].append({
    "date": "2025-10-05T10:00:00Z",
    "old_confidence": 0.75,
    "new_confidence": 0.85
})
```

## Expected Results

All tests should pass with the new implementation:

```
test_batch_update_features.py::TestBatchUpdate::test_update_sections_batch PASSED
test_batch_update_features.py::TestBatchUpdate::test_threshold_calculation PASSED
test_batch_update_features.py::TestSectionTracking::test_chunk_with_section_id PASSED
test_batch_update_features.py::TestSectionTracking::test_get_chunks_by_section_id PASSED
test_batch_update_features.py::TestUpdateCountTracking::test_increment_update_count PASSED
test_batch_update_features.py::TestUpdateCountTracking::test_reset_update_count PASSED
test_batch_update_features.py::TestMetadataTracking::test_entity_metadata_updates PASSED
test_batch_update_features.py::TestConsolidationWorkflow::test_selective_consolidation PASSED
test_batch_update_features.py::TestConsolidationWorkflow::test_promotion_threshold_check PASSED
test_batch_update_features.py::TestConcurrencyControl::test_consolidation_lock PASSED
test_batch_update_features.py::TestBackwardCompatibility::test_single_section_update_delegates_to_batch PASSED
test_batch_update_features.py::TestResetOperations::test_reset_section_counts PASSED
test_batch_update_features.py::TestResetOperations::test_delete_all_chunks PASSED
test_batch_update_features.py::TestEndToEndWorkflow::test_batch_update_triggers_consolidation PASSED

test_mcp_batch_update.py::TestMCPBatchUpdate::test_handle_update_memory_sections_success PASSED
test_mcp_batch_update.py::TestMCPBatchUpdate::test_handle_update_memory_sections_validation PASSED
test_mcp_batch_update.py::TestMCPBatchUpdate::test_handle_update_memory_sections_invalid_section PASSED
test_mcp_batch_update.py::TestMCPBatchUpdate::test_handle_update_memory_sections_empty_content PASSED
test_mcp_batch_update.py::TestMCPBatchUpdate::test_consolidation_info_calculation PASSED

==================== XX passed in X.XXs ====================
```

## Troubleshooting

### Import Errors
```bash
# Make sure agent_mem is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Async Test Failures
```bash
# Install pytest-asyncio if not already installed
pip install pytest-asyncio
```

### Mock Issues
```bash
# Ensure unittest.mock is available (Python 3.3+)
python --version
```

## CI/CD Integration

Add to your CI pipeline:

```yaml
# .github/workflows/test.yml
- name: Run Batch Update Tests
  run: |
    pytest tests/test_batch_update_features.py -v
    pytest tests/test_mcp_batch_update.py -v
```

## Future Test Additions

### Recommended Additional Tests:
1. **Performance Tests** - Measure batch update vs loop performance
2. **Load Tests** - Test with many sections/concurrent updates
3. **Failure Recovery** - Test rollback on partial failure
4. **Memory Leak Tests** - Verify lock cleanup
5. **Race Condition Tests** - More thorough concurrency testing

## Contributing

When adding new features:
1. Add corresponding tests in appropriate file
2. Update this README with new test descriptions
3. Ensure all tests pass before PR
4. Aim for >80% code coverage

## Questions?

See main documentation:
- `docs/IMPLEMENTATION_PROGRESS.md` - Implementation details
- `docs/PHASE_6_10_SUMMARY.md` - Feature summary
- `docs/BATCH_UPDATE_AND_CONSOLIDATION_REFACTOR_PLAN.md` - Original plan
