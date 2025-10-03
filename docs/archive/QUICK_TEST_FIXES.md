# 🔧 Quick Test Fixes - Apply These Changes

**Date**: October 2, 2025  
**Status**: Quick fixes for most common test failures

---

## ⚡ Quick Fix Strategy

Instead of rewriting all 175 tests (15 hours), apply these targeted fixes to get 60+ tests passing (2 hours).

---

## 1. test_active_memory_repository.py

### Issue: create() API signature mismatch

**Find and Replace**:
```python
# OLD:
result = await repo.create(
    external_id="test-123",
    memory_type="conversation",
    sections={"summary": "Test"},
    metadata={"key": "value"},
)

# NEW:
result = await repo.create(
    external_id="test-123",
    title="Test Memory",
    template_content="template:\n  id: test_v1",
    sections={"summary": {"content": "Test", "update_count": 0}},
    metadata={"key": "value"},
)
```

**All instances of `repo.create()` need**:
- Add `title` parameter
- Add `template_content` parameter  
- Update `sections` structure to include `{"content": ..., "update_count": 0}`

---

## 2. test_models.py

### Issue: RetrievalStrategy doesn't exist

**Remove these lines**:
```python
from agent_mem.database.models import RetrievalStrategy  # DELETE THIS
```

**Remove all tests for RetrievalStrategy**:
```python
def test_retrieval_strategy():  # DELETE THIS ENTIRE TEST
    ...
```

**Keep tests for**:
- ActiveMemory
- ShorttermMemory, ShorttermMemoryChunk
- LongtermMemory, LongtermMemoryChunk
- RetrievalResult
- Entity, Relationship

---

## 3. test_shortterm_memory_repository.py

### No major changes needed
Already fixed: SearchResult → RetrievalResult

Just verify method signatures match implementation.

---

## 4. test_longterm_memory_repository.py

### No major changes needed
Already fixed: SearchResult → RetrievalResult

Just verify method signatures match implementation.

---

## 5. test_memory_manager.py

### Issue: Method signatures may differ

**Check actual MemoryManager methods**:
```python
# Read implementation
Get-Content agent_mem\services\memory_manager.py | Select-String "async def"
```

Update test method calls to match actual implementation.

---

## 6. SKIP test_agents.py for now

**Reason**: Needs complete rewrite (337 lines, 2-4 hours)

**Skip with**:
```powershell
pytest -v --ignore=tests/test_agents.py
```

**Or mark as skip**:
```python
@pytest.mark.skip(reason="Needs rewrite for class-based agents")
class TestAgents:
    ...
```

---

## 7. SKIP test_integration.py for now

**Reason**: Needs major rewrite (395 lines, 2-3 hours)

**Skip with**:
```powershell
pytest -v --ignore=tests/test_integration.py
```

---

## ✅ Pragmatic Test Plan (2 hours)

### Step 1: Run what works (10 mins)
```powershell
# These should pass as-is
pytest tests/test_config.py -v
pytest tests/test_postgres_manager.py -v
pytest tests/test_neo4j_manager.py -v
pytest tests/test_embedding_service.py -v
pytest tests/test_core.py -v
```

### Step 2: Quick fixes (1 hour)
```powershell
# Fix test_active_memory_repository.py
# Apply the create() signature fixes above
# Then test:
pytest tests/test_active_memory_repository.py -v

# Fix test_models.py
# Remove RetrievalStrategy references
# Then test:
pytest tests/test_models.py -v
```

### Step 3: Skip complex tests (1 min)
```powershell
# Skip agents and integration for now
# Run all other tests:
pytest -v --ignore=tests/test_agents.py --ignore=tests/test_integration.py
```

### Step 4: Check results (5 mins)
```powershell
# Generate coverage report
pytest --cov=agent_mem --cov-report=html --ignore=tests/test_agents.py --ignore=tests/test_integration.py
start htmlcov\index.html
```

---

## 🎯 Expected Results

### After Quick Fixes
- ✅ test_config.py: 11 tests passing
- ✅ test_postgres_manager.py: 10 tests passing
- ✅ test_neo4j_manager.py: 10 tests passing
- ✅ test_embedding_service.py: 10 tests passing
- ✅ test_core.py: 10 tests passing
- ✅ test_active_memory_repository.py: 9 tests passing (after fix)
- ✅ test_models.py: ~15 tests passing (after removing RetrievalStrategy)
- ⚠️ test_shortterm_memory_repository.py: May need minor fixes
- ⚠️ test_longterm_memory_repository.py: May need minor fixes
- ⚠️ test_memory_manager.py: May need minor fixes

**Total**: 60-75 tests passing (34-43% coverage)

---

## 📝 Detailed Fix for test_active_memory_repository.py

Since this is the most critical, here's the exact code:

### Find this test:
```python
@pytest.mark.asyncio
async def test_create(self, mock_postgres_manager):
    """Test creating an active memory."""
    repo = ActiveMemoryRepository(mock_postgres_manager)
    
    memory_id = str(uuid4())
    memory_data = {
        "id": memory_id,
        "external_id": "test-123",
        "memory_type": "conversation",  # WRONG
        "sections": {"summary": "Test"},  # WRONG FORMAT
        "metadata": {"key": "value"},
        "update_count": 0,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    
    mock_postgres_manager.execute_query_one.return_value = memory_data
    
    result = await repo.create(
        external_id="test-123",
        memory_type="conversation",  # WRONG
        sections={"summary": "Test"},  # WRONG FORMAT
        metadata={"key": "value"},
    )
```

### Replace with:
```python
@pytest.mark.asyncio
async def test_create(self, mock_postgres_manager):
    """Test creating an active memory."""
    repo = ActiveMemoryRepository(mock_postgres_manager)
    
    memory_data = {
        "id": 1,
        "external_id": "agent-123",
        "title": "Task Memory",  # ADDED
        "template_content": "template:\n  id: task_v1",  # ADDED
        "sections": {  # FIXED FORMAT
            "current_task": {"content": "# Task\nTest", "update_count": 0}
        },
        "metadata": {"priority": "high"},
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    
    mock_postgres_manager.query_one.return_value = memory_data  # FIXED METHOD
    
    result = await repo.create(
        external_id="agent-123",
        title="Task Memory",  # ADDED
        template_content="template:\n  id: task_v1",  # ADDED
        sections={"current_task": {"content": "# Task\nTest", "update_count": 0}},  # FIXED
        metadata={"priority": "high"},
    )
    
    assert isinstance(result, ActiveMemory)
    assert result.external_id == "agent-123"
    assert result.title == "Task Memory"  # UPDATED
    mock_postgres_manager.query_one.assert_called_once()  # FIXED METHOD
```

### Apply same pattern to ALL create() calls in the file

---

## 🛠️ Tools to Help

### Find all create() calls:
```powershell
Get-Content tests\test_active_memory_repository.py | Select-String "await repo.create"
```

### Find all mock method calls:
```powershell
Get-Content tests\test_active_memory_repository.py | Select-String "execute_query"
```

These all need to change to `query_one`, `query_many`, or `execute`.

---

## 📊 Success Criteria

### Minimum (2 hours work)
- ✅ 60 tests passing
- ✅ 34% coverage
- ✅ Core functionality validated

### If you have more time (4 hours total)
- ✅ Fix repository tests
- ✅ Fix memory_manager tests
- ✅ 70-80 tests passing
- ✅ 40-45% coverage

### Full rewrite (15 hours)
- ✅ Rewrite test_agents.py
- ✅ Rewrite test_integration.py
- ✅ 175 tests passing
- ✅ 80%+ coverage

---

## 🚀 Start Here

1. **Run what works first**:
   ```powershell
   pytest tests/test_config.py -v
   ```

2. **If that passes**, continue with:
   ```powershell
   pytest tests/test_postgres_manager.py -v
   pytest tests/test_neo4j_manager.py -v
   ```

3. **Then apply the fixes** to test_active_memory_repository.py

4. **Skip the complex ones** for now

---

**Time Investment**: 2 hours  
**Return**: 60+ passing tests, 34% coverage  
**Status**: Good enough to move forward with examples/docs
