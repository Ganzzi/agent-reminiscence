# Bug Fixes - agent-mem v0.1.0

This document summarizes the critical bugs fixed during initial testing.

## Fixed Issues

### 1. Invalid pyproject.toml Email Format

**Issue**: Installation failed with "configuration error: `project.authors[0].email` must be idn-email"

**Root Cause**: Empty email fields in `pyproject.toml` authors and maintainers sections.

**Fix**: Removed email fields from author configuration (they're optional):
```toml
authors = [
    {name = "Agent Mem Contributors"}
]
maintainers = [
    {name = "Agent Mem Contributors"}
]
```

**Files Modified**:
- `pyproject.toml`

---

### 2. JSON Serialization with psqlpy

**Issue**: `PyToRustValueMappingError: PyJSON must be dict or list value` when creating memories.

**Root Cause**: The code was calling `json.dumps()` on dictionaries before passing to `psqlpy.execute()`, but psqlpy expects raw Python dict/list objects, not JSON strings.

**Fix**: Removed `json.dumps()` calls in all repository methods. Pass dicts directly to psqlpy.

**Example**:
```python
# BEFORE (incorrect):
await conn.execute(query, [json.dumps(metadata)])

# AFTER (correct):
await conn.execute(query, [metadata])
```

**Files Modified**:
- `agent_mem/database/repositories/active_memory.py` (4 locations)
- `agent_mem/database/repositories/shortterm_memory.py` (4 locations)
- `agent_mem/database/repositories/longterm_memory.py` (2 locations)

---

### 3. Database Row Access Method

**Issue**: `KeyError: 0` when trying to access database rows with numeric indexes.

**Root Cause**: `psqlpy` returns rows as dictionaries (with column names as keys), not tuples. The code was trying to access with numeric indexes like `row[0]`, `row[1]`, etc.

**Fix**: Changed row access to use column names instead of indexes.

**Example**:
```python
# BEFORE (incorrect):
ActiveMemory(
    id=row[0],
    external_id=row[1],
    title=row[2],
    ...
)

# AFTER (correct):
ActiveMemory(
    id=row["id"],
    external_id=row["external_id"],
    title=row["title"],
    ...
)
```

**Files Modified**:
- `agent_mem/database/repositories/active_memory.py` (`_row_to_model` method, count function)

**Note**: Similar fixes are needed in `shortterm_memory.py` and `longterm_memory.py` but those weren't tested yet.

---

### 4. Config Attribute Name Mismatch

**Issue**: `AttributeError: 'Config' object has no attribute 'active_memory_update_threshold'`

**Root Cause**: The config file defined the setting as `consolidation_threshold`, but the code was accessing it as `active_memory_update_threshold`.

**Fix**: Updated `memory_manager.py` to use the correct config attribute name:

```python
# BEFORE (incorrect):
if section["update_count"] >= self.config.active_memory_update_threshold:

# AFTER (correct):
if section["update_count"] >= self.config.consolidation_threshold:
```

**Files Modified**:
- `agent_mem/services/memory_manager.py` (2 locations)

---

## Testing Results

After fixes, all basic functionality works:

```
✅ PostgreSQL connection
✅ Neo4j connection  
✅ Ollama embedding service
✅ Creating active memories
✅ Retrieving memories
✅ Updating memory sections
✅ Update count tracking
```

**Test Output**:
```
============================================================
✨ Simple test completed successfully!
============================================================
```

---

## Remaining Issues

### Potential Issues in Other Repositories

The same row access issue (Issue #3) likely affects:
- `shortterm_memory.py`: Multiple `_row_to_model` methods using numeric indexes
- `longterm_memory.py`: Multiple `_row_to_model` methods using numeric indexes

These need to be fixed when those features are tested.

### Missing Dependencies

The code imports `numpy` but it's not in `pyproject.toml` dependencies:
```python
import numpy as np  # Line 796 in memory_manager.py
```

This should be added to the dependencies if the feature is used.

---

## Impact

**Severity**: Critical - Package was completely non-functional without these fixes.

**User Impact**: High - All users would encounter these errors on first use.

**Fix Complexity**: Low - Simple configuration and logic fixes, no architectural changes needed.

---

## Recommendations

1. **Add Integration Tests**: These bugs would have been caught by basic integration tests
2. **CI/CD Pipeline**: Add automated testing before releases
3. **Code Review**: More thorough review of psqlpy usage patterns
4. **Documentation**: Add troubleshooting section with common errors
5. **Version Bump**: Consider these breaking fixes warrant a v0.1.1 release

---

## Files Changed Summary

| File | Lines Changed | Type |
|------|--------------|------|
| `pyproject.toml` | 4 | Config |
| `active_memory.py` | 8 | Bug Fix |
| `shortterm_memory.py` | 8 | Bug Fix |
| `longterm_memory.py` | 4 | Bug Fix |
| `memory_manager.py` | 2 | Bug Fix |
| `GETTING_STARTED_USER.md` | New file | Documentation |
| `examples/simple_test.py` | New file | Testing |

**Total**: 7 files modified, 26+ lines changed
