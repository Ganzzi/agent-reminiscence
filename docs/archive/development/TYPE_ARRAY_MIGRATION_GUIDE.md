# Shortterm/Longterm Repository Type Array Migration Guide

## Overview
This guide documents the remaining changes needed to migrate `shortterm_memory.py` and `longterm_memory.py` to use `types: List[str]` instead of `type: str`.

## Files Status

### ✅ Completed
- `agent_mem/database/models.py` - All models updated to use `types: List[str]`
  - ShorttermEntity
  - ShorttermRelationship
  - LongtermEntity
  - LongtermRelationship
  - Entity
  - Relationship

### 🔄 Partially Complete
- `agent_mem/database/repositories/shortterm_memory.py`
  - ✅ `create_entity()` - Updated
  - ✅ `get_entity()` - Updated
  - ✅ `get_entities_by_memory()` - Updated
  - ✅ `update_entity()` - Updated (added types parameter)
  - ✅ `create_relationship()` - Updated (changed to types)
  - ❌ Remaining relationship methods need updates (see below)

### ❌ Not Started
- `agent_mem/database/repositories/longterm_memory.py` - All entity/relationship methods

## Remaining Changes Needed

### shortterm_memory.py - Lines to Update

**Line 1097:** Change `r.type AS type` to `r.types AS types` in `get_relationship()` query

**Line 1118:** Change `type=record["type"]` to `types=record["types"]` in `get_relationship()` model creation

**Line 1144:** Change `r.type AS type` to `r.types AS types` in `get_relationships_by_memory()` query

**Line 1164:** Change `type=record["type"]` to `types=record["types"]` in model list comprehension

**Line 1232:** Change `r.type AS type` to `r.types AS types` in `update_relationship()` query

**Line 1253:** Change `type=record["type"]` to `types=record["types"]` in `update_relationship()` model creation

**update_relationship() method:** Add `types` parameter similar to `update_entity()`

### longterm_memory.py - All Methods Need Updates

Similar to shortterm_memory.py, all entity and relationship methods need to be updated:

**Entity Methods:**
- `create_entity()` - Change `entity_type: str` to `types: List[str]`
- `get_entity()` - Update query and model creation
- `get_valid_entities_by_external_id()` - Update query and model creation
- `update_entity()` - Add `types` parameter

**Relationship Methods:**
- `create_relationship()` - Change `relationship_type: str` to `types: List[str]`
- `get_relationship()` - Update query and model creation
- `get_relationships_by_memory()` - Update query and model creation
- `update_relationship()` - Add `types` parameter

## Search/Replace Patterns

For shortterm_memory.py:
```
# Query changes
r.type AS type  →  r.types AS types

# Model instantiation changes
type=record["type"]  →  types=record["types"]

# Relationship creation changes
type: $relationship_type  →  types: $types
relationship_type=relationship_type  →  types=types
```

For longterm_memory.py:
```
# Same patterns as shortterm_memory.py
# Plus change relationship type from RELATES_TO to LONGTERM_RELATES
```

## Verification Checklist

After completing changes:
- [ ] All entity queries return `e.types AS types`
- [ ] All relationship queries return `r.types AS types`
- [ ] All model instantiations use `types=record["types"]`
- [ ] All create methods accept `types: List[str]`
- [ ] All update methods accept optional `types: Optional[List[str]]`
- [ ] Relationship type changed to SHORTTERM_RELATES and LONGTERM_RELATES
- [ ] Run tests: `pytest tests/test_shortterm_memory.py tests/test_longterm_memory.py`

## Testing Strategy

After changes complete:
```python
# Test entity with multiple types
entity = await repo.create_entity(
    external_id="agent-123",
    shortterm_memory_id=1,
    name="John Doe",
    types=["PERSON", "DEVELOPER", "ENGINEER"],
    confidence=0.9
)
assert entity.types == ["PERSON", "DEVELOPER", "ENGINEER"]

# Test relationship with multiple types
rel = await repo.create_relationship(
    external_id="agent-123",
    shortterm_memory_id=1,
    from_entity_id=1,
    to_entity_id=2,
    types=["WORKS_WITH", "COLLABORATES_ON"],
    confidence=0.8,
    strength=0.9
)
assert rel.types == ["WORKS_WITH", "COLLABORATES_ON"]
```

## Migration SQL for Neo4j (Already in neo4j_schema.cypher)

```cypher
// Migrate existing single-type entities to array
MATCH (e:ShorttermEntity)
WHERE e.type IS NOT NULL AND e.types IS NULL
SET e.types = [e.type]
REMOVE e.type;

// Migrate existing single-type relationships to array
MATCH ()-[r:SHORTTERM_RELATES]-()
WHERE r.type IS NOT NULL AND r.types IS NULL
SET r.types = [r.type]
REMOVE r.type;
```
