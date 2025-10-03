# Quick Fix Plan for Agent Mem
## Critical Issues - Immediate Action Required

**Time Estimate**: 4-6 hours  
**Impact**: Makes package usable for development

---

## ✅ Fix 1: Replace deprecated `datetime.utcnow()` (30 minutes)

### Files to Update:

**1. agent_mem/services/memory_manager.py** (15 occurrences)
```python
# Add at top of file
from datetime import datetime, timezone

# Replace all instances:
datetime.utcnow() → datetime.now(timezone.utc)
datetime.utcnow().isoformat() → datetime.now(timezone.utc).isoformat()
```

**Lines to fix**: 567, 621, 640, 657, 687, 954, 959, 1018, 1022, 1040, 1041, 1045, 1093, 1094, 1098

**2. agent_mem/database/repositories/shortterm_memory.py** (4 occurrences)
```python
# Lines: 620, 775, 885, 1070
datetime.utcnow() → datetime.now(timezone.utc)
```

**3. agent_mem/database/repositories/longterm_memory.py** (2 occurrences)
```python
# Lines: 106, 326
datetime.utcnow() → datetime.now(timezone.utc)
```

**4. All test files using datetime.utcnow()**
- tests/test_active_memory_repository.py
- tests/test_core.py
- tests/test_integration.py

---

## ✅ Fix 2: Fix Pydantic V2 Config (20 minutes)

**agent_mem/config/settings.py** (Line 96-99)

```python
# BEFORE
class Config:
    """Pydantic config."""
    env_file = ".env"
    env_file_encoding = "utf-8"

# AFTER
from pydantic import ConfigDict

model_config = ConfigDict(
    env_file=".env",
    env_file_encoding="utf-8"
)
```

---

## ✅ Fix 3: Fix Broken Test Import (10 minutes)

**Option A: Fix the import**
1. Check what the actual function name is in `agent_mem/agents/memory_retriever.py`
2. Update line 12 in `tests/test_agents_skip.py`

**Option B: Skip the file for now**
```bash
# Rename to prevent import
mv tests/test_agents_skip.py tests/test_agents_skip.py.disabled
```

---

## ✅ Fix 4: Fix Integration Test (30 minutes)

**tests/test_integration.py** (Lines 34-43)

```python
# BEFORE (wrong structure)
memory = ActiveMemory(
    id=uuid4(),
    external_id="test-agent",
    sections={"summary": "Initial conversation"},
    metadata={"test": True},
    created_at=datetime.utcnow(),
    updated_at=datetime.utcnow(),
)

# AFTER (correct structure)
memory = ActiveMemory(
    id=1,  # Integer, not UUID
    external_id="test-agent",
    title="Test Memory",  # Required field
    template_content="template:\n  id: test",  # Required field
    sections={
        "summary": {  # Dict, not string
            "content": "Initial conversation",
            "update_count": 0
        }
    },
    metadata={"test": True},
    created_at=datetime.now(timezone.utc),
    updated_at=datetime.now(timezone.utc),
)
```

---

## ✅ Fix 5: Add Input Validation (1 hour)

### 5a. Validate Empty Passwords

**agent_mem/config/settings.py**

```python
from pydantic import field_validator

class Config(BaseModel):
    # ... existing fields ...
    
    @field_validator('postgres_password', 'neo4j_password')
    @classmethod
    def validate_password(cls, v: str, info) -> str:
        if not v or v.strip() == "":
            field_name = info.field_name
            raise ValueError(
                f"{field_name} is required. Set {field_name.upper()} "
                f"environment variable or pass in config."
            )
        return v
```

### 5b. Validate Core API Inputs

**agent_mem/core.py - in create_active_memory method**

```python
async def create_active_memory(
    self,
    external_id: str | UUID | int,
    title: str,
    template_content: str,
    initial_sections: Optional[Dict[str, Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ActiveMemory:
    """..."""
    self._ensure_initialized()
    
    # ADD VALIDATION
    if not title or not title.strip():
        raise ValueError("title cannot be empty")
    
    if len(title) > 500:
        raise ValueError("title too long (max 500 characters)")
    
    if not template_content or not template_content.strip():
        raise ValueError("template_content cannot be empty")
    
    external_id_str = str(external_id)
    # ... rest of method
```

---

## ✅ Fix 6: Handle TODOs (2 hours)

**Option A: Implement Entity/Relationship Extraction**

**agent_mem/services/memory_manager.py** (Lines 300-350)

```python
async def retrieve_memories(
    self,
    external_id: str,
    query: str,
    search_shortterm: bool = True,
    search_longterm: bool = True,
    limit: int = 10,
) -> RetrievalResult:
    """..."""
    
    # ... existing code ...
    
    # REPLACE THE TODO SECTION
    # Extract entities and relationships from Neo4j
    entities: List[Entity] = []
    relationships: List[Relationship] = []
    
    if search_shortterm:
        # Get entities from shortterm memory
        st_entities = await self.shortterm_repo.get_entities_by_external_id(
            external_id=external_id,
            limit=limit
        )
        entities.extend(st_entities)
        
        # Get relationships
        st_rels = await self.shortterm_repo.get_relationships_by_external_id(
            external_id=external_id,
            limit=limit
        )
        relationships.extend(st_rels)
    
    if search_longterm:
        # Get entities from longterm memory  
        lt_entities = await self.longterm_repo.get_entities_by_external_id(
            external_id=external_id,
            limit=limit
        )
        entities.extend(lt_entities)
        
        # Get relationships
        lt_rels = await self.longterm_repo.get_relationships_by_external_id(
            external_id=external_id,
            limit=limit
        )
        relationships.extend(lt_rels)
    
    # Use the retrieve agent with extracted data
    result = await self.retriever_agent.retrieve(
        # ... existing parameters ...
        entities=entities,
        relationships=relationships,
    )
```

**Option B: Document Limitation**

Add to docstring:
```python
"""
...

Note: Entity and relationship extraction from Neo4j is not yet implemented.
The retrieve_memories method will return entities and relationships from 
the shortterm and longterm repositories, but advanced graph queries are 
planned for future versions.

...
"""
```

---

## ✅ Fix 7: Improve Error Handling (2 hours)

### 7a. Add Connection Retry Logic

**agent_mem/database/postgres_manager.py**

```python
import asyncio
from typing import Optional

async def initialize(self) -> None:
    """Initialize connection pool with retry logic."""
    if self._initialized:
        return

    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            dsn = (
                f"host={self.config.postgres_host} "
                f"port={self.config.postgres_port} "
                f"user={self.config.postgres_user} "
                f"password={self.config.postgres_password} "
                f"dbname={self.config.postgres_db}"
            )

            self._pool = ConnectionPool(
                dsn=dsn,
                max_db_pool_size=10,
            )

            self._initialized = True
            logger.info("PostgreSQL connection pool initialized (max_size=10)")
            return
            
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"PostgreSQL connection attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {retry_delay}s..."
                )
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(f"PostgreSQL connection failed after {max_retries} attempts")
                raise RuntimeError(
                    f"Failed to connect to PostgreSQL: {e}. "
                    f"Check host={self.config.postgres_host}:{self.config.postgres_port}"
                )
```

### 7b. Fix Embedding Service Error Handling

**agent_mem/services/embedding.py**

```python
async def get_embedding(
    self, 
    text: str, 
    timeout: int = 30,
    allow_fallback: bool = False  # NEW PARAMETER
) -> List[float]:
    """
    Generate an embedding for the given text.
    
    Args:
        text: Text to embed
        timeout: Request timeout in seconds
        allow_fallback: If True, return zero vector on error. 
                       If False, raise exception (recommended).
    
    Raises:
        RuntimeError: If embedding generation fails and allow_fallback=False
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for embedding")
        if allow_fallback:
            return [0.0] * self.vector_dimension
        raise ValueError("Cannot generate embedding for empty text")

    try:
        # ... existing embedding code ...
        
    except aiohttp.ClientError as e:
        error_msg = (
            f"Error connecting to Ollama API at {self.base_url}: {e}. "
            f"Ensure Ollama is running and {self.model} model is installed."
        )
        logger.error(error_msg)
        
        if allow_fallback:
            logger.warning("Returning zero vector as fallback")
            return [0.0] * self.vector_dimension
        else:
            raise RuntimeError(error_msg) from e

    except Exception as e:
        logger.error(f"Unexpected error generating embedding: {e}")
        if allow_fallback:
            return [0.0] * self.vector_dimension
        else:
            raise RuntimeError(f"Failed to generate embedding: {e}") from e
```

---

## ✅ Fix 8: Add Pagination (1 hour)

**agent_mem/core.py**

```python
async def get_active_memories(
    self,
    external_id: str | UUID | int,
    limit: Optional[int] = None,  # NEW
    offset: int = 0,  # NEW
) -> List[ActiveMemory]:
    """
    Get active memories for a specific agent.
    
    Args:
        external_id: Unique identifier for the agent
        limit: Maximum number of memories to return (None = all)
        offset: Number of memories to skip (for pagination)
    
    Returns:
        List of ActiveMemory objects
        
    Example:
        # Get first 10 memories
        page1 = await agent_mem.get_active_memories("agent-1", limit=10, offset=0)
        
        # Get next 10 memories
        page2 = await agent_mem.get_active_memories("agent-1", limit=10, offset=10)
    """
    self._ensure_initialized()
    external_id_str = str(external_id)

    logger.info(
        f"Retrieving active memories for {external_id_str} "
        f"(limit={limit}, offset={offset})"
    )
    
    memories = await self._memory_manager.get_active_memories(
        external_id=external_id_str
    )
    
    # Apply pagination
    if limit is not None:
        memories = memories[offset:offset + limit]
    elif offset > 0:
        memories = memories[offset:]
    
    return memories
```

---

## 📋 Quick Verification Tests

After fixes, run these commands:

```bash
# 1. Check no datetime.utcnow() left
grep -r "datetime.utcnow()" agent_mem/ --include="*.py"
# Should return nothing

# 2. Run tests
python -m pytest tests/ --ignore=tests/test_agents_skip.py -v

# 3. Check test coverage
python -m pytest tests/ --ignore=tests/test_agents_skip.py --cov=agent_mem --cov-report=term-missing

# 4. Run the quick test
python quick_test.py

# 5. Test with real usage
python examples/basic_usage.py
```

---

## 🎯 Success Criteria

After these fixes:
- [ ] No deprecation warnings
- [ ] All tests passing (120/120)
- [ ] No import errors
- [ ] Input validation working
- [ ] Better error messages
- [ ] Pagination working

---

## ⏭️ Next Steps After Quick Fixes

1. Increase test coverage to 80%+
2. Add comprehensive integration tests
3. Security audit for SQL injection
4. Add transaction management
5. Performance testing
6. Documentation updates

---

**Estimated Total Time**: 4-6 hours for all quick fixes  
**Priority Order**: Fix 1 → Fix 2 → Fix 3 → Fix 4 → Fix 5 → Fix 6 → Fix 7 → Fix 8

**Start with**: Fix 1 (datetime) - it's the quickest and most impactful.
