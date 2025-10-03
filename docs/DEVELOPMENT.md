# Agent Mem Development Guide

## Setup for Development

### 1. Clone and Install

```bash
cd libs/agent_mem
pip install -e ".[dev]"
```

### 2. Set Up Dependencies

**PostgreSQL with Extensions:**
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_tokenizer CASCADE;
CREATE EXTENSION IF NOT EXISTS vchord_bm25 CASCADE;
```

**Neo4j:**
```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5-community
```

**Ollama:**
```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull embedding model
ollama pull nomic-embed-text
```

### 3. Configure Environment

Create `.env`:
```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=agent_mem

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j

OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
VECTOR_DIMENSION=768
```

### 4. Initialize Schema

```bash
# PostgreSQL
psql -U postgres -d agent_mem < agent_mem/sql/schema.sql

# Neo4j (run in Neo4j browser)
# Copy constraints from docs/ARCHITECTURE.md
```

## Project Structure (Complete)

```
agent_mem/
├── __init__.py                    # Public API
├── core.py                        # AgentMem main class
├── config/
│   ├── __init__.py
│   ├── settings.py               # Pydantic configuration
│   └── constants.py              # System constants
├── database/
│   ├── __init__.py
│   ├── models.py                 # Pydantic data models
│   ├── postgres_manager.py      # PostgreSQL connection pool
│   ├── neo4j_manager.py         # Neo4j connection manager
│   └── repositories/
│       ├── __init__.py
│       ├── active_memory.py     # Active memory CRUD
│       ├── shortterm_memory.py  # Shortterm memory CRUD
│       └── longterm_memory.py   # Longterm memory CRUD
├── services/
│   ├── __init__.py
│   ├── embedding.py             # Ollama embedding service
│   └── memory_manager.py        # Core orchestration
├── agents/
│   ├── __init__.py
│   ├── memory_updater.py        # Memory Update Agent
│   ├── memorizer.py             # Memorizer Agent
│   └── memory_retriever.py      # Memory Retrieve Agent
├── utils/
│   ├── __init__.py
│   └── helpers.py               # Utility functions
└── sql/
    ├── schema.sql               # Full PostgreSQL schema
    └── migrations/
        └── 001_initial.sql
```

## Implementation Checklist

### ✅ Completed

- [x] Package structure and pyproject.toml
- [x] README.md with quick start guide
- [x] Configuration system (settings.py, constants.py)
- [x] Core AgentMem class interface
- [x] Pydantic data models
- [x] Architecture documentation

### 🚧 To Implement (Priority Order)

#### 1. Database Layer (HIGH PRIORITY)

**Files to create:**

- [ ] `database/postgres_manager.py` - Connection pooling
- [ ] `database/neo4j_manager.py` - Neo4j sessions
- [ ] `database/repositories/active_memory.py` - Active memory CRUD
- [ ] `database/repositories/shortterm_memory.py` - Shortterm with vectors
- [ ] `database/repositories/longterm_memory.py` - Longterm with temporal
- [ ] `database/__init__.py` - Export managers and repositories
- [ ] `sql/schema.sql` - Complete PostgreSQL schema

**Reference files from main codebase:**
- `database/postgres_conn.py` → adapt to PSQLPy
- `database/neo4j_conn.py` → adapt connection handling
- `database/memories/*_repository.py` → adapt CRUD logic

#### 2. Embedding Service (HIGH PRIORITY)

**Files to create:**

- [ ] `services/embedding.py` - Ollama API client

**Reference:** `database/utility_services/embedding_service.py`

**Key adaptations:**
- Make async
- Add batch support
- Add retry logic
- Validate dimensions

#### 3. Memory Manager (HIGH PRIORITY)

**Files to create:**

- [ ] `services/memory_manager.py` - Core orchestration

**Must implement:**
```python
class MemoryManager:
    async def initialize()
    async def close()
    
    # Public methods (called by AgentMem)
    async def create_active_memory(...)
    async def get_active_memories(...)
    async def update_active_memory(...)
    async def retrieve_memories(...)
    
    # Internal methods
    async def _check_consolidation_threshold(...)
    async def _consolidate_to_shortterm(...)
    async def _promote_to_longterm(...)
```

#### 4. Pydantic AI Agents (MEDIUM PRIORITY)

**Files to create:**

- [ ] `agents/memory_updater.py` - Memory Update Agent
- [ ] `agents/memorizer.py` - Memorizer Agent
- [ ] `agents/memory_retriever.py` - Memory Retrieve Agent
- [ ] `agents/__init__.py` - Export agents

**Reference files:**
- `agents/predefined_agents/memory_updater_agent.py`
- `agents/predefined_agents/memorizer_agent.py`
- `agents/predefined_agents/memory_retrievor_agent.py`
- `config/agents/predefined/*.py` for system prompts

**Key adaptations:**
- Use external_id instead of worker_id
- Simplify dependencies (no monitor, no complex deps)
- Make self-contained (include system prompts in code)

#### 5. Utilities (LOW PRIORITY)

**Files to create:**

- [ ] `utils/helpers.py` - Chunking, text processing

**Useful functions:**
```python
def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]
def normalize_vector(vector: List[float]) -> List[float]
def parse_external_id(external_id: Any) -> str
```

#### 6. Testing (MEDIUM PRIORITY)

**Files to create:**

- [ ] `tests/test_config.py`
- [ ] `tests/test_models.py`
- [ ] `tests/test_core.py`
- [ ] `tests/test_embedding_service.py`
- [ ] `tests/test_memory_manager.py`
- [ ] `tests/test_repositories.py`
- [ ] `tests/test_agents.py`
- [ ] `tests/conftest.py` - Pytest fixtures

#### 7. Examples (MEDIUM PRIORITY)

**Files to create:**

- [ ] `examples/basic_usage.py`
- [ ] `examples/custom_configuration.py`
- [ ] `examples/batch_operations.py`
- [ ] `examples/advanced_search.py`

#### 8. Documentation (LOW PRIORITY)

**Files to create:**

- [ ] `docs/API.md` - Complete API reference
- [ ] `docs/AGENTS.md` - Agent system documentation
- [ ] `docs/SEARCH.md` - Search capabilities guide
- [ ] `docs/DEPLOYMENT.md` - Production deployment
- [ ] `docs/TROUBLESHOOTING.md` - Common issues
- [ ] `CONTRIBUTING.md` - Contribution guidelines
- [ ] `LICENSE` - MIT license

## Implementation Notes

### External ID Handling

**Key principle:** Use `external_id` everywhere instead of `worker_id`.

**In SQL:**
```sql
-- Add index for external_id in all tables
CREATE INDEX idx_active_memory_external_id ON active_memory(external_id);
CREATE INDEX idx_shortterm_memory_external_id ON shortterm_memory(external_id);
CREATE INDEX idx_longterm_chunk_external_id ON longterm_memory_chunk(external_id);
```

**In Neo4j:**
```cypher
// Store external_id in all entities and relationships
CREATE (e:ShorttermEntity {external_id: $external_id, ...})
CREATE (e)-[r:RELATES_TO {external_id: $external_id, ...}]->(e2)
```

**In Python:**
```python
# Convert any ID type to string
external_id = str(external_id)  # Works for UUID, int, string

# Use consistently in all queries
await connection.execute(
    "SELECT * FROM active_memory WHERE external_id = $1",
    [external_id]
)
```

### Agent System Prompts

**Embed prompts directly in agent files** (no separate config files):

```python
# agents/memory_updater.py

MEMORY_UPDATE_SYSTEM_PROMPT = """
You are a Memory Consolidation Specialist...
[Full prompt here]
"""

class MemoryUpdateAgent:
    def __init__(self, model: str):
        self.agent = Agent(
            model=model,
            system_prompt=MEMORY_UPDATE_SYSTEM_PROMPT,
            tools=[...],
        )
```

### Embedding Service

**Key features to implement:**

1. **Async/await support:**
```python
async def get_embedding(self, text: str) -> List[float]:
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            data = await response.json()
            return data["embedding"]
```

2. **Batch processing:**
```python
async def get_embeddings_batch(
    self, 
    texts: List[str], 
    batch_size: int = 10
) -> List[List[float]]:
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = await asyncio.gather(
            *[self.get_embedding(text) for text in batch]
        )
        embeddings.extend(batch_embeddings)
    return embeddings
```

3. **Error handling:**
```python
try:
    embedding = await self.get_embedding(text)
except Exception as e:
    logger.error(f"Embedding failed: {e}")
    # Return zero vector as fallback
    return [0.0] * self.config.vector_dimension
```

### Database Repositories

**Pattern for all repositories:**

```python
class ActiveMemoryRepository:
    """Repository for active memory CRUD operations."""
    
    def __init__(self, postgres_manager: PostgreSQLManager):
        self.postgres = postgres_manager
    
    async def create(
        self, 
        external_id: str, 
        title: str, 
        content: str,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> ActiveMemory:
        """Create new active memory."""
        conn = await self.postgres.get_connection()
        
        query = """
            INSERT INTO active_memory 
            (external_id, title, content, description, metadata)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING *
        """
        
        result = await conn.execute(
            query,
            [external_id, title, content, description, metadata or {}]
        )
        
        row = result.result()[0]
        return ActiveMemory(**dict(row))
    
    async def get_all(self, external_id: str) -> List[ActiveMemory]:
        """Get all active memories for external_id."""
        # Implementation...
    
    async def update(self, memory_id: int, **kwargs) -> ActiveMemory:
        """Update active memory."""
        # Implementation...
    
    async def delete(self, memory_id: int) -> bool:
        """Delete active memory."""
        # Implementation...
```

### Memory Manager Consolidation

**Consolidation workflow:**

```python
async def _consolidate_to_shortterm(
    self, 
    active_memory_id: int
) -> ShorttermMemory:
    """
    Consolidate active memory to shortterm.
    
    Steps:
    1. Get active memory content
    2. Find or create shortterm memory
    3. Invoke Memorizer Agent
    4. Agent calls auto_resolve
    5. Agent updates chunks and entities
    6. Mark active memory as consolidated
    """
    
    # 1. Get active memory
    active_memory = await self.active_repo.get_by_id(active_memory_id)
    
    # 2. Find best shortterm memory match
    shortterm_memory = await self._find_best_shortterm_match(
        active_memory
    )
    
    if not shortterm_memory:
        # Create new shortterm memory
        shortterm_memory = await self.shortterm_repo.create(
            external_id=active_memory.external_id,
            title=active_memory.title,
            summary=active_memory.description
        )
    
    # 3. Invoke Memorizer Agent
    from agent_mem.agents.memorizer import MemorizerAgent
    
    agent = MemorizerAgent(model=self.config.memorizer_agent_model)
    result = await agent.consolidate(
        active_memory=active_memory,
        shortterm_memory=shortterm_memory
    )
    
    # 4. Mark as consolidated (optional: delete or archive)
    await self.active_repo.mark_consolidated(active_memory_id)
    
    return shortterm_memory
```

## Testing Strategy

### Unit Tests

```python
# tests/test_active_memory_repository.py

import pytest
from agent_mem.database.repositories import ActiveMemoryRepository
from agent_mem.database.postgres_manager import PostgreSQLManager

@pytest.mark.asyncio
async def test_create_active_memory(postgres_manager):
    """Test creating an active memory."""
    repo = ActiveMemoryRepository(postgres_manager)
    
    memory = await repo.create(
        external_id="test-agent-1",
        title="Test Memory",
        content="Test content",
    )
    
    assert memory.id is not None
    assert memory.title == "Test Memory"
    assert memory.external_id == "test-agent-1"
```

### Integration Tests

```python
# tests/test_integration.py

import pytest
from agent_mem import AgentMem

@pytest.mark.asyncio
async def test_full_workflow():
    """Test complete workflow from creation to retrieval."""
    
    async with AgentMem(external_id="test-agent") as agent_mem:
        # Create memory
        memory = await agent_mem.create_active_memory(
            title="Test",
            content="Implementation details..."
        )
        
        # Update multiple times to trigger consolidation
        for i in range(6):
            await agent_mem.update_active_memory(
                memory_id=memory.id,
                content=f"Updated content {i}"
            )
        
        # Retrieve
        result = await agent_mem.retrieve_memories(
            query="implementation details"
        )
        
        assert len(result.shortterm_chunks) > 0
```

### Agent Tests

```python
# tests/test_agents.py

from pydantic_ai.models.test import TestModel
from agent_mem.agents.memory_updater import MemoryUpdateAgent

@pytest.mark.asyncio
async def test_memory_update_agent():
    """Test Memory Update Agent."""
    
    agent = MemoryUpdateAgent(model="test")
    
    with agent.override(model=TestModel()):
        result = await agent.update_memory(
            active_summaries=[],
            prompt="Create memory for new task"
        )
        
        assert result is not None
```

## Quick Start for Contributors

1. **Pick a file to implement** from the checklist above
2. **Reference similar files** from the main codebase (`ai-army/`)
3. **Adapt for standalone package**:
   - Change `worker_id` to `external_id`
   - Remove monitor dependencies
   - Simplify agent configurations
   - Make self-contained
4. **Write tests** for your implementation
5. **Update this guide** with any issues or clarifications

## Common Patterns

### Async Context Manager

```python
class MyManager:
    async def initialize(self):
        """Initialize resources."""
        pass
    
    async def close(self):
        """Clean up resources."""
        pass
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
```

### Error Handling

```python
import logging

logger = logging.getLogger(__name__)

async def safe_operation():
    """Operation with proper error handling."""
    try:
        result = await risky_operation()
        return result
    except SpecificException as e:
        logger.error(f"Specific error: {e}")
        # Handle or re-raise
        raise
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        # Return default or raise
        raise RuntimeError(f"Operation failed: {e}")
```

### Type Hints

```python
from typing import List, Optional, Dict, Any, Union
from uuid import UUID

async def typed_function(
    required_param: str,
    optional_param: Optional[int] = None,
    list_param: List[str] = None,
    dict_param: Dict[str, Any] = None,
    union_param: Union[str, int, UUID] = None,
) -> Optional[Dict[str, Any]]:
    """Function with complete type hints."""
    pass
```

## Next Steps

1. **Implement database layer** - Start with PostgreSQLManager
2. **Create SQL schema** - Define all tables and indexes
3. **Build repositories** - One at a time (active → shortterm → longterm)
4. **Add embedding service** - Ollama integration
5. **Build memory manager** - Core orchestration
6. **Implement agents** - Memory Update → Memorizer → Retrieve
7. **Write tests** - As you implement each component
8. **Create examples** - Demonstrate usage patterns
9. **Complete documentation** - API reference and guides

## Support

For questions or issues during development:
- Review the main codebase in `ai-army/` for reference
- Check existing documentation in `docs/`
- Look at the original agent implementations
- Refer to Pydantic AI docs: https://ai.pydantic.dev/
- Check PSQLPy docs for PostgreSQL interactions
- Review Neo4j Python driver docs

## Summary

This package is designed to be:
- **Simple**: 4 public methods, clear API
- **Standalone**: No dependencies on external systems (except DBs)
- **Flexible**: Generic external_id, configurable everything
- **Intelligent**: Pydantic AI agents for smart memory management
- **Production-ready**: Proper connection pooling, error handling, testing

Follow the checklist, reference the main codebase, and build incrementally. Good luck! 🚀
