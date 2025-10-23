# Agent Mem Architecture

## Overview

Agent Mem is a standalone Python package providing hierarchical memory management for AI agents. It implements a three-tier memory system (Active → Shortterm → Longterm) with dual storage (PostgreSQL + Neo4j), vector search capabilities, and intelligent consolidation powered by Pydantic AI agents.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      AGENT MEM API                          │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  AgentMem (core.py)                                  │  │
│  │  - create_active_memory()                            │  │
│  │  - get_active_memories()                             │  │
│  │  - update_active_memory()                            │  │
│  │  - retrieve_memories()                               │  │
│  └─────────────────────┬────────────────────────────────┘  │
│                        │                                    │
│                        ↓                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  MemoryManager (services/memory_manager.py)          │  │
│  │  - Orchestrates memory operations                    │  │
│  │  - Manages consolidation workflows                   │  │
│  │  - Coordinates agents and repositories               │  │
│  └────┬─────────────┬──────────────┬───────────────────┘  │
│       │             │              │                        │
└───────┼─────────────┼──────────────┼────────────────────────┘
        │             │              │
        ↓             ↓              ↓
┌───────────┐  ┌──────────┐  ┌──────────────┐
│  Pydantic │  │ Database │  │  Embedding   │
│  AI Agents│  │  Layer   │  │   Service    │
└───────────┘  └──────────┘  └──────────────┘
```

## Layer Architecture

### 1. Public API Layer (`core.py`)

**Purpose**: Simple, user-friendly interface hiding internal complexity.

**Components**:
- **AgentMem**: Main class with 4 public methods
  - Handles initialization and cleanup
  - Validates inputs
  - Delegates to MemoryManager

**Design Principles**:
- Minimal surface area (only 4 methods)
- Clear, descriptive method names
- Type hints for all parameters
- Comprehensive docstrings
- Context manager support (`async with`)

### 2. Service Layer (`services/`)

**Purpose**: Business logic and orchestration.

#### MemoryManager (`services/memory_manager.py`)

Central orchestrator for all memory operations:

```python
class MemoryManager:
    """
    Orchestrates memory operations across all tiers.
    
    Responsibilities:
    - Create/update/retrieve memories
    - Trigger consolidation workflows
    - Coordinate agents and repositories
    - Manage transactions
    """
    
    # Core operations
    async def create_active_memory(...) -> ActiveMemory
    async def get_active_memories(...) -> List[ActiveMemory]
    async def update_active_memory(...) -> ActiveMemory
    async def retrieve_memories(...) -> RetrievalResult
    
    # Internal consolidation (triggered automatically)
    async def _consolidate_to_shortterm(...)
    async def _promote_to_longterm(...)
```

**Key Workflows**:

1. **Active Memory Creation**:
   ```
   User → AgentMem.create_active_memory()
     → MemoryManager.create_active_memory()
       → ActiveMemoryRepository.create()
         → PostgreSQL INSERT
   ```

2. **Memory Retrieval**:
   ```
   User → AgentMem.retrieve_memories()
     → MemoryManager.retrieve_memories()
       → Memory Retrieve Agent
         → Search repositories (Active, Shortterm, Longterm)
           → PostgreSQL (vector + BM25 search)
           → Neo4j (entity/relationship queries)
         → Synthesize results
       → Return RetrievalResult
   ```

3. **Automatic Consolidation**:
   ```
   update_active_memory() triggers consolidation check
     → If update_count >= threshold:
       → Memorizer Agent
         → auto_resolve() extracts entities/relationships
         → Update/add shortterm chunks
         → Resolve entity/relationship conflicts
         → Mark active memory as consolidated
   ```

#### EmbeddingService (`services/embedding.py`)

Generates vector embeddings using Ollama:

```python
class EmbeddingService:
    """Generate embeddings via Ollama API."""
    
    async def get_embedding(text: str) -> List[float]
    async def get_embeddings_batch(texts: List[str]) -> List[List[float]]
```

**Features**:
- Async/await support
- Batch processing
- Automatic retries
- Dimension validation
- Fallback to zero vectors on error

### 3. Database Layer (`database/`)

**Purpose**: Data persistence and retrieval.

#### Connection Managers

**PostgreSQLManager** (`database/postgres_manager.py`):
```python
class PostgreSQLManager:
    """Connection pool manager for PostgreSQL."""
    
    async def initialize()
    async def get_connection() -> Connection
    async def close()
```

**Neo4jManager** (`database/neo4j_manager.py`):
```python
class Neo4jManager:
    """Connection manager for Neo4j."""
    
    async def initialize()
    async def get_write_session() -> AsyncSession
    async def get_read_session() -> AsyncSession
    async def close()
```

#### Repositories

Each repository handles CRUD operations for a specific memory tier:

**ActiveMemoryRepository** (`database/repositories/active_memory.py`):
- `create()`, `get_by_id()`, `get_all()`, `update()`, `delete()`
- No vectors, no entities (simple text storage)

**ShorttermMemoryRepository** (`database/repositories/shortterm_memory.py`):
- Chunk management with vectors
- Hybrid search (vector + BM25)
- Entity/relationship management via Neo4j
- Consolidation from active memory

**LongtermMemoryRepository** (`database/repositories/longterm_memory.py`):
- Chunk management with temporal tracking
- Hybrid search with confidence filtering
- Entity/relationship management with importance scores
- Promotion from shortterm memory

### 4. Agent Layer (`agents/`)

**Purpose**: Intelligent memory management powered by LLMs.

#### Memory Update Agent (`agents/memory_updater.py`)

**Role**: Maintains active memory integrity.

```python
class MemoryUpdateAgent:
    """
    Analyzes message history and updates active memories.
    
    Tools:
    - get_detailed_active_memories
    - create_active_memory
    - update_active_memory
    """
```

**Workflow**:
1. Analyze message history
2. Identify new information
3. Decide: create new or update existing
4. Execute memory operation
5. Confirm action

#### Memorizer Agent (`agents/memorizer.py`)

**Role**: Consolidates active → shortterm memory.

```python
class MemorizerAgent:
    """
    Consolidates active memories into shortterm with conflict resolution.
    
    Tools:
    - auto_resolve (extracts entities/relationships)
    - update_shortterm_memory_chunks
    - add_new_shortterm_memory_chunks
    - update_shortterm_memory_entities_relationships
    """
```

**Workflow**:
1. Auto-resolve differences
2. Update similar chunks
3. Add new chunks (with context)
4. Resolve entity/relationship conflicts

#### Memory Retrieve Agent (`agents/memory_retriever.py`)

**Role**: Searches and synthesizes information.

```python
class MemoryRetrieveAgent:
    """
    Intelligent search across all memory tiers.
    
    Tools:
    - get_detailed_active_memories
    - search_shortterm_memory
    - search_longterm_memory
    """
```

**Workflow**:
1. Analyze query intent
2. Select appropriate memory tiers
3. Execute searches
4. Aggregate results
5. Synthesize response

### 5. Configuration Layer (`config/`)

**settings.py**: Pydantic-based configuration
- Environment variable loading
- Type validation
- Default values
- Global config instance

**constants.py**: System constants
- Memory tier names
- Entity/relationship types
- Thresholds and limits
- Timeouts

## Data Flow

### Creating and Consolidating Memory

```
┌──────────────────────────────────────────────────────┐
│ 1. User creates active memory                        │
│    agent_reminiscence.create_active_memory(...)               │
└─────────────────┬────────────────────────────────────┘
                  │
                  ↓
┌──────────────────────────────────────────────────────┐
│ 2. Stored in PostgreSQL (active_memory table)        │
│    No vectors, no entities yet                       │
└─────────────────┬────────────────────────────────────┘
                  │
                  ↓
┌──────────────────────────────────────────────────────┐
│ 3. User updates memory multiple times                │
│    agent_reminiscence.update_active_memory(...)               │
│    Update count increments                           │
└─────────────────┬────────────────────────────────────┘
                  │
                  ↓ (when update_count >= threshold)
┌──────────────────────────────────────────────────────┐
│ 4. Automatic consolidation triggered                 │
│    MemoryManager._consolidate_to_shortterm()         │
└─────────────────┬────────────────────────────────────┘
                  │
                  ↓
┌──────────────────────────────────────────────────────┐
│ 5. Memorizer Agent processes                         │
│    - Calls auto_resolve()                            │
│    - Extracts entities and relationships             │
│    - Updates/adds shortterm chunks                   │
│    - Resolves conflicts                              │
└─────────────────┬────────────────────────────────────┘
                  │
                  ↓
┌──────────────────────────────────────────────────────┐
│ 6. Shortterm memory created                          │
│    PostgreSQL: chunks with vectors + BM25            │
│    Neo4j: entities and relationships as graph        │
└─────────────────┬────────────────────────────────────┘
                  │
                  ↓ (based on importance score)
┌──────────────────────────────────────────────────────┐
│ 7. Promotion to longterm (background process)        │
│    - Chunks with temporal tracking                   │
│    - High-confidence entities                        │
│    - Validated relationships                         │
└──────────────────────────────────────────────────────┘
```

### Retrieving Memory

```
┌──────────────────────────────────────────────────────┐
│ 1. User requests information                         │
│    result = agent_reminiscence.retrieve_memories(query)       │
└─────────────────┬────────────────────────────────────┘
                  │
                  ↓
┌──────────────────────────────────────────────────────┐
│ 2. MemoryManager invokes Memory Retrieve Agent       │
└─────────────────┬────────────────────────────────────┘
                  │
                  ↓
┌──────────────────────────────────────────────────────┐
│ 3. Agent analyzes query and selects tiers            │
│    - Current task? → Search active                   │
│    - Recent work? → Search shortterm                 │
│    - General knowledge? → Search longterm            │
└─────────────────┬────────────────────────────────────┘
                  │
                  ↓
┌──────────────────────────────────────────────────────┐
│ 4. Parallel searches executed                        │
│    ┌─────────────────────────────────────────────┐  │
│    │ Active: PostgreSQL text search              │  │
│    └─────────────────────────────────────────────┘  │
│    ┌─────────────────────────────────────────────┐  │
│    │ Shortterm: Hybrid search (vector + BM25)    │  │
│    │            Entity/relationship query (Neo4j) │  │
│    └─────────────────────────────────────────────┘  │
│    ┌─────────────────────────────────────────────┐  │
│    │ Longterm: Hybrid search with confidence     │  │
│    │           Entity/relationship query (Neo4j)  │  │
│    └─────────────────────────────────────────────┘  │
└─────────────────┬────────────────────────────────────┘
                  │
                  ↓
┌──────────────────────────────────────────────────────┐
│ 5. Agent synthesizes results                         │
│    - Aggregates chunks, entities, relationships      │
│    - Generates human-readable summary                │
└─────────────────┬────────────────────────────────────┘
                  │
                  ↓
┌──────────────────────────────────────────────────────┐
│ 6. Return RetrievalResult                            │
│    - Matched chunks with scores                      │
│    - Related entities                                │
│    - Related relationships                           │
│    - Synthesized response                            │
└──────────────────────────────────────────────────────┘
```

## Database Schema

### PostgreSQL Tables

```sql
-- Active Memory (working memory)
CREATE TABLE active_memory (
    id SERIAL PRIMARY KEY,
    external_id VARCHAR(255) NOT NULL,  -- Generic ID (not worker_id)
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    description TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_external_id (external_id)
);

-- Shortterm Memory (searchable chunks)
CREATE TABLE shortterm_memory (
    id SERIAL PRIMARY KEY,
    external_id VARCHAR(255) NOT NULL,
    title VARCHAR(500) NOT NULL,
    summary TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_external_id (external_id)
);

CREATE TABLE shortterm_memory_chunk (
    id SERIAL PRIMARY KEY,
    shortterm_memory_id INTEGER REFERENCES shortterm_memory(id) ON DELETE CASCADE,
    external_id VARCHAR(255) NOT NULL,
    chunk_order INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(768),  -- Configurable dimension
    content_bm25 bm25vector,  -- Auto-generated via trigger
    metadata JSONB DEFAULT '{}',
    INDEX idx_external_id (external_id),
    INDEX idx_embedding USING ivfflat (embedding vector_cosine_ops),
    INDEX idx_content_bm25 USING bm25 (content_bm25 bm25_ops)
);

-- Longterm Memory (consolidated knowledge)
CREATE TABLE longterm_memory_chunk (
    id SERIAL PRIMARY KEY,
    external_id VARCHAR(255) NOT NULL,
    shortterm_memory_id INTEGER,  -- Reference to source
    chunk_order INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(768),
    content_bm25 bm25vector,
    confidence_score FLOAT DEFAULT 0.5,
    start_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    end_date TIMESTAMP,  -- NULL = still valid
    metadata JSONB DEFAULT '{}',
    INDEX idx_external_id (external_id),
    INDEX idx_valid_chunks (external_id, start_date, end_date),
    INDEX idx_embedding USING ivfflat (embedding vector_cosine_ops),
    INDEX idx_content_bm25 USING bm25 (content_bm25 bm25_ops)
);
```

### Neo4j Schema

```cypher
// Shortterm Entity
CREATE (e:ShorttermEntity {
    id: INTEGER,
    external_id: STRING,
    shortterm_memory_id: INTEGER,
    name: STRING,
    type: STRING,
    description: STRING,
    confidence: FLOAT,
    first_seen: DATETIME,
    last_seen: DATETIME,
    metadata: MAP
})

// Longterm Entity
CREATE (e:LongtermEntity {
    id: INTEGER,
    external_id: STRING,
    name: STRING,
    type: STRING,
    description: STRING,
    confidence: FLOAT,
    importance: FLOAT,
    first_seen: DATETIME,
    last_seen: DATETIME,
    metadata: MAP
})

// Relationships
CREATE (e1)-[r:RELATES_TO {
    id: INTEGER,
    external_id: STRING,
    type: STRING,
    description: STRING,
    confidence: FLOAT,
    strength: FLOAT,
    // Shortterm: first_observed, last_observed
    // Longterm: start_date, last_updated
    metadata: MAP
}]->(e2)

// Constraints
CREATE CONSTRAINT shortterm_entity_unique IF NOT EXISTS
FOR (e:ShorttermEntity) REQUIRE e.id IS UNIQUE;

CREATE CONSTRAINT longterm_entity_unique IF NOT EXISTS
FOR (e:LongtermEntity) REQUIRE e.id IS UNIQUE;
```

## Key Design Decisions

### 1. Generic External ID

**Decision**: Use `external_id` instead of `worker_id`.

**Rationale**:
- Package is standalone and domain-agnostic
- Supports any ID type (UUID, string, int)
- Allows use beyond "worker" concept
- User can map their own entity IDs

### 2. Dual Storage (PostgreSQL + Neo4j)

**Decision**: Use both databases, not just one.

**Rationale**:
- PostgreSQL excels at vector search and text search
- Neo4j excels at graph queries and relationship traversal
- Each database handles what it does best
- Better performance than forcing one to do both

### 3. Only 4 Public Methods

**Decision**: Minimal public API with 4 methods.

**Rationale**:
- Simplicity for users
- Hide internal complexity
- Clear use cases
- Easy to learn and remember
- Consolidation happens automatically

### 4. Pydantic AI for Agents

**Decision**: Use Pydantic AI framework for agents.

**Rationale**:
- Type-safe tool calling
- Built-in validation
- Model-agnostic (supports multiple LLM providers)
- Structured outputs with Pydantic
- Testing support with TestModel/FunctionModel

### 5. Ollama for Embeddings

**Decision**: Use Ollama for embedding generation.

**Rationale**:
- Local execution (privacy)
- Free and open-source
- Multiple model support
- Simple HTTP API
- Consistent with existing codebase

### 6. Automatic Consolidation

**Decision**: Consolidate automatically based on thresholds.

**Rationale**:
- Users don't need to manage consolidation manually
- Happens at optimal times (update threshold)
- Can be customized via configuration
- Reduces cognitive load on users

## Extension Points

### Custom Embedding Models

```python
from agent_reminiscence.services import EmbeddingService

class CustomEmbeddingService(EmbeddingService):
    async def get_embedding(self, text: str) -> List[float]:
        # Use OpenAI, Cohere, or custom model
        pass

# Use custom service
memory_manager._embedding_service = CustomEmbeddingService()
```

### Custom Agent Models

```python
config = Config(
    memory_update_agent_model="openai:gpt-4",
    memorizer_agent_model="anthropic:claude-3-sonnet",
    memory_retrieve_agent_model="openai:gpt-4",
)
```

### Custom Consolidation Logic

```python
config = Config(
    active_memory_update_threshold=10,  # Higher threshold
    shortterm_promotion_threshold=0.8,  # Higher importance required
)
```

## Performance Considerations

### Connection Pooling

- PostgreSQL: PSQLPy connection pool (configurable size)
- Neo4j: Session pooling with read/write separation

### Indexing

- Vector indexes: IVFFlat for approximate nearest neighbor
- BM25 indexes: Specialized text search index
- Neo4j indexes: On entity IDs, names, types
- Composite indexes: On (external_id, date) for temporal queries

### Caching

- Embedding cache: Avoid regenerating same embeddings
- Configuration cache: Load once, reuse
- Connection pool: Reuse database connections

### Batch Operations

- Batch embedding generation
- Batch entity/relationship creation
- Transaction batching for consistency

## Testing Strategy

### Unit Tests

- Test each repository independently
- Mock database connections
- Test Pydantic models validation
- Test utility functions

### Integration Tests

- Test with real PostgreSQL and Neo4j
- Test agent workflows end-to-end
- Test consolidation workflows
- Test search accuracy

### Agent Testing

- Use Pydantic AI's TestModel
- Use FunctionModel for custom logic
- Capture and verify message flows
- Test tool calling

## Deployment

### Docker Compose Setup

```yaml
version: '3.8'
services:
  postgres:
    image: ankane/pgvector:latest
    environment:
      POSTGRES_DB: agent_reminiscence
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
  
  neo4j:
    image: neo4j:5-community
    environment:
      NEO4J_AUTH: neo4j/password
    ports:
      - "7687:7687"
      - "7474:7474"
  
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
```

### Environment Configuration

Provide `.env` file with all required credentials.

### Schema Migration

Run SQL and Cypher scripts to initialize schema.

## Future Enhancements

1. **Async Background Workers**: Separate process for consolidation
2. **Streaming Responses**: Stream large retrieval results
3. **Multi-Modal Support**: Images, audio embeddings
4. **Distributed Deployment**: Multi-node PostgreSQL and Neo4j
5. **Monitoring Dashboard**: Visualize memory usage and performance
6. **Advanced Search**: Temporal queries, graph pattern matching
7. **Export/Import**: Backup and restore memories
8. **Multi-Tenancy**: Support multiple agents efficiently

## Summary

Agent Mem provides a clean, simple API backed by sophisticated memory management. The architecture separates concerns clearly, uses appropriate technologies for each task, and provides intelligent automation through Pydantic AI agents. The design is extensible, performant, and production-ready.

