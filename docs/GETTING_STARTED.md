# Getting Started with Agent Mem Development

## Current Status 🚀

The agent_mem package now has:

✅ **Core Infrastructure**
- PostgreSQL connection manager with PSQLPy
- Neo4j connection manager
- Complete SQL schema with vector + BM25 support
- Active Memory Repository (full CRUD)
- Configuration system with Pydantic
- Async embedding service with Ollama
- Text chunking and utility helpers
- Core AgentMem class interface

🚧 **In Progress**
- Shortterm Memory Repository (TODO)
- Longterm Memory Repository (TODO)
- Pydantic AI agents (TODO)
- Full memory consolidation workflow (TODO)

## Quick Setup

### 1. Install Dependencies

```bash
cd libs/agent_mem
pip install -e .
```

### 2. Set Up Databases

**PostgreSQL with Extensions:**

```bash
# Install PostgreSQL if needed
# Then install required extensions (requires superuser):

psql -U postgres -d agent_mem << EOF
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_tokenizer CASCADE;
CREATE EXTENSION IF NOT EXISTS vchord_bm25 CASCADE;
EOF
```

**Run Schema:**

```bash
psql -U postgres -d agent_mem < agent_mem/sql/schema.sql
```

**Neo4j:**

```bash
docker run -d \
  --name neo4j-agent-mem \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5-community
```

**Ollama:**

```bash
# Install Ollama from https://ollama.ai

# Pull embedding model
ollama pull nomic-embed-text
```

### 3. Configure Environment

Create `.env` in `libs/agent_mem/`:

```env
# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=agent_mem

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
VECTOR_DIMENSION=768

# Agent Models (optional)
MEMORY_UPDATE_AGENT_MODEL=google-gla:gemini-2.0-flash
MEMORIZER_AGENT_MODEL=google-gla:gemini-2.0-flash
MEMORY_RETRIEVE_AGENT_MODEL=google-gla:gemini-2.0-flash
```

### 4. Run Example

```bash
cd libs/agent_mem
python examples/basic_usage.py
```

Expected output:
```
=== Agent Mem Basic Usage Example ===

Initializing Agent Mem...
✓ Initialized successfully

1. Creating active memory...
✓ Created memory ID 1: Project Requirements

2. Creating another active memory...
✓ Created memory ID 2: Technical Stack

3. Retrieving all active memories...
✓ Retrieved 2 active memories:
   - [1] Project Requirements (updates: 0)
   - [2] Technical Stack (updates: 0)

4. Updating memory...
✓ Updated memory 1, update_count: 1

5. Updating memory multiple times...
   - Update 1: update_count = 2
   - Update 2: update_count = 3
   - Update 3: update_count = 4

6. Searching memories...
✓ Search result: Retrieved 2 active memories. Full search implementation coming soon.
   Active memories found: 2

=== Example Complete ===
```

## Testing the Package

### Test PostgreSQL Connection

```python
import asyncio
from agent_mem.database import PostgreSQLManager
from agent_mem.config import get_config

async def test_postgres():
    config = get_config()
    manager = PostgreSQLManager(config)
    
    await manager.initialize()
    
    # Test query
    result = await manager.execute("SELECT 1 as test")
    print(f"PostgreSQL test: {result.result()}")
    
    # Check extensions
    await manager.ensure_extensions()
    
    # Check tables
    exists = await manager.table_exists("active_memory")
    print(f"active_memory table exists: {exists}")
    
    await manager.close()

asyncio.run(test_postgres())
```

### Test Neo4j Connection

```python
import asyncio
from agent_mem.database import Neo4jManager
from agent_mem.config import get_config

async def test_neo4j():
    config = get_config()
    manager = Neo4jManager(config)
    
    await manager.initialize()
    
    # Test query
    result = await manager.execute_read("RETURN 1 as test")
    print(f"Neo4j test: {result}")
    
    # Create constraints
    await manager.ensure_constraints()
    await manager.create_indexes()
    
    await manager.close()

asyncio.run(test_neo4j())
```

### Test Embedding Service

```python
import asyncio
from agent_mem.services import EmbeddingService
from agent_mem.config import get_config

async def test_embeddings():
    config = get_config()
    service = EmbeddingService(config)
    
    # Verify connection
    ok = await service.verify_connection()
    print(f"Ollama available: {ok}")
    
    # Generate embedding
    embedding = await service.get_embedding("Hello world")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Batch embeddings
    texts = ["Hello", "World", "Test"]
    embeddings = await service.get_embeddings_batch(texts)
    print(f"Generated {len(embeddings)} embeddings")

asyncio.run(test_embeddings())
```

### Test Active Memory Repository

```python
import asyncio
from agent_mem.database import PostgreSQLManager, ActiveMemoryRepository
from agent_mem.config import get_config

async def test_active_memory():
    config = get_config()
    postgres = PostgreSQLManager(config)
    await postgres.initialize()
    
    repo = ActiveMemoryRepository(postgres)
    
    # Create
    memory = await repo.create(
        external_id="test-agent",
        title="Test Memory",
        content="Test content"
    )
    print(f"Created memory: {memory.id}")
    
    # Get all
    memories = await repo.get_all_by_external_id("test-agent")
    print(f"Total memories: {len(memories)}")
    
    # Update
    updated = await repo.update(memory.id, content="Updated content")
    print(f"Update count: {updated.update_count}")
    
    # Delete
    deleted = await repo.delete(memory.id)
    print(f"Deleted: {deleted}")
    
    await postgres.close()

asyncio.run(test_active_memory())
```

## Current API

### AgentMem Class

```python
from agent_mem import AgentMem

# Initialize
agent_mem = AgentMem(external_id="agent-123")
await agent_mem.initialize()

# Create active memory
memory = await agent_mem.create_active_memory(
    title="Task",
    content="Implementation details...",
    description="Description",
    metadata={"key": "value"}
)

# Get all active memories
memories = await agent_mem.get_active_memories()

# Update active memory
updated = await agent_mem.update_active_memory(
    memory_id=1,
    content="New content"
)

# Retrieve memories (stub for now)
result = await agent_mem.retrieve_memories(
    query="What did I implement?"
)

# Close
await agent_mem.close()
```

## Next Development Steps

See `docs/DEVELOPMENT.md` for the complete implementation checklist.

**High Priority:**
1. Implement ShorttermMemoryRepository with vector search
2. Implement LongtermMemoryRepository with temporal tracking
3. Build MemoryManager consolidation workflow
4. Implement 3 Pydantic AI agents

**Medium Priority:**
5. Add comprehensive tests
6. Create more examples
7. Add API documentation

**Low Priority:**
8. Performance optimization
9. Monitoring and observability
10. Advanced features

## Troubleshooting

### PostgreSQL Extensions Not Installing

If you get permission errors installing extensions:

```bash
# Connect as superuser
sudo -u postgres psql agent_mem

# Then run:
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_tokenizer CASCADE;
CREATE EXTENSION IF NOT EXISTS vchord_bm25 CASCADE;
```

### Ollama Model Not Found

```bash
# List installed models
ollama list

# Pull the model
ollama pull nomic-embed-text

# Verify it works
ollama run nomic-embed-text "test"
```

### Neo4j Connection Failed

Check Neo4j is running:
```bash
docker ps | grep neo4j
```

Access Neo4j Browser: http://localhost:7474

Default credentials: neo4j/password (change in `.env`)

### Import Errors

Make sure package is installed in editable mode:
```bash
cd libs/agent_mem
pip install -e .
```

## Development Workflow

1. **Pick a task** from `docs/DEVELOPMENT.md` checklist
2. **Reference** similar code from main codebase (`ai-army/`)
3. **Adapt** for standalone package (external_id, no monitor deps)
4. **Test** your implementation
5. **Update** documentation

## Resources

- **Architecture**: See `docs/ARCHITECTURE.md`
- **Development Guide**: See `docs/DEVELOPMENT.md`
- **Main Codebase**: Reference `../../` for examples
- **Pydantic AI Docs**: https://ai.pydantic.dev/
- **PSQLPy Docs**: https://github.com/qaspen-python/psqlpy

Happy coding! 🎉
