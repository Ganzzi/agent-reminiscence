# API Reference

Complete API documentation for agent-reminiscence v0.2.0

## Table of Contents

1. [Core API Methods](#core-api-methods)
2. [Search Methods](#search-methods)
3. [Memory Management](#memory-management)
4. [Data Models](#data-models)
5. [Error Handling](#error-handling)
6. [Examples](#examples)

## Core API Methods

### AgentMem Class

Entry point for all memory operations. Provides stateless interface for managing agent memories.

```python
from agent_reminiscence import AgentMem, Config

config = Config(
    postgres_host="localhost",
    postgres_port=5432,
    postgres_user="postgres",
    postgres_password="password",
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    ollama_base_url="http://localhost:11434"
)

agent_mem = AgentMem(config=config)
await agent_mem.initialize()
```

### Initialization

**Method**: `async def initialize() -> None`

Initializes database connections and services. Must be called before using any memory operations.

**Parameters**: None

**Returns**: None

**Raises**: 
- `RuntimeError`: If initialization fails
- Database connection errors

**Example**:
```python
try:
    await agent_mem.initialize()
    print("AgentMem initialized successfully")
except RuntimeError as e:
    print(f"Failed to initialize: {e}")
```

---

## Search Methods

### search_memories()

Fast pointer-based memory retrieval without AI synthesis.

**Method**: 
```python
async def search_memories(
    external_id: str | UUID | int,
    query: str,
    limit: int = 10
) -> RetrievalResult
```

**Parameters**:
- `external_id` (str | UUID | int): **Required**. Agent identifier. Can be string, UUID, or integer.
- `query` (str): **Required**. Natural language search query describing what you're looking for.
- `limit` (int): Optional. Maximum number of results to return. Default: 10. Range: 1-100.

**Returns**: `RetrievalResult` containing:
- `mode` (str): "pointer" - indicates fast search mode
- `chunks` (List[RetrievedChunk]): Matched memory chunks with scores
- `entities` (List[Entity]): Related entities from knowledge graph
- `relationships` (List[Relationship]): Related entity relationships
- `synthesis` (str | None): None for pointer mode
- `metadata` (dict): Additional metadata

**Use Cases**:
- Quick lookups of recent memories
- Direct reference searches
- Fast response time requirements (< 200ms)
- When full synthesis is not needed

**Example**:
```python
# Fast search for specific information
result = await agent_mem.search_memories(
    external_id="agent-123",
    query="JWT authentication implementation",
    limit=5
)

# Process results
for chunk in result.chunks:
    print(f"Found: {chunk.content} (score: {chunk.score})")

print(f"Found {len(result.entities)} entities")
```

**Performance**: < 200ms typical response time

---

### deep_search_memories()

Comprehensive memory retrieval with full AI synthesis, entity extraction, and relationship analysis.

**Method**:
```python
async def deep_search_memories(
    external_id: str | UUID | int,
    query: str,
    limit: int = 10
) -> RetrievalResult
```

**Parameters**:
- `external_id` (str | UUID | int): **Required**. Agent identifier. Can be string, UUID, or integer.
- `query` (str): **Required**. Natural language search query. Can be complex and multi-part.
- `limit` (int): Optional. Maximum number of base results to retrieve before synthesis. Default: 10. Range: 1-100.

**Returns**: `RetrievalResult` containing:
- `mode` (str): "synthesis" - indicates deep search mode
- `chunks` (List[RetrievedChunk]): Matched memory chunks with scores
- `entities` (List[Entity]): Extracted and analyzed entities
- `relationships` (List[Relationship]): Extracted entity relationships
- `synthesis` (str): **AI-generated summary connecting findings**
- `metadata` (dict): Search metadata including confidence scores

**Use Cases**:
- Comprehensive context retrieval
- Understanding relationships between concepts
- Complex multi-part queries
- Generating AI summaries of findings
- Entity relationship analysis
- Decision support

**Example**:
```python
# Deep search with synthesis
result = await agent_mem.deep_search_memories(
    external_id="agent-123",
    query="How does JWT authentication relate to our API security design?",
    limit=10
)

# AI-synthesized summary of findings
print("Summary:", result.synthesis)

# Analyze relationships
for rel in result.relationships:
    print(f"{rel.from_entity_name} -{rel.types}-> {rel.to_entity_name}")
    print(f"  Description: {rel.description}")

# Entity insights
for entity in result.entities:
    print(f"Entity: {entity.name} (Types: {entity.types})")
    print(f"  Importance: {entity.importance}")
```

**Performance**: 500ms - 2s typical response time (includes AI synthesis)

**Token Usage**: May consume LLM tokens. Check `result.metadata` for token usage information.

---

### retrieve_memories()

Legacy method for backward compatibility. Searches across multiple memory tiers.

**Method**:
```python
async def retrieve_memories(
    external_id: str | UUID | int,
    query: str,
    use_active: bool = True,
    use_shortterm: bool = True,
    use_longterm: bool = True,
    synthesis: bool = False
) -> RetrievalResult
```

**Parameters**:
- `external_id`: Agent identifier
- `query`: Search query
- `use_active`: Include active memory tier
- `use_shortterm`: Include shortterm memory tier
- `use_longterm`: Include longterm memory tier
- `synthesis`: Generate AI synthesis

**Returns**: `RetrievalResult`

**Note**: Use `search_memories()` or `deep_search_memories()` for new code.

---

## Memory Management

### create_active_memory()

Create a new active memory with template-driven sections.

**Method**:
```python
async def create_active_memory(
    external_id: str | UUID | int,
    title: str,
    template_content: dict,
    initial_sections: dict | None = None,
    metadata: dict | None = None
) -> ActiveMemory
```

**Parameters**:
- `external_id`: Agent identifier
- `title`: Human-readable memory title
- `template_content`: Dict with template and sections definition
- `initial_sections`: Optional dict with initial content for sections
- `metadata`: Optional additional metadata

**Returns**: `ActiveMemory` object

**Example**:
```python
# Create a task memory
result = await agent_mem.create_active_memory(
    external_id="agent-123",
    title="Current Task Progress",
    template_content={
        "template": {
            "id": "task_memory_v1",
            "name": "Task Memory"
        },
        "sections": [
            {
                "id": "current_task",
                "description": "What is being worked on now"
            },
            {
                "id": "progress",
                "description": "Status and completion tracking"
            }
        ]
    },
    initial_sections={
        "current_task": {
            "content": "Implementing authentication system"
        },
        "progress": {
            "content": "Just started, 0% complete"
        }
    }
)

print(f"Created memory: {result.id}")
```

---

### get_active_memories()

Retrieve all active memories for an agent.

**Method**:
```python
async def get_active_memories(
    external_id: str | UUID | int
) -> list[ActiveMemory]
```

**Parameters**:
- `external_id`: Agent identifier

**Returns**: List of `ActiveMemory` objects

**Example**:
```python
# Get all memories for agent
memories = await agent_mem.get_active_memories(
    external_id="agent-123"
)

print(f"Agent has {len(memories)} active memories")
for memory in memories:
    print(f"  - {memory.title} ({len(memory.sections)} sections)")
```

---

### update_active_memory()

Update sections in an active memory.

**Method**:
```python
async def update_active_memory(
    external_id: str | UUID | int,
    memory_id: int,
    sections: list[dict]
) -> ActiveMemory
```

**Parameters**:
- `external_id`: Agent identifier
- `memory_id`: ID of memory to update
- `sections`: List of section updates

**Returns**: Updated `ActiveMemory` object

**Example**:
```python
# Update memory sections
result = await agent_mem.update_active_memory(
    external_id="agent-123",
    memory_id=1,
    sections=[
        {
            "section_id": "progress",
            "action": "replace",
            "new_content": "30% complete - authentication module done"
        },
        {
            "section_id": "current_task",
            "action": "replace",
            "new_content": "Implementing authorization system"
        }
    ]
)

print(f"Updated memory: {result.id}")
```

---

### delete_active_memory()

Delete an active memory.

**Method**:
```python
async def delete_active_memory(
    external_id: str | UUID | int,
    memory_id: int
) -> bool
```

**Parameters**:
- `external_id`: Agent identifier
- `memory_id`: ID of memory to delete

**Returns**: True if successful

**Example**:
```python
# Delete a memory
success = await agent_mem.delete_active_memory(
    external_id="agent-123",
    memory_id=1
)

if success:
    print("Memory deleted")
```

---

## Data Models

### RetrievalResult

Result object from search operations.

```python
class RetrievalResult:
    mode: str  # "pointer" or "synthesis"
    chunks: list[RetrievedChunk]
    entities: list[Entity]
    relationships: list[Relationship]
    synthesis: str | None
    metadata: dict
```

### RetrievedChunk

A chunk of retrieved memory content.

```python
class RetrievedChunk:
    id: str
    content: str
    tier: str  # "active", "shortterm", or "longterm"
    score: float  # Relevance score 0.0-1.0
    importance: float | None
    start_date: datetime | None
```

### Entity

Entity from the knowledge graph.

```python
class Entity:
    id: str
    name: str
    types: list[str]
    description: str | None
    tier: str
    importance: float
```

### Relationship

Relationship between entities.

```python
class Relationship:
    id: str
    from_entity_name: str
    to_entity_name: str
    types: list[str]
    description: str | None
    tier: str
    importance: float
```

### ActiveMemory

Active memory structure with template-driven sections.

```python
class ActiveMemory:
    id: int
    external_id: str
    title: str
    sections: dict[str, Section]
    metadata: dict
    created_at: datetime
    updated_at: datetime
```

---

## Error Handling

### Common Errors

**RuntimeError**: AgentMem not initialized
```python
try:
    result = await agent_mem.search_memories(...)
except RuntimeError as e:
    print("Error: AgentMem not initialized")
    print(e)
```

**ValueError**: Invalid parameters
```python
try:
    result = await agent_mem.search_memories(
        external_id="",  # Empty ID
        query="test"
    )
except ValueError as e:
    print("Error: Invalid parameters")
    print(e)
```

**Database Errors**: Connection issues
```python
try:
    result = await agent_mem.search_memories(...)
except Exception as e:
    print(f"Database error: {e}")
```

### Best Practices

1. Always initialize before use
2. Validate external_id is not empty
3. Handle timeouts for deep_search operations
4. Check result.synthesis before using (may be None for pointer mode)
5. Use try-catch for all async operations

---

## Examples

### Example 1: Complete Search Workflow

```python
import asyncio
from agent_reminiscence import AgentMem, Config

async def main():
    # Initialize
    config = Config(...)
    agent_mem = AgentMem(config=config)
    await agent_mem.initialize()
    
    # Fast search
    fast_result = await agent_mem.search_memories(
        external_id="agent-1",
        query="database configuration",
        limit=5
    )
    print(f"Fast search: {len(fast_result.chunks)} results")
    
    # Deep search
    deep_result = await agent_mem.deep_search_memories(
        external_id="agent-1",
        query="How does caching relate to our database strategy?",
        limit=10
    )
    print(f"Summary: {deep_result.synthesis}")

asyncio.run(main())
```

### Example 2: Memory Management

```python
# Create memory
memory = await agent_mem.create_active_memory(
    external_id="agent-1",
    title="Project Status",
    template_content={...}
)

# Update sections
updated = await agent_mem.update_active_memory(
    external_id="agent-1",
    memory_id=memory.id,
    sections=[...]
)

# Retrieve and search
memories = await agent_mem.get_active_memories(
    external_id="agent-1"
)

# Delete if needed
success = await agent_mem.delete_active_memory(
    external_id="agent-1",
    memory_id=memory.id
)
```

---

## Version

- API Version: 2.0.0
- Package Version: 0.2.0
- Last Updated: November 16, 2025
