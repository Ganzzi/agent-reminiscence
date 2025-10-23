# Memory Tiers

Agent Mem organizes memories into three tiers, each optimized for different use cases and lifespans.

## Active Memory

**Purpose**: Template-driven working memory for current tasks

**Characteristics**:
- **Lifetime**: Hours to days
- **Storage**: PostgreSQL (JSONB for sections)
- **Structure**: YAML template + structured sections
- **Update Frequency**: Very frequent (per-section tracking)
- **Consolidation**: Section-level, triggered by update_count

**Storage Schema**:
```
active_memories
├── id (serial)
├── external_id (text) - Agent identifier
├── title (text)
├── template_id (text)
├── template_version (text)
├── sections (jsonb) - {section_id: {content, update_count}}
├── metadata (jsonb)
├── created_at (timestamp)
└── updated_at (timestamp)
```

**When to Use**:
- Current task context
- Ongoing work tracking
- Structured progress updates
- Real-time collaboration state

**Example**:
```python
memory = await agent_reminiscence.create_active_memory(
    external_id="agent-123",
    title="API Development",
    template_content=API_TEMPLATE,
    initial_sections={
        "endpoints": {
            "content": "POST /api/users\nGET /api/users/:id",
            "update_count": 0
        },
        "status": {
            "content": "In progress - 60% complete",
            "update_count": 0
        }
    }
)
```

## Shortterm Memory

**Purpose**: Searchable recent knowledge with entity relationships

**Characteristics**:
- **Lifetime**: Days to weeks
- **Storage**: PostgreSQL (vectors + BM25) + Neo4j (graph)
- **Structure**: Text chunks + entities + relationships
- **Update Frequency**: Occasional (from consolidation)
- **Search**: Vector similarity + BM25 keyword + graph queries

**Storage Schema**:
```
# PostgreSQL
shortterm_memories
├── id (serial)
├── external_id (text)
├── content (text)
├── embedding (vector) - For semantic search
├── chunk_index (int)
├── source_id (int) - References active_memory
├── importance_score (float)
├── access_count (int)
├── created_at (timestamp)
└── last_accessed_at (timestamp)

# Neo4j
(:Entity {name, type, external_id, memory_tier: "shortterm"})
(:Entity)-[:RELATES_TO {type, strength}]->(:Entity)
```

**When to Use**:
- Recent implementations
- Recent conversations
- Research findings
- Technical decisions (recent)

**Example**:
```python
# Automatic consolidation from active memory
await agent_reminiscence.update_active_memory_section(
    external_id="agent-123",
    memory_id=1,
    section_id="endpoints",
    new_content="Added DELETE endpoint..."
)
# After threshold reached, section consolidates to shortterm

# Manual retrieval
results = await agent_reminiscence.retrieve_memories(
    external_id="agent-123",
    query="API endpoint implementations",
    search_shortterm=True,
    search_longterm=False
)
```

## Longterm Memory

**Purpose**: Consolidated, persistent knowledge base

**Characteristics**:
- **Lifetime**: Persistent (never expires)
- **Storage**: PostgreSQL (vectors + BM25) + Neo4j (graph)
- **Structure**: Consolidated chunks + entities + relationships
- **Update Frequency**: Rare (promoted from shortterm)
- **Search**: Vector similarity + BM25 keyword + graph queries

**Storage Schema**:
```
# PostgreSQL
longterm_memories
├── id (serial)
├── external_id (text)
├── content (text)
├── embedding (vector)
├── source_ids (int[]) - References shortterm memories
├── consolidation_count (int)
├── importance_score (float)
├── access_count (int)
├── created_at (timestamp)
└── last_accessed_at (timestamp)

# Neo4j
(:Entity {name, type, external_id, memory_tier: "longterm"})
(:Entity)-[:RELATES_TO {type, strength}]->(:Entity)
```

**When to Use**:
- Core system knowledge
- Established patterns
- Historical decisions
- Domain expertise

**Example**:
```python
# Automatic promotion from shortterm
# (Happens internally based on importance_score >= threshold)

# Retrieval includes longterm by default
results = await agent_reminiscence.retrieve_memories(
    external_id="agent-123",
    query="system architecture patterns",
    search_longterm=True
)
```

## Memory Lifecycle

### 1. Creation (Active)

```python
memory = await agent_reminiscence.create_active_memory(
    external_id="agent-123",
    title="Feature Development",
    template_content=TEMPLATE,
    initial_sections={"task": {"content": "...", "update_count": 0}}
)
```

### 2. Updates (Active → Shortterm)

```python
# Each update increments section's update_count
await agent_reminiscence.update_active_memory_section(
    external_id="agent-123",
    memory_id=memory.id,
    section_id="task",
    new_content="Updated task description"
)

# When update_count >= ACTIVE_MEMORY_UPDATE_THRESHOLD:
# → Section consolidates to shortterm memory
# → Entities/relationships extracted
# → Embeddings generated
```

### 3. Consolidation (Active → Shortterm)

Triggered automatically when:
- Section's `update_count >= ACTIVE_MEMORY_UPDATE_THRESHOLD` (default: 5)

Process:
1. Extract section content
2. Generate text chunks
3. Create embeddings
4. Extract entities and relationships (using AI agent)
5. Store in shortterm memory (PostgreSQL + Neo4j)
6. Reset section's update_count

### 4. Promotion (Shortterm → Longterm)

Triggered automatically when:
- `importance_score >= SHORTTERM_PROMOTION_THRESHOLD` (default: 0.7)
- Access patterns indicate high value
- Memory age exceeds threshold

Process:
1. Consolidate related shortterm memories
2. Merge entities and relationships
3. Update embeddings
4. Store in longterm memory
5. Optionally archive shortterm entries

## Comparison Table

| Feature | Active | Shortterm | Longterm |
|---------|--------|-----------|----------|
| **Lifetime** | Hours-Days | Days-Weeks | Persistent |
| **Structure** | Template+Sections | Chunks+Entities | Consolidated |
| **Updates** | Very Frequent | Occasional | Rare |
| **Search** | Direct access | Vector+BM25+Graph | Vector+BM25+Graph |
| **Size** | Small | Medium | Large |
| **Use Case** | Current work | Recent knowledge | Core knowledge |

## Configuration

Control memory behavior with environment variables:

```env
# Active memory consolidation threshold
ACTIVE_MEMORY_UPDATE_THRESHOLD=5

# Shortterm promotion threshold
SHORTTERM_PROMOTION_THRESHOLD=0.7
```

Or in code:

```python
config = Config(
    active_memory_update_threshold=5,
    shortterm_promotion_threshold=0.7
)
agent_reminiscence = AgentMem(config=config)
```

## Next Steps

- **[Active Memory Guide](active-memory.md)**: Working with templates
- **[Memory Retrieval](memory-retrieval.md)**: Advanced search
- **[Best Practices](best-practices.md)**: Optimization tips

