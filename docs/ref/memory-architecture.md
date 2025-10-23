# Memory Architecture: Active, Shortterm, and Longterm Memory

## Overview

The AI Army application implements a three-tier hierarchical memory system with dual storage (PostgreSQL and Neo4j). This architecture enables efficient memory management with different levels of persistence, detail, and temporal validity.

## Table of Contents

1. [Memory Tiers](#memory-tiers)
2. [Dual Storage Architecture](#dual-storage-architecture)
3. [Active Memory](#active-memory)
4. [Shortterm Memory](#shortterm-memory)
5. [Longterm Memory](#longterm-memory)
6. [Migration Algorithms](#migration-algorithms)
7. [Memory Flow](#memory-flow)
8. [Implementation Details](#implementation-details)

## Memory Tiers

### Hierarchy Overview

```
┌─────────────────────────────────────────────────┐
│              ACTIVE MEMORY                      │
│  - Summary-level information                    │
│  - Frequently updated                           │
│  - Worker-specific context                      │
│  - No vector search                             │
└─────────────────────────────────────────────────┘
                    │
                    │ Migration (via Memorizer Agent)
                    ↓
┌─────────────────────────────────────────────────┐
│            SHORTTERM MEMORY                     │
│  - Detailed chunks with embeddings              │
│  - Entities and relationships in Neo4j          │
│  - Vector + BM25 search enabled                 │
│  - Time-bounded relevance                       │
└─────────────────────────────────────────────────┘
                    │
                    │ Promotion (based on importance)
                    ↓
┌─────────────────────────────────────────────────┐
│            LONGTERM MEMORY                      │
│  - Consolidated knowledge                       │
│  - High importance content                      │
│  - Temporal validity tracking                   │
│  - Entities with confidence scores              │
└─────────────────────────────────────────────────┘
```

### Memory Characteristics

| Feature | Active Memory | Shortterm Memory | Longterm Memory |
|---------|--------------|------------------|-----------------|
| **Granularity** | Summary | Detailed chunks | Consolidated |
| **Storage** | PostgreSQL only | PostgreSQL + Neo4j | PostgreSQL + Neo4j |
| **Vector Search** | No | Yes | Yes |
| **BM25 Search** | No | Yes | Yes |
| **Entities** | Yes (extracted) | Yes (graph nodes) | Yes (graph nodes) |
| **Relationships** | Yes (extracted) | Yes (graph edges) | Yes (graph edges) |
| **Temporal Validity** | No | No | Yes |
| **Update Frequency** | Very High | High | Low |
| **Retention** | Short (task-specific) | Medium (days-weeks) | Long (persistent) |

## Dual Storage Architecture

### PostgreSQL Storage

PostgreSQL stores textual content, chunks, and metadata:

```sql
-- Active Memory (summary-level)
CREATE TABLE active_memory (
    id SERIAL PRIMARY KEY,
    worker_id INTEGER NOT NULL,
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Shortterm Memory (summary-level)
CREATE TABLE shortterm_memory (
    id SERIAL PRIMARY KEY,
    worker_id INTEGER NOT NULL,
    title VARCHAR(500) NOT NULL,
    summary TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Shortterm Memory Chunks (with vectors)
CREATE TABLE shortterm_memory_chunk (
    id SERIAL PRIMARY KEY,
    shortterm_memory_id INTEGER NOT NULL,
    chunk_order INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1024),  -- Vector embeddings
    metadata JSONB DEFAULT '{}'
);

-- Longterm Memory (summary-level)
CREATE TABLE longterm_memory (
    id SERIAL PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    summary TEXT NOT NULL,
    importance_score FLOAT DEFAULT 0.5,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Longterm Memory Chunks (with vectors and temporal validity)
CREATE TABLE longterm_memory_chunk (
    id SERIAL PRIMARY KEY,
    longterm_memory_id INTEGER NOT NULL,
    chunk_order INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1024),
    importance_score FLOAT DEFAULT 0.5,
    confidence_score FLOAT DEFAULT 0.5,
    valid_from TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    valid_until TIMESTAMP,  -- NULL = still valid
    metadata JSONB DEFAULT '{}'
);
```

### Neo4j Storage

Neo4j stores entities and their relationships as a graph:

```cypher
// Shortterm Entity
CREATE (e:ShorttermEntity {
    id: INTEGER,           // References PostgreSQL entity ID
    name: STRING,
    type: STRING,
    description: STRING,
    first_seen: DATETIME,
    last_seen: DATETIME,
    confidence: FLOAT,
    metadata: MAP
})

// Longterm Entity
CREATE (e:LongtermEntity {
    id: INTEGER,
    name: STRING,
    type: STRING,
    description: STRING,
    confidence: FLOAT,
    importance: FLOAT,
    valid_from: DATETIME,
    valid_until: DATETIME,  // NULL = still valid
    metadata: MAP
})

// Relationships
CREATE (e1)-[r:RELATES_TO {
    id: INTEGER,
    type: STRING,
    description: STRING,
    confidence: FLOAT,
    strength: FLOAT,
    first_observed: DATETIME,
    last_observed: DATETIME,
    metadata: MAP
}]->(e2)
```

### Why Dual Storage?

1. **PostgreSQL Strengths**:
   - Excellent for vector similarity search (pgvector)
   - Efficient full-text search (BM25)
   - ACID compliance for transactional data
   - Structured data with strong typing
   - Efficient chunk storage and retrieval

2. **Neo4j Strengths**:
   - Native graph queries for entity relationships
   - Path finding between entities
   - Relationship pattern matching
   - Flexible schema for evolving entity types
   - Graph traversal algorithms

3. **Combined Benefits**:
   - Search chunks in PostgreSQL (vector + BM25)
   - Explore entity relationships in Neo4j
   - Maintain referential integrity via IDs
   - Optimize each storage for its strengths

## Active Memory

### Purpose

Active memory serves as working memory for workers (agents). It contains:
- Current task context stored as sections
- Recent activities
- Summary-level information
- Plain text content (no vector embeddings)

### Structure

```python
class ActiveMemory:
    id: int
    worker_id: int
    title: str
    description: str
    template_content: str        # YAML template stored as text
    sections: Dict[str, Dict]    # JSONB: section_name -> {"content": str, "update_count": int}
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
```

**SQL Schema:**

```sql
CREATE TABLE active_memory (
    id SERIAL PRIMARY KEY,
    worker_id INTEGER NOT NULL,
    title VARCHAR(255),
    description TEXT,
    template_content TEXT,      -- YAML template as text
    sections JSONB NOT NULL,    -- {"section_name": {"content": "markdown", "update_count": 0}}
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Example sections structure:
{
    "current_task": {
        "content": "# Current Task\n\nImplementing authentication...",
        "update_count": 3
    },
    "progress": {
        "content": "# Progress\n\n- [x] Set up database\n- [ ] Create API endpoints",
        "update_count": 5
    },
    "blockers": {
        "content": "# Blockers\n\nWaiting for API keys",
        "update_count": 1
    }
}
```

**Template Content Format:**

The `template_content` field stores the YAML template as text:

```yaml
template:
  id: "coding_agent_activity_v1"
  name: "Coding Agent Activity"
  version: "1.0.0"

sections:
  - id: "current_task"
    name: "Current Task"
    required: true
  - id: "progress"
    name: "Progress"
    required: true
  - id: "blockers"
    name: "Blockers"
    required: false
```

### Characteristics

- **Template-Driven**: Uses YAML templates stored in `template_content` to define structure
- **Markdown Sections**: Each section contains `content` (markdown) and `update_count` (integer)
- **Update Tracking**: `update_count` increments on each section update, used for migration eligibility
- **Auto-initialization**: Sections created automatically from template on memory creation
- **No Vector Search**: Active memory is summary-level, not chunked or embedded
- **No Entities/Relationships Stored**: Entities and relationships are NOT stored in active memory
- **Entity Extraction on Migration**: When migrating to shortterm, content is used to extract entities and relationships
- **Section-Based Updates**: Sections can be updated independently via JSONB operations
- **Frequent Updates**: Updated continuously as workers perform tasks
- **Worker-Specific**: Each worker has its own active memories
- **Short-Lived**: Migrated to shortterm memory based on update count criteria

### Operations

```python
from memory.memory_manager import memory_manager

# Create active memory with template and sections
template_yaml = """
template:
  id: "coding_agent_activity_v1"
  name: "Coding Agent Activity"
  version: "1.0.0"
sections:
  - id: "current_task"
    name: "Current Task"
  - id: "progress"
    name: "Progress"
"""

active_memory = await memory_manager.create_active_memory(
    worker_id=1,
    title="Current Task Context",
    template_content=template_yaml,
    # Sections are auto-initialized from template with empty content and update_count=0
    # Alternatively, provide initial content:
    initial_sections={
        "current_task": "# Current Task\n\nImplementing authentication",
        "progress": "# Progress\n\n- [x] Set up database"
    },
    metadata={"task_id": 123}
)
# Resulting sections:
# {
#     "current_task": {"content": "# Current Task\n\nImplementing authentication", "update_count": 0},
#     "progress": {"content": "# Progress\n\n- [x] Set up database", "update_count": 0}
# }

# Update a specific section (increments update_count)
await memory_manager.update_active_memory_section(
    memory_id=active_memory.id,
    section_name="progress",
    new_content="# Progress\n\n- [x] Set up database\n- [x] Create API endpoints"
)
# Resulting section:
# "progress": {"content": "...", "update_count": 1}

# Get all active memories for a worker
memories = await memory_manager.get_active_memories_by_worker(worker_id=1)
```

## Shortterm Memory

### Purpose

Shortterm memory stores detailed, searchable information:
- Detailed task results
- Conversation history
- Code analysis results
- Research findings
- Recent context (hours to days)

### Structure

**PostgreSQL**:
```python
class ShorttermMemory:
    id: int
    worker_id: int
    title: str
    summary: str
    created_at: datetime
    last_updated: datetime
    metadata: Dict[str, Any]

class ShorttermMemoryChunk:
    id: int
    shortterm_id: int         # Reference to shortterm_memory.id
    content: str
    content_vector: List[float]  # 768-dimensional vector
    content_bm25: bm25vector     # Auto-generated BM25 vector
    metadata: Dict[str, Any]
```

**Important Notes:**
- Active memory sections are chunked **only if they are large**
- Small sections are embedded and stored as single chunks
- Chunks are linked to shortterm_memory via `shortterm_id`

**Neo4j**:
```python
class ShorttermEntity:
    id: int
    shortterm_memory_id: int  # Reference to PostgreSQL shortterm_memory.id
    worker_id: int
    name: str
    type: str
    description: str
    first_seen: datetime
    last_seen: datetime
    confidence: float
    metadata: Dict[str, Any]

class ShorttermRelationship:
    id: int
    shortterm_memory_id: int  # Reference to PostgreSQL shortterm_memory.id
    worker_id: int
    from_entity_id: int
    to_entity_id: int
    type: str
    description: str
    confidence: float
    strength: float
    first_observed: datetime
    last_observed: datetime
```

### Characteristics

- **Chunked Content**: Large sections are split into chunks; small sections stored as-is
- **Vector Embeddings**: Each chunk has a vector embedding (768-dim)
- **BM25 Vector**: Auto-generated via trigger when content is inserted/updated
- **Graph Entities**: Entities stored as Neo4j nodes with shortterm_memory_id reference
- **Graph Relationships**: Relationships as Neo4j edges with shortterm_memory_id reference
- **Medium Retention**: Kept for days to weeks before promotion to longterm

### Search Operations

```python
from memory.memory_manager import memory_manager

# Hybrid search (vector + BM25)
results = await memory_manager.search_shortterm_memory(
    worker_id=1,
    query_text="user authentication implementation",
    limit=10,
    min_similarity=0.7
)

# Results include matched chunks, entities, and relationships
for result in results:
    print(f"Chunk: {result.chunk.content}")
    print(f"Similarity: {result.similarity_score}")
    print(f"Entities: {[e.name for e in result.entities]}")
```

## Longterm Memory

### Purpose

Longterm memory stores consolidated, important knowledge across different time periods:
- Core domain knowledge
- Important facts and concepts
- Historical relationships
- Temporal evolution of information

### Structure

**PostgreSQL**:
```python
class LongtermMemoryChunk:
    id: int
    worker_id: int            # Direct reference to worker, no longterm_memory table
    shortterm_memory_id: int  # Reference to source shortterm_memory
    content: str
    content_vector: List[float]  # 768-dimensional vector
    content_bm25: bm25vector     # Auto-generated BM25 vector
    confidence_score: float
    start_date: datetime      # When this version became valid
    end_date: datetime        # When this version ended (NULL = still valid)
    metadata: Dict[str, Any]
```

**Important Design Notes:**
- **No longterm_memory table**: Only the chunk table exists
- **No longterm_memory_id**: Chunks reference worker_id and shortterm_memory_id directly
- **Temporal tracking**: `start_date` and `end_date` track when chunks are valid
- **Version history**: Multiple chunks can exist for the same content across different time periods

**Neo4j**:
```python
class LongtermEntity:
    id: int
    worker_id: int
    name: str
    type: str
    description: str
    first_seen: datetime
    last_seen: datetime
    confidence: float
    importance: float
    metadata: Dict[str, Any]

class LongtermRelationship:
    id: int
    worker_id: int
    from_entity_id: int
    to_entity_id: int
    type: str
    description: str
    confidence: float
    strength: float
    start_date: datetime      # When this relationship version started
    last_updated: datetime    # When relationship was last updated
    metadata: Dict[str, Any]
```

**Longterm Relationship Constraints:**
```cypher
// Prevent duplicate relationships
CREATE CONSTRAINT relationship_temporal_unique IF NOT EXISTS 
FOR ()-[r:RELATES_TO]-()
REQUIRE (r.from_entity_id, r.to_entity_id, r.type, r.start_date) IS UNIQUE;
```

This allows the same relationship type between two entities to exist across different time periods, each uniquely identified by its `start_date`. The `last_updated` timestamp tracks when the relationship was last modified.

### Characteristics

- **No Summary Table**: Longterm memory consists only of chunks, not a summary table
- **Temporal Tracking**: `start_date` and `end_date` track validity periods
- **Version History**: Same content can exist across multiple time periods
- **Source Tracking**: `shortterm_memory_id` links back to the source shortterm memory
- **Consolidated**: Duplicate/similar info merged across time
- **Long Retention**: Kept indefinitely with temporal validity

### Query Operations

```python
from memory.memory_manager import memory_manager

# Search longterm memory chunks (currently valid)
results = await memory_manager.search_longterm_chunks(
    worker_id=1,
    query_text="authentication patterns",
    valid_at=datetime.now(),  # Only get currently valid chunks
    limit=10
)

# Get historical versions of a chunk
versions = await memory_manager.get_chunk_history(
    worker_id=1,
    shortterm_memory_id=123
)
```

## Migration Algorithms

### Active to Shortterm Migration

#### Trigger Conditions

1. **Update Count Threshold**: Sections with `update_count >= 1` are eligible for migration
2. **Worker Memory Threshold**: Migration starts when aggregate update activity reaches threshold
3. **Manual Trigger**: Can be triggered manually by agents or on task completion

#### Algorithm Overview

```python
async def migrate_active_to_shortterm(worker_id: int):
    """Migrate active memory sections to shortterm memory."""
    
    # Step 1: Get active memory
    active_memory = await get_active_memory(worker_id)
    
    # Step 2: Filter sections with update_count >= 1
    eligible_sections = {
        name: data for name, data in active_memory.sections.items()
        if data["update_count"] >= 1
    }
    
    if not eligible_sections:
        return  # No sections to migrate
    
    # Step 3: For each eligible section
    for section_name, section_data in eligible_sections.items():
        section_content = section_data["content"]
        
        # Get existing shortterm memory chunks for this worker
        existing_chunks = await search_shortterm_memory_semantic(
            worker_id=worker_id,
            query_text=section_content,
            limit=5,
            similarity_threshold=0.85
        )
        
        # Step 4: Extract entities and relationships from section
        section_entities = await extract_entities(section_content)
        section_relationships = await extract_relationships(section_content)
        
        # Step 5: Compare with existing chunks
        similar_chunks = []
        for chunk in existing_chunks:
            similarity = calculate_semantic_similarity(
                section_content, 
                chunk.content
            )
            
            if similarity >= 0.85:
                # Get entities/relationships from existing chunk
                chunk_entities = await get_entities_by_shortterm_id(chunk.id)
                chunk_relationships = await get_relationships_by_shortterm_id(chunk.id)
                
                # Compare entities and relationships
                entity_overlap = calculate_entity_overlap(
                    section_entities, 
                    chunk_entities
                )
                relationship_overlap = calculate_relationship_overlap(
                    section_relationships,
                    chunk_relationships
                )
                
                similar_chunks.append({
                    "chunk": chunk,
                    "similarity": similarity,
                    "entity_overlap": entity_overlap,
                    "relationship_overlap": relationship_overlap
                })
        
        # Step 6: Decide merge or create new
        if similar_chunks:
            # Check for conflicts
            has_conflicts = await check_for_conflicts(
                section_entities,
                section_relationships,
                similar_chunks
            )
            
            if has_conflicts:
                # Manual merge required - mark for human review
                await mark_for_manual_merge(
                    worker_id=worker_id,
                    section=section,
                    conflicting_chunks=similar_chunks,
                    reason="Conflicting information detected"
                )
            else:
                # Auto-resolve: update existing chunk
                best_match = max(
                    similar_chunks, 
                    key=lambda x: x["similarity"]
                )
                await update_shortterm_chunk(
                    chunk_id=best_match["chunk"].id,
                    new_content=merge_content(
                        best_match["chunk"].content,
                        section_content
                    )
                )
                # Update entities and relationships
                await update_entities(best_match["chunk"].id, section_entities)
                await update_relationships(best_match["chunk"].id, section_relationships)
        else:
            # No similar content - create new shortterm memory
            # Step 7a: Chunk if large (>500 tokens), otherwise store as single chunk
            if len(section_content) > 500:  # Token estimate
                chunks = semantic_chunk(section_content)
            else:
                chunks = [section_content]
            
            # Step 7b: Create shortterm memory entry
            shortterm_id = await create_shortterm_memory(
                worker_id=worker_id,
                content=section_content,
                metadata={"section_name": section_name}
            )
            
            # Step 7c: Store chunks with embeddings
            for chunk_text in chunks:
                await create_shortterm_chunk(
                    shortterm_memory_id=shortterm_id,
                    worker_id=worker_id,
                    content=chunk_text
                )
            
            # Step 7d: Store extracted entities and relationships in Neo4j
            for entity in section_entities:
                await create_shortterm_entity(
                    worker_id=worker_id,
                    shortterm_memory_id=shortterm_id,
                    **entity
                )
            
            for rel in section_relationships:
                await create_shortterm_relationship(
                    worker_id=worker_id,
                    shortterm_memory_id=shortterm_id,
                    **rel
                )
        
        # Step 8: Reset section update_count after successful migration
        active_memory.sections[section_name]["update_count"] = 0
    
    # Step 9: Save updated active memory with reset counts
    await update_active_memory(worker_id, active_memory)
```

#### Auto-Resolution vs Manual Merge

**Auto-resolution occurs when**:
- High semantic similarity (>= 0.85)
- High entity overlap (>= 0.7)
- No conflicting facts or relationships
- Information can be safely merged

**Manual merge required when**:
- Contradictory information detected
- Low confidence in entity/relationship matching
- Significant semantic differences despite similarity
- Complex merge logic needed

#### Section Update Count Reset

After successful migration (whether merged or created new), the section's `update_count` is reset to 0. This ensures sections won't be migrated again until they accumulate new updates.

**Implementation:**
```python
# After migration
active_memory.sections[section_name]["update_count"] = 0
await save_active_memory(active_memory)
```

### Shortterm to Longterm Promotion

Shortterm memories are promoted to longterm based on **importance scoring** and **temporal tracking**.

#### Algorithm Overview

```python
async def promote_shortterm_to_longterm(
    shortterm_memory_id: int,
    importance_threshold: float = 0.7
) -> None:
    """
    Promote shortterm memory chunks to longterm memory.
    
    Process:
    1. Get shortterm memory and its chunks
    2. Calculate importance score
    3. Check if meets threshold
    4. For each chunk, add to longterm with temporal tracking
    5. Update end_date of previous longterm chunks from same source
    6. Promote entities and relationships with temporal tracking
    """
    
    # Step 1: Get shortterm memory
    shortterm_memory = await shortterm_memory_service.get_by_id(
        shortterm_memory_id
    )
    
    # Step 2: Calculate importance
    importance_score = await calculate_importance_score(shortterm_memory)
    
    if importance_score < importance_threshold:
        return  # Not important enough
    
    # Step 3: Process each shortterm chunk
    for chunk in shortterm_memory.chunks:
        # Step 3a: Find the latest longterm chunk from this shortterm_memory_id
        latest_longterm_chunk = await find_latest_longterm_chunk(
            worker_id=shortterm_memory.worker_id,
            shortterm_memory_id=shortterm_memory_id
        )
        
        # Step 3b: Update end_date of previous chunk
        if latest_longterm_chunk and latest_longterm_chunk.end_date is None:
            await update_longterm_chunk(
                chunk_id=latest_longterm_chunk.id,
                end_date=datetime.now()
            )
        
        # Step 3c: Create new longterm chunk with start_date
        await create_longterm_chunk(
            worker_id=shortterm_memory.worker_id,
            shortterm_memory_id=shortterm_memory_id,
            content=chunk.content,
            confidence_score=importance_score,
            start_date=datetime.now(),
            end_date=None,  # NULL = currently valid
            metadata=chunk.metadata
        )
    
    # Step 4: Process entities
    entities = await get_entities_by_shortterm_id(shortterm_memory_id)
    for entity in entities:
        # Check if entity already exists in longterm
        existing_longterm_entity = await find_longterm_entity(
            worker_id=shortterm_memory.worker_id,
            name=entity.name,
            type=entity.type
        )
        
        if existing_longterm_entity:
            # Update confidence and temporal tracking
            new_confidence = calculate_updated_confidence(
                existing_confidence=existing_longterm_entity.confidence,
                new_evidence=entity.confidence
            )
            await update_longterm_entity(
                entity_id=existing_longterm_entity.id,
                confidence=new_confidence,
                last_seen=datetime.now()
            )
        else:
            # Create new longterm entity
            await create_longterm_entity(
                worker_id=shortterm_memory.worker_id,
                name=entity.name,
                type=entity.type,
                description=entity.description,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                confidence=entity.confidence,
                importance=importance_score,
                metadata=entity.metadata
            )
    
    # Step 5: Process relationships with temporal tracking
    relationships = await get_relationships_by_shortterm_id(shortterm_memory_id)
    for rel in relationships:
        # Check if relationship already exists with same start_date
        existing_rel = await find_longterm_relationship(
            worker_id=shortterm_memory.worker_id,
            from_entity_id=rel.from_entity_id,
            to_entity_id=rel.to_entity_id,
            type=rel.type,
            start_date=datetime.now().date()
        )
        
        if existing_rel:
            # Update existing relationship
            await update_longterm_relationship(
                relationship_id=existing_rel.id,
                confidence=max(existing_rel.confidence, rel.confidence),
                strength=max(existing_rel.strength, rel.strength),
                last_updated=datetime.now()
            )
        else:
            # Create new relationship version
            await create_longterm_relationship(
                worker_id=shortterm_memory.worker_id,
                from_entity_id=rel.from_entity_id,
                to_entity_id=rel.to_entity_id,
                type=rel.type,
                description=rel.description,
                confidence=rel.confidence,
                strength=rel.strength,
                start_date=datetime.now(),
                last_updated=datetime.now(),
                metadata=rel.metadata
            )
```

#### Temporal Chunk Tracking

**Key Principles:**
- Each chunk promotion creates a new longterm chunk with `start_date = now()`
- Previous chunks from the same `shortterm_memory_id` have their `end_date` updated
- This creates a version history: multiple chunks can exist for the same source across different time periods
- Queries can filter by temporal validity: `end_date IS NULL` for current, or `start_date <= X AND (end_date IS NULL OR end_date >= X)` for historical

**Example:**
```sql
-- Find current longterm chunks for a worker
SELECT * FROM longterm_memory_chunk
WHERE worker_id = 1
  AND end_date IS NULL;

-- Find chunks valid at a specific time
SELECT * FROM longterm_memory_chunk
WHERE worker_id = 1
  AND start_date <= '2024-01-15 10:00:00'
  AND (end_date IS NULL OR end_date >= '2024-01-15 10:00:00');

-- Find historical versions of chunks from same shortterm source
SELECT * FROM longterm_memory_chunk
WHERE worker_id = 1
  AND shortterm_memory_id = 123
ORDER BY start_date DESC;
```

#### Entity Confidence Recalculation

When promoting entities to longterm, confidence is recalculated:

```python
def calculate_updated_confidence(
    existing_confidence: float,
    new_evidence: float
) -> float:
    """
    Update confidence score based on new evidence.
    
    Uses Bayesian update:
    - Multiple confirmations increase confidence
    - Contradictions decrease confidence
    """
    # Simple weighted average favoring accumulated evidence
    weight = 0.7  # Weight for existing confidence
    return weight * existing_confidence + (1 - weight) * new_evidence
```

#### Relationship Temporal Tracking

Relationships use `start_date` and `last_updated` to track temporal validity:

- **Unique Constraint**: `(from_entity_id, to_entity_id, type, start_date)` must be unique
- **Version History**: Same relationship can exist across multiple time periods
- **Last Updated**: `last_updated` timestamp shows when the relationship was last modified
- **Historical Queries**: Filter by temporal range to see relationship evolution

## Memory Flow

### Complete Flow Diagram

```
┌──────────────┐
│   Worker     │
│  Activity    │
└──────┬───────┘
       │
       │ Creates/Updates
       ↓
┌──────────────────────┐
│  Active Memory       │
│  - Summary content   │
│  - Entities extracted│
│  - Relationships     │
└──────┬───────────────┘
       │
       │ Periodic Migration
       │ (Memorizer Agent)
       ↓
┌────────────────────────────┐
│  Shortterm Memory          │
│  PostgreSQL:               │
│  - Chunks with embeddings  │
│  Neo4j:                    │
│  - Entity nodes            │
│  - Relationship edges      │
└──────┬─────────────────────┘
       │
       │ Importance-Based Promotion
       │ (Background Process)
       ↓
┌────────────────────────────┐
│  Longterm Memory           │
│  PostgreSQL:               │
│  - Consolidated chunks     │
│  - Temporal validity       │
│  Neo4j:                    │
│  - Validated entities      │
│  - Strong relationships    │
└────────────────────────────┘
```

### Flow Example

```python
# 1. Worker creates active memory
active_memory = await memory_manager.create_active_memory(
    worker_id=1,
    title="User Auth Implementation",
    content="Implemented JWT-based authentication with refresh tokens..."
)

# 2. Periodic migration (automated)
# Triggered every N minutes or when active memory threshold reached
shortterm_memory = await memory_migration_manager.migrate_active_to_shortterm(
    active_memory_id=active_memory.id
)

# 3. Background promotion (automated)
# Triggered daily or weekly for high-importance memories
longterm_memory = await memory_migration_manager.promote_shortterm_to_longterm(
    shortterm_memory_id=shortterm_memory.id
)
```

## Implementation Details

### Services

```python
# Active Memory Service
from database.memories.active_memory_service import ActiveMemoryService
active_service = ActiveMemoryService()

# Shortterm Memory Service
from database.memories.shortterm_memory_service import ShorttermMemoryService
shortterm_service = ShorttermMemoryService()

# Longterm Memory Service
from database.memories.longterm_memory_service import LongtermMemoryService
longterm_service = LongtermMemoryService()

# Unified Memory Manager
from memory.memory_manager import memory_manager
```

### Configuration

```python
# config/constants.py

# Chunking parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Similarity thresholds
CHUNK_SIMILARITY_THRESHOLD = 0.85
ENTITY_SIMILARITY_THRESHOLD = 0.90

# Importance thresholds
LONGTERM_PROMOTION_THRESHOLD = 0.7
HIGH_IMPORTANCE_THRESHOLD = 0.8

# Temporal settings
SHORTTERM_RETENTION_DAYS = 30
LONGTERM_RETENTION_YEARS = 10
```

## Summary

- **Three-tier hierarchy**: Active → Shortterm → Longterm
- **Dual storage**: PostgreSQL (chunks, vectors) + Neo4j (entities, relationships)
- **Active memory**: Summary-level, frequently updated working memory
- **Shortterm memory**: Detailed, searchable chunks with graph entities
- **Longterm memory**: Consolidated, important knowledge with temporal validity
- **Migration**: Memorizer Agent structures active → shortterm
- **Promotion**: Importance-based shortterm → longterm
- **Conflict resolution**: Auto-resolve duplicates, manual agent for complex cases

For implementation details, see:
- `memory/memory_manager.py`
- `memory/managers/memory_migration_manager.py`
- `memory/managers/memory_auto_resolve_manager.py`
- `database/memories/*_service.py`

