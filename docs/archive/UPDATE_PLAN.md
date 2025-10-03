# AgentMem Architecture Update Plan

## Critical Changes Required

### 1. **AgentMem API - Make Stateless** ✅ HIGH PRIORITY

**Current Problem:**
- `external_id` is set in `__init__()`
- One AgentMem instance per worker/agent
- Cannot serve multiple workers

**Required Change:**
- Remove `external_id` from `__init__()`
- Add `external_id` parameter to all 4 methods:
  - `create_active_memory(external_id, ...)`
  - `get_active_memories(external_id)`
  - `update_active_memory(external_id, memory_id, section_id, ...)`
  - `retrieve_memories(external_id, query, ...)`

**Files to Update:**
- `agent_mem/core.py` - AgentMem class
- `agent_mem/services/memory_manager.py` - MemoryManager (remove external_id from init)
- `examples/basic_usage.py` - Update example
- All documentation

### 2. **Active Memory Structure** ✅ HIGH PRIORITY

**Current Problem:**
- Simple `content` field
- No template support
- No section-based structure

**Required Structure (from memory-architecture.md):**
```python
class ActiveMemory:
    id: int
    external_id: str  # worker_id equivalent
    title: str
    template_content: str  # YAML template as text
    sections: Dict[str, Dict[str, Any]]  # {section_id: {content: str, update_count: int}}
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
```

**SQL Schema:**
```sql
CREATE TABLE active_memory (
    id SERIAL PRIMARY KEY,
    external_id VARCHAR(255) NOT NULL,
    title VARCHAR(500) NOT NULL,
    template_content TEXT NOT NULL,  -- YAML template
    sections JSONB DEFAULT '{}',  -- {section_id: {content, update_count}}
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

**Files to Update:**
- ✅ `agent_mem/database/models.py` - ActiveMemory model
- ✅ `agent_mem/sql/schema.sql` - active_memory table
- `agent_mem/database/repositories/active_memory.py` - CRUD methods
- `agent_mem/services/memory_manager.py` - Usage

### 3. **Shortterm Memory Structure** 🚧 MEDIUM PRIORITY

**Reference from memory-architecture.md:**

**PostgreSQL:**
```sql
CREATE TABLE shortterm_memory (
    id SERIAL PRIMARY KEY,
    worker_id INTEGER NOT NULL,  -- Change to external_id VARCHAR
    title VARCHAR(255) NOT NULL,
    summary TEXT,
    metadata JSONB DEFAULT '{}',
    start_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE shortterm_memory_chunk (
    id SERIAL PRIMARY KEY,
    shortterm_id INTEGER NOT NULL REFERENCES shortterm_memory(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    content_vector vector(768),
    content_bm25 bm25vector,
    metadata JSONB DEFAULT '{}'
);
```

**Neo4j:**
```cypher
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

CREATE (e1)-[r:RELATES_TO {
    id: INTEGER,
    external_id: STRING,
    shortterm_memory_id: INTEGER,
    type: STRING,
    description: STRING,
    confidence: FLOAT,
    strength: FLOAT,
    first_observed: DATETIME,
    last_observed: DATETIME,
    metadata: MAP
}]->(e2)
```

**Models:**
```python
class ShorttermMemory:
    id: int
    external_id: str
    title: str
    summary: Optional[str]
    metadata: Dict[str, Any]
    start_date: datetime
    last_updated: datetime

class ShorttermMemoryChunk:
    id: int
    shortterm_memory_id: int
    external_id: str
    chunk_order: int
    content: str
    embedding: Optional[List[float]]  # Vector
    metadata: Dict[str, Any]
    
class ShorttermEntity:
    id: int
    external_id: str
    shortterm_memory_id: int
    name: str
    type: str
    description: str
    confidence: float
    first_seen: datetime
    last_seen: datetime
    metadata: Dict[str, Any]

class ShorttermRelationship:
    id: int
    external_id: str
    shortterm_memory_id: int
    from_entity_id: int
    to_entity_id: int
    type: str
    description: str
    confidence: float
    strength: float
    first_observed: datetime
    last_observed: datetime
    metadata: Dict[str, Any]
```

**Files to Update:**
- `agent_mem/database/models.py` - Add entity/relationship models
- `agent_mem/sql/schema.sql` - Adjust shortterm tables (already mostly correct)
- Create `agent_mem/database/repositories/shortterm_memory.py`
- Update Neo4j schema documentation

### 4. **Longterm Memory Structure** 🚧 MEDIUM PRIORITY

**Reference from memory-architecture.md:**

**PostgreSQL:**
```sql
CREATE TABLE longterm_memory_chunk (
    id SERIAL PRIMARY KEY,
    worker_id INTEGER NOT NULL,  -- Change to external_id VARCHAR
    content TEXT NOT NULL,
    content_vector vector(768),
    content_bm25 bm25vector,
    confidence_score REAL DEFAULT 0.5,
    metadata JSONB DEFAULT '{}',
    start_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    end_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

**Neo4j:**
```cypher
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

CREATE (e1)-[r:RELATES_TO {
    id: INTEGER,
    external_id: STRING,
    type: STRING,
    description: STRING,
    confidence: FLOAT,
    strength: FLOAT,
    start_date: DATETIME,
    last_updated: DATETIME,
    metadata: MAP
}]->(e2)
```

**Files to Update:**
- `agent_mem/database/models.py` - Add longterm entity/relationship models
- `agent_mem/sql/schema.sql` - Adjust longterm tables
- Create `agent_mem/database/repositories/longterm_memory.py`

### 5. **Update Active Memory Operations** ✅ HIGH PRIORITY

**New API:**
```python
# Create with template and initial sections
await agent_mem.create_active_memory(
    external_id="agent-123",
    title="Task Memory",
    template_content="""
template:
  id: "task_memory_v1"
  name: "Task Memory"
  version: "1.0.0"
sections:
  - id: "current_task"
    title: "Current Task"
  - id: "progress"
    title: "Progress"
""",
    initial_sections={
        "current_task": {"content": "# Task\nImplement feature", "update_count": 0},
        "progress": {"content": "# Progress\n- Started", "update_count": 0}
    }
)

# Update a specific section
await agent_mem.update_active_memory_section(
    external_id="agent-123",
    memory_id=1,
    section_id="progress",
    new_content="# Progress\n- Completed step 1\n- Working on step 2"
)
# This increments section's update_count

# Get all sections
memory = await agent_mem.get_active_memory(external_id="agent-123", memory_id=1)
# Returns ActiveMemory with full sections dict

# Check if section needs consolidation
if memory.sections["progress"]["update_count"] >= threshold:
    # Trigger consolidation
```

**Files to Update:**
- `agent_mem/core.py` - Update method signatures
- `agent_mem/database/repositories/active_memory.py` - Add section operations
- `agent_mem/services/memory_manager.py` - Update orchestration

## Implementation Order

### Phase 1: Core API Changes (Day 1)
1. ✅ Update models.py - ActiveMemory structure
2. ✅ Update schema.sql - active_memory table
3. Update core.py - Remove external_id from init, add to methods
4. Update memory_manager.py - Make stateless
5. Update active_memory.py repository - Handle sections
6. Update basic_usage.py example
7. Update all documentation

### Phase 2: Complete Models (Day 2)
1. Add all entity/relationship models to models.py
2. Ensure schema.sql matches memory-architecture.md
3. Update documentation with correct structure

### Phase 3: Repositories (Day 3-4)
1. Complete active_memory.py with section operations
2. Implement shortterm_memory.py repository
3. Implement longterm_memory.py repository

### Phase 4: Testing & Documentation (Day 5)
1. Update all examples
2. Create migration tests
3. Update ARCHITECTURE.md
4. Update DEVELOPMENT.md
5. Update README.md

## Key Principles

1. **External ID is generic** - worker_id → external_id everywhere
2. **Stateless AgentMem** - Can serve multiple agents
3. **Template-driven Active Memory** - YAML templates + sections
4. **Section-level tracking** - update_count per section, not per memory
5. **Match main codebase architecture** - memory-architecture.md is source of truth
6. **Dual storage** - PostgreSQL for chunks, Neo4j for entities/relationships
7. **Preserve existing good work** - Keep connection managers, embedding service, etc.

## Status

- ✅ ActiveMemory model updated
- ✅ active_memory table schema updated (partial - needs cleanup)
- ⏳ Core API refactoring - IN PROGRESS
- ⏳ Repository updates - PENDING
- ⏳ Documentation updates - PENDING
