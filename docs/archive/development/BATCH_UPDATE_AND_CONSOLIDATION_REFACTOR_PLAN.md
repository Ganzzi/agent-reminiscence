# Batch Update and Consolidation Refactor Plan

**Date:** October 4, 2025  
**Status:** Planning  
**Purpose:** Implement batch section updates, improved consolidation with AI-assisted conflict resolution, and enhanced promotion workflow

---

## Overview

This plan outlines modifications to support:
1. **Batch section updates** - Update multiple sections in one call
2. **Smart consolidation threshold** - Based on total memory sections and average update count
3. **AI-assisted conflict resolution** - Use Memorizer agent for merging conflicts
4. **Section-based chunking** - Track which chunks belong to which sections
5. **Update count tracking** - Track shortterm and longterm memory updates
6. **Enhanced promotion** - Better entity/relationship handling with metadata updates

---

## Files to Modify

### Core Files
1. ✅ `agent_mem/core.py` - Main AgentMem interface
2. ✅ `agent_mem/services/memory_manager.py` - Memory orchestration
3. ✅ `agent_mem/database/repositories/active_memory.py` - Active memory operations
4. ✅ `agent_mem/database/repositories/shortterm_memory.py` - Shortterm memory operations
5. ✅ `agent_mem/database/repositories/longterm_memory.py` - Longterm memory operations

### Database Schema
6. ✅ `agent_mem/sql/schema.sql` - Database schema changes

### Configuration
7. ✅ `agent_mem/config/settings.py` - Add new config parameters

### Agent Files
8. ✅ `agent_mem/agents/memorizer.py` - Conflict resolution agent

### Models
9. ✅ `agent_mem/database/models.py` - Add new models for conflicts

### MCP Server
10. ✅ `agent_mem_mcp/server.py` - Update MCP tools
11. ✅ `agent_mem_mcp/schemas.py` - Update schemas

### Tests
12. ✅ `tests/test_core.py` - Update core tests
13. ✅ `tests/test_memory_manager.py` - Update manager tests
14. ✅ `tests/test_active_memory_repo.py` - Update repository tests
15. ✅ `agent_mem_mcp/tests/test_batch_update.py` - Batch update tests

### Documentation
16. ✅ `README.md` - Update API documentation
17. ✅ `examples/basic_usage.py` - Update examples
18. ✅ `examples/simple_test.py` - Update examples

---

## Implementation Checklist

### Phase 1: Database Schema Changes

#### 1.1 Add Columns to Tables
- [ ] **shortterm_memory_chunk table**
  - [ ] Add `section_id TEXT` column (nullable, references active memory section)
  - [ ] Add index on `section_id` for faster lookups
  
- [ ] **shortterm_memory table**
  - [ ] Add `update_count INTEGER DEFAULT 0` column
  
- [ ] **longterm_memory_chunk table**
  - [ ] Add `last_updated TIMESTAMP` column (nullable, tracks last update time)
  - [ ] Add index on `last_updated` for temporal queries

**Files to modify:**
- `agent_mem/sql/schema.sql`

**SQL Changes:**
```sql
-- In shortterm_memory_chunk table
ALTER TABLE shortterm_memory_chunk 
ADD COLUMN section_id TEXT;

CREATE INDEX idx_shortterm_chunk_section 
ON shortterm_memory_chunk(shortterm_memory_id, section_id);

-- In shortterm_memory table
ALTER TABLE shortterm_memory 
ADD COLUMN update_count INTEGER DEFAULT 0;

-- In longterm_memory_chunk table
ALTER TABLE longterm_memory_chunk 
ADD COLUMN last_updated TIMESTAMP;

CREATE INDEX idx_longterm_chunk_updated 
ON longterm_memory_chunk(last_updated);
```

---

### Phase 2: Configuration Updates

#### 2.1 Add New Config Parameters
- [ ] Add `avg_section_update_count_for_consolidation` (float, default 5.0)
- [ ] Add `shortterm_update_count_threshold` (int, default 10)
- [ ] Rename/keep `consolidation_threshold` for backward compatibility

**Files to modify:**
- `agent_mem/config/settings.py`

**Changes:**
```python
# Memory Configuration
avg_section_update_count_for_consolidation: float = Field(
    default_factory=lambda: float(os.getenv("AVG_SECTION_UPDATE_COUNT", "5.0")),
    description="Average update count per section before consolidation trigger",
)

shortterm_update_count_threshold: int = Field(
    default_factory=lambda: int(os.getenv("SHORTTERM_UPDATE_THRESHOLD", "10")),
    description="Number of shortterm memory updates before longterm promotion",
)

# Keep for backward compatibility
consolidation_threshold: int = Field(
    default_factory=lambda: int(os.getenv("ACTIVE_MEMORY_UPDATE_THRESHOLD", "5")),
    description="[DEPRECATED] Use avg_section_update_count_for_consolidation",
)
```

---

### Phase 3: Data Models

#### 3.1 Create Conflict Resolution Models
- [ ] Create `ConflictSection` class - Holds section + referencing chunks
- [ ] Create `ConflictEntity` class - Holds conflicting entities
- [ ] Create `ConflictRelationship` class - Holds conflicting relationships
- [ ] Create `ConsolidationConflicts` class - Container for all conflicts

**Files to modify:**
- `agent_mem/database/models.py`

**New Classes:**
```python
class ConflictSection(BaseModel):
    """Section with conflicting chunks."""
    section_id: str
    section_content: str
    update_count: int
    existing_chunks: List[ShorttermMemoryChunk] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ConflictEntity(BaseModel):
    """Entity conflict between active and shortterm."""
    active_entity: ExtractedEntity
    shortterm_entities: List[ShorttermEntity]
    similarity_scores: Dict[int, float] = Field(default_factory=dict)  # entity_id -> score
    recommended_action: str  # "merge", "create", "manual"

class ConflictRelationship(BaseModel):
    """Relationship conflict between active and shortterm."""
    active_relationship: ExtractedRelationship
    shortterm_relationships: List[ShorttermRelationship]
    recommended_action: str

class ConsolidationConflicts(BaseModel):
    """All conflicts for consolidation."""
    external_id: str
    active_memory_id: int
    shortterm_memory_id: int
    sections: List[ConflictSection] = Field(default_factory=list)
    entities: List[ConflictEntity] = Field(default_factory=list)
    relationships: List[ConflictRelationship] = Field(default_factory=list)
    created_at: datetime
```

---

### Phase 4: Core Interface Changes

#### 4.1 Replace Single Section Update with Batch Update
- [ ] Rename `update_active_memory_section` to `update_active_memory_section` (keep for backward compatibility)
- [ ] Create new `update_active_memory_sections` method for batch updates
- [ ] Update method to accept `List[Dict]` with section updates
- [ ] Return updated memory after all sections updated

**Files to modify:**
- `agent_mem/core.py`

**Changes:**
```python
async def update_active_memory_sections(
    self,
    external_id: str | UUID | int,
    memory_id: int,
    sections: List[Dict[str, str]],  # [{"section_id": "x", "new_content": "..."}]
) -> ActiveMemory:
    """
    Update multiple sections in an active memory (batch update).
    
    After updating all sections, checks if total update count across all sections
    exceeds threshold for consolidation.
    
    Args:
        external_id: Agent identifier
        memory_id: Memory ID
        sections: List of section updates with section_id and new_content
    
    Returns:
        Updated ActiveMemory object
    """
    # Implementation in memory_manager
    
# Keep backward compatibility
async def update_active_memory_section(
    self,
    external_id: str | UUID | int,
    memory_id: int,
    section_id: str,
    new_content: str,
) -> ActiveMemory:
    """Update a single section (calls batch method with single section)."""
    return await self.update_active_memory_sections(
        external_id=external_id,
        memory_id=memory_id,
        sections=[{"section_id": section_id, "new_content": new_content}]
    )
```

---

### Phase 5: Repository Changes

#### 5.1 Active Memory Repository - Batch Update
- [ ] Create `update_sections` method (plural) for batch updates
- [ ] Keep `update_section` (singular) for backward compatibility
- [ ] Update query to handle multiple sections in one transaction

**Files to modify:**
- `agent_mem/database/repositories/active_memory.py`

**New Method:**
```python
async def update_sections(
    self,
    memory_id: int,
    section_updates: List[Dict[str, str]],  # [{"section_id": "x", "new_content": "..."}]
) -> Optional[ActiveMemory]:
    """
    Update multiple sections in an active memory (batch update).
    
    All updates are done in a single transaction.
    Increments update_count for each section.
    """
    # Implementation with jsonb_set for multiple sections
```

#### 5.2 Shortterm Memory Repository - Section Reference
- [ ] Update `create_chunk` to accept `section_id` parameter
- [ ] Add `get_chunks_by_section_id` method
- [ ] Update `update_memory` to increment `update_count`

**Files to modify:**
- `agent_mem/database/repositories/shortterm_memory.py`

**Changes:**
```python
async def create_chunk(
    self,
    shortterm_memory_id: int,
    external_id: str,
    content: str,
    chunk_order: int,
    embedding: Optional[List[float]] = None,
    section_id: Optional[str] = None,  # NEW
    metadata: Optional[Dict[str, Any]] = None,
) -> ShorttermMemoryChunk:
    # Add section_id to INSERT

async def get_chunks_by_section_id(
    self,
    shortterm_memory_id: int,
    section_id: str
) -> List[ShorttermMemoryChunk]:
    """Get all chunks that reference a specific section."""
    # Implementation

async def increment_update_count(
    self,
    memory_id: int
) -> Optional[ShorttermMemory]:
    """Increment update count for shortterm memory."""
    # Implementation
```

#### 5.3 Longterm Memory Repository - Update Tracking
- [ ] Update `create_chunk` to set `last_updated` to current time
- [ ] Add `update_chunk_timestamps` method for batch updates
- [ ] Add queries for entity/relationship metadata updates

**Files to modify:**
- `agent_mem/database/repositories/longterm_memory.py`

**Changes:**
```python
async def update_chunk_timestamps(
    self,
    shortterm_memory_id: int
) -> int:
    """
    Update last_updated for all chunks from a shortterm memory.
    
    Returns number of chunks updated.
    """
    # Implementation

async def update_entity_with_metadata(
    self,
    entity_id: int,
    confidence: Optional[float] = None,
    importance: Optional[float] = None,
    metadata_update: Optional[Dict[str, Any]] = None,
) -> Optional[LongtermEntity]:
    """Update entity and append to metadata.updates array."""
    # Implementation
```

---

### Phase 6: Memory Manager - Batch Update Logic

#### 6.1 Update Method with New Threshold Logic
- [ ] Implement `update_active_memory_sections` for batch updates
- [ ] Calculate total update count threshold: `avg_section_update_count * total_sections`
- [ ] Check if sum of all section update counts >= threshold
- [ ] Trigger consolidation in background thread if threshold exceeded
- [ ] Add lock mechanism to prevent concurrent consolidation for same memory

**Files to modify:**
- `agent_mem/services/memory_manager.py`

**New/Modified Methods:**
```python
import asyncio
from threading import Lock

class MemoryManager:
    def __init__(self, config: Config):
        # ... existing code ...
        self._consolidation_locks: Dict[int, Lock] = {}  # memory_id -> lock
    
    async def update_active_memory_sections(
        self,
        external_id: str,
        memory_id: int,
        sections: List[Dict[str, str]],
    ) -> ActiveMemory:
        """
        Update multiple sections in batch.
        
        Workflow:
        1. Update all sections in repository (transactional)
        2. Get updated memory
        3. Calculate total update count threshold
        4. If threshold exceeded, trigger consolidation in background
        5. Return updated memory immediately (don't wait for consolidation)
        """
        # 1. Update all sections
        memory = await self.active_repo.update_sections(
            memory_id=memory_id,
            section_updates=sections,
        )
        
        if not memory:
            raise ValueError(f"Active memory {memory_id} not found")
        
        # 2. Calculate threshold
        num_sections = len(memory.sections)
        threshold = self.config.avg_section_update_count_for_consolidation * num_sections
        
        # 3. Calculate total update count
        total_update_count = sum(
            section["update_count"] 
            for section in memory.sections.values()
        )
        
        # 4. Check threshold and trigger consolidation
        if total_update_count >= threshold:
            logger.info(
                f"Total update count ({total_update_count}) >= threshold ({threshold}). "
                f"Triggering consolidation in background..."
            )
            
            # Start consolidation in background (non-blocking)
            asyncio.create_task(
                self._consolidate_with_lock(external_id, memory.id)
            )
        
        return memory
    
    async def _consolidate_with_lock(
        self,
        external_id: str,
        memory_id: int
    ) -> None:
        """
        Consolidate with lock to prevent concurrent consolidation.
        """
        # Get or create lock for this memory
        if memory_id not in self._consolidation_locks:
            self._consolidation_locks[memory_id] = Lock()
        
        lock = self._consolidation_locks[memory_id]
        
        # Try to acquire lock (non-blocking)
        if not lock.acquire(blocking=False):
            logger.info(f"Consolidation already in progress for memory {memory_id}, skipping")
            return
        
        try:
            await self._consolidate_to_shortterm(external_id, memory_id)
        finally:
            lock.release()
            # Clean up lock if no longer needed
            if memory_id in self._consolidation_locks:
                del self._consolidation_locks[memory_id]
```

---

### Phase 7: Enhanced Consolidation Workflow

#### 7.1 Refactor `_consolidate_to_shortterm`
- [ ] Extract sections with update_count > 0
- [ ] For each section, check if chunks with section_id reference exist
- [ ] If no chunks: Create new chunks with section_id reference
- [ ] If chunks exist: Add to conflicts for manual merge
- [ ] Extract entities from all updated sections
- [ ] Compare entities with existing shortterm entities
- [ ] Add conflicting entities to conflicts class
- [ ] Same for relationships
- [ ] Pass ConsolidationConflicts to Memorizer agent
- [ ] Execute agent's resolution plan
- [ ] Increment shortterm memory update_count
- [ ] Check if update_count >= threshold for promotion
- [ ] If yes, call `_promote_to_longterm` then delete shortterm chunks
- [ ] Reset all section update counts to 0

**Files to modify:**
- `agent_mem/services/memory_manager.py`

**Refactored Method:**
```python
async def _consolidate_to_shortterm(
    self, 
    external_id: str, 
    active_memory_id: int
) -> Optional[ShorttermMemory]:
    """
    Enhanced consolidation with conflict resolution.
    
    Workflow:
    1. Get active memory and updated sections (update_count > 0)
    2. Find or create shortterm memory
    3. Build ConsolidationConflicts:
       a. For each section:
          - Get existing chunks referencing this section
          - If no chunks: Ready to create new
          - If chunks exist: Add to conflicts
       b. Extract entities from all updated sections
       c. Compare with existing shortterm entities
       d. Add conflicts
    4. Pass conflicts to Memorizer agent for resolution
    5. Execute agent's plan (create/update chunks, entities, relationships)
    6. Increment shortterm memory update_count
    7. Check promotion threshold
    8. If threshold met: promote to longterm, delete shortterm chunks
    9. Reset section update counts to 0
    """
    # Implementation details below
```

#### 7.2 Build Conflicts
- [ ] Create helper method `_build_consolidation_conflicts`
- [ ] Extract sections with update_count > 0
- [ ] Query chunks by section_id for each section
- [ ] Extract entities using ER agent
- [ ] Compare with existing shortterm entities
- [ ] Build ConflictSection, ConflictEntity, ConflictRelationship objects

**New Helper Method:**
```python
async def _build_consolidation_conflicts(
    self,
    external_id: str,
    active_memory: ActiveMemory,
    shortterm_memory: ShorttermMemory,
) -> ConsolidationConflicts:
    """
    Build conflicts object for memorizer agent.
    
    Returns ConsolidationConflicts with:
    - Sections needing consolidation
    - Entity conflicts
    - Relationship conflicts
    """
    conflicts = ConsolidationConflicts(
        external_id=external_id,
        active_memory_id=active_memory.id,
        shortterm_memory_id=shortterm_memory.id,
        created_at=datetime.now(timezone.utc),
    )
    
    # 1. Process sections
    for section_id, section_data in active_memory.sections.items():
        if section_data.get("update_count", 0) > 0:
            # Get existing chunks referencing this section
            existing_chunks = await self.shortterm_repo.get_chunks_by_section_id(
                shortterm_memory_id=shortterm_memory.id,
                section_id=section_id
            )
            
            conflict_section = ConflictSection(
                section_id=section_id,
                section_content=section_data.get("content", ""),
                update_count=section_data.get("update_count", 0),
                existing_chunks=existing_chunks,
            )
            conflicts.sections.append(conflict_section)
    
    # 2. Extract entities from all updated sections
    all_section_content = "\n\n".join([
        s.section_content for s in conflicts.sections
    ])
    
    extraction_result = await extract_entities_and_relationships(all_section_content)
    
    # 3. Compare entities
    existing_entities = await self.shortterm_repo.get_entities_by_memory_id(
        shortterm_memory.id
    )
    
    for extracted_entity in extraction_result.entities:
        # Find matching shortterm entities
        matches = []
        for existing in existing_entities:
            if existing.name.lower() == extracted_entity.name.lower():
                similarity = await self._calculate_semantic_similarity(
                    extracted_entity.name, existing.name
                )
                matches.append((existing, similarity))
        
        if matches:
            conflict_entity = ConflictEntity(
                active_entity=extracted_entity,
                shortterm_entities=[m[0] for m in matches],
                similarity_scores={m[0].id: m[1] for m in matches},
                recommended_action="merge" if matches[0][1] >= 0.85 else "manual"
            )
            conflicts.entities.append(conflict_entity)
    
    # 4. Same for relationships
    # ... similar logic ...
    
    return conflicts
```

#### 7.3 Integrate Memorizer Agent
- [ ] Pass conflicts to Memorizer agent
- [ ] Get consolidation plan from agent
- [ ] Execute plan: create/update chunks, entities, relationships
- [ ] Handle errors gracefully

**Integration Code:**
```python
# In _consolidate_to_shortterm method

# Build conflicts
conflicts = await self._build_consolidation_conflicts(
    external_id=external_id,
    active_memory=active_memory,
    shortterm_memory=shortterm_memory,
)

# Get resolution plan from Memorizer agent
from agent_mem.agents.memorizer import MemorizerAgent

memorizer = MemorizerAgent(
    config=self.config,
    memory_manager=self,
)

consolidation_plan = await memorizer.resolve_conflicts(
    conflicts=conflicts
)

# Execute plan
await self._execute_consolidation_plan(
    external_id=external_id,
    shortterm_memory=shortterm_memory,
    plan=consolidation_plan,
)
```

#### 7.4 Update Count and Promotion Check
- [ ] After successful consolidation, increment shortterm update_count
- [ ] Check if update_count >= shortterm_update_count_threshold
- [ ] If yes, call _promote_to_longterm
- [ ] After promotion, delete all chunks from shortterm memory
- [ ] Reset all active memory section update counts to 0

**Code:**
```python
# Increment shortterm memory update count
updated_shortterm = await self.shortterm_repo.increment_update_count(
    shortterm_memory.id
)

# Check promotion threshold
if updated_shortterm.update_count >= self.config.shortterm_update_count_threshold:
    logger.info(
        f"Shortterm memory {shortterm_memory.id} reached promotion threshold "
        f"({updated_shortterm.update_count} >= {self.config.shortterm_update_count_threshold}). "
        f"Promoting to longterm..."
    )
    
    await self._promote_to_longterm(external_id, shortterm_memory.id)
    
    # Delete shortterm chunks after promotion
    await self.shortterm_repo.delete_all_chunks(shortterm_memory.id)
    
    # Reset shortterm update count
    await self.shortterm_repo.reset_update_count(shortterm_memory.id)

# Reset active memory section update counts
await self.active_repo.reset_all_section_counts(active_memory_id)
```

---

### Phase 8: Enhanced Promotion Workflow

#### 8.1 Refactor `_promote_to_longterm`
- [ ] Update last_updated for all chunks from shortterm_memory_id where last_updated is null
- [ ] Copy all chunks to longterm (not just filtered by importance)
- [ ] For entities: compare with longterm entities by name+type
- [ ] If no match: create new entity in longterm
- [ ] If match with different confidence/importance: update longterm entity
- [ ] Add to metadata.updates array: `{date, old_confidence, new_confidence, old_importance, new_importance}`
- [ ] Same for relationships: compare by from_entity+to_entity+type
- [ ] Update if description/confidence/strength differ
- [ ] Add to metadata.updates array

**Files to modify:**
- `agent_mem/services/memory_manager.py`

**Refactored Method:**
```python
async def _promote_to_longterm(
    self, 
    external_id: str, 
    shortterm_memory_id: int
) -> List[LongtermMemoryChunk]:
    """
    Enhanced promotion with metadata tracking.
    
    Workflow:
    1. Update last_updated for all chunks from this shortterm memory
    2. Copy all chunks to longterm (no filtering)
    3. Process entities:
       - If no match: create new
       - If match with different values: update + add to metadata.updates
    4. Process relationships:
       - If no match: create new
       - If match with different values: update + add to metadata.updates
    """
    # 1. Update timestamps for chunks
    await self.longterm_repo.update_chunk_timestamps(shortterm_memory_id)
    
    # 2. Get and copy all chunks
    shortterm_chunks = await self.shortterm_repo.get_chunks_by_memory_id(
        shortterm_memory_id
    )
    
    longterm_chunks = []
    for chunk in shortterm_chunks:
        embedding = await self.embedding_service.get_embedding(chunk.content)
        
        longterm_chunk = await self.longterm_repo.create_chunk(
            external_id=external_id,
            shortterm_memory_id=shortterm_memory_id,
            content=chunk.content,
            chunk_order=chunk.chunk_order,
            embedding=embedding,
            confidence_score=0.85,
            importance_score=chunk.metadata.get("importance_score", 0.75),
            start_date=datetime.now(timezone.utc),
            end_date=None,
            metadata={
                **chunk.metadata,
                "promoted_from_shortterm": shortterm_memory_id,
                "promoted_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        longterm_chunks.append(longterm_chunk)
    
    # 3. Process entities with update tracking
    shortterm_entities = await self.shortterm_repo.get_entities_by_memory_id(
        shortterm_memory_id
    )
    longterm_entities = await self.longterm_repo.get_entities_by_external_id(
        external_id
    )
    
    for st_entity in shortterm_entities:
        # Find match by name + type
        lt_match = None
        for lt_entity in longterm_entities:
            if (lt_entity.name.lower() == st_entity.name.lower() and 
                lt_entity.type == st_entity.type):
                lt_match = lt_entity
                break
        
        if lt_match:
            # Check if update needed
            confidence_changed = abs(lt_match.confidence - st_entity.confidence) > 0.05
            importance_changed = abs(
                lt_match.importance - self._calculate_importance(st_entity)
            ) > 0.05
            
            if confidence_changed or importance_changed:
                # Prepare update metadata
                update_entry = {
                    "date": datetime.now(timezone.utc).isoformat(),
                }
                if confidence_changed:
                    update_entry["old_confidence"] = lt_match.confidence
                    update_entry["new_confidence"] = st_entity.confidence
                if importance_changed:
                    update_entry["old_importance"] = lt_match.importance
                    update_entry["new_importance"] = self._calculate_importance(st_entity)
                
                # Get existing updates array
                updates_array = lt_match.metadata.get("updates", [])
                updates_array.append(update_entry)
                
                # Update entity
                await self.longterm_repo.update_entity_with_metadata(
                    entity_id=lt_match.id,
                    confidence=st_entity.confidence if confidence_changed else None,
                    importance=self._calculate_importance(st_entity) if importance_changed else None,
                    metadata_update={"updates": updates_array},
                )
        else:
            # Create new entity
            await self.longterm_repo.create_entity(
                external_id=external_id,
                name=st_entity.name,
                entity_type=st_entity.type,
                description=st_entity.description or "",
                confidence=st_entity.confidence,
                importance=self._calculate_importance(st_entity),
                first_seen=datetime.now(timezone.utc),
                last_seen=datetime.now(timezone.utc),
                metadata={
                    **st_entity.metadata,
                    "promoted_from_shortterm": shortterm_memory_id,
                    "promoted_at": datetime.now(timezone.utc).isoformat(),
                    "updates": [],  # Initialize updates array
                },
            )
    
    # 4. Process relationships (similar logic)
    # ... implementation ...
    
    return longterm_chunks
```

---

### Phase 9: Memorizer Agent Updates

#### 9.1 Add Conflict Resolution Method
- [ ] Add `resolve_conflicts` method that takes ConsolidationConflicts
- [ ] Return ConsolidationPlan with chunk, entity, relationship operations
- [ ] Update system prompt to handle conflicts

**Files to modify:**
- `agent_mem/agents/memorizer.py`

**New Method:**
```python
async def resolve_conflicts(
    self,
    conflicts: ConsolidationConflicts,
) -> ConsolidationPlan:
    """
    Resolve consolidation conflicts using AI.
    
    Args:
        conflicts: All conflicts between active and shortterm
    
    Returns:
        ConsolidationPlan with operations to execute
    """
    # Build context for agent
    context = self._build_conflict_context(conflicts)
    
    # Call agent
    result = await self.agent.run(
        user_prompt=f"Resolve these consolidation conflicts:\n\n{context}",
        deps=MemorizerDeps(
            external_id=conflicts.external_id,
            active_memory_id=conflicts.active_memory_id,
            shortterm_memory_id=conflicts.shortterm_memory_id,
            memory_manager=self.memory_manager,
        ),
    )
    
    return result.data
```

---

### Phase 10: MCP Server Updates

#### 10.1 Update update_memory_section Tool
- [ ] Keep existing tool for backward compatibility
- [ ] Update to call new batch method internally

#### 10.2 Add update_memory_sections Tool (Optional)
- [ ] Create new tool for batch updates
- [ ] Update schema

**Files to modify:**
- `agent_mem_mcp/server.py`
- `agent_mem_mcp/schemas.py`

**Schema:**
```python
UPDATE_MEMORY_SECTIONS_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "external_id": {
            "type": "string",
            "description": "Agent identifier"
        },
        "memory_id": {
            "type": "integer",
            "description": "Memory ID"
        },
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "section_id": {"type": "string"},
                    "new_content": {"type": "string"}
                },
                "required": ["section_id", "new_content"]
            },
            "description": "List of sections to update"
        }
    },
    "required": ["external_id", "memory_id", "sections"]
}
```

---

### Phase 11: Testing

#### 11.1 Unit Tests
- [ ] Test batch update in active_memory repo
- [ ] Test section_id in shortterm chunks
- [ ] Test update_count tracking
- [ ] Test consolidation threshold calculation
- [ ] Test conflict resolution
- [ ] Test promotion with metadata updates

**Files to create/modify:**
- `tests/test_active_memory_repo.py` - batch update tests
- `tests/test_shortterm_memory_repo.py` - section_id tests
- `tests/test_memory_manager.py` - consolidation/promotion tests
- `tests/test_memorizer_agent.py` - conflict resolution tests

#### 11.2 Integration Tests
- [ ] Test end-to-end batch update → consolidation → promotion
- [ ] Test concurrent consolidation handling
- [ ] Test backward compatibility with single section updates

**Files to modify:**
- `tests/test_integration.py`

#### 11.3 MCP Tests
- [ ] Update existing batch update test
- [ ] Test new MCP tool (if added)

**Files to modify:**
- `agent_mem_mcp/tests/test_batch_update.py`

---

### Phase 12: Documentation

#### 12.1 Update API Documentation
- [ ] Document new `update_active_memory_sections` method
- [ ] Document backward compatibility
- [ ] Update configuration docs
- [ ] Add examples

**Files to modify:**
- `README.md`
- `docs/GETTING_STARTED.md`

#### 12.2 Update Examples
- [ ] Update basic_usage.py with batch update example
- [ ] Update simple_test.py with batch update example

**Files to modify:**
- `examples/basic_usage.py`
- `examples/simple_test.py`

---

## Implementation Order

### Week 1: Foundation
1. ✅ Phase 1: Database schema changes
2. ✅ Phase 2: Configuration updates
3. ✅ Phase 3: Data models

### Week 2: Core Logic
4. ✅ Phase 4: Core interface changes
5. ✅ Phase 5: Repository changes
6. ✅ Phase 6: Memory manager batch update

### Week 3: Enhanced Workflows
7. ✅ Phase 7: Enhanced consolidation
8. ✅ Phase 8: Enhanced promotion
9. ✅ Phase 9: Memorizer agent updates

### Week 4: Integration & Testing
10. ✅ Phase 10: MCP server updates
11. ✅ Phase 11: Testing
12. ✅ Phase 12: Documentation

---

## Rollback Plan

If issues arise:
1. Database schema changes are additive (new columns nullable)
2. Code changes maintain backward compatibility
3. Can disable new features via config flags
4. Can revert to old consolidation logic by feature flag

---

## Success Criteria

- [ ] Batch updates work correctly with multiple sections
- [ ] Consolidation threshold calculated based on total sections
- [ ] Consolidation runs in background without blocking updates
- [ ] Concurrent consolidation attempts properly handled
- [ ] Conflicts detected and resolved by AI agent
- [ ] Chunks track which section they came from
- [ ] Shortterm memory update count tracked and triggers promotion
- [ ] Longterm entities/relationships track update history
- [ ] All tests pass
- [ ] Backward compatibility maintained
- [ ] Documentation updated

---

## Notes

- **Thread Safety**: Use asyncio.Lock instead of threading.Lock for async code
- **Transaction Safety**: Batch updates should be atomic
- **Error Handling**: Failed consolidation shouldn't fail the update
- **Performance**: Background consolidation prevents blocking
- **AI Agent Costs**: Memorizer agent called on every consolidation (monitor costs)
- **Metadata Growth**: Entity/relationship update arrays could grow large (consider pruning strategy)

---

## Questions to Resolve

1. **Should we delete shortterm chunks after promotion?**
   - Current plan: YES, delete after promotion to longterm
   
2. **How to handle very large metadata.updates arrays?**
   - Consider keeping only last N updates
   - Or archive old updates to separate table

3. **Should batch update be transactional?**
   - YES - all sections updated or none

4. **What if memorizer agent fails?**
   - Fallback to automatic resolution (current logic)
   - Log for manual review

5. **How to test background consolidation?**
   - Use asyncio.wait_for with timeout in tests
   - Mock asyncio.create_task to run synchronously

---

**END OF PLAN**
