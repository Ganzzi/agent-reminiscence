# Adjustment Plan for Phase 4: Using ER Extractor Agent

## Summary of Required Changes

Based on user feedback, we need to adjust the Phase 4 implementation to:

1. **Replace MemorizerAgent with ER Extractor Agent**
   - Use the existing `er_extractor_agent` from `agents/predefined_agents/er_extractor_agent.py`
   - Remove the custom MemorizerAgent implementation
   - The ER Extractor is a specialized agent focused solely on entity/relationship extraction

2. **Enhance Consolidation (Active → Shortterm)**
   - Extract entities and relationships using ER Extractor Agent
   - Compare extracted entities with existing shortterm entities
   - Implement auto-resolution logic from memory-architecture.md:
     - Merge if semantic similarity >= 0.85 AND entity overlap >= 0.7
     - Create new if below thresholds or conflicts detected
   - Store entities and relationships in Neo4j

3. **Enhance Promotion (Shortterm → Longterm)**
   - Get entities/relationships from shortterm memory
   - Compare with existing longterm entities
   - Update confidence scores for existing entities using the formula:
     ```python
     weight = 0.7
     new_confidence = weight * existing_confidence + (1 - weight) * new_evidence
     ```
   - Add new entities/relationships with temporal tracking (start_date, end_date)
   - Update end_date for superseded entities/relationships

## Implementation Steps

### Step 1: Update memory_manager.py Imports

```python
# Remove
from agent_mem.agents import MemorizerAgent, MemoryRetrieveAgent

# Add
from agent_mem.agents import MemoryRetrieveAgent
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from agents.predefined_agents.er_extractor_agent import er_extractor_agent
```

### Step 2: Remove MemorizerAgent Initialization

Remove from `__init__` and `initialize()`:
```python
# Remove
self.memorizer_agent: Optional[MemorizerAgent] = None
self.memorizer_agent = MemorizerAgent(self.config)
```

### Step 3: Replace _consolidate_to_shortterm Method

The new method should:
1. Get active memory and create/find shortterm memory
2. Extract and chunk content
3. Store chunks with embeddings
4. **Use ER Extractor Agent** to extract entities/relationships
5. Compare with existing shortterm entities
6. Merge or create entities based on similarity
7. Create relationships in Neo4j

Pseudo-code:
```python
async def _consolidate_to_shortterm(self, external_id: str, active_memory_id: int):
    # Get active memory
    active_memory = await self.active_repo.get_by_id(active_memory_id)
    
    # Create/find shortterm memory
    shortterm_memory = await self._find_or_create_shortterm_memory(...)
    
    # Extract and chunk content
    content = self._extract_content_from_sections(active_memory)
    chunks = chunk_text(content, self.config.chunk_size, self.config.chunk_overlap)
    
    # Store chunks with embeddings
    for i, chunk in enumerate(chunks):
        embedding = await self.embedding_service.get_embedding(chunk)
        await self.shortterm_repo.create_chunk(...)
    
    # Extract entities and relationships using ER Extractor Agent
    extraction_result = await er_extractor_agent.run(content)
    
    # Get existing entities for comparison
    existing_entities = await self.shortterm_repo.get_entities_by_memory_id(shortterm_memory.id)
    
    # Process entities with auto-resolution
    entity_map = {}  # name -> id mapping for relationships
    for extracted_entity in extraction_result.data.entities:
        # Check if entity exists
        existing_match = find_matching_entity(extracted_entity, existing_entities)
        
        if existing_match:
            # Calculate semantic similarity
            similarity = calculate_similarity(extracted_entity, existing_match)
            
            # Calculate entity overlap (check if names/types match)
            overlap = calculate_entity_overlap(extracted_entity, existing_match)
            
            # Auto-resolution criteria
            if similarity >= 0.85 and overlap >= 0.7:
                # Merge: Update existing entity
                updated = await self.shortterm_repo.update_entity(
                    entity_id=existing_match.id,
                    confidence=max(existing_match.confidence, extracted_entity.confidence),
                    ...
                )
                entity_map[extracted_entity.name] = updated.id
            else:
                # Conflict detected: Create new entity (manual merge required)
                created = await self.shortterm_repo.create_entity(...)
                entity_map[extracted_entity.name] = created.id
        else:
            # No existing entity: Create new
            created = await self.shortterm_repo.create_entity(...)
            entity_map[extracted_entity.name] = created.id
    
    # Create relationships
    for rel in extraction_result.data.relationships:
        from_id = entity_map.get(rel.source)
        to_id = entity_map.get(rel.target)
        if from_id and to_id:
            await self.shortterm_repo.create_relationship(...)
```

### Step 4: Enhance _promote_to_longterm Method

The method should handle entity/relationship promotion:

```python
async def _promote_to_longterm(self, external_id: str, shortterm_memory_id: int):
    # Get shortterm chunks and filter by importance
    shortterm_chunks = await self.shortterm_repo.get_chunks_by_memory_id(shortterm_memory_id)
    important_chunks = [c for c in shortterm_chunks if c.importance >= threshold]
    
    # Promote chunks to longterm
    for chunk in important_chunks:
        # Create longterm chunk with temporal tracking
        await self.longterm_repo.create_chunk(
            start_date=datetime.utcnow(),
            end_date=None,  # Currently valid
            ...
        )
        
        # Update previous chunks from same source (set end_date)
        await self.longterm_repo.supersede_chunks(
            shortterm_memory_id=shortterm_memory_id,
            end_date=datetime.utcnow()
        )
    
    # Get entities from shortterm
    shortterm_entities = await self.shortterm_repo.get_entities_by_memory_id(shortterm_memory_id)
    
    # Get existing longterm entities for comparison
    longterm_entities = await self.longterm_repo.get_entities_by_external_id(external_id)
    
    # Process entities
    for st_entity in shortterm_entities:
        # Find matching longterm entity
        lt_match = find_matching_longterm_entity(st_entity, longterm_entities)
        
        if lt_match:
            # Update existing entity confidence
            weight = 0.7
            new_confidence = weight * lt_match.confidence + (1 - weight) * st_entity.confidence
            
            await self.longterm_repo.update_entity(
                entity_id=lt_match.id,
                confidence=new_confidence,
                last_seen=datetime.utcnow(),
                ...
            )
        else:
            # Create new longterm entity
            await self.longterm_repo.create_entity(
                name=st_entity.name,
                entity_type=st_entity.type,
                confidence=st_entity.confidence,
                importance=calculate_importance(st_entity),
                valid_from=datetime.utcnow(),
                valid_until=None,  # Currently valid
                ...
            )
    
    # Promote relationships with temporal tracking
    shortterm_relationships = await self.shortterm_repo.get_relationships_by_memory_id(shortterm_memory_id)
    
    for st_rel in shortterm_relationships:
        # Check if relationship already exists in longterm
        lt_rel = find_matching_longterm_relationship(st_rel, longterm_relationships)
        
        if lt_rel:
            # Update last_updated timestamp
            await self.longterm_repo.update_relationship(
                relationship_id=lt_rel.id,
                last_updated=datetime.utcnow(),
                strength=max(lt_rel.strength, st_rel.strength),
                ...
            )
        else:
            # Create new longterm relationship with start_date
            await self.longterm_repo.create_relationship(
                from_entity_id=...,
                to_entity_id=...,
                relationship_type=st_rel.type,
                start_date=datetime.utcnow(),
                last_updated=datetime.utcnow(),
                ...
            )
```

### Step 5: Helper Functions Needed

Add these helper functions to memory_manager.py:

```python
def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts using embeddings."""
    # Get embeddings
    emb1 = await self.embedding_service.get_embedding(text1)
    emb2 = await self.embedding_service.get_embedding(text2)
    
    # Calculate cosine similarity
    from numpy import dot
    from numpy.linalg import norm
    similarity = dot(emb1, emb2) / (norm(emb1) * norm(emb2))
    return similarity

def _calculate_entity_overlap(self, entity1, entity2) -> float:
    """Calculate overlap between two entities based on name and type."""
    # Check name similarity
    name_match = entity1.name.lower() == entity2.name.lower()
    
    # Check type match
    type_match = entity1.type == entity2.type
    
    # Return overlap score
    if name_match and type_match:
        return 1.0
    elif name_match or type_match:
        return 0.5
    else:
        return 0.0

def _calculate_importance(self, entity) -> float:
    """Calculate importance score for entity promotion."""
    # Factors:
    # - Entity confidence
    # - Number of relationships
    # - Frequency of mentions
    # - Entity type (some types are more important)
    
    base_score = entity.confidence
    
    # Adjust based on entity type
    if entity.type in ["PERSON", "ORGANIZATION", "TECHNOLOGY"]:
        base_score *= 1.2
    elif entity.type in ["CONCEPT", "PROJECT"]:
        base_score *= 1.1
    
    return min(base_score, 1.0)
```

## Files to Update

1. **agent_mem/services/memory_manager.py**:
   - Update imports
   - Remove MemorizerAgent initialization
   - Replace _consolidate_to_shortterm method
   - Enhance _promote_to_longterm method
   - Add helper functions

2. **agent_mem/agents/__init__.py**:
   - Remove MemorizerAgent export
   - Keep only MemoryRetrieveAgent and MemoryUpdateAgent

3. **Delete agent_mem/agents/memorizer.py**:
   - No longer needed

4. **Documentation**:
   - Update PHASE4_COMPLETE.md
   - Update PHASE4_INTEGRATION_SUMMARY.md
   - Update IMPLEMENTATION_CHECKLIST.md

## Testing

After implementing changes:

1. Test consolidation with entity extraction
2. Test auto-resolution logic (similarity and overlap thresholds)
3. Test promotion with confidence updates
4. Test temporal tracking for longterm entities
5. Verify Neo4j graph structure

## Migration from Current State

Since MemorizerAgent is already integrated, we need to:

1. Remove all MemorizerAgent code
2. Simplify consolidation to use ER Extractor Agent directly
3. Add entity/relationship handling to promotion
4. Update tests and examples

This approach is simpler and uses the specialized ER Extractor Agent that's already part of the ai-army system, making it more maintainable and consistent with the existing architecture.
