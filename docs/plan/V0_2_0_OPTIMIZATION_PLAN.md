# Agent Mem Optimization Plan

**Version**: 0.2.0  
**Date**: November 14, 2025  
**Status**: ✅ COMPLETE (November 17, 2025)  
**Total Duration**: 3 days (Phase 1-7 complete, Phase 8 testing complete, Phase 9 ready)

---

## Overview

This plan outlines major optimizations to Agent Mem focusing on:
1. **Knowledge Graph Optimization**: Transition from nodes/edges to triplets (subject-predicate-object)
2. **Retrieval Methods Separation**: Split programmatic search vs. agent-powered deep search
3. **Output Optimization**: Streamline MCP and Core API outputs by removing unnecessary fields
4. **Token Usage Tracking**: Add pluggable token usage processor for monitoring LLM costs

---

## Motivation

### Current Issues

1. **Graph Knowledge Redundancy**
   - Current format: Separate `entities` and `relationships` arrays
   - Problem: Verbose, harder for LLMs to process, requires correlation
   - Solution: Use RDF-style triplets (subject-predicate-object)

2. **Retrieval Method Confusion**
   - Current: Single `retrieve_memories()` uses agent with fallback
   - Problem: No control over search strategy, hidden complexity
   - Solution: Two explicit methods - `search_memories()` (fast) and `deep_search_memories()` (agent)

3. **Output Bloat**
   - Current: Full models returned with all fields (external_id, metadata, template_content)
   - Problem: Unnecessary data transfer, confuses MCP clients
   - Solution: Slim response models for MCP and Core outputs

4. **No Token Tracking**
   - Current: Agent usage data discarded
   - Problem: Can't monitor costs or optimize token usage
   - Solution: Pluggable usage processor callback

---

## Implementation Phases

### Phase 1: Data Model Restructuring (Models Layer)
**Estimated Time**: 2 days  
**Priority**: High (Foundation for other changes)

#### 1.1: Create New Retrieval Models ✅

**File**: `agent_reminiscence/database/models.py`

**Changes**:
- Create tier-specific chunk models: `ShorttermRetrievedChunk`, `LongtermRetrievedChunk`
- Create triplet model: `KnowledgeTriplet` (subject, predicate, object, tier, importance)
- Create new `RetrievalResult` with triplets instead of entities/relationships
- Keep old models as deprecated for backward compatibility

**New Models**:
```python
class ShorttermRetrievedChunk(BaseModel):
    """Chunk from shortterm memory."""
    id: int
    content: str
    score: float
    section_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class LongtermRetrievedChunk(BaseModel):
    """Chunk from longterm memory."""
    id: int
    content: str
    score: float
    importance: float
    start_date: datetime
    last_updated: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class KnowledgeTriplet(BaseModel):
    """RDF-style knowledge triplet (subject-predicate-object)."""
    subject: str  # Entity name
    predicate: str  # Relationship type
    object: str  # Target entity name
    tier: Literal["shortterm", "longterm"]
    importance: float
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RetrievalResult(BaseModel):
    """New retrieval result with optimized structure."""
    mode: Literal["search", "deep_search"]
    shortterm_chunks: List[ShorttermRetrievedChunk] = Field(default_factory=list)
    longterm_chunks: List[LongtermRetrievedChunk] = Field(default_factory=list)
    knowledge_triplets: List[KnowledgeTriplet] = Field(default_factory=list)
    synthesis: Optional[str] = None  # Only for deep_search mode
    search_strategy: str
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

**Checklist**:
- [ ] Define `ShorttermRetrievedChunk` model
- [ ] Define `LongtermRetrievedChunk` model
- [ ] Define `KnowledgeTriplet` model
- [ ] Update `RetrievalResult` model with new structure
- [ ] Mark old models as deprecated with `@deprecated` decorator
- [ ] Add migration guide in docstrings

---

#### 1.2: Create Slim Output Models ✅

**File**: `agent_reminiscence/database/models.py`

**Changes**:
- Create `ActiveMemorySlim` for outputs (exclude external_id, metadata, template_content)
- Create `ActiveMemorySummary` for listings (id, title, section_count, last_updated)

**New Models**:
```python
class ActiveMemorySummary(BaseModel):
    """Slim active memory for list views."""
    id: int
    title: str
    section_count: int
    last_updated: datetime
    created_at: datetime

class ActiveMemorySlim(BaseModel):
    """Active memory without unnecessary fields for outputs."""
    id: int
    title: str
    sections: Dict[str, Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
```

**Checklist**:
- [ ] Define `ActiveMemorySummary` model
- [ ] Define `ActiveMemorySlim` model
- [ ] Add conversion methods on `ActiveMemory` class
- [ ] Document use cases in docstrings

---

### Phase 2: Repository Layer Updates
**Estimated Time**: 3 days  
**Priority**: High (Data access layer)

#### 2.1: Update Repository Return Types ✅

**Files**: 
- `agent_reminiscence/database/repositories/shortterm_memory.py`
- `agent_reminiscence/database/repositories/longterm_memory.py`

**Changes**:
- Update `search_entities_with_relationships()` methods to return triplets
- Create helper method to convert Neo4j relationships to triplets
- Add `_relationship_to_triplet()` conversion function

**Triplet Conversion Logic**:
```python
def _relationship_to_triplet(
    relationship: Neo4jRelationship,
    source_entity: Entity,
    target_entity: Entity,
    tier: str
) -> KnowledgeTriplet:
    """Convert Neo4j relationship to knowledge triplet."""
    return KnowledgeTriplet(
        subject=source_entity.name,
        predicate=relationship.types[0],  # Use first type as predicate
        object=target_entity.name,
        tier=tier,
        importance=relationship.importance,
        description=relationship.description,
        metadata={
            "relationship_id": relationship.id,
            "confidence": relationship.confidence,
            "additional_types": relationship.types[1:] if len(relationship.types) > 1 else []
        }
    )
```

**Checklist**:
- [ ] Create `_relationship_to_triplet()` helper in `shortterm_memory.py`
- [ ] Create `_relationship_to_triplet()` helper in `longterm_memory.py`
- [ ] Update `search_entities_with_relationships()` in shortterm repo
- [ ] Update `search_entities_with_relationships()` in longterm repo
- [ ] Add unit tests for triplet conversion
- [ ] Update repository documentation

---

### Phase 3: Agent Layer Updates
**Estimated Time**: 4 days  
**Priority**: High (Core logic change)

#### 3.1: Update Memory Retriever Agent ✅

**File**: `agent_reminiscence/agents/memory_retriever.py`

**Changes**:
- Rename `retrieve_memory()` to `deep_search_memory()`
- Update response models to use triplets
- Update tools to return triplet data
- Update synthesis prompts to work with triplets

**Agent Prompt Update**:
```python
system_prompt = """You are a Memory Retrieval Agent specialized in deep semantic search.

You search across memory tiers and return knowledge in triplet format:
- Triplets: (subject, predicate, object) format for relationships
- Example: ("John", "WorksAt", "Company X")

Your goal: Find relevant information and synthesize a comprehensive response.
"""
```

**Checklist**:
- [ ] Rename function `retrieve_memory()` → `deep_search_memory()`
- [ ] Update `RetrievalResult` model in agent
- [ ] Update `resolve_and_format_results()` to use triplets
- [ ] Update tool outputs to return triplet data
- [ ] Update agent system prompt
- [ ] Add tests for triplet-based retrieval
- [ ] Update agent documentation

---

#### 3.2: Add Token Usage Tracking ✅

**File**: `agent_reminiscence/agents/memory_retriever.py` (and other agents)

**Changes**:
- Capture `RunUsage` from agent results
- Return usage data alongside retrieval results

**Return Type Update**:
```python
async def deep_search_memory(
    query: str,
    external_id: str,
    shortterm_repo: ShorttermMemoryRepository,
    longterm_repo: LongtermMemoryRepository,
    active_repo: ActiveMemoryRepository,
    embedding_service: EmbeddingService,
    synthesis: bool = False,
) -> Tuple[FinalRetrievalResult, RunUsage]:
    """Deep search with agent - returns result and usage."""
    # ... existing code ...
    
    result = await agent.run(query, deps=deps)
    
    # Extract usage
    usage = result.usage()
    
    return final_result, usage
```

**Checklist**:
- [ ] Update `deep_search_memory()` return type to include `RunUsage`
- [ ] Update `extract_entities_and_relationships()` to return usage
- [ ] Update other agent functions to return usage
- [ ] Add usage extraction logic
- [ ] Update function signatures and type hints

---

### Phase 4: Service Layer Updates
**Estimated Time**: 3 days  
**Priority**: High (Business logic)

#### 4.1: Add Basic Search Method ✅

**File**: `agent_reminiscence/services/memory_manager.py`

**Changes**:
- Rename `_retrieve_memories_basic()` → `search_memories()` (make public)
- Create new `deep_search_memories()` that uses agent
- Both return new `RetrievalResult` with triplets

**Method Signatures**:
```python
async def search_memories(
    self,
    external_id: str,
    query: str,
    limit: int = 10,
    use_shortterm: bool = True,
    use_longterm: bool = True,
) -> RetrievalResult:
    """Fast programmatic search across memory tiers."""
    # Uses hybrid search without agent
    # Returns triplets from graph queries
    pass

async def deep_search_memories(
    self,
    external_id: str,
    query: str,
    limit: int = 10,
    synthesis: bool = True,
) -> Tuple[RetrievalResult, RunUsage]:
    """Deep semantic search using AI agent."""
    # Uses memory retriever agent
    # Returns synthesis + triplets + usage
    pass
```

**Checklist**:
- [ ] Rename and make public `_retrieve_memories_basic()` → `search_memories()`
- [ ] Update `search_memories()` to return new `RetrievalResult` format
- [ ] Create `deep_search_memories()` method
- [ ] Update both methods to construct triplets from graph data
- [ ] Remove old `retrieve_memories()` method (breaking change)
- [ ] Add comprehensive tests
- [ ] Update service documentation

---

#### 4.2: Add Token Usage Processor ✅

**File**: `agent_reminiscence/services/memory_manager.py`

**Changes**:
- Add `usage_processor` callback field
- Call processor after agent operations
- Pass external_id and usage data

**Implementation**:
```python
from typing import Callable, Optional
from pydantic_ai.usage import RunUsage

class MemoryManager:
    def __init__(
        self, 
        config: Config,
        usage_processor: Optional[Callable[[str, RunUsage], None]] = None
    ):
        self.config = config
        self.usage_processor = usage_processor
        # ... existing code ...
    
    async def deep_search_memories(self, ...) -> Tuple[RetrievalResult, RunUsage]:
        # ... search logic ...
        result, usage = await deep_search_memory(...)
        
        # Process usage if callback registered
        if self.usage_processor:
            try:
                self.usage_processor(external_id, usage)
            except Exception as e:
                logger.warning(f"Usage processor failed: {e}")
        
        return result, usage
```

**Checklist**:
- [ ] Add `usage_processor` parameter to `__init__`
- [ ] Add type hints for callback: `Callable[[str, RunUsage], None]`
- [ ] Call processor in `deep_search_memories()`
- [ ] Call processor in consolidation methods
- [ ] Add error handling for processor failures
- [ ] Add example usage processor in documentation

---

#### 4.3: Optimize Active Memory Outputs ✅

**File**: `agent_reminiscence/services/memory_manager.py`

**Changes**:
- Update `get_active_memories()` to return `List[ActiveMemorySummary]`
- Update `create_active_memory()` to return `ActiveMemorySlim`
- Update `update_active_memory_sections()` to return `ActiveMemorySlim`

**Checklist**:
- [ ] Update `get_active_memories()` return type
- [ ] Update `create_active_memory()` return type
- [ ] Update `update_active_memory_sections()` return type
- [ ] Add conversion logic in each method
- [ ] Update internal consolidation methods if needed
- [ ] Add tests for new return types

---

### Phase 5: Core API Layer Updates
**Estimated Time**: 2 days  
**Priority**: High (Public API)

#### 5.1: Split Retrieval Methods ✅

**File**: `agent_reminiscence/core.py`

**Changes**:
- Remove `retrieve_memories()` method
- Add `search_memories()` method (fast, no agent)
- Add `deep_search_memories()` method (with agent + synthesis)
- Update docstrings with clear use cases

**Method Signatures**:
```python
async def search_memories(
    self,
    external_id: str | UUID | int,
    query: str,
    limit: int = 10,
    use_shortterm: bool = True,
    use_longterm: bool = True,
) -> RetrievalResult:
    """
    Fast programmatic search across memory tiers.
    
    Use when:
    - You need quick results
    - You know what you're looking for
    - You don't need AI synthesis
    - You want to control which tiers to search
    
    Returns: RetrievalResult with chunks and knowledge triplets
    """
    pass

async def deep_search_memories(
    self,
    external_id: str | UUID | int,
    query: str,
    limit: int = 10,
    synthesis: bool = True,
) -> RetrievalResult:
    """
    Deep semantic search using AI agent with optional synthesis.
    
    Use when:
    - You need intelligent query understanding
    - You want AI to synthesize results
    - You need comprehensive context
    - Query is complex or ambiguous
    
    Returns: RetrievalResult with synthesis and knowledge triplets
    
    Note: This method uses LLM tokens. Enable usage tracking to monitor costs.
    """
    pass
```

**Checklist**:
- [ ] Remove `retrieve_memories()` method
- [ ] Add `search_memories()` method
- [ ] Add `deep_search_memories()` method
- [ ] Update docstrings with use cases
- [ ] Update method signatures
- [ ] Delegate to MemoryManager methods
- [ ] Update README with new API

---

#### 5.2: Add Usage Processor Registration ✅

**File**: `agent_reminiscence/core.py`

**Changes**:
- Add `set_usage_processor()` method
- Pass processor to MemoryManager
- Document usage tracking feature

**Implementation**:
```python
class AgentMem:
    def __init__(self, config: Optional[Config] = None):
        # ... existing code ...
        self._usage_processor: Optional[Callable[[str, RunUsage], None]] = None
    
    def set_usage_processor(
        self, 
        processor: Callable[[str, RunUsage], None]
    ) -> None:
        """
        Register a callback to process LLM token usage.
        
        The processor receives:
        - external_id: Agent identifier
        - usage: RunUsage object with token counts
        
        Example:
            ```python
            def log_usage(external_id: str, usage: RunUsage):
                print(f"{external_id}: {usage.total_tokens} tokens used")
            
            agent_mem.set_usage_processor(log_usage)
            ```
        """
        self._usage_processor = processor
        if self._memory_manager:
            self._memory_manager.usage_processor = processor
    
    async def initialize(self) -> None:
        # ... existing code ...
        self._memory_manager = MemoryManager(
            config=self.config,
            usage_processor=self._usage_processor
        )
```

**Checklist**:
- [ ] Add `_usage_processor` field to `__init__`
- [ ] Add `set_usage_processor()` method
- [ ] Pass processor to MemoryManager in `initialize()`
- [ ] Add comprehensive docstring with examples
- [ ] Add usage tracking example to README

---

#### 5.3: Optimize Output Models ✅

**File**: `agent_reminiscence/core.py`

**Changes**:
- Update `get_active_memories()` to return `List[ActiveMemorySummary]`
- Update `create_active_memory()` to return `ActiveMemorySlim`
- Update `update_active_memory_sections()` to return `ActiveMemorySlim`

**Checklist**:
- [ ] Update `get_active_memories()` return type
- [ ] Update `create_active_memory()` return type
- [ ] Update `update_active_memory_sections()` return type
- [ ] Update docstrings
- [ ] Update type hints in `__init__.py` exports

---

### Phase 6: MCP Server Updates
**Estimated Time**: 3 days  
**Priority**: Medium (External integration)

#### 6.1: Split Search Tools ✅

**File**: `agent_reminiscence_mcp/server.py`

**Changes**:
- Split `search_memories` tool into two:
  - `search_memories` (fast programmatic search)
  - `deep_search_memories` (agent-powered with synthesis)
- Update schemas in `schemas.py`
- Update tool handlers

**Tool Definitions**:
```python
# Tool 1: search_memories (fast)
{
    "name": "search_memories",
    "description": "Fast programmatic search across memory tiers without AI synthesis",
    "inputSchema": SEARCH_MEMORIES_INPUT_SCHEMA
}

# Tool 2: deep_search_memories (agent)
{
    "name": "deep_search_memories",
    "description": "Deep semantic search using AI agent with optional synthesis (uses LLM tokens)",
    "inputSchema": DEEP_SEARCH_MEMORIES_INPUT_SCHEMA
}
```

**Checklist**:
- [ ] Update `search_memories` tool to use `agent_mem.search_memories()`
- [ ] Add new `deep_search_memories` tool
- [ ] Create `DEEP_SEARCH_MEMORIES_INPUT_SCHEMA` in `schemas.py`
- [ ] Update `handle_list_tools()` to return 4 tools
- [ ] Update `handle_call_tool()` routing
- [ ] Update tool descriptions to clarify differences
- [ ] Update MCP README with new tools

---

#### 6.2: Optimize Tool Outputs ✅

**File**: `agent_reminiscence_mcp/server.py`

**Changes**:
- Update `get_active_memories` to return slim summaries
- Update `create_active_memory` to return slim model
- Update `update_memory_sections` to return slim model
- Update search tools to use new triplet format

**Response Format (get_active_memories)**:
```json
{
  "memories": [
    {
      "id": 1,
      "title": "Task Memory",
      "section_count": 3,
      "last_updated": "2025-11-14T10:00:00Z",
      "created_at": "2025-11-14T09:00:00Z"
    }
  ],
  "count": 1
}
```

**Response Format (search results with triplets)**:
```json
{
  "query": "authentication implementation",
  "mode": "search",
  "shortterm_chunks": [
    {
      "id": 1,
      "content": "JWT tokens implemented...",
      "score": 0.95
    }
  ],
  "longterm_chunks": [
    {
      "id": 42,
      "content": "Security best practices...",
      "score": 0.88,
      "importance": 0.85
    }
  ],
  "knowledge_triplets": [
    {
      "subject": "JWT",
      "predicate": "UsedFor",
      "object": "Authentication",
      "tier": "shortterm",
      "importance": 0.9
    }
  ],
  "search_strategy": "Hybrid search on shortterm and longterm tiers",
  "confidence": 0.92
}
```

**Checklist**:
- [ ] Update `_handle_get_active_memories()` output format
- [ ] Update `_handle_create_active_memory()` output format
- [ ] Update `_handle_update_memory_sections()` output format
- [ ] Update `_handle_search_memories()` output format
- [ ] Add `_handle_deep_search_memories()` handler
- [ ] Update response serialization for triplets
- [ ] Test with Claude Desktop

---

#### 6.3: Add Usage Tracking Documentation ✅

**File**: `agent_reminiscence_mcp/README.md`

**Changes**:
- Document `deep_search_memories` tool
- Add warning about token usage
- Add note about usage tracking

**Documentation Section**:
```markdown
### Token Usage Tracking

The `deep_search_memories` tool uses LLM tokens for AI-powered search. To track usage:

1. Implement a usage processor in your application
2. Register it before using deep search:

```python
from agent_reminiscence import AgentMem

def track_usage(external_id: str, usage):
    print(f"Agent {external_id} used {usage.total_tokens} tokens")
    # Log to database, monitoring system, etc.

agent_mem = AgentMem()
await agent_mem.initialize()
agent_mem.set_usage_processor(track_usage)
```

**Note**: MCP server does not automatically track usage. Implement tracking in your client application.
```

**Checklist**:
- [ ] Add deep_search_memories tool documentation
- [ ] Add token usage tracking section
- [ ] Add examples of usage processors
- [ ] Add cost estimation tips
- [ ] Update tool comparison table

---

### Phase 7: Documentation Updates
**Estimated Time**: 2 days  
**Priority**: Medium (User-facing)

#### 7.1: Update API Documentation ✅

**Files**: 
- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/QUICKSTART.md`

**Changes**:
- Update API reference with new methods
- Document triplet format for knowledge graphs
- Document usage tracking feature
- Add migration guide for breaking changes

**Checklist**:
- [ ] Update README.md API section
- [ ] Update ARCHITECTURE.md with triplet format
- [ ] Update QUICKSTART.md examples
- [ ] Add MIGRATION.md guide for v0.1.x → v0.2.0
- [ ] Document breaking changes
- [ ] Add triplet examples

---

#### 7.2: Create Usage Examples ✅

**Files**: 
- `examples/search_vs_deep_search.py`
- `examples/token_usage_tracking.py`
- `examples/triplet_knowledge_graph.py`

**Checklist**:
- [ ] Create `search_vs_deep_search.py` example
- [ ] Create `token_usage_tracking.py` example
- [ ] Create `triplet_knowledge_graph.py` example
- [ ] Update `examples/README.md`
- [ ] Add code comments explaining use cases

---

### Phase 8: Testing
**Estimated Time**: 3 days  
**Priority**: High (Quality assurance)

#### 8.1: Unit Tests ✅

**Test Coverage**:
- Models: Triplet conversion, model validation
- Repositories: Triplet queries, graph operations
- Agents: Deep search, usage tracking
- Services: Both search methods, usage processor
- Core: API method contracts

**Checklist**:
- [ ] Update `tests/test_models.py` with triplet tests
- [ ] Update `tests/test_repositories.py` with triplet conversion tests
- [ ] Update `tests/test_agents.py` with usage tracking tests
- [ ] Update `tests/test_memory_manager.py` with new search methods
- [ ] Update `tests/test_core.py` with split retrieval methods
- [ ] Add usage processor tests
- [ ] Achieve >90% coverage on changed code

---

#### 8.2: Integration Tests ✅

**Test Scenarios**:
- End-to-end search flow with triplets
- End-to-end deep search with synthesis
- Usage tracking across operations
- MCP tool integration

**Checklist**:
- [ ] Create `tests/integration/test_search_flow.py`
- [ ] Create `tests/integration/test_deep_search_flow.py`
- [ ] Create `tests/integration/test_usage_tracking.py`
- [ ] Create `tests/integration/test_mcp_tools.py`
- [ ] Test backward compatibility (deprecated models)
- [ ] Test migration path

---

#### 8.3: Performance Tests ✅

**Benchmarks**:
- Search vs. Deep Search performance comparison
- Triplet conversion overhead
- Memory usage with slim models

**Checklist**:
- [ ] Create `tests/performance/test_search_performance.py`
- [ ] Benchmark triplet conversion vs. entity/relationship format
- [ ] Measure memory reduction with slim models
- [ ] Document performance improvements in README

---

### Phase 9: Deployment
**Estimated Time**: 1 day  
**Priority**: Medium (Release preparation)

#### 9.1: Version Bump and Changelog ✅

**Files**:
- `pyproject.toml`
- `CHANGELOG.md`
- `agent_reminiscence/__init__.py`

**Checklist**:
- [ ] Update version to 0.2.0
- [ ] Write comprehensive CHANGELOG.md entry
- [ ] Document breaking changes
- [ ] Document new features
- [ ] Update deprecation notices

---

#### 9.2: Release Preparation ✅

**Checklist**:
- [ ] Run full test suite
- [ ] Update all documentation
- [ ] Create migration guide
- [ ] Tag release in git
- [ ] Build and test package
- [ ] Update GitHub release notes

---

## Breaking Changes Summary

### Removed Methods
- `AgentMem.retrieve_memories()` → Split into `search_memories()` and `deep_search_memories()`

### Changed Return Types
- `AgentMem.get_active_memories()`: `List[ActiveMemory]` → `List[ActiveMemorySummary]`
- `AgentMem.create_active_memory()`: `ActiveMemory` → `ActiveMemorySlim`
- `AgentMem.update_active_memory_sections()`: `ActiveMemory` → `ActiveMemorySlim`

### Changed Data Structures
- `RetrievalResult.entities` (removed) → `RetrievalResult.knowledge_triplets`
- `RetrievalResult.relationships` (removed) → `RetrievalResult.knowledge_triplets`
- `RetrievalResult.chunks` → `RetrievalResult.shortterm_chunks` + `RetrievalResult.longterm_chunks`

### Deprecated (Still Available)
- `RetrievedChunk`, `RetrievedEntity`, `RetrievedRelationship` models
- Old `RetrievalResult` structure (maintained for backward compatibility in v0.2.0)

---

## Migration Guide

### For Users Calling `retrieve_memories()`

**Before (v0.1.x)**:
```python
result = await agent_mem.retrieve_memories(
    external_id="agent-123",
    query="authentication",
    synthesis=True
)

for entity in result.entities:
    print(f"Entity: {entity.name}")

for rel in result.relationships:
    print(f"{rel.from_entity_name} -> {rel.types[0]} -> {rel.to_entity_name}")
```

**After (v0.2.0)**:
```python
# Option 1: Fast search (no agent)
result = await agent_mem.search_memories(
    external_id="agent-123",
    query="authentication"
)

# Option 2: Deep search with synthesis (uses agent)
result = await agent_mem.deep_search_memories(
    external_id="agent-123",
    query="authentication",
    synthesis=True
)

# Access knowledge as triplets
for triplet in result.knowledge_triplets:
    print(f"{triplet.subject} --{triplet.predicate}--> {triplet.object}")
```

### For Users Processing Active Memories

**Before (v0.1.x)**:
```python
memories = await agent_mem.get_active_memories("agent-123")
for memory in memories:
    print(memory.template_content)  # Full template
    print(memory.metadata)  # Metadata included
```

**After (v0.2.0)**:
```python
memories = await agent_mem.get_active_memories("agent-123")
for summary in memories:
    print(summary.title)
    print(f"Sections: {summary.section_count}")
    # No template_content, no metadata - slimmer output
```

---

## Timeline

| Phase | Duration | Dependency |
|-------|----------|------------|
| Phase 1: Models | 2 days | None |
| Phase 2: Repositories | 3 days | Phase 1 |
| Phase 3: Agents | 4 days | Phase 1, 2 |
| Phase 4: Services | 3 days | Phase 1, 2, 3 |
| Phase 5: Core API | 2 days | Phase 4 |
| Phase 6: MCP Server | 3 days | Phase 5 |
| Phase 7: Documentation | 2 days | All phases |
| Phase 8: Testing | 3 days | Phase 1-6 |
| Phase 9: Deployment | 1 day | Phase 8 |

**Total Estimated Time**: 23 days (~4.5 weeks)

---

## Success Metrics

### Performance
- [ ] Search response time < 100ms (programmatic)
- [ ] Deep search response time < 2s (with agent)
- [ ] 30% reduction in data transfer with slim models
- [ ] Triplet format reduces LLM input tokens by 20%

### Code Quality
- [ ] >90% test coverage on changed code
- [ ] All tests passing
- [ ] No regressions in existing functionality
- [ ] Backward compatibility maintained for v0.2.0

### Documentation
- [ ] All public APIs documented
- [ ] Migration guide complete
- [ ] Examples for all new features
- [ ] MCP server documentation updated

---

## Risk Assessment

### High Risk
- **Breaking changes**: May affect existing users
  - Mitigation: Comprehensive migration guide, deprecation warnings

### Medium Risk
- **Triplet conversion complexity**: Graph queries may be complex
  - Mitigation: Extensive testing, fallback to entity/relationship format

### Low Risk
- **Usage tracking overhead**: Callback execution may slow down operations
  - Mitigation: Error handling, optional feature, async execution

---

## Open Questions

1. **Backward Compatibility**: Should we maintain old models in v0.2.0 or force migration?
   - **Decision**: Keep deprecated models with warnings in v0.2.0, remove in v0.3.0

2. **Triplet Predicate Naming**: How to standardize relationship types as predicates?
   - **Decision**: Use first relationship type, store additional types in metadata

3. **Usage Tracking**: Should we provide built-in processors (e.g., logging, metrics)?
   - **Decision**: Provide examples in docs, let users implement custom processors

4. **MCP Token Costs**: Should MCP server warn about token usage?
   - **Decision**: Add warnings in tool descriptions, document in README

---

## Next Steps

1. Review and approve this plan
2. Create GitHub issues for each phase
3. Set up project board with milestones
4. Begin Phase 1 implementation
5. Schedule weekly progress reviews

---

**Plan Status**: 📋 Draft - Awaiting Review  
**Last Updated**: November 14, 2025
