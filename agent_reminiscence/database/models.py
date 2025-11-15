"""Pydantic models for Agent Mem."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Literal
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict


class ActiveMemory(BaseModel):
    """
    Active memory model representing working memory.

    Uses template-driven structure with sections:
    - template_content: JSON template with section definitions and defaults
    - sections: JSONB with section_id -> {content, update_count, awake_update_count, last_updated}
    """

    id: int
    external_id: str  # worker_id equivalent - generic identifier
    title: str
    template_content: Dict[str, Any]  # Changed from str to Dict
    sections: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict
    )  # {section_id: {content, update_count, awake_update_count, last_updated}}
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ShorttermMemoryChunk(BaseModel):
    """Shortterm memory chunk with embeddings."""

    id: int
    shortterm_memory_id: int
    content: str
    section_id: Optional[str] = None  # Reference to active memory section
    similarity_score: Optional[float] = None
    bm25_score: Optional[float] = None
    access_count: int = 0
    last_access: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class ShorttermMemory(BaseModel):
    """Shortterm memory model."""

    id: int
    external_id: str
    title: str
    summary: Optional[str] = None
    chunks: List[ShorttermMemoryChunk] = Field(default_factory=list)
    update_count: int = 0  # Track number of consolidations
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    last_updated: datetime

    model_config = ConfigDict(from_attributes=True)


class LongtermMemoryChunk(BaseModel):
    """Longterm memory chunk with temporal validity."""

    id: int
    external_id: str
    shortterm_memory_id: Optional[int] = None
    content: str
    importance: float
    start_date: datetime
    last_updated: Optional[datetime] = None  # Track when chunk was last updated from shortterm
    similarity_score: Optional[float] = None
    bm25_score: Optional[float] = None
    access_count: int = 0
    last_access: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class LongtermMemory(BaseModel):
    """Longterm memory model (aggregated from chunks)."""

    chunks: List[LongtermMemoryChunk]
    external_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class ShorttermEntity(BaseModel):
    """Shortterm entity model for graph nodes."""

    id: str  # Neo4j elementId (string)
    external_id: str
    shortterm_memory_id: int
    name: str
    types: List[str] = Field(default_factory=list)  # Multiple types supported
    description: Optional[str] = None
    importance: float = Field(ge=0.0, le=1.0)
    access_count: int = 0
    last_access: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class ShorttermRelationship(BaseModel):
    """Shortterm relationship model for graph edges."""

    id: str  # Neo4j elementId (string)
    external_id: str
    shortterm_memory_id: int
    from_entity_id: str  # Neo4j elementId (string)
    to_entity_id: str  # Neo4j elementId (string)
    from_entity_name: Optional[str] = None
    to_entity_name: Optional[str] = None
    types: List[str] = Field(default_factory=list)  # Multiple types supported
    description: Optional[str] = None
    importance: float = Field(ge=0.0, le=1.0)
    access_count: int = 0
    last_access: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class LongtermEntity(BaseModel):
    """Longterm entity model for graph nodes."""

    id: str  # Neo4j elementId (string)
    external_id: str
    name: str
    types: List[str] = Field(default_factory=list)  # Multiple types supported
    description: Optional[str] = None
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    access_count: int = 0
    last_access: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class LongtermRelationship(BaseModel):
    """Longterm relationship model for graph edges."""

    id: str  # Neo4j elementId (string)
    external_id: str
    from_entity_id: str  # Neo4j elementId (string)
    to_entity_id: str  # Neo4j elementId (string)
    from_entity_name: Optional[str] = None
    to_entity_name: Optional[str] = None
    types: List[str] = Field(default_factory=list)  # Multiple types supported
    description: Optional[str] = None
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    start_date: datetime
    access_count: int = 0
    last_access: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# RETRIEVAL RESULT MODELS
# ============================================================================


class RetrievedChunk(BaseModel):
    """Resolved chunk data from retrieval."""

    id: int
    content: str
    tier: Literal["shortterm", "longterm"]
    score: float
    importance: Optional[float] = None  # Only for longterm chunks
    start_date: Optional[datetime] = None  # Only for longterm chunks

    model_config = ConfigDict(from_attributes=True)


class RetrievedEntity(BaseModel):
    """Resolved entity data from retrieval."""

    id: str
    name: str
    types: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    tier: Literal["shortterm", "longterm"]
    importance: float

    model_config = ConfigDict(from_attributes=True)


class RetrievedRelationship(BaseModel):
    """Resolved relationship data from retrieval."""

    id: str
    from_entity_name: Optional[str] = None
    to_entity_name: Optional[str] = None
    types: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    tier: Literal["shortterm", "longterm"]
    importance: float

    model_config = ConfigDict(from_attributes=True)


class RetrievalResult(BaseModel):
    """
    Result from memory retrieval with resolved data.

    Mode determines behavior:
    - pointer: Returns resolved data from pointer IDs
    - synthesis: Returns synthesized summary with resolved data
    """

    mode: Literal["pointer", "synthesis"] = Field(
        description="Result mode: pointer (resolved IDs) or synthesis (with summary)"
    )
    chunks: List[RetrievedChunk] = Field(default_factory=list, description="Resolved chunk data")
    entities: List[RetrievedEntity] = Field(
        default_factory=list, description="Resolved entity data"
    )
    relationships: List[RetrievedRelationship] = Field(
        default_factory=list, description="Resolved relationship data"
    )
    synthesis: Optional[str] = Field(
        default=None, description="Natural language synthesis (only in synthesis mode)"
    )
    search_strategy: str = Field(description="Brief explanation of search approach and decisions")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in result relevance (0-1)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the search (counts, timing, etc.)",
    )

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# CONFLICT RESOLUTION MODELS (for batch update and consolidation)
# ============================================================================


class ConflictEntityDetail(BaseModel):
    """Detailed entity conflict information."""

    name: str
    shortterm_types: List[str] = Field(default_factory=list)
    active_types: List[str] = Field(default_factory=list)
    merged_types: List[str] = Field(default_factory=list)
    shortterm_importance: float
    active_importance: float
    merged_importance: float
    shortterm_description: Optional[str] = None
    active_description: Optional[str] = None
    merged_description: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class ConflictRelationshipDetail(BaseModel):
    """Detailed relationship conflict information."""

    from_entity: str
    to_entity: str
    shortterm_types: List[str] = Field(default_factory=list)
    active_types: List[str] = Field(default_factory=list)
    merged_types: List[str] = Field(default_factory=list)
    shortterm_importance: float
    active_importance: float
    merged_importance: float
    shortterm_strength: float
    active_strength: float
    merged_strength: float

    model_config = ConfigDict(from_attributes=True)


class ConflictSection(BaseModel):
    """Section with potentially conflicting chunks."""

    section_id: str
    section_content: str
    update_count: int
    existing_chunks: List[ShorttermMemoryChunk] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class ConsolidationConflicts(BaseModel):
    """Comprehensive conflict tracking for consolidation."""

    external_id: str
    active_memory_id: int
    shortterm_memory_id: int
    created_at: datetime
    total_conflicts: int = 0

    # Enhanced conflict tracking
    sections: List[ConflictSection] = Field(default_factory=list)
    entity_conflicts: List[ConflictEntityDetail] = Field(default_factory=list)
    relationship_conflicts: List[ConflictRelationshipDetail] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# SEARCH RESULT MODELS (for enhanced search features)
# ============================================================================


class ShorttermEntityRelationshipSearchResult(BaseModel):
    """
    Result from entity/relationship graph search in shortterm memory.

    Represents a subgraph centered around entities matching the search query.
    """

    query_entity_names: List[str] = Field(description="Original entity names used in search")
    external_id: str = Field(description="Agent identifier")
    shortterm_memory_id: Optional[int] = Field(
        default=None, description="Optional memory ID filter"
    )
    matched_entities: List[ShorttermEntity] = Field(
        default_factory=list, description="Entities directly matching query names"
    )
    related_entities: List[ShorttermEntity] = Field(
        default_factory=list, description="Entities connected via relationships"
    )
    relationships: List[ShorttermRelationship] = Field(
        default_factory=list,
        description="All relationships connecting matched and related entities",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Search metadata (filters, timing, etc.)"
    )

    model_config = ConfigDict(from_attributes=True)


class LongtermEntityRelationshipSearchResult(BaseModel):
    """
    Result from entity/relationship graph search in longterm memory.

    Similar to shortterm but without memory_id constraint.
    """

    query_entity_names: List[str] = Field(description="Original entity names used in search")
    external_id: str = Field(description="Agent identifier")
    matched_entities: List[LongtermEntity] = Field(
        default_factory=list, description="Entities directly matching query names"
    )
    related_entities: List[LongtermEntity] = Field(
        default_factory=list, description="Entities connected via relationships"
    )
    relationships: List[LongtermRelationship] = Field(
        default_factory=list,
        description="All relationships connecting matched and related entities",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Search metadata (filters, timing, etc.)"
    )

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# NEW v0.2.0 MODELS - OPTIMIZED FOR LLM AND OUTPUT
# ============================================================================


class ShorttermKnowledgeTriplet(BaseModel):
    """
    Shortterm knowledge triplet (subject-predicate-object).

    Represents recent relationships extracted from shortterm memory.
    Optimized for LLM processing and knowledge graph representation.

    Example:
        {"subject": "JWT", "predicate": "UsedFor", "object": "Authentication",
         "importance": 0.9, "access_count": 3}
    """

    subject: str = Field(description="Subject entity name")
    predicate: str = Field(description="Relationship type/predicate")
    object: str = Field(description="Object entity name")
    importance: float = Field(ge=0.0, le=1.0, description="Triplet importance score")
    shortterm_memory_id: Optional[int] = Field(
        default=None, description="Source shortterm memory ID"
    )
    access_count: int = Field(default=0, description="Number of times accessed")
    description: Optional[str] = Field(
        default=None, description="Optional relationship description"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata (confidence, additional_types, etc.)"
    )

    model_config = ConfigDict(from_attributes=True)


class LongtermKnowledgeTriplet(BaseModel):
    """
    Longterm knowledge triplet (subject-predicate-object).

    Represents consolidated relationships from longterm memory with temporal validity.
    Optimized for LLM processing and knowledge graph representation.

    Example:
        {"subject": "JWT", "predicate": "UsedFor", "object": "Authentication",
         "importance": 0.95, "start_date": "2025-11-15T00:00:00Z", "temporal_validity": "evergreen"}
    """

    subject: str = Field(description="Subject entity name")
    predicate: str = Field(description="Relationship type/predicate")
    object: str = Field(description="Object entity name")
    importance: float = Field(ge=0.0, le=1.0, description="Triplet importance score")
    start_date: datetime = Field(description="When triplet became valid")
    temporal_validity: Optional[str] = Field(
        default=None, description="Validity period (e.g., 'evergreen', 'until_2025-12-01')"
    )
    access_count: int = Field(default=0, description="Number of times accessed")
    description: Optional[str] = Field(
        default=None, description="Optional relationship description"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata (confidence, additional_types, etc.)"
    )

    model_config = ConfigDict(from_attributes=True)


class ShorttermRetrievedChunk(BaseModel):
    """
    Chunk from shortterm memory (optimized for retrieval results).

    Slim version without embedding vectors - only relevant fields for LLM consumption.
    """

    id: int = Field(description="Chunk ID")
    content: str = Field(description="Chunk text content")
    score: float = Field(ge=0.0, le=1.0, description="Relevance score (vector + BM25)")
    section_id: Optional[str] = Field(
        default=None, description="Reference to active memory section"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")

    model_config = ConfigDict(from_attributes=True)


class LongtermRetrievedChunk(BaseModel):
    """
    Chunk from longterm memory (optimized for retrieval results).

    Includes importance and temporal tracking for consolidated knowledge.
    """

    id: int = Field(description="Chunk ID")
    content: str = Field(description="Chunk text content")
    score: float = Field(ge=0.0, le=1.0, description="Relevance score (vector + BM25)")
    importance: float = Field(ge=0.0, le=1.0, description="Chunk importance score")
    start_date: datetime = Field(description="When chunk became valid")
    last_updated: Optional[datetime] = Field(default=None, description="Last update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")

    model_config = ConfigDict(from_attributes=True)


class RetrievalResultV2(BaseModel):
    """
    Optimized retrieval result for v0.2.0.

    Uses separated chunks by tier and triplet-based knowledge representation.

    Attributes:
        mode: "search" for fast programmatic search, "deep_search" for agent-powered search
        shortterm_chunks: Relevant chunks from shortterm memory
        longterm_chunks: Relevant chunks from longterm memory
        shortterm_triplets: Knowledge triplets from shortterm memory
        longterm_triplets: Knowledge triplets from longterm memory
        synthesis: AI-generated summary (only in deep_search mode)
        search_strategy: Explanation of search approach used
        confidence: Overall confidence in result relevance (0-1)
        metadata: Additional search metadata (counts, timing, etc.)
    """

    mode: Literal["search", "deep_search"] = Field(
        description="Search mode: fast programmatic or agent-powered"
    )
    shortterm_chunks: List[ShorttermRetrievedChunk] = Field(
        default_factory=list, description="Chunks from shortterm memory tier"
    )
    longterm_chunks: List[LongtermRetrievedChunk] = Field(
        default_factory=list, description="Chunks from longterm memory tier"
    )
    shortterm_triplets: List[ShorttermKnowledgeTriplet] = Field(
        default_factory=list,
        description="Knowledge triplets from shortterm memory (subject-predicate-object)",
    )
    longterm_triplets: List[LongtermKnowledgeTriplet] = Field(
        default_factory=list,
        description="Knowledge triplets from longterm memory (subject-predicate-object)",
    )
    synthesis: Optional[str] = Field(
        default=None,
        description="Natural language synthesis (only in deep_search mode with synthesis=True)",
    )
    search_strategy: str = Field(description="Brief explanation of search approach and decisions")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in result relevance (0-1)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (result_counts, timing, filters_applied, etc.)",
    )

    model_config = ConfigDict(from_attributes=True)


class ActiveMemorySummary(BaseModel):
    """
    Slim active memory for list views and summary endpoints.

    Excludes heavy fields like template_content, metadata, and sections detail.
    Use when listing or summarizing memories.
    """

    id: int = Field(description="Memory ID")
    title: str = Field(description="Memory title")
    section_count: int = Field(description="Number of sections in memory")
    last_updated: datetime = Field(description="Last update timestamp")
    created_at: datetime = Field(description="Creation timestamp")

    model_config = ConfigDict(from_attributes=True)


class ActiveMemorySlim(BaseModel):
    """
    Slim active memory for output without metadata and external_id.

    Use when returning created/updated memories to keep payload minimal.
    Excludes: external_id, metadata, template_content (structure only).
    """

    id: int = Field(description="Memory ID")
    title: str = Field(description="Memory title")
    sections: Dict[str, Dict[str, Any]] = Field(
        description="Memory sections: {section_id: {content, update_count, ...}}"
    )
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# DEPRECATED MODELS (v0.1.x - kept for backward compatibility)
# ============================================================================

from typing_extensions import deprecated


@deprecated(
    "RetrievedChunk is deprecated in v0.2.0. Use ShorttermRetrievedChunk or "
    "LongtermRetrievedChunk instead. Triplet-based knowledge representation is preferred. "
    "See docs/MIGRATION_v0.1_to_v0.2.md for migration guide."
)
class RetrievedChunkDeprecated(BaseModel):
    """DEPRECATED: Use ShorttermRetrievedChunk or LongtermRetrievedChunk."""

    id: int
    content: str
    tier: Literal["shortterm", "longterm"]
    score: float
    importance: Optional[float] = None
    start_date: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


@deprecated(
    "RetrievedEntity is deprecated in v0.2.0. Use KnowledgeTriplet instead for "
    "triplet-based knowledge representation. See docs/MIGRATION_v0.1_to_v0.2.md"
)
class RetrievedEntityDeprecated(BaseModel):
    """DEPRECATED: Use KnowledgeTriplet instead."""

    id: str
    name: str
    types: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    tier: Literal["shortterm", "longterm"]
    importance: float

    model_config = ConfigDict(from_attributes=True)


@deprecated(
    "RetrievedRelationship is deprecated in v0.2.0. Use KnowledgeTriplet instead "
    "for triplet-based knowledge representation. See docs/MIGRATION_v0.1_to_v0.2.md"
)
class RetrievedRelationshipDeprecated(BaseModel):
    """DEPRECATED: Use KnowledgeTriplet instead."""

    id: str
    from_entity_name: Optional[str] = None
    to_entity_name: Optional[str] = None
    types: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    tier: Literal["shortterm", "longterm"]
    importance: float

    model_config = ConfigDict(from_attributes=True)


# Keep old RetrievalResult as alias for backward compatibility
class RetrievalResultLegacy(BaseModel):
    """
    DEPRECATED: Legacy retrieval result from v0.1.x.

    Use RetrievalResultV2 instead. This model is kept for backward compatibility
    and will be removed in v0.3.0. See docs/MIGRATION_v0.1_to_v0.2.md
    """

    mode: Literal["pointer", "synthesis"] = Field(
        description="DEPRECATED: Use RetrievalResultV2 with 'search' or 'deep_search'"
    )
    chunks: List["RetrievedChunkDeprecated"] = Field(default_factory=list)
    entities: List["RetrievedEntityDeprecated"] = Field(default_factory=list)
    relationships: List["RetrievedRelationshipDeprecated"] = Field(default_factory=list)
    synthesis: Optional[str] = None
    search_strategy: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


# Add extension methods to ActiveMemory for conversion
def _add_conversion_methods():
    """Add to_summary() and to_slim() methods to ActiveMemory."""

    def to_summary(self) -> ActiveMemorySummary:
        """Convert to summary view (for listings)."""
        return ActiveMemorySummary(
            id=self.id,
            title=self.title,
            section_count=len(self.sections),
            last_updated=self.updated_at,
            created_at=self.created_at,
        )

    def to_slim(self) -> ActiveMemorySlim:
        """Convert to slim view (for outputs without metadata)."""
        return ActiveMemorySlim(
            id=self.id,
            title=self.title,
            sections=self.sections,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    ActiveMemory.to_summary = to_summary
    ActiveMemory.to_slim = to_slim


# Call at module load to add methods
_add_conversion_methods()
