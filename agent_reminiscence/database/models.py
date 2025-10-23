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


