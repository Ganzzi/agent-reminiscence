"""Pydantic models for Agent Mem."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict


class ActiveMemory(BaseModel):
    """
    Active memory model representing working memory.

    Uses template-driven structure with sections:
    - template_content: YAML template defining structure
    - sections: JSONB with section_id -> {content: str, update_count: int}
    """

    id: int
    external_id: str  # worker_id equivalent - generic identifier
    title: str
    template_content: str  # YAML template as text
    sections: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict
    )  # {section_id: {content, update_count}}
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ShorttermMemoryChunk(BaseModel):
    """Shortterm memory chunk with embeddings."""

    id: int
    shortterm_memory_id: int
    content: str
    chunk_order: int
    similarity_score: Optional[float] = None
    bm25_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class ShorttermMemory(BaseModel):
    """Shortterm memory model."""

    id: int
    external_id: str
    title: str
    summary: Optional[str] = None
    chunks: List[ShorttermMemoryChunk] = Field(default_factory=list)
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
    chunk_order: int
    confidence_score: float
    start_date: datetime
    end_date: Optional[datetime] = None
    similarity_score: Optional[float] = None
    bm25_score: Optional[float] = None
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

    id: int
    external_id: str
    shortterm_memory_id: int
    name: str
    type: str
    description: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    first_seen: datetime
    last_seen: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class ShorttermRelationship(BaseModel):
    """Shortterm relationship model for graph edges."""

    id: int
    external_id: str
    shortterm_memory_id: int
    from_entity_id: int
    to_entity_id: int
    from_entity_name: Optional[str] = None
    to_entity_name: Optional[str] = None
    type: str
    description: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    strength: float = Field(ge=0.0, le=1.0)
    first_observed: datetime
    last_observed: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class LongtermEntity(BaseModel):
    """Longterm entity model for graph nodes."""

    id: int
    external_id: str
    name: str
    type: str
    description: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    first_seen: datetime
    last_seen: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class LongtermRelationship(BaseModel):
    """Longterm relationship model for graph edges."""

    id: int
    external_id: str
    from_entity_id: int
    to_entity_id: int
    from_entity_name: Optional[str] = None
    to_entity_name: Optional[str] = None
    type: str
    description: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    strength: float = Field(ge=0.0, le=1.0)
    start_date: datetime
    last_updated: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class Entity(BaseModel):
    """Entity model for graph nodes (generic)."""

    id: int
    external_id: str
    name: str
    type: str
    description: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    importance: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    first_seen: datetime
    last_seen: datetime
    memory_tier: str  # 'shortterm' or 'longterm'
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class Relationship(BaseModel):
    """Relationship model for graph edges (generic)."""

    id: int
    external_id: str
    from_entity_id: int
    to_entity_id: int
    from_entity_name: Optional[str] = None
    to_entity_name: Optional[str] = None
    type: str
    description: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    strength: float = Field(ge=0.0, le=1.0)
    first_observed: Optional[datetime] = None
    last_observed: Optional[datetime] = None
    start_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    memory_tier: str  # 'shortterm' or 'longterm'
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class RetrievalResult(BaseModel):
    """Result from memory retrieval."""

    query: str
    active_memories: List[ActiveMemory] = Field(default_factory=list)
    shortterm_chunks: List[ShorttermMemoryChunk] = Field(default_factory=list)
    longterm_chunks: List[LongtermMemoryChunk] = Field(default_factory=list)
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    synthesized_response: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class ChunkUpdateData(BaseModel):
    """Data for updating a chunk."""

    chunk_id: int
    new_content: str
    metadata: Optional[Dict[str, Any]] = None


class NewChunkData(BaseModel):
    """Data for creating a new chunk."""

    content: str
    chunk_order: int
    metadata: Optional[Dict[str, Any]] = None


class EntityUpdateData(BaseModel):
    """Data for updating an entity."""

    entity_id: int
    name: Optional[str] = None
    description: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class RelationshipUpdateData(BaseModel):
    """Data for updating a relationship."""

    relationship_id: int
    type: Optional[str] = None
    description: Optional[str] = None
    confidence: Optional[float] = None
    strength: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

