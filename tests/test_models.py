"""
Tests for Pydantic data models (database/models.py).
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from agent_mem.database.models import (
    ActiveMemory,
    ShorttermMemory,
    ShorttermMemoryChunk,
    LongtermMemoryChunk,
    LongtermMemory,
    ShorttermEntity,
    ShorttermRelationship,
    LongtermEntity,
    LongtermRelationship,
    Entity,
    Relationship,
    RetrievalResult,
    ChunkUpdateData,
    NewChunkData,
    EntityUpdateData,
    RelationshipUpdateData,
)


class TestActiveMemory:
    """Test ActiveMemory model."""

    def test_active_memory_creation(self):
        """Test creating an active memory."""
        now = datetime.now(timezone.utc)
        memory = ActiveMemory(
            id=1,
            external_id="test-123",
            title="Test Memory",
            template_content="# Template",
            sections={"summary": {"content": "Test summary", "update_count": 0}},
            metadata={"key": "value"},
            created_at=now,
            updated_at=now,
        )
        assert memory.external_id == "test-123"
        assert memory.title == "Test Memory"
        assert memory.template_content == "# Template"
        assert memory.sections == {"summary": {"content": "Test summary", "update_count": 0}}
        assert memory.metadata == {"key": "value"}
        assert isinstance(memory.created_at, datetime)

    def test_active_memory_validation(self):
        """Test active memory validation."""
        now = datetime.now(timezone.utc)
        # Missing required fields
        with pytest.raises(ValidationError):
            ActiveMemory()

        with pytest.raises(ValidationError):
            ActiveMemory(external_id="test-123")

        # Valid memory
        memory = ActiveMemory(
            id=1,
            external_id="test-123",
            title="Test",
            template_content="# Test",
            created_at=now,
            updated_at=now,
        )
        assert memory.external_id == "test-123"


class TestShorttermMemory:
    """Test ShorttermMemory model."""

    def test_shortterm_memory_creation(self):
        """Test creating a shortterm memory."""
        now = datetime.now(timezone.utc)
        memory = ShorttermMemory(
            id=1,
            external_id="test-123",
            title="Test Memory",
            summary="Test summary",
            metadata={"key": "value"},
            created_at=now,
            last_updated=now,
        )
        assert memory.external_id == "test-123"
        assert memory.title == "Test Memory"
        assert memory.summary == "Test summary"
        assert memory.metadata == {"key": "value"}
        assert isinstance(memory.created_at, datetime)

    def test_shortterm_memory_with_chunks(self):
        """Test shortterm memory with chunks."""
        now = datetime.now(timezone.utc)
        chunk = ShorttermMemoryChunk(
            id=1, shortterm_memory_id=1, content="Test chunk", chunk_order=0, metadata={}
        )
        memory = ShorttermMemory(
            id=1,
            external_id="test-123",
            title="Test",
            chunks=[chunk],
            created_at=now,
            last_updated=now,
        )
        assert len(memory.chunks) == 1
        assert memory.chunks[0].content == "Test chunk"


class TestShorttermMemoryChunk:
    """Test ShorttermMemoryChunk model."""

    def test_chunk_creation(self):
        """Test creating a chunk."""
        chunk = ShorttermMemoryChunk(
            id=1,
            shortterm_memory_id=1,
            content="This is a test chunk.",
            chunk_order=0,
            metadata={},
        )
        assert chunk.shortterm_memory_id == 1
        assert chunk.content == "This is a test chunk."
        assert chunk.chunk_order == 0

    def test_chunk_with_scores(self):
        """Test chunk with similarity scores."""
        chunk = ShorttermMemoryChunk(
            id=1,
            shortterm_memory_id=1,
            content="Test",
            chunk_order=0,
            similarity_score=0.95,
            bm25_score=10.5,
            metadata={"source": "test"},
        )
        assert chunk.similarity_score == 0.95
        assert chunk.bm25_score == 10.5
        assert chunk.metadata == {"source": "test"}

    def test_chunk_validation(self):
        """Test chunk validation."""
        # Missing required fields
        with pytest.raises(ValidationError):
            ShorttermMemoryChunk()

        # Valid chunk
        chunk = ShorttermMemoryChunk(
            id=1,
            shortterm_memory_id=1,
            content="Test",
            chunk_order=0,
        )
        assert chunk.content == "Test"


class TestLongtermMemoryChunk:
    """Test LongtermMemoryChunk model."""

    def test_longterm_chunk_creation(self):
        """Test creating a longterm chunk."""
        now = datetime.now(timezone.utc)
        chunk = LongtermMemoryChunk(
            id=1,
            external_id="test-123",
            shortterm_memory_id=1,
            content="Validated knowledge",
            chunk_order=0,
            confidence_score=0.9,
            start_date=now,
            metadata={},
        )
        assert chunk.external_id == "test-123"
        assert chunk.confidence_score == 0.9
        assert chunk.end_date is None

    def test_longterm_chunk_temporal(self):
        """Test longterm chunk temporal tracking."""
        now = datetime.now(timezone.utc)
        chunk = LongtermMemoryChunk(
            id=1,
            external_id="test-123",
            content="Test",
            chunk_order=0,
            confidence_score=0.8,
            start_date=now,
            end_date=now,
        )
        assert chunk.start_date == now
        assert chunk.end_date == now

    def test_longterm_chunk_validation(self):
        """Test longterm chunk validation."""
        now = datetime.now(timezone.utc)

        # Valid chunk
        chunk = LongtermMemoryChunk(
            id=1,
            external_id="test-123",
            content="Test",
            chunk_order=0,
            confidence_score=0.8,
            start_date=now,
        )
        assert chunk.confidence_score == 0.8


class TestLongtermMemory:
    """Test LongtermMemory model."""

    def test_longterm_memory_creation(self):
        """Test creating a longterm memory."""
        now = datetime.now(timezone.utc)
        chunk = LongtermMemoryChunk(
            id=1,
            external_id="test-123",
            content="Test",
            chunk_order=0,
            confidence_score=0.8,
            start_date=now,
        )
        memory = LongtermMemory(
            chunks=[chunk],
            external_id="test-123",
            metadata={"key": "value"},
        )
        assert len(memory.chunks) == 1
        assert memory.external_id == "test-123"
        assert memory.metadata == {"key": "value"}


class TestShorttermEntity:
    """Test ShorttermEntity model."""

    def test_shortterm_entity_creation(self):
        """Test creating a shortterm entity."""
        now = datetime.now(timezone.utc)
        entity = ShorttermEntity(
            id=1,
            external_id="test-123",
            shortterm_memory_id=1,
            name="Python",
            type="TECHNOLOGY",
            description="Programming language",
            confidence=0.9,
            first_seen=now,
            last_seen=now,
            metadata={},
        )
        assert entity.name == "Python"
        assert entity.type == "TECHNOLOGY"
        assert entity.confidence == 0.9

    def test_shortterm_entity_validation(self):
        """Test shortterm entity validation."""
        now = datetime.now(timezone.utc)

        # Invalid confidence (must be 0-1)
        with pytest.raises(ValidationError):
            ShorttermEntity(
                id=1,
                external_id="test-123",
                shortterm_memory_id=1,
                name="Test",
                type="TYPE",
                confidence=2.0,
                first_seen=now,
                last_seen=now,
            )


class TestShorttermRelationship:
    """Test ShorttermRelationship model."""

    def test_shortterm_relationship_creation(self):
        """Test creating a shortterm relationship."""
        now = datetime.now(timezone.utc)
        rel = ShorttermRelationship(
            id=1,
            external_id="test-123",
            shortterm_memory_id=1,
            from_entity_id=1,
            to_entity_id=2,
            from_entity_name="Entity1",
            to_entity_name="Entity2",
            type="RELATED_TO",
            description="Test relationship",
            confidence=0.8,
            strength=0.7,
            first_observed=now,
            last_observed=now,
            metadata={},
        )
        assert rel.from_entity_id == 1
        assert rel.to_entity_id == 2
        assert rel.type == "RELATED_TO"
        assert rel.confidence == 0.8
        assert rel.strength == 0.7

    def test_shortterm_relationship_validation(self):
        """Test shortterm relationship validation."""
        now = datetime.now(timezone.utc)

        # Invalid strength (must be 0-1)
        with pytest.raises(ValidationError):
            ShorttermRelationship(
                id=1,
                external_id="test-123",
                shortterm_memory_id=1,
                from_entity_id=1,
                to_entity_id=2,
                type="RELATED_TO",
                confidence=0.8,
                strength=1.5,
                first_observed=now,
                last_observed=now,
            )


class TestLongtermEntity:
    """Test LongtermEntity model."""

    def test_longterm_entity_creation(self):
        """Test creating a longterm entity."""
        now = datetime.now(timezone.utc)
        entity = LongtermEntity(
            id=1,
            external_id="test-123",
            name="AI",
            type="CONCEPT",
            description="Artificial Intelligence",
            confidence=0.95,
            importance=0.9,
            first_seen=now,
            last_seen=now,
            metadata={},
        )
        assert entity.name == "AI"
        assert entity.type == "CONCEPT"
        assert entity.confidence == 0.95
        assert entity.importance == 0.9


class TestLongtermRelationship:
    """Test LongtermRelationship model."""

    def test_longterm_relationship_creation(self):
        """Test creating a longterm relationship."""
        now = datetime.now(timezone.utc)
        rel = LongtermRelationship(
            id=1,
            external_id="test-123",
            from_entity_id=1,
            to_entity_id=2,
            from_entity_name="Entity1",
            to_entity_name="Entity2",
            type="USES",
            description="Uses relationship",
            confidence=0.9,
            strength=0.85,
            start_date=now,
            last_updated=now,
            metadata={},
        )
        assert rel.type == "USES"
        assert rel.confidence == 0.9
        assert rel.strength == 0.85


class TestEntity:
    """Test Entity model (generic)."""

    def test_entity_creation(self):
        """Test creating a generic entity."""
        now = datetime.now(timezone.utc)
        entity = Entity(
            id=1,
            external_id="test-123",
            name="Python",
            type="TECHNOLOGY",
            description="Programming language",
            confidence=0.9,
            importance=0.85,
            first_seen=now,
            last_seen=now,
            memory_tier="shortterm",
            metadata={},
        )
        assert entity.name == "Python"
        assert entity.type == "TECHNOLOGY"
        assert entity.confidence == 0.9
        assert entity.memory_tier == "shortterm"

    def test_entity_validation(self):
        """Test entity validation."""
        now = datetime.now(timezone.utc)

        # Invalid confidence
        with pytest.raises(ValidationError):
            Entity(
                id=1,
                external_id="test-123",
                name="Test",
                type="TYPE",
                confidence=2.0,
                first_seen=now,
                last_seen=now,
                memory_tier="shortterm",
            )


class TestRelationship:
    """Test Relationship model (generic)."""

    def test_relationship_creation(self):
        """Test creating a generic relationship."""
        now = datetime.now(timezone.utc)
        rel = Relationship(
            id=1,
            external_id="test-123",
            from_entity_id=1,
            to_entity_id=2,
            from_entity_name="Entity1",
            to_entity_name="Entity2",
            type="RELATED_TO",
            description="Test",
            confidence=0.8,
            strength=0.7,
            first_observed=now,
            last_observed=now,
            memory_tier="shortterm",
            metadata={},
        )
        assert rel.from_entity_id == 1
        assert rel.to_entity_id == 2
        assert rel.type == "RELATED_TO"
        assert rel.memory_tier == "shortterm"


class TestRetrievalResult:
    """Test RetrievalResult model."""

    def test_retrieval_result_creation(self):
        """Test creating a retrieval result."""
        now = datetime.now(timezone.utc)
        result = RetrievalResult(
            query="test query",
            active_memories=[],
            shortterm_chunks=[],
            longterm_chunks=[],
            entities=[],
            relationships=[],
            synthesized_response="Test response",
        )
        assert result.query == "test query"
        assert result.synthesized_response == "Test response"
        assert len(result.active_memories) == 0

    def test_retrieval_result_with_data(self):
        """Test retrieval result with actual data."""
        now = datetime.now(timezone.utc)

        active_mem = ActiveMemory(
            id=1,
            external_id="test-123",
            title="Test",
            template_content="# Test",
            created_at=now,
            updated_at=now,
        )

        chunk = ShorttermMemoryChunk(
            id=1,
            shortterm_memory_id=1,
            content="Test chunk",
            chunk_order=0,
            similarity_score=0.95,
        )

        entity = Entity(
            id=1,
            external_id="test-123",
            name="Test",
            type="TYPE",
            confidence=0.9,
            first_seen=now,
            last_seen=now,
            memory_tier="shortterm",
        )

        result = RetrievalResult(
            query="test",
            active_memories=[active_mem],
            shortterm_chunks=[chunk],
            entities=[entity],
        )

        assert len(result.active_memories) == 1
        assert len(result.shortterm_chunks) == 1
        assert len(result.entities) == 1


class TestChunkUpdateData:
    """Test ChunkUpdateData model."""

    def test_chunk_update_data(self):
        """Test creating chunk update data."""
        data = ChunkUpdateData(
            chunk_id=1,
            new_content="Updated content",
            metadata={"updated": True},
        )
        assert data.chunk_id == 1
        assert data.new_content == "Updated content"
        assert data.metadata == {"updated": True}


class TestNewChunkData:
    """Test NewChunkData model."""

    def test_new_chunk_data(self):
        """Test creating new chunk data."""
        data = NewChunkData(
            content="New chunk content",
            chunk_order=5,
            metadata={"source": "test"},
        )
        assert data.content == "New chunk content"
        assert data.chunk_order == 5
        assert data.metadata == {"source": "test"}


class TestEntityUpdateData:
    """Test EntityUpdateData model."""

    def test_entity_update_data(self):
        """Test creating entity update data."""
        data = EntityUpdateData(
            entity_id=1,
            name="Updated Name",
            description="Updated description",
            confidence=0.95,
            metadata={"updated": True},
        )
        assert data.entity_id == 1
        assert data.name == "Updated Name"
        assert data.confidence == 0.95


class TestRelationshipUpdateData:
    """Test RelationshipUpdateData model."""

    def test_relationship_update_data(self):
        """Test creating relationship update data."""
        data = RelationshipUpdateData(
            relationship_id=1,
            type="NEW_TYPE",
            description="Updated",
            confidence=0.9,
            strength=0.85,
            metadata={"updated": True},
        )
        assert data.relationship_id == 1
        assert data.type == "NEW_TYPE"
        assert data.confidence == 0.9
        assert data.strength == 0.85

