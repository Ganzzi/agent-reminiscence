"""
Tests for Pydantic data models (database/models.py).
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from agent_reminiscence.database.models import (
    ActiveMemory,
    ShorttermMemory,
    ShorttermMemoryChunk,
    LongtermMemoryChunk,
    LongtermMemory,
    ShorttermEntity,
    ShorttermRelationship,
    LongtermEntity,
    LongtermRelationship,
    RetrievalResult,
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
            template_content={
                "template": {"id": "test", "name": "Test"},
                "sections": [{"id": "summary", "description": "Test summary"}]
            },
            sections={"summary": {"content": "Test summary", "update_count": 0}},
            metadata={"key": "value"},
            created_at=now,
            updated_at=now,
        )
        assert memory.external_id == "test-123"
        assert memory.title == "Test Memory"
        assert memory.template_content["template"]["id"] == "test"
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
            template_content={"template": {"id": "test", "name": "Test"}},
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
        chunk = ShorttermMemoryChunk(id=1, shortterm_memory_id=1, content="Test chunk", metadata={})
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
            metadata={},
        )
        assert chunk.shortterm_memory_id == 1
        assert chunk.content == "This is a test chunk."

    def test_chunk_with_scores(self):
        """Test chunk with similarity scores."""
        chunk = ShorttermMemoryChunk(
            id=1,
            shortterm_memory_id=1,
            content="Test",
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
            importance=0.9,
            start_date=now,
            metadata={},
        )
        assert chunk.external_id == "test-123"
        assert chunk.importance == 0.9

    def test_longterm_chunk_temporal(self):
        """Test longterm chunk temporal tracking."""
        now = datetime.now(timezone.utc)
        chunk = LongtermMemoryChunk(
            id=1,
            external_id="test-123",
            content="Test",
            importance=0.8,
            start_date=now,
        )
        assert chunk.start_date == now

    def test_longterm_chunk_validation(self):
        """Test longterm chunk validation."""
        now = datetime.now(timezone.utc)

        # Valid chunk
        chunk = LongtermMemoryChunk(
            id=1,
            external_id="test-123",
            content="Test",
            importance=0.8,
            start_date=now,
        )
        assert chunk.importance == 0.8


class TestLongtermMemory:
    """Test LongtermMemory model."""

    def test_longterm_memory_creation(self):
        """Test creating a longterm memory."""
        now = datetime.now(timezone.utc)
        chunk = LongtermMemoryChunk(
            id=1,
            external_id="test-123",
            content="Test",
            importance=0.8,
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
            id="1",
            external_id="test-123",
            shortterm_memory_id=1,
            name="Python",
            types=["TECHNOLOGY"],
            description="Programming language",
            importance=0.9,
            metadata={},
        )
        assert entity.name == "Python"
        assert entity.types == ["TECHNOLOGY"]
        assert entity.importance == 0.9

    def test_shortterm_entity_validation(self):
        """Test shortterm entity validation."""
        now = datetime.now(timezone.utc)

        # Invalid importance (must be 0-1)
        with pytest.raises(ValidationError):
            ShorttermEntity(
                id="1",
                external_id="test-123",
                shortterm_memory_id=1,
                name="Test",
                types=["TYPE"],
                importance=2.0,
            )


class TestShorttermRelationship:
    """Test ShorttermRelationship model."""

    def test_shortterm_relationship_creation(self):
        """Test creating a shortterm relationship."""
        now = datetime.now(timezone.utc)
        rel = ShorttermRelationship(
            id="1",
            external_id="test-123",
            shortterm_memory_id=1,
            from_entity_id="1",
            to_entity_id="2",
            from_entity_name="Entity1",
            to_entity_name="Entity2",
            types=["RELATED_TO"],
            description="Test relationship",
            importance=0.8,
            metadata={},
        )
        assert rel.from_entity_id == "1"
        assert rel.to_entity_id == "2"
        assert rel.types == ["RELATED_TO"]
        assert rel.importance == 0.8

    def test_shortterm_relationship_validation(self):
        """Test shortterm relationship validation."""
        now = datetime.now(timezone.utc)

        # Invalid importance (must be 0-1)
        with pytest.raises(ValidationError):
            ShorttermRelationship(
                id="1",
                external_id="test-123",
                shortterm_memory_id=1,
                from_entity_id="1",
                to_entity_id="2",
                types=["RELATED_TO"],
                importance=1.5,
            )


class TestLongtermEntity:
    """Test LongtermEntity model."""

    def test_longterm_entity_creation(self):
        """Test creating a longterm entity."""
        now = datetime.now(timezone.utc)
        entity = LongtermEntity(
            id="1",
            external_id="test-123",
            name="AI",
            types=["CONCEPT"],
            description="Artificial Intelligence",
            importance=0.9,
            metadata={},
        )
        assert entity.name == "AI"
        assert entity.types == ["CONCEPT"]
        assert entity.importance == 0.9


class TestLongtermRelationship:
    """Test LongtermRelationship model."""

    def test_longterm_relationship_creation(self):
        """Test creating a longterm relationship."""
        now = datetime.now(timezone.utc)
        rel = LongtermRelationship(
            id="1",
            external_id="test-123",
            from_entity_id="1",
            to_entity_id="2",
            from_entity_name="Entity1",
            to_entity_name="Entity2",
            types=["USES"],
            description="Uses relationship",
            importance=0.9,
            start_date=now,
            metadata={},
        )
        assert rel.types == ["USES"]
        assert rel.importance == 0.9


# NOTE: The following test classes are commented out because they test models that no longer exist
# (Entity, Relationship, ChunkUpdateData, NewChunkData, EntityUpdateData, RelationshipUpdateData)
# These were removed during model refactoring. Tests should be updated to test actual models.

# class TestEntity:
#     """Test Entity model (generic)."""
#     (tests omitted - model doesn't exist)

# class TestRelationship:
#     """Test Relationship model (generic)."""
#     (tests omitted - model doesn't exist)

# The RetrievalResult tests that use Entity have also been removed.

# class TestChunkUpdateData:
#     """Test ChunkUpdateData model."""
#     (tests omitted - model doesn't exist)

# class TestNewChunkData:
#     """Test NewChunkData model."""
#     (tests omitted - model doesn't exist)

# class TestEntityUpdateData:
#     """Test EntityUpdateData model."""
#     (tests omitted - model doesn't exist)

# class TestRelationshipUpdateData:
#     """Test RelationshipUpdateData model."""
#     (tests omitted - model doesn't exist)


