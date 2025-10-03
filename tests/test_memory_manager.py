"""
Tests for Memory Manager (services/memory_manager.py).

Note: These are unit tests with mocked dependencies.
Integration tests are in test_integration.py.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from agent_mem.services.memory_manager import MemoryManager
from agent_mem.database.models import ActiveMemory


class TestMemoryManagerInit:
    """Test MemoryManager initialization."""

    def test_initialization(self, test_config):
        """Test manager initialization."""
        with (
            patch("agent_mem.services.memory_manager.PostgreSQLManager"),
            patch("agent_mem.services.memory_manager.Neo4jManager"),
            patch("agent_mem.services.memory_manager.EmbeddingService"),
        ):

            manager = MemoryManager(test_config)

            assert manager.config == test_config
            assert not manager._initialized


class TestMemoryManagerActiveMemory:
    """Test active memory operations."""

    @pytest.mark.asyncio
    async def test_create_active_memory(self, test_config):
        """Test creating active memory."""
        with (
            patch("agent_mem.services.memory_manager.PostgreSQLManager"),
            patch("agent_mem.services.memory_manager.Neo4jManager"),
            patch("agent_mem.services.memory_manager.EmbeddingService"),
        ):

            manager = MemoryManager(test_config)
            manager._initialized = True  # Bypass initialization check

            # Mock the repository
            mock_repo = MagicMock()
            mock_memory = ActiveMemory(
                id=1,
                external_id="test-123",
                title="Test",
                template_content="# Template",
                sections={"summary": {"content": "Test", "update_count": 0}},
                metadata={},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            mock_repo.create = AsyncMock(return_value=mock_memory)
            manager.active_repo = mock_repo

            result = await manager.create_active_memory(
                external_id="test-123",
                title="Test",
                template_content="# Template",
                initial_sections={"summary": {"content": "Test", "update_count": 0}},
                metadata={},
            )

            assert result == mock_memory
            mock_repo.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_active_memory(self, test_config):
        """Test getting active memories."""
        with (
            patch("agent_mem.services.memory_manager.PostgreSQLManager"),
            patch("agent_mem.services.memory_manager.Neo4jManager"),
            patch("agent_mem.services.memory_manager.EmbeddingService"),
        ):

            manager = MemoryManager(test_config)
            manager._initialized = True  # Bypass initialization check

            # Mock the repository
            mock_repo = MagicMock()
            mock_memories = [
                ActiveMemory(
                    id=1,
                    external_id="test-123",
                    title="Test 1",
                    template_content="# Template",
                    sections={"summary": {"content": "Test 1", "update_count": 0}},
                    metadata={},
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
            ]
            mock_repo.get_all_by_external_id = AsyncMock(return_value=mock_memories)
            manager.active_repo = mock_repo

            result = await manager.get_active_memories("test-123")

            assert len(result) == 1
            assert result == mock_memories

    @pytest.mark.asyncio
    async def test_update_active_memory_no_consolidation(self, test_config):
        """Test updating active memory without triggering consolidation."""
        with (
            patch("agent_mem.services.memory_manager.PostgreSQLManager"),
            patch("agent_mem.services.memory_manager.Neo4jManager"),
            patch("agent_mem.services.memory_manager.EmbeddingService"),
        ):

            manager = MemoryManager(test_config)
            manager._initialized = True  # Bypass initialization check

            # Replace config with a MagicMock that allows attribute assignment
            mock_config = MagicMock()
            mock_config.active_memory_update_threshold = 5
            manager.config = mock_config

            # Mock the repository
            mock_repo = MagicMock()
            memory_id = 1
            updated_memory = ActiveMemory(
                id=memory_id,
                external_id="test-123",
                title="Test",
                template_content="# Template",
                sections={"summary": {"content": "Updated", "update_count": 2}},
                metadata={},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            mock_repo.update_section = AsyncMock(return_value=updated_memory)
            manager.active_repo = mock_repo

            result = await manager.update_active_memory_section(
                external_id="test-123",
                memory_id=memory_id,
                section_id="summary",
                new_content="Updated",
            )

            assert result == updated_memory

    @pytest.mark.asyncio
    async def test_update_active_memory_triggers_consolidation(self, test_config):
        """Test that updating active memory triggers consolidation after threshold."""
        with (
            patch("agent_mem.services.memory_manager.PostgreSQLManager"),
            patch("agent_mem.services.memory_manager.Neo4jManager"),
            patch("agent_mem.services.memory_manager.EmbeddingService"),
        ):

            manager = MemoryManager(test_config)
            manager._initialized = True  # Bypass initialization check

            # Replace config with a MagicMock that allows attribute assignment
            mock_config = MagicMock()
            mock_config.active_memory_update_threshold = 3
            manager.config = mock_config

            # Mock consolidation method
            manager._consolidate_to_shortterm = AsyncMock()

            # Mock the repository
            mock_repo = MagicMock()
            memory_id = 1
            updated_memory = ActiveMemory(
                id=memory_id,
                external_id="test-123",
                title="Test",
                template_content="# Template",
                sections={"summary": {"content": "Updated", "update_count": 4}},
                metadata={},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            mock_repo.update_section = AsyncMock(return_value=updated_memory)
            manager.active_repo = mock_repo

            await manager.update_active_memory_section(
                external_id="test-123",
                memory_id=memory_id,
                section_id="summary",
                new_content="Updated",
            )

            # Check that consolidation was triggered
            manager._consolidate_to_shortterm.assert_called_once_with("test-123", memory_id)


class TestMemoryManagerConsolidation:
    """Test consolidation workflow."""

    @pytest.mark.asyncio
    async def test_consolidate_to_shortterm(self, test_config):
        """Test consolidation workflow (internal method)."""
        with (
            patch("agent_mem.services.memory_manager.PostgreSQLManager"),
            patch("agent_mem.services.memory_manager.Neo4jManager"),
            patch("agent_mem.services.memory_manager.EmbeddingService"),
        ):

            manager = MemoryManager(test_config)
            manager._initialized = True  # Bypass initialization check

            # Verify that the internal consolidation method exists
            assert hasattr(manager, "_consolidate_to_shortterm")
            assert callable(manager._consolidate_to_shortterm)


class TestMemoryManagerPromotion:
    """Test promotion workflow."""

    @pytest.mark.asyncio
    async def test_promote_to_longterm(self, test_config):
        """Test promotion to longterm memory (internal method)."""
        with (
            patch("agent_mem.services.memory_manager.PostgreSQLManager"),
            patch("agent_mem.services.memory_manager.Neo4jManager"),
            patch("agent_mem.services.memory_manager.EmbeddingService"),
        ):

            manager = MemoryManager(test_config)
            manager._initialized = True  # Bypass initialization check

            # Verify that the internal promotion method exists
            assert hasattr(manager, "_promote_to_longterm")
            assert callable(manager._promote_to_longterm)


class TestMemoryManagerRetrieval:
    """Test memory retrieval."""

    @pytest.mark.asyncio
    async def test_retrieve_memories_basic(self, test_config):
        """Test basic memory retrieval."""
        with (
            patch("agent_mem.services.memory_manager.PostgreSQLManager"),
            patch("agent_mem.services.memory_manager.Neo4jManager"),
            patch("agent_mem.services.memory_manager.EmbeddingService"),
        ):

            from agent_mem.database.models import RetrievalResult

            manager = MemoryManager(test_config)
            manager._initialized = True  # Bypass initialization check

            # Mock all repositories with AsyncMock for async methods
            manager.active_repo = MagicMock()
            manager.active_repo.get_all_by_external_id = AsyncMock(return_value=[])

            manager.shortterm_repo = MagicMock()
            manager.shortterm_repo.hybrid_search = AsyncMock(return_value=[])

            manager.longterm_repo = MagicMock()
            manager.longterm_repo.hybrid_search = AsyncMock(return_value=[])

            # Mock embedding service
            manager.embedding_service = MagicMock()
            manager.embedding_service.get_embedding = AsyncMock(return_value=[0.1] * 768)

            # Mock the retriever agent
            mock_agent = MagicMock()
            mock_agent.determine_strategy = AsyncMock(
                side_effect=Exception("Force basic retrieval")
            )
            manager.retriever_agent = mock_agent

            result = await manager.retrieve_memories(query="test query", external_id="test-123")

            assert isinstance(result, RetrievalResult)
            assert result.query == "test query"


class TestMemoryManagerHelpers:
    """Test helper methods."""

    @pytest.mark.asyncio
    async def test_calculate_semantic_similarity(self, test_config):
        """Test semantic similarity calculation."""
        with (
            patch("agent_mem.services.memory_manager.PostgreSQLManager"),
            patch("agent_mem.services.memory_manager.Neo4jManager"),
            patch("agent_mem.services.memory_manager.EmbeddingService") as mock_embedding,
        ):

            # Setup mock embedding service
            mock_embedding_instance = mock_embedding.return_value
            mock_embedding_instance.get_embedding = AsyncMock(
                side_effect=[
                    [1.0, 0.0, 0.0],  # First embedding
                    [1.0, 0.0, 0.0],  # Second embedding
                ]
            )

            manager = MemoryManager(test_config)

            # Test calculation with text inputs
            similarity = await manager._calculate_semantic_similarity("test1", "test2")

            assert 0.0 <= similarity <= 1.0

    def test_calculate_entity_overlap_exact_match(self, test_config):
        """Test entity overlap calculation with exact match."""
        with (
            patch("agent_mem.services.memory_manager.PostgreSQLManager"),
            patch("agent_mem.services.memory_manager.Neo4jManager"),
            patch("agent_mem.services.memory_manager.EmbeddingService"),
        ):

            manager = MemoryManager(test_config)

            # Create mock entities with name and type attributes
            entity1 = MagicMock()
            entity1.name = "Python"
            entity1.type = "TECHNOLOGY"

            entity2 = MagicMock()
            entity2.name = "Python"
            entity2.type = "TECHNOLOGY"

            overlap = manager._calculate_entity_overlap(entity1, entity2)

            assert overlap == 1.0

    def test_calculate_entity_overlap_no_match(self, test_config):
        """Test entity overlap with no match."""
        with (
            patch("agent_mem.services.memory_manager.PostgreSQLManager"),
            patch("agent_mem.services.memory_manager.Neo4jManager"),
            patch("agent_mem.services.memory_manager.EmbeddingService"),
        ):

            manager = MemoryManager(test_config)

            # Create mock entities with different names
            entity1 = MagicMock()
            entity1.name = "Python"
            entity1.type = "TECHNOLOGY"

            entity2 = MagicMock()
            entity2.name = "Java"
            entity2.type = "TECHNOLOGY"

            overlap = manager._calculate_entity_overlap(entity1, entity2)

            # Same type but different name = 0.5 (partial match)
            assert overlap == 0.5

    def test_calculate_importance_with_multiplier(self, test_config):
        """Test importance calculation with type multiplier."""
        with (
            patch("agent_mem.services.memory_manager.PostgreSQLManager"),
            patch("agent_mem.services.memory_manager.Neo4jManager"),
            patch("agent_mem.services.memory_manager.EmbeddingService"),
        ):

            manager = MemoryManager(test_config)

            # Create mock entity with confidence and type
            entity = MagicMock()
            entity.confidence = 0.5
            entity.type = "PERSON"  # Has multiplier of 1.2

            importance = manager._calculate_importance(entity)

            # 0.5 * 1.2 = 0.6
            assert importance == 0.6

    def test_calculate_importance_capped(self, test_config):
        """Test importance calculation is capped at 1.0."""
        with (
            patch("agent_mem.services.memory_manager.PostgreSQLManager"),
            patch("agent_mem.services.memory_manager.Neo4jManager"),
            patch("agent_mem.services.memory_manager.EmbeddingService"),
        ):

            manager = MemoryManager(test_config)

            # Create mock entity with high confidence and type multiplier
            entity = MagicMock()
            entity.confidence = 0.9
            entity.type = "PERSON"  # Has multiplier of 1.2

            importance = manager._calculate_importance(entity)

            # 0.9 * 1.2 = 1.08, but should be capped at 1.0
            assert importance == 1.0

