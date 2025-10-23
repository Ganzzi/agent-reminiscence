"""
Tests for Memory Manager (services/memory_manager.py).

Note: These are unit tests with mocked dependencies.
Integration tests are in test_integration.py.
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from agent_reminiscence.services.memory_manager import MemoryManager
from agent_reminiscence.database.models import ActiveMemory


class TestMemoryManagerInit:
    """Test MemoryManager initialization."""

    def test_initialization(self, test_config):
        """Test manager initialization."""
        with (
            patch("agent_reminiscence.services.memory_manager.PostgreSQLManager"),
            patch("agent_reminiscence.services.memory_manager.Neo4jManager"),
            patch("agent_reminiscence.services.memory_manager.EmbeddingService"),
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
            patch("agent_reminiscence.services.memory_manager.PostgreSQLManager"),
            patch("agent_reminiscence.services.memory_manager.Neo4jManager"),
            patch("agent_reminiscence.services.memory_manager.EmbeddingService"),
        ):

            manager = MemoryManager(test_config)
            manager._initialized = True  # Bypass initialization check

            # Mock the repository
            mock_repo = MagicMock()
            mock_memory = ActiveMemory(
                id=1,
                external_id="test-123",
                title="Test",
                template_content={
                    "template": {"id": "test-template", "name": "Test Template"},
                    "sections": [{"id": "summary", "description": "Summary section"}]
                },
                sections={
                    "summary": {
                        "content": "Test", 
                        "update_count": 0,
                        "awake_update_count": 0,
                        "last_updated": None
                    }
                },
                metadata={},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            mock_repo.create = AsyncMock(return_value=mock_memory)
            manager.active_repo = mock_repo

            result = await manager.create_active_memory(
                external_id="test-123",
                title="Test",
                template_content={
                    "template": {"id": "test-template", "name": "Test Template"},
                    "sections": [{"id": "summary", "description": "Summary section"}]
                },
                initial_sections={
                    "summary": {
                        "content": "Test", 
                        "update_count": 0,
                        "awake_update_count": 0,
                        "last_updated": None
                    }
                },
                metadata={},
            )

            assert result == mock_memory
            mock_repo.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_active_memory(self, test_config):
        """Test getting active memories."""
        with (
            patch("agent_reminiscence.services.memory_manager.PostgreSQLManager"),
            patch("agent_reminiscence.services.memory_manager.Neo4jManager"),
            patch("agent_reminiscence.services.memory_manager.EmbeddingService"),
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
                    template_content={
                        "template": {"id": "test-template", "name": "Test Template"},
                        "sections": [{"id": "summary", "description": "Summary section"}]
                    },
                    sections={
                        "summary": {
                            "content": "Test 1", 
                            "update_count": 0,
                            "awake_update_count": 0,
                            "last_updated": None
                        }
                    },
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
            patch("agent_reminiscence.services.memory_manager.PostgreSQLManager"),
            patch("agent_reminiscence.services.memory_manager.Neo4jManager"),
            patch("agent_reminiscence.services.memory_manager.EmbeddingService"),
        ):

            manager = MemoryManager(test_config)
            manager._initialized = True  # Bypass initialization check

            # Replace config with a MagicMock that allows attribute assignment
            mock_config = MagicMock()
            mock_config.consolidation_threshold = 5
            mock_config.avg_section_update_count_for_consolidation = 3.0  # Add this property
            manager.config = mock_config

            # Mock the repository
            mock_repo = MagicMock()
            memory_id = 1
            updated_memory = ActiveMemory(
                id=memory_id,
                external_id="test-123",
                title="Test",
                template_content={
                    "template": {"id": "test-template", "name": "Test Template"},
                    "sections": [{"id": "summary", "description": "Summary section"}]
                },
                sections={
                    "summary": {
                        "content": "Updated", 
                        "update_count": 2,
                        "awake_update_count": 2,
                        "last_updated": datetime.now(timezone.utc)
                    }
                },
                metadata={},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            mock_repo.upsert_sections = AsyncMock(return_value=updated_memory)
            manager.active_repo = mock_repo

            result = await manager.update_active_memory_sections(
                external_id="test-123",
                memory_id=memory_id,
                sections=[{
                    "section_id": "summary", 
                    "new_content": "Updated",
                    "action": "replace"
                }],
            )

            assert result == updated_memory

    @pytest.mark.asyncio
    async def test_update_active_memory_triggers_consolidation(self, test_config):
        """Test that updating active memory triggers consolidation after threshold."""
        with (
            patch("agent_reminiscence.services.memory_manager.PostgreSQLManager"),
            patch("agent_reminiscence.services.memory_manager.Neo4jManager"),
            patch("agent_reminiscence.services.memory_manager.EmbeddingService"),
            patch("asyncio.create_task") as mock_create_task,
        ):

            manager = MemoryManager(test_config)
            manager._initialized = True  # Bypass initialization check

            # Configure consolidation threshold
            test_config.avg_section_update_count_for_consolidation = 3.0

            # Mock the repository
            mock_repo = MagicMock()
            memory_id = 1
            # 2 sections with 3 updates each = 6 total updates (>= 3*2=6 threshold)
            updated_memory = ActiveMemory(
                id=memory_id,
                external_id="test-123",
                title="Test",
                template_content={
                    "template": {"id": "test-template", "name": "Test Template"},
                    "sections": [
                        {"id": "summary", "description": "Summary section"},
                        {"id": "context", "description": "Context section"}
                    ]
                },
                sections={
                    "summary": {
                        "content": "Updated", 
                        "update_count": 3,
                        "awake_update_count": 3,
                        "last_updated": datetime.now(timezone.utc)
                    },
                    "context": {
                        "content": "Context", 
                        "update_count": 3,
                        "awake_update_count": 3,
                        "last_updated": datetime.now(timezone.utc)
                    },
                },
                metadata={},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            mock_repo.upsert_sections = AsyncMock(return_value=updated_memory)
            # Note: We'll assume reset happens in the background task
            manager.active_repo = mock_repo

            await manager.update_active_memory_sections(
                external_id="test-123",
                memory_id=memory_id,
                sections=[{"section_id": "summary", "new_content": "Updated"}],
            )

            # Check that consolidation task was created
            mock_create_task.assert_called_once()
            # Note: reset happens in the background consolidation task

    @pytest.mark.asyncio
    async def test_update_active_memory_below_threshold(self, test_config):
        """Test updating active memory below consolidation threshold."""
        with (
            patch("agent_reminiscence.services.memory_manager.PostgreSQLManager"),
            patch("agent_reminiscence.services.memory_manager.Neo4jManager"),
            patch("agent_reminiscence.services.memory_manager.EmbeddingService"),
            patch("asyncio.create_task") as mock_create_task,
        ):

            manager = MemoryManager(test_config)
            manager._initialized = True  # Bypass initialization check

            # Configure consolidation threshold
            test_config.avg_section_update_count_for_consolidation = 5.0

            # Mock the repository
            mock_repo = MagicMock()
            memory_id = 1
            # 2 sections with 2 updates each = 4 total updates (< 5*2=10 threshold)
            updated_memory = ActiveMemory(
                id=memory_id,
                external_id="test-123",
                title="Test",
                template_content={
                    "template": {"id": "test-template", "name": "Test Template"},
                    "sections": [
                        {"id": "summary", "description": "Summary section"},
                        {"id": "context", "description": "Context section"}
                    ]
                },
                sections={
                    "summary": {
                        "content": "Updated", 
                        "update_count": 2,
                        "awake_update_count": 2,
                        "last_updated": datetime.now(timezone.utc)
                    },
                    "context": {
                        "content": "Context", 
                        "update_count": 2,
                        "awake_update_count": 2,
                        "last_updated": datetime.now(timezone.utc)
                    },
                },
                metadata={},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            mock_repo.upsert_sections = AsyncMock(return_value=updated_memory)
            manager.active_repo = mock_repo

            result = await manager.update_active_memory_sections(
                external_id="test-123",
                memory_id=memory_id,
                sections=[{"section_id": "summary", "new_content": "Updated"}],
            )

            # Check that consolidation was NOT triggered
            mock_create_task.assert_not_called()
            assert result == updated_memory

    @pytest.mark.asyncio
    async def test_delete_active_memory_success(self, test_config):
        """Test successful deletion of active memory."""
        with (
            patch("agent_reminiscence.services.memory_manager.PostgreSQLManager"),
            patch("agent_reminiscence.services.memory_manager.Neo4jManager"),
            patch("agent_reminiscence.services.memory_manager.EmbeddingService"),
        ):

            manager = MemoryManager(test_config)
            manager._initialized = True  # Bypass initialization check

            # Mock the repository
            mock_repo = MagicMock()
            memory_id = 1
            mock_memory = ActiveMemory(
                id=memory_id,
                external_id="test-123",
                title="Test Memory",
                template_content={
                    "template": {"id": "test-template", "name": "Test Template"},
                    "sections": [{"id": "summary", "description": "Summary section"}]
                },
                sections={
                    "summary": {
                        "content": "Test", 
                        "update_count": 0,
                        "awake_update_count": 0,
                        "last_updated": None
                    }
                },
                metadata={},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            mock_repo.get_by_id = AsyncMock(return_value=mock_memory)
            mock_repo.delete = AsyncMock(return_value=True)
            manager.active_repo = mock_repo

            result = await manager.delete_active_memory(
                external_id="test-123",
                memory_id=memory_id,
            )

            assert result is True
            mock_repo.get_by_id.assert_called_once_with(memory_id)
            mock_repo.delete.assert_called_once_with(memory_id)

    @pytest.mark.asyncio
    async def test_delete_active_memory_not_found(self, test_config):
        """Test deletion when memory is not found."""
        with (
            patch("agent_reminiscence.services.memory_manager.PostgreSQLManager"),
            patch("agent_reminiscence.services.memory_manager.Neo4jManager"),
            patch("agent_reminiscence.services.memory_manager.EmbeddingService"),
        ):

            manager = MemoryManager(test_config)
            manager._initialized = True  # Bypass initialization check

            # Mock the repository
            mock_repo = MagicMock()
            memory_id = 999

            mock_repo.get_by_id = AsyncMock(return_value=None)
            manager.active_repo = mock_repo

            result = await manager.delete_active_memory(
                external_id="test-123",
                memory_id=memory_id,
            )

            assert result is False
            mock_repo.get_by_id.assert_called_once_with(memory_id)
            mock_repo.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_active_memory_wrong_agent(self, test_config):
        """Test deletion when memory belongs to different agent."""
        with (
            patch("agent_reminiscence.services.memory_manager.PostgreSQLManager"),
            patch("agent_reminiscence.services.memory_manager.Neo4jManager"),
            patch("agent_reminiscence.services.memory_manager.EmbeddingService"),
        ):

            manager = MemoryManager(test_config)
            manager._initialized = True  # Bypass initialization check

            # Mock the repository
            mock_repo = MagicMock()
            memory_id = 1
            mock_memory = ActiveMemory(
                id=memory_id,
                external_id="different-agent",  # Different external_id
                title="Test Memory",
                template_content={
                    "template": {"id": "test-template", "name": "Test Template"},
                    "sections": [{"id": "summary", "description": "Summary section"}]
                },
                sections={
                    "summary": {
                        "content": "Test", 
                        "update_count": 0,
                        "awake_update_count": 0,
                        "last_updated": None
                    }
                },
                metadata={},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            mock_repo.get_by_id = AsyncMock(return_value=mock_memory)
            manager.active_repo = mock_repo

            result = await manager.delete_active_memory(
                external_id="test-123",
                memory_id=memory_id,
            )

            assert result is False
            mock_repo.get_by_id.assert_called_once_with(memory_id)
            mock_repo.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_active_memory_delete_fails(self, test_config):
        """Test deletion when repository delete operation fails."""
        with (
            patch("agent_reminiscence.services.memory_manager.PostgreSQLManager"),
            patch("agent_reminiscence.services.memory_manager.Neo4jManager"),
            patch("agent_reminiscence.services.memory_manager.EmbeddingService"),
        ):

            manager = MemoryManager(test_config)
            manager._initialized = True  # Bypass initialization check

            # Mock the repository
            mock_repo = MagicMock()
            memory_id = 1
            mock_memory = ActiveMemory(
                id=memory_id,
                external_id="test-123",
                title="Test Memory",
                template_content={
                    "template": {"id": "test-template", "name": "Test Template"},
                    "sections": [{"id": "summary", "description": "Summary section"}]
                },
                sections={
                    "summary": {
                        "content": "Test", 
                        "update_count": 0,
                        "awake_update_count": 0,
                        "last_updated": None
                    }
                },
                metadata={},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            mock_repo.get_by_id = AsyncMock(return_value=mock_memory)
            mock_repo.delete = AsyncMock(return_value=False)  # Delete fails
            manager.active_repo = mock_repo

            result = await manager.delete_active_memory(
                external_id="test-123",
                memory_id=memory_id,
            )

            assert result is False
            mock_repo.get_by_id.assert_called_once_with(memory_id)
            mock_repo.delete.assert_called_once_with(memory_id)


class TestMemoryManagerConsolidation:
    """Test consolidation workflow."""

    @pytest.mark.asyncio
    async def test_consolidate_to_shortterm(self, test_config):
        """Test consolidation workflow (internal method)."""
        with (
            patch("agent_reminiscence.services.memory_manager.PostgreSQLManager"),
            patch("agent_reminiscence.services.memory_manager.Neo4jManager"),
            patch("agent_reminiscence.services.memory_manager.EmbeddingService"),
        ):

            manager = MemoryManager(test_config)
            manager._initialized = True  # Bypass initialization check

            # Verify that the internal consolidation method exists
            assert hasattr(manager, "_consolidate_to_shortterm")
            assert callable(manager._consolidate_to_shortterm)

    @pytest.mark.asyncio
    async def test_consolidate_with_lock_success(self, test_config):
        """Test successful consolidation with lock mechanism."""
        with (
            patch("agent_reminiscence.services.memory_manager.PostgreSQLManager"),
            patch("agent_reminiscence.services.memory_manager.Neo4jManager"),
            patch("agent_reminiscence.services.memory_manager.EmbeddingService"),
        ):

            manager = MemoryManager(test_config)
            manager._initialized = True  # Bypass initialization check

            # Mock the internal consolidation method
            manager._consolidate_to_shortterm = AsyncMock()

            # Test consolidation with lock
            await manager._consolidate_with_lock("test-agent", 123)

            # Verify consolidation was called
            manager._consolidate_to_shortterm.assert_called_once_with("test-agent", 123)

            # Verify lock was cleaned up (no longer in locks dict)
            assert 123 not in manager._consolidation_locks

    @pytest.mark.asyncio
    async def test_consolidate_with_lock_concurrent_access(self, test_config):
        """Test that concurrent consolidation attempts are handled correctly."""
        with (
            patch("agent_reminiscence.services.memory_manager.PostgreSQLManager"),
            patch("agent_reminiscence.services.memory_manager.Neo4jManager"),
            patch("agent_reminiscence.services.memory_manager.EmbeddingService"),
        ):

            manager = MemoryManager(test_config)
            manager._initialized = True  # Bypass initialization check

            # Create a slow consolidation method
            consolidation_called_count = 0

            async def slow_consolidation(external_id, memory_id):
                nonlocal consolidation_called_count
                consolidation_called_count += 1
                await asyncio.sleep(0.1)  # Simulate slow operation

            manager._consolidate_to_shortterm = slow_consolidation

            # Start two consolidation tasks concurrently for same memory
            task1 = asyncio.create_task(manager._consolidate_with_lock("test-agent", 123))
            task2 = asyncio.create_task(manager._consolidate_with_lock("test-agent", 123))

            # Wait for both to complete
            await asyncio.gather(task1, task2)

            # Only one consolidation should have been executed
            assert consolidation_called_count == 1

    @pytest.mark.asyncio
    async def test_consolidate_with_lock_error_handling(self, test_config):
        """Test error handling in consolidation with lock."""
        with (
            patch("agent_reminiscence.services.memory_manager.PostgreSQLManager"),
            patch("agent_reminiscence.services.memory_manager.Neo4jManager"),
            patch("agent_reminiscence.services.memory_manager.EmbeddingService"),
        ):

            manager = MemoryManager(test_config)
            manager._initialized = True  # Bypass initialization check

            # Mock consolidation to raise an exception
            manager._consolidate_to_shortterm = AsyncMock(
                side_effect=Exception("Consolidation failed")
            )

            # This should not raise an exception
            await manager._consolidate_with_lock("test-agent", 123)

            # Verify consolidation was attempted
            manager._consolidate_to_shortterm.assert_called_once_with("test-agent", 123)

            # Verify lock was cleaned up even after error
            assert 123 not in manager._consolidation_locks


class TestMemoryManagerPromotion:
    """Test promotion workflow."""

    @pytest.mark.asyncio
    async def test_promote_to_longterm(self, test_config):
        """Test promotion to longterm memory (internal method)."""
        with (
            patch("agent_reminiscence.services.memory_manager.PostgreSQLManager"),
            patch("agent_reminiscence.services.memory_manager.Neo4jManager"),
            patch("agent_reminiscence.services.memory_manager.EmbeddingService"),
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
            patch("agent_reminiscence.services.memory_manager.PostgreSQLManager"),
            patch("agent_reminiscence.services.memory_manager.Neo4jManager"),
            patch("agent_reminiscence.services.memory_manager.EmbeddingService"),
            patch("agent_reminiscence.services.memory_manager.retrieve_memory") as mock_retrieve,
        ):

            from agent_reminiscence.database.models import RetrievalResult

            manager = MemoryManager(test_config)
            manager._initialized = True  # Bypass initialization check

            # Mock repositories
            manager.active_repo = MagicMock()
            manager.shortterm_repo = MagicMock()
            manager.longterm_repo = MagicMock()
            manager.embedding_service = MagicMock()

            # Mock retrieve_memory function to return a RetrievalResult
            mock_result = RetrievalResult(
                mode="pointer",
                chunks=[],
                entities=[],
                relationships=[],
                synthesis=None,
                search_strategy="Test search strategy",
                confidence=0.9,
                metadata={"test": True},
            )
            mock_retrieve.return_value = mock_result

            result = await manager.retrieve_memories(query="test query", external_id="test-123")

            assert isinstance(result, RetrievalResult)
            assert result.mode == "pointer"
            assert result.search_strategy == "Test search strategy"
            assert result.confidence == 0.9
            mock_retrieve.assert_called_once()


class TestMemoryManagerHelpers:
    """Test helper methods."""

    def test_calculate_importance_with_multiplier(self, test_config):
        """Test importance calculation with type multiplier."""
        with (
            patch("agent_reminiscence.services.memory_manager.PostgreSQLManager"),
            patch("agent_reminiscence.services.memory_manager.Neo4jManager"),
            patch("agent_reminiscence.services.memory_manager.EmbeddingService"),
        ):

            manager = MemoryManager(test_config)

            # Create mock entity with importance and types
            entity = MagicMock()
            entity.importance = 0.5
            entity.type = "PERSON"  # Has multiplier of 1.2

            importance = manager._calculate_importance(entity)

            # 0.5 * 1.2 = 0.6
            assert importance == 0.6

    def test_calculate_importance_capped(self, test_config):
        """Test importance calculation is capped at 1.0."""
        with (
            patch("agent_reminiscence.services.memory_manager.PostgreSQLManager"),
            patch("agent_reminiscence.services.memory_manager.Neo4jManager"),
            patch("agent_reminiscence.services.memory_manager.EmbeddingService"),
        ):

            manager = MemoryManager(test_config)

            # Create mock entity with high importance and types multiplier
            entity = MagicMock()
            entity.importance = 0.9
            entity.type = "PERSON"  # Has multiplier of 1.2

            importance = manager._calculate_importance(entity)

            # 0.9 * 1.2 = 1.08, but should be capped at 1.0
            assert importance == 1.0


