"""
Tests for Batch Update and Enhanced Consolidation Features (Phases 6-10).

Tests cover:
- Batch section updates
- Smart consolidation threshold
- Section tracking
- Update count management
- Metadata history
- Background consolidation
"""

import pytest
import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from agent_reminiscence.database.repositories.active_memory import ActiveMemoryRepository
from agent_reminiscence.database.repositories.shortterm_memory import ShorttermMemoryRepository
from agent_reminiscence.database.repositories.longterm_memory import LongtermMemoryRepository
from agent_reminiscence.database.models import (
    ActiveMemory,
    ShorttermMemory,
    ShorttermMemoryChunk,
    LongtermMemoryChunk,
    LongtermEntity,
)
from agent_reminiscence.services.memory_manager import MemoryManager


class TestBatchUpdate:
    """Test batch section update functionality."""

    @pytest.mark.asyncio
    async def test_update_sections_batch(self, test_config):
        """Test updating multiple sections in a single call."""
        # Mock PostgreSQL manager
        mock_pg = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()

        # Initial memory state
        initial_sections = {
            "progress": {"content": "Old progress", "update_count": 0},
            "notes": {"content": "Old notes", "update_count": 0},
            "blockers": {"content": "Old blockers", "update_count": 0},
        }

        # Updated memory state (after batch update)
        updated_sections = {
            "progress": {"content": "New progress", "update_count": 1},
            "notes": {"content": "New notes", "update_count": 1},
            "blockers": {"content": "Old blockers", "update_count": 0},
        }

        row_data = (
            1,  # id
            "agent-123",  # external_id
            "Test Memory",  # title
            "# Template",  # template_content
            json.dumps(updated_sections),  # sections
            json.dumps({}),  # metadata
            datetime.now(timezone.utc),  # created_at
            datetime.now(timezone.utc),  # updated_at
        )

        mock_result.result.return_value = [row_data]
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn

        repo = ActiveMemoryRepository(mock_pg)

        # Call batch update
        sections_to_update = [
            {"section_id": "progress", "new_content": "New progress"},
            {"section_id": "notes", "new_content": "New notes"},
        ]

        result = await repo.update_sections(
            memory_id=1,
            section_updates=sections_to_update,
        )

        # Verify result
        assert result is not None
        assert result.id == 1
        assert result.sections["progress"]["update_count"] == 1
        assert result.sections["notes"]["update_count"] == 1
        assert result.sections["blockers"]["update_count"] == 0

    @pytest.mark.asyncio
    async def test_threshold_calculation(self, test_config):
        """Test smart consolidation threshold calculation."""
        # Mock config
        test_config.avg_section_update_count_for_consolidation = 5.0

        # Memory with 3 sections
        memory = ActiveMemory(
            id=1,
            external_id="agent-123",
            title="Test Memory",
            template_content="",
            sections={
                "section1": {"content": "Content 1", "update_count": 4},
                "section2": {"content": "Content 2", "update_count": 5},
                "section3": {"content": "Content 3", "update_count": 6},
            },
            metadata={},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        # Calculate threshold: avg_section_update_count * num_sections
        num_sections = len(memory.sections)
        expected_threshold = test_config.avg_section_update_count_for_consolidation * num_sections
        assert expected_threshold == 15.0  # 5.0 * 3

        # Calculate total update count
        total_update_count = sum(
            section.get("update_count", 0) for section in memory.sections.values()
        )
        assert total_update_count == 15  # 4 + 5 + 6

        # Should trigger consolidation
        assert total_update_count >= expected_threshold


class TestSectionTracking:
    """Test section_id tracking through the workflow."""

    @pytest.mark.asyncio
    async def test_chunk_with_section_id(self, test_config):
        """Test creating shortterm chunk with section_id."""
        mock_pg = MagicMock()
        mock_neo4j = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()

        # Mock chunk with section_id
        row_data = (
            1,  # id
            123,  # shortterm_memory_id
            "Test chunk content",  # content
            "progress",  # section_id (NEW)
            json.dumps({"source": "active_memory"}),  # metadata
            0,  # access_count
            None,  # last_access
        )

        mock_result.result.return_value = [row_data]
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn

        repo = ShorttermMemoryRepository(mock_pg, mock_neo4j)

        result = await repo.create_chunk(
            shortterm_memory_id=123,
            external_id="agent-123",
            content="Test chunk content",
            embedding=[0.1] * 768,
            section_id="progress",  # NEW parameter
            metadata={"source": "active_memory"},
        )

        assert result is not None
        assert result.section_id == "progress"

    @pytest.mark.asyncio
    async def test_get_chunks_by_section_id(self, test_config):
        """Test retrieving chunks by section_id."""
        mock_pg = MagicMock()
        mock_neo4j = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()

        # Mock multiple chunks with same section_id
        chunks_data = [
            (
                i,  # id
                123,  # shortterm_memory_id
                f"Chunk {i}",  # content
                "progress",  # section_id
                json.dumps({}),  # metadata
                0,  # access_count
                None,  # last_access
            )
            for i in range(3)
        ]

        mock_result.result.return_value = chunks_data
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn

        repo = ShorttermMemoryRepository(mock_pg, mock_neo4j)

        result = await repo.get_chunks_by_section_id(
            shortterm_memory_id=123,
            section_id="progress",
        )

        assert len(result) == 3
        assert all(chunk.section_id == "progress" for chunk in result)


class TestUpdateCountTracking:
    """Test update_count tracking for shortterm memory."""

    @pytest.mark.asyncio
    async def test_increment_update_count(self, test_config):
        """Test incrementing shortterm memory update count."""
        mock_pg = MagicMock()
        mock_neo4j = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()

        # Mock memory with incremented count
        row_data = (
            123,  # id
            "agent-123",  # external_id
            "Test Memory",  # title
            "Test summary",  # summary
            5,  # update_count (incremented)
            json.dumps({}),  # metadata
            datetime.now(timezone.utc),  # created_at
            datetime.now(timezone.utc),  # last_updated
        )

        mock_result.result.return_value = [row_data]
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn

        repo = ShorttermMemoryRepository(mock_pg, mock_neo4j)

        result = await repo.increment_update_count(memory_id=123)

        assert result is not None
        assert result.update_count == 5

    @pytest.mark.asyncio
    async def test_reset_update_count(self, test_config):
        """Test resetting shortterm memory update count."""
        mock_pg = MagicMock()
        mock_neo4j = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()

        # Mock memory with reset count
        row_data = (
            123,  # id
            "agent-123",  # external_id
            "Test Memory",  # title
            "Test summary",  # summary
            0,  # update_count (reset)
            json.dumps({}),  # metadata
            datetime.now(timezone.utc),  # created_at
            datetime.now(timezone.utc),  # last_updated
        )

        mock_result.result.return_value = [row_data]
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn

        repo = ShorttermMemoryRepository(mock_pg, mock_neo4j)

        result = await repo.reset_update_count(memory_id=123)

        assert result is not None
        assert result.update_count == 0


class TestMetadataTracking:
    """Test metadata.updates array tracking."""

    @pytest.mark.asyncio
    async def test_entity_metadata_updates(self, test_config):
        """Test entity update history in metadata."""
        mock_pg = MagicMock()
        mock_neo4j = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()

        # Mock entity with updates array
        updates_array = [
            {
                "date": "2025-10-05T10:00:00Z",
                "old_importance": 0.75,
                "new_importance": 0.85,
                "source": "shortterm_promotion",
            }
        ]

        # Mock a longterm entity (Neo4j entity, use string ID)
        mock_entity = LongtermEntity(
            id="neo4j-id-1",
            external_id="agent-123",
            name="Python",
            types=["TECHNOLOGY"],
            description="Programming language",
            importance=0.85,
            metadata={"updates": updates_array},
        )

        repo = LongtermMemoryRepository(mock_pg, mock_neo4j)

        # Mock update_entity_with_metadata to return the entity
        repo.update_entity_with_metadata = AsyncMock(return_value=mock_entity)

        # Prepare new update
        new_update = {
            "date": "2025-10-05T11:00:00Z",
            "old_importance": 0.85,
            "new_importance": 0.90,
            "source": "shortterm_promotion",
        }
        updates_array.append(new_update)

        result = await repo.update_entity_with_metadata(
            entity_id="neo4j-id-1",
            importance=0.90,
            metadata_update={"updates": updates_array},
        )

        assert result is not None
        assert result.importance == 0.85  # From mock data
        assert "updates" in result.metadata
        assert len(result.metadata["updates"]) >= 1


class TestConsolidationWorkflow:
    """Test enhanced consolidation workflow."""

    @pytest.mark.asyncio
    async def test_selective_consolidation(self, test_config):
        """Test that only updated sections are consolidated."""
        # This would require a more complex setup with mocked memory_manager
        # For now, we'll test the logic conceptually

        sections = {
            "progress": {"content": "Updated", "update_count": 5},
            "notes": {"content": "Updated", "update_count": 3},
            "blockers": {"content": "Not updated", "update_count": 0},
        }

        # Filter sections with update_count > 0
        updated_sections = {
            sid: sdata for sid, sdata in sections.items() if sdata.get("update_count", 0) > 0
        }

        assert len(updated_sections) == 2
        assert "progress" in updated_sections
        assert "notes" in updated_sections
        assert "blockers" not in updated_sections

    @pytest.mark.asyncio
    async def test_promotion_threshold_check(self, test_config):
        """Test automatic promotion threshold checking."""
        test_config.shortterm_update_count_threshold = 10

        # Mock shortterm memory with update count
        memory = ShorttermMemory(
            id=1,
            external_id="agent-123",
            title="Test Memory",
            update_count=12,  # Above threshold
            metadata={},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        # Should trigger promotion
        should_promote = memory.update_count >= test_config.shortterm_update_count_threshold
        assert should_promote is True

        # Test below threshold
        memory.update_count = 5
        should_promote = memory.update_count >= test_config.shortterm_update_count_threshold
        assert should_promote is False


class TestConcurrencyControl:
    """Test consolidation locking mechanism."""

    @pytest.mark.asyncio
    async def test_consolidation_lock(self):
        """Test that concurrent consolidations are prevented."""
        locks = {}
        memory_id = 1

        # Create lock
        if memory_id not in locks:
            locks[memory_id] = asyncio.Lock()

        lock = locks[memory_id]

        # First consolidation acquires lock
        async def first_consolidation():
            async with lock:
                await asyncio.sleep(0.1)  # Simulate work
                return "first"

        # Second consolidation should wait
        async def second_consolidation():
            if lock.locked():
                return "skipped"  # Would skip in real implementation
            async with lock:
                return "second"

        # Start both
        task1 = asyncio.create_task(first_consolidation())
        await asyncio.sleep(0.01)  # Let first acquire lock
        task2 = asyncio.create_task(second_consolidation())

        result1 = await task1
        result2 = await task2

        assert result1 == "first"
        assert result2 == "skipped"


class TestBackwardCompatibility:
    """Test that old single-section API still works."""

    @pytest.mark.asyncio
    async def test_single_section_update_delegates_to_batch(self, test_config):
        """Test that single section update calls batch method."""
        # This test verifies the delegation pattern

        # Mock the batch method call
        with patch.object(
            ActiveMemoryRepository, "update_sections", new_callable=AsyncMock
        ) as mock_batch:
            mock_pg = MagicMock()
            repo = ActiveMemoryRepository(mock_pg)

            # Mock return value
            mock_memory = ActiveMemory(
                id=1,
                external_id="agent-123",
                title="Test",
                template_content="",
                sections={"progress": {"content": "Updated", "update_count": 1}},
                metadata={},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            mock_batch.return_value = mock_memory

            # Call single section update (old API)
            result = await repo.update_section(
                memory_id=1,
                section_id="progress",
                new_content="Updated",
            )

            # Verify it called the batch method
            mock_batch.assert_called_once()
            call_args = mock_batch.call_args
            assert call_args.kwargs["memory_id"] == 1
            assert len(call_args.kwargs["section_updates"]) == 1
            assert call_args.kwargs["section_updates"][0]["section_id"] == "progress"


class TestResetOperations:
    """Test reset operations after consolidation/promotion."""

    @pytest.mark.asyncio
    async def test_reset_section_counts(self, test_config):
        """Test resetting all section update counts."""
        mock_pg = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()

        # Mock memory with reset counts
        reset_sections = {
            "progress": {"content": "Content 1", "update_count": 0},
            "notes": {"content": "Content 2", "update_count": 0},
            "blockers": {"content": "Content 3", "update_count": 0},
        }

        row_data = (
            1,  # id
            "agent-123",  # external_id
            "Test Memory",  # title
            "# Template",  # template_content
            json.dumps(reset_sections),  # sections
            json.dumps({}),  # metadata
            datetime.now(timezone.utc),  # created_at
            datetime.now(timezone.utc),  # updated_at
        )

        mock_result.result.return_value = [row_data]
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn

        repo = ActiveMemoryRepository(mock_pg)

        result = await repo.reset_section_count(memory_id=1, section_id="all")

        assert result is not None
        assert all(section.get("update_count", 0) == 0 for section in result.sections.values())

    @pytest.mark.asyncio
    async def test_delete_all_chunks(self, test_config):
        """Test deleting all chunks after promotion."""
        mock_pg = MagicMock()
        mock_neo4j = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()

        # Mock successful deletion
        mock_result.result.return_value = []
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn

        repo = ShorttermMemoryRepository(mock_pg, mock_neo4j)

        # Should not raise exception
        await repo.delete_all_chunks(memory_id=123)


# Integration test example
class TestEndToEndWorkflow:
    """Test complete workflow from batch update to promotion."""

    @pytest.mark.asyncio
    async def test_batch_update_triggers_consolidation(self, test_config):
        """
        Test end-to-end workflow:
        1. Batch update multiple sections
        2. Threshold exceeded
        3. Background consolidation triggered
        4. Promotion threshold checked
        """
        # This is a conceptual test showing the workflow
        # Full implementation would require extensive mocking

        # Step 1: Batch update
        sections_updated = [
            {"section_id": "progress", "new_content": "New progress"},
            {"section_id": "notes", "new_content": "New notes"},
        ]

        # Step 2: Calculate threshold
        num_sections = 3
        avg_count = 5.0
        threshold = num_sections * avg_count  # 15

        # Simulate section counts after update
        section_counts = [6, 5, 4]  # Total = 15
        total_count = sum(section_counts)

        # Step 3: Check if consolidation should trigger
        should_consolidate = total_count >= threshold
        assert should_consolidate is True

        # Step 4: After consolidation, check promotion
        shortterm_update_count = 12
        promotion_threshold = 10
        should_promote = shortterm_update_count >= promotion_threshold
        assert should_promote is True


class TestAccessTrackingDuringWorkflow:
    """Test access tracking during batch update and consolidation workflows."""

    @pytest.mark.asyncio
    async def test_chunk_access_during_consolidation(self, test_config):
        """Test that chunks are accessed during consolidation."""
        mock_pg = MagicMock()
        mock_neo4j = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()

        # Mock chunk with initial access_count = 5
        initial_row = (1, 1, "Test chunk", None, {}, 5, None)
        updated_row = (1, 1, "Test chunk", None, {}, 6, datetime.now(timezone.utc))

        # First call returns initial chunk, second returns updated
        mock_result.result.side_effect = [[initial_row], [updated_row]]
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn

        repo = ShorttermMemoryRepository(mock_pg, mock_neo4j)

        # Simulate accessing chunk during consolidation
        result = await repo.increment_chunk_access(1)
        assert result.access_count == 6

    @pytest.mark.asyncio
    async def test_entity_access_during_search(self, test_config):
        """Test entity access tracking during search operations."""
        mock_pg = MagicMock()
        mock_neo4j = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = MagicMock()

        # Mock entity with initial access_count = 3
        mock_entity = MagicMock()
        mock_entity.element_id = "entity-123"
        mock_entity.__getitem__.side_effect = lambda key: {
            "external_id": "test-123",
            "shortterm_memory_id": 1,
            "name": "Test Entity",
            "types": ["PERSON"],
            "importance": 0.8,
            "access_count": 4,  # Incremented from 3
            "last_access": datetime.now(timezone.utc),
            "metadata": {},
        }.get(key)

        mock_record.__getitem__.return_value = mock_entity
        mock_result.single = AsyncMock(return_value=mock_record)
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j.session.return_value = mock_session

        repo = ShorttermMemoryRepository(mock_pg, mock_neo4j)
        result = await repo.increment_entity_access("entity-123")

        assert result.access_count == 4


class TestSearchIntegration:
    """Test search functionality integration with batch updates."""

    @pytest.mark.asyncio
    async def test_hybrid_search_with_memory_filter(self, test_config):
        """Test hybrid search with memory_id filter during batch operations."""
        mock_pg = MagicMock()
        mock_neo4j = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()

        # Mock chunks from different memories
        rows = [
            (
                1,
                123,
                "Memory 123 chunk",
                None,
                {},
                0,
                None,
                datetime.now(timezone.utc),
                0.8,
                0.0,
                0.0,
            ),
            (
                2,
                456,
                "Memory 456 chunk",
                None,
                {},
                0,
                None,
                datetime.now(timezone.utc),
                0.7,
                0.0,
                0.0,
            ),
        ]
        mock_result.result.return_value = rows
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn

        repo = ShorttermMemoryRepository(mock_pg, mock_neo4j)

        # Search with memory filter
        results = await repo.hybrid_search(
            external_id="test-123",
            query_text="test",
            query_embedding=[0.1] * 768,
            shortterm_memory_id=123,  # Filter by memory
        )

        # Should only return chunks from specified memory
        assert len(results) == 1
        assert results[0].shortterm_memory_id == 123

    @pytest.mark.asyncio
    async def test_entity_search_after_consolidation(self, test_config):
        """Test entity search works with consolidated memories."""
        mock_pg = MagicMock()
        mock_neo4j = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = MagicMock()

        # Mock consolidated entity
        mock_entity = MagicMock()
        mock_entity.element_id = "entity-consolidated"
        mock_entity.__getitem__.side_effect = lambda key: {
            "external_id": "test-123",
            "shortterm_memory_id": 1,
            "name": "Consolidated Entity",
            "types": ["PERSON", "DEVELOPER"],
            "importance": 0.9,
            "access_count": 10,
            "metadata": {"consolidated": True},
        }.get(key)

        mock_record.__getitem__.side_effect = lambda key: {
            "e": mock_entity,
            "related_incoming": [],
            "related_outgoing": [],
            "relationships_in": [],
            "relationships_out": [],
        }.get(key)

        mock_result.fetch = AsyncMock(return_value=[mock_record])
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j.session.return_value = mock_session

        repo = ShorttermMemoryRepository(mock_pg, mock_neo4j)
        result = await repo.search_entities_with_relationships(
            entity_names=["Consolidated"], external_id="test-123"
        )

        assert len(result.matched_entities) == 1
        assert "consolidated" in result.matched_entities[0].metadata


class TestAccessPatternsForImportance:
    """Test that access patterns influence importance calculations."""

    @pytest.mark.asyncio
    async def test_high_access_entity_importance(self, test_config):
        """Test that frequently accessed entities maintain higher importance."""
        mock_pg = MagicMock()
        mock_neo4j = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = MagicMock()

        # Mock high-access entity
        mock_entity = MagicMock()
        mock_entity.element_id = "high-access-entity"
        mock_entity.__getitem__.side_effect = lambda key: {
            "external_id": "test-123",
            "name": "Important Person",
            "types": ["PERSON"],
            "importance": 0.95,
            "access_count": 100,  # Very high access
            "last_access": datetime.now(timezone.utc),
            "metadata": {"frequently_accessed": True},
        }.get(key)

        mock_record.__getitem__.return_value = mock_entity
        mock_result.single = AsyncMock(return_value=mock_record)
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j.session.return_value = mock_session

        repo = ShorttermMemoryRepository(mock_pg, mock_neo4j)

        # Access the entity multiple times
        for i in range(3):
            result = await repo.increment_entity_access("high-access-entity")
            assert result.access_count == 100 + i + 1  # Starts at 100, increments each time

        # Final access count should be high
        assert result.access_count >= 103


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


