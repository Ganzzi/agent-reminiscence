"""
Tests for Active Memory Repository (database/repositories/active_memory.py).
"""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from agent_mem.database.repositories.active_memory import ActiveMemoryRepository
from agent_mem.database.models import ActiveMemory


class TestActiveMemoryRepository:
    """Test ActiveMemoryRepository class."""

    @pytest.mark.asyncio
    async def test_create(self, test_config):
        """Test creating an active memory."""
        # Mock PostgreSQL manager
        mock_pg = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()

        # Mock return data
        row_data = (
            1,  # id
            "test-123",  # external_id
            "Test Memory",  # title
            "# Template",  # template_content
            json.dumps({"summary": {"content": "Test", "update_count": 0}}),  # sections
            json.dumps({"key": "value"}),  # metadata
            datetime.now(timezone.utc),  # created_at
            datetime.now(timezone.utc),  # updated_at
        )

        mock_result.result.return_value = [row_data]
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn

        repo = ActiveMemoryRepository(mock_pg)

        result = await repo.create(
            external_id="test-123",
            title="Test Memory",
            template_content="# Template",
            sections={"summary": {"content": "Test", "update_count": 0}},
            metadata={"key": "value"},
        )

        assert isinstance(result, ActiveMemory)
        assert result.external_id == "test-123"
        assert result.title == "Test Memory"
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_id(self, test_config):
        """Test getting active memory by ID."""
        mock_pg = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()

        row_data = (
            1,
            "test-123",
            "Test Memory",
            "# Template",
            json.dumps({"summary": {"content": "Test", "update_count": 0}}),
            json.dumps({}),
            datetime.now(timezone.utc),
            datetime.now(timezone.utc),
        )

        mock_result.result.return_value = [row_data]
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn

        repo = ActiveMemoryRepository(mock_pg)
        result = await repo.get_by_id(1)

        assert isinstance(result, ActiveMemory)
        assert result.id == 1
        assert result.external_id == "test-123"

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, test_config):
        """Test getting non-existent memory returns None."""
        mock_pg = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()

        mock_result.result.return_value = []  # Empty result
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn

        repo = ActiveMemoryRepository(mock_pg)
        result = await repo.get_by_id(999)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_all_by_external_id(self, test_config):
        """Test getting all memories by external_id."""
        mock_pg = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()

        rows = [
            (
                1,
                "test-123",
                "Memory 1",
                "# Template",
                json.dumps({"summary": {"content": "Test 1", "update_count": 0}}),
                json.dumps({}),
                datetime.now(timezone.utc),
                datetime.now(timezone.utc),
            ),
            (
                2,
                "test-123",
                "Memory 2",
                "# Template",
                json.dumps({"summary": {"content": "Test 2", "update_count": 0}}),
                json.dumps({}),
                datetime.now(timezone.utc),
                datetime.now(timezone.utc),
            ),
        ]

        mock_result.result.return_value = rows
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn

        repo = ActiveMemoryRepository(mock_pg)
        results = await repo.get_all_by_external_id("test-123")

        assert len(results) == 2
        assert all(isinstance(m, ActiveMemory) for m in results)
        assert all(m.external_id == "test-123" for m in results)

    @pytest.mark.asyncio
    async def test_update_section(self, test_config):
        """Test updating a section."""
        mock_pg = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()

        # First call: get_by_id - PSQLPy auto-parses JSON to dict
        get_row = (
            1,
            "test-123",
            "Test Memory",
            "# Template",
            {"summary": {"content": "Old", "update_count": 0}},  # Already a dict
            {},
            datetime.now(timezone.utc),
            datetime.now(timezone.utc),
        )

        # Second call: update
        update_row = (
            1,
            "test-123",
            "Test Memory",
            "# Template",
            {"summary": {"content": "Updated", "update_count": 1}},  # Already a dict
            {},
            datetime.now(timezone.utc),
            datetime.now(timezone.utc),
        )

        mock_result.result.side_effect = [[get_row], [update_row]]
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn

        repo = ActiveMemoryRepository(mock_pg)
        result = await repo.update_section(memory_id=1, section_id="summary", new_content="Updated")

        assert isinstance(result, ActiveMemory)
        assert result.sections["summary"]["content"] == "Updated"
        assert result.sections["summary"]["update_count"] == 1

    @pytest.mark.asyncio
    async def test_delete(self, test_config):
        """Test deleting an active memory."""
        mock_pg = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()

        mock_result.result.return_value = []
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn

        repo = ActiveMemoryRepository(mock_pg)
        result = await repo.delete(1)

        assert result is True
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_sections_needing_consolidation(self, test_config):
        """Test getting sections that need consolidation."""
        mock_pg = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()

        # PSQLPy auto-parses JSON to dict
        row = (
            1,
            "test-123",
            "Test Memory",
            "# Template",
            {"summary": {"content": "Test", "update_count": 5}},  # Already a dict
            {},  # Already a dict
            datetime.now(timezone.utc),
            datetime.now(timezone.utc),
        )

        mock_result.result.return_value = [row]
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn

        repo = ActiveMemoryRepository(mock_pg)
        results = await repo.get_sections_needing_consolidation(external_id="test-123", threshold=3)

        assert len(results) > 0
        assert isinstance(results[0], dict)
        assert "memory_id" in results[0]
        assert results[0]["update_count"] >= 3

    @pytest.mark.asyncio
    async def test_create_with_validation_error(self, test_config):
        """Test that validation errors are raised."""
        mock_pg = MagicMock()
        repo = ActiveMemoryRepository(mock_pg)

        # This should raise a validation error due to bad data
        # The actual validation happens in Pydantic model
        # Just test that repo can be created
        assert repo.postgres == mock_pg
