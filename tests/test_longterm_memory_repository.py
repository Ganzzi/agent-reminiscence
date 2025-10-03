import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from agent_mem.database.repositories.longterm_memory import LongtermMemoryRepository
from agent_mem.database.models import LongtermMemoryChunk


class TestLongtermMemoryRepository:
    @pytest.mark.asyncio
    async def test_create_chunk(self):
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_conn, mock_result = MagicMock(), MagicMock()
        now = datetime.now(timezone.utc)
        row_data = (1, "test-123", None, 0, "Test", 0.8, 0.9, now, None, {}, now)
        mock_result.result.return_value = [row_data]
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn
        repo = LongtermMemoryRepository(mock_pg, mock_neo4j)
        result = await repo.create_chunk(external_id="test-123", content="Test", chunk_order=0)
        assert isinstance(result, LongtermMemoryChunk)

    @pytest.mark.asyncio
    async def test_get_chunk_by_id(self):
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_conn, mock_result = MagicMock(), MagicMock()
        now = datetime.now(timezone.utc)
        row_data = (1, "test-123", None, 0, "Test", 0.8, 0.9, now, None, {}, now)
        mock_result.result.return_value = [row_data]
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn
        repo = LongtermMemoryRepository(mock_pg, mock_neo4j)
        result = await repo.get_chunk_by_id(1)
        assert result is not None

    @pytest.mark.asyncio
    async def test_get_valid_chunks_by_external_id(self):
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_conn, mock_result = MagicMock(), MagicMock()
        now = datetime.now(timezone.utc)
        rows = [
            (1, "test-123", None, 0, "C1", 0.8, 0.9, now, None, {}, now),
            (2, "test-123", None, 1, "C2", 0.7, 0.8, now, None, {}, now),
        ]
        mock_result.result.return_value = rows
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn
        repo = LongtermMemoryRepository(mock_pg, mock_neo4j)
        results = await repo.get_valid_chunks_by_external_id("test-123")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_get_chunks_by_temporal_range(self):
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_conn, mock_result = MagicMock(), MagicMock()
        now = datetime.now(timezone.utc)
        rows = [(1, "test-123", None, 0, "C1", 0.8, 0.9, now, None, {}, now)]
        mock_result.result.return_value = rows
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn
        repo = LongtermMemoryRepository(mock_pg, mock_neo4j)
        results = await repo.get_chunks_by_temporal_range("test-123", start_date=now, end_date=now)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_supersede_chunk(self):
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_conn, mock_result = MagicMock(), MagicMock()
        mock_result.result.return_value = []
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn
        repo = LongtermMemoryRepository(mock_pg, mock_neo4j)
        result = await repo.supersede_chunk(1)
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_chunk(self):
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_conn, mock_result = MagicMock(), MagicMock()
        mock_result.result.return_value = []
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn
        repo = LongtermMemoryRepository(mock_pg, mock_neo4j)
        result = await repo.delete_chunk(1)
        assert result is True

