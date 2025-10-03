import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from agent_mem.database.repositories.shortterm_memory import ShorttermMemoryRepository
from agent_mem.database.models import ShorttermMemory, ShorttermMemoryChunk


class TestShorttermMemoryRepository:
    @pytest.mark.asyncio
    async def test_create_memory(self):
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_conn, mock_result = MagicMock(), MagicMock()
        row_data = (
            1,
            "test-123",
            "Test",
            "Summary",
            {},
            datetime.now(timezone.utc),
            datetime.now(timezone.utc),
        )
        mock_result.result.return_value = [row_data]
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn
        repo = ShorttermMemoryRepository(mock_pg, mock_neo4j)
        result = await repo.create_memory(external_id="test-123", title="Test")
        assert isinstance(result, ShorttermMemory)
        assert result.id == 1

    @pytest.mark.asyncio
    async def test_get_memory_by_id(self):
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_conn, mock_result = MagicMock(), MagicMock()
        row_data = (
            1,
            "test-123",
            "Test",
            "Summary",
            {},
            datetime.now(timezone.utc),
            datetime.now(timezone.utc),
        )
        mock_result.result.return_value = [row_data]
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn
        repo = ShorttermMemoryRepository(mock_pg, mock_neo4j)
        result = await repo.get_memory_by_id(1)
        assert result is not None and result.id == 1

    @pytest.mark.asyncio
    async def test_get_memories_by_external_id(self):
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_conn, mock_result = MagicMock(), MagicMock()
        rows = [
            (1, "test-123", "M1", "S1", {}, datetime.now(timezone.utc), datetime.now(timezone.utc)),
            (2, "test-123", "M2", "S2", {}, datetime.now(timezone.utc), datetime.now(timezone.utc)),
        ]
        mock_result.result.return_value = rows
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn
        repo = ShorttermMemoryRepository(mock_pg, mock_neo4j)
        results = await repo.get_memories_by_external_id("test-123")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_create_chunk(self):
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_conn, mock_result = MagicMock(), MagicMock()
        row_data = (1, 1, None, "Test chunk", 0, {})
        mock_result.result.return_value = [row_data]
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn
        repo = ShorttermMemoryRepository(mock_pg, mock_neo4j)
        result = await repo.create_chunk(
            shortterm_memory_id=1,
            external_id="test-123",
            content="Test",
            chunk_order=0,
            embedding=[0.1],
        )
        assert isinstance(result, ShorttermMemoryChunk)

    @pytest.mark.asyncio
    async def test_get_chunk_by_id(self):
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_conn, mock_result = MagicMock(), MagicMock()
        row_data = (1, 1, None, "Test chunk", 0, {})
        mock_result.result.return_value = [row_data]
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn
        repo = ShorttermMemoryRepository(mock_pg, mock_neo4j)
        result = await repo.get_chunk_by_id(1)
        assert result is not None

    @pytest.mark.asyncio
    async def test_get_chunks_by_memory_id(self):
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_conn, mock_result = MagicMock(), MagicMock()
        rows = [(1, 1, None, "C1", 0, {}), (2, 1, None, "C2", 1, {})]
        mock_result.result.return_value = rows
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn
        repo = ShorttermMemoryRepository(mock_pg, mock_neo4j)
        results = await repo.get_chunks_by_memory_id(1)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_delete_chunk(self):
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_conn, mock_result = MagicMock(), MagicMock()
        mock_result.result.return_value = []
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn
        repo = ShorttermMemoryRepository(mock_pg, mock_neo4j)
        result = await repo.delete_chunk(1)
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_memory(self):
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_conn, mock_result = MagicMock(), MagicMock()
        mock_result.result.return_value = []
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn
        repo = ShorttermMemoryRepository(mock_pg, mock_neo4j)
        result = await repo.delete_memory(1)
        assert result is True
