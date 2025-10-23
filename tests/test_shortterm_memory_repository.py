import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from agent_reminiscence.database.repositories.shortterm_memory import ShorttermMemoryRepository
from agent_reminiscence.database.models import (
    ShorttermMemory,
    ShorttermMemoryChunk,
    ShorttermEntityRelationshipSearchResult,
)


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
            0,  # update_count
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
            0,  # update_count
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
            (
                1,
                "test-123",
                "M1",
                "S1",
                0,
                {},
                datetime.now(timezone.utc),
                datetime.now(timezone.utc),
            ),
            (
                2,
                "test-123",
                "M2",
                "S2",
                0,
                {},
                datetime.now(timezone.utc),
                datetime.now(timezone.utc),
            ),
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
        row_data = (1, 1, "Test chunk", None, {}, 0, None)
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
            embedding=[0.1],
        )
        assert isinstance(result, ShorttermMemoryChunk)

    @pytest.mark.asyncio
    async def test_get_chunk_by_id(self):
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_conn, mock_result = MagicMock(), MagicMock()
        row_data = (1, 1, "Test chunk", None, {}, 0, None)
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
        rows = [(1, 1, "C1", None, {}, 0, None), (2, 1, "C2", None, {}, 0, None)]
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

    @pytest.mark.asyncio
    async def test_increment_chunk_access(self):
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_conn, mock_result = MagicMock(), MagicMock()
        row_data = (1, 1, "Test chunk", None, {}, 6, datetime.now(timezone.utc))
        mock_result.result.return_value = [row_data]
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn
        repo = ShorttermMemoryRepository(mock_pg, mock_neo4j)
        result = await repo.increment_chunk_access(1)
        assert result is not None
        assert result.access_count == 6

    @pytest.mark.asyncio
    async def test_search_entities_with_relationships(self):
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = MagicMock()

        # Mock entity data
        mock_entity = MagicMock()
        mock_entity.element_id = "entity-123"
        mock_entity.__getitem__ = MagicMock(
            side_effect=lambda key: {
                "external_id": "test-123",
                "shortterm_memory_id": 1,
                "name": "Test Entity",
                "types": ["PERSON"],
                "description": "A test entity",
                "importance": 0.8,
                "access_count": 5,
                "last_access": None,
                "metadata": {},
            }[key]
        )
        mock_entity.get = MagicMock(
            side_effect=lambda key, default=None: {
                "external_id": "test-123",
                "shortterm_memory_id": 1,
                "name": "Test Entity",
                "types": ["PERSON"],
                "description": "A test entity",
                "importance": 0.8,
                "access_count": 5,
                "last_access": None,
                "metadata": {},
            }.get(key, default)
        )

        # Mock empty collections for relationships
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
            entity_names=["Test"], external_id="test-123", limit=10
        )

        assert result is not None
        assert len(result.matched_entities) == 1
        assert result.matched_entities[0].name == "Test Entity"

    @pytest.mark.asyncio
    async def test_increment_entity_access(self):
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = MagicMock()

        mock_entity = MagicMock()
        mock_entity.element_id = "entity-123"
        mock_entity.__getitem__.side_effect = lambda key: {
            "external_id": "test-123",
            "shortterm_memory_id": 1,
            "name": "Test Entity",
            "types": ["PERSON"],
            "description": "A test entity",
            "importance": 0.8,
            "access_count": 6,
            "last_access": None,
            "metadata": {},
        }.get(key)
        mock_entity.get = lambda key, default=None: {
            "external_id": "test-123",
            "shortterm_memory_id": 1,
            "name": "Test Entity",
            "types": ["PERSON"],
            "description": "A test entity",
            "importance": 0.8,
            "access_count": 6,
            "last_access": None,
            "metadata": {},
        }.get(key, default)

        mock_record.__getitem__.return_value = mock_entity
        mock_result.single = AsyncMock(return_value=mock_record)
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j.session.return_value = mock_session

        repo = ShorttermMemoryRepository(mock_pg, mock_neo4j)
        result = await repo.increment_entity_access("entity-123")

        assert result is not None
        assert result.access_count == 6

    @pytest.mark.asyncio
    async def test_increment_relationship_access(self):
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = MagicMock()

        mock_relationship = MagicMock()
        mock_relationship.element_id = "rel-123"
        mock_relationship.__getitem__.side_effect = lambda key: {
            "external_id": "test-123",
            "shortterm_memory_id": 1,
            "from_entity_id": "entity-1",
            "to_entity_id": "entity-2",
            "from_entity_name": "Entity 1",
            "to_entity_name": "Entity 2",
            "types": ["WORKS_WITH"],
            "description": "Test relationship",
            "importance": 0.7,
            "access_count": 3,
            "last_access": None,
            "metadata": {},
        }.get(key)
        mock_relationship.get = lambda key, default=None: {
            "external_id": "test-123",
            "shortterm_memory_id": 1,
            "from_entity_id": "entity-1",
            "to_entity_id": "entity-2",
            "from_entity_name": "Entity 1",
            "to_entity_name": "Entity 2",
            "types": ["WORKS_WITH"],
            "description": "Test relationship",
            "importance": 0.7,
            "access_count": 3,
            "last_access": None,
            "metadata": {},
        }.get(key, default)

        mock_record.__getitem__.return_value = mock_relationship
        mock_result.single = AsyncMock(return_value=mock_record)
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j.session.return_value = mock_session

        repo = ShorttermMemoryRepository(mock_pg, mock_neo4j)
        result = await repo.increment_relationship_access("rel-123")

        assert result is not None
        assert result.access_count == 3

    @pytest.mark.asyncio
    async def test_hybrid_search_with_memory_id_filter(self):
        """Test hybrid_search filters chunks by shortterm_memory_id"""
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_conn, mock_result = MagicMock(), MagicMock()

        # Mock chunks from memory_id=1 only
        # Format: id, shortterm_memory_id, external_id, content, section_id, metadata, access_count, last_access, created_at, combined_score, vector_score, bm25_score
        now = datetime.now(timezone.utc)
        rows = [
            (
                1,
                1,
                "test-123",
                "Chunk from memory 1",
                None,
                {},
                5,
                now,
                now,
                0.9,
                0.8,
                0.7,
            ),
            (
                2,
                1,
                "test-123",
                "Another chunk from memory 1",
                None,
                {},
                3,
                now,
                now,
                0.85,
                0.75,
                0.65,
            ),
        ]
        mock_result.result.return_value = rows
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn

        repo = ShorttermMemoryRepository(mock_pg, mock_neo4j)
        results = await repo.hybrid_search(
            external_id="test-123",
            query_text="test query",
            query_embedding=[0.1] * 384,
            limit=10,
            shortterm_memory_id=1,
        )

        assert len(results) == 2
        assert all(chunk.shortterm_memory_id == 1 for chunk in results)
        # Verify that memory_id filter was passed in SQL query
        call_args = mock_conn.execute.call_args[0]
        assert call_args[1][-3] == 1  # shortterm_memory_id parameter

    @pytest.mark.asyncio
    async def test_hybrid_search_with_similarity_thresholds(self):
        """Test hybrid_search filters by min_similarity_score and min_bm25_score"""
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_conn, mock_result = MagicMock(), MagicMock()

        # Mock chunks with scores above thresholds
        now = datetime.now(timezone.utc)
        rows = [
            (
                1,
                1,
                "test-123",
                "High scoring chunk",
                None,
                {},
                5,
                now,
                now,
                0.9,
                0.85,
                0.8,
            ),
        ]
        mock_result.result.return_value = rows
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn

        repo = ShorttermMemoryRepository(mock_pg, mock_neo4j)
        results = await repo.hybrid_search(
            external_id="test-123",
            query_text="test query",
            query_embedding=[0.1] * 384,
            limit=10,
            min_similarity_score=0.7,
            min_bm25_score=0.6,
        )

        assert len(results) == 1
        # Verify thresholds were passed to SQL query
        call_args = mock_conn.execute.call_args[0]
        assert call_args[1][-2] == 0.7  # min_similarity_score
        assert call_args[1][-1] == 0.6  # min_bm25_score

    @pytest.mark.asyncio
    async def test_hybrid_search_returns_access_count(self):
        """Test hybrid_search returns access_count in chunk models"""
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_conn, mock_result = MagicMock(), MagicMock()

        now = datetime.now(timezone.utc)
        rows = [
            (
                1,
                1,
                "test-123",
                "Test chunk",
                None,
                {},
                10,
                now,
                now,
                0.9,
                0.8,
                0.7,
            ),
        ]
        mock_result.result.return_value = rows
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn

        repo = ShorttermMemoryRepository(mock_pg, mock_neo4j)
        results = await repo.hybrid_search(
            external_id="test-123", query_text="test query", query_embedding=[0.1] * 384, limit=10
        )

        assert len(results) == 1
        assert results[0].access_count == 10
        assert results[0].last_access is not None

    @pytest.mark.asyncio
    async def test_hybrid_search_combined_filters(self):
        """Test hybrid_search with multiple filters combined"""
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_conn, mock_result = MagicMock(), MagicMock()

        now = datetime.now(timezone.utc)
        rows = [
            (
                1,
                2,
                "test-123",
                "Filtered chunk",
                None,
                {},
                7,
                now,
                now,
                0.95,
                0.9,
                0.85,
            ),
        ]
        mock_result.result.return_value = rows
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn

        repo = ShorttermMemoryRepository(mock_pg, mock_neo4j)
        results = await repo.hybrid_search(
            external_id="test-123",
            query_text="test query",
            query_embedding=[0.1] * 384,
            limit=5,
            shortterm_memory_id=2,
            min_similarity_score=0.8,
            min_bm25_score=0.7,
        )

        assert len(results) == 1
        assert results[0].shortterm_memory_id == 2
        # Verify all filters were passed
        call_args = mock_conn.execute.call_args[0]
        params = call_args[1]
        assert params[-3] == 2  # shortterm_memory_id
        assert params[-2] == 0.8  # min_similarity_score
        assert params[-1] == 0.7  # min_bm25_score


