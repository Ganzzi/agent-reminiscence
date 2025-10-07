import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from agent_mem.database.repositories.longterm_memory import LongtermMemoryRepository
from agent_mem.database.models import LongtermMemoryChunk, LongtermEntityRelationshipSearchResult


class TestLongtermMemoryRepository:
    @pytest.mark.asyncio
    async def test_create_chunk(self):
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_conn, mock_result = MagicMock(), MagicMock()
        now = datetime.now(timezone.utc)
        row_data = (1, "test-123", None, "Test", 0.8, now, None, 0, None, {})
        mock_result.result.return_value = [row_data]
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn
        repo = LongtermMemoryRepository(mock_pg, mock_neo4j)
        result = await repo.create_chunk(external_id="test-123", content="Test", importance=0.8)
        assert isinstance(result, LongtermMemoryChunk)

    @pytest.mark.asyncio
    async def test_get_chunk_by_id(self):
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_conn, mock_result = MagicMock(), MagicMock()
        now = datetime.now(timezone.utc)
        row_data = (1, "test-123", None, "Test", 0.8, now, None, 0, None, {})
        mock_result.result.return_value = [row_data]
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn
        repo = LongtermMemoryRepository(mock_pg, mock_neo4j)
        result = await repo.get_chunk_by_id(1)
        assert result is not None

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

    @pytest.mark.asyncio
    async def test_increment_chunk_access(self):
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_conn, mock_result = MagicMock(), MagicMock()
        now = datetime.now(timezone.utc)
        row_data = (1, "test-123", None, "Test", 0.8, now, None, 6, now, {})
        mock_result.result.return_value = [row_data]
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn
        repo = LongtermMemoryRepository(mock_pg, mock_neo4j)
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

        repo = LongtermMemoryRepository(mock_pg, mock_neo4j)
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

        repo = LongtermMemoryRepository(mock_pg, mock_neo4j)
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
            "from_entity_id": "entity-1",
            "to_entity_id": "entity-2",
            "from_entity_name": "Entity 1",
            "to_entity_name": "Entity 2",
            "types": ["WORKS_WITH"],
            "description": "Test relationship",
            "importance": 0.7,
            "start_date": datetime.now(timezone.utc),
            "access_count": 3,
            "last_access": None,
            "metadata": {},
        }.get(key)
        mock_relationship.get = lambda key, default=None: {
            "external_id": "test-123",
            "from_entity_id": "entity-1",
            "to_entity_id": "entity-2",
            "from_entity_name": "Entity 1",
            "to_entity_name": "Entity 2",
            "types": ["WORKS_WITH"],
            "description": "Test relationship",
            "importance": 0.7,
            "start_date": datetime.now(timezone.utc),
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

        repo = LongtermMemoryRepository(mock_pg, mock_neo4j)
        result = await repo.increment_relationship_access("rel-123")

        assert result is not None
        assert result.access_count == 3

    @pytest.mark.asyncio
    async def test_hybrid_search_with_memory_id_filter(self):
        """Test hybrid_search filters chunks by shortterm_memory_id"""
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_conn, mock_result = MagicMock(), MagicMock()

        # Mock chunks from memory_id=1 only
        # Format: id, external_id, shortterm_memory_id, content, importance, start_date, last_updated, access_count, last_access, metadata, created_at, combined_score, vector_score, bm25_score
        now = datetime.now(timezone.utc)
        rows = [
            (
                1,
                "test-123",
                1,
                "Chunk from memory 1",
                0.8,
                now,
                now,
                5,
                now,
                {},
                now,
                0.9,
                0.8,
                0.7,
            ),
            (
                2,
                "test-123",
                1,
                "Another chunk from memory 1",
                0.7,
                now,
                now,
                3,
                now,
                {},
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

        repo = LongtermMemoryRepository(mock_pg, mock_neo4j)
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
        assert call_args[1][-5] == 1  # shortterm_memory_id parameter

    @pytest.mark.asyncio
    async def test_hybrid_search_with_temporal_filters(self):
        """Test hybrid_search filters by start_date and end_date"""
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_conn, mock_result = MagicMock(), MagicMock()

        now = datetime.now(timezone.utc)
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2024, 12, 31, tzinfo=timezone.utc)

        rows = [
            (
                1,
                "test-123",
                None,
                "Temporal chunk",
                0.8,
                start_date,
                end_date,
                5,
                now,
                {},
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

        repo = LongtermMemoryRepository(mock_pg, mock_neo4j)
        results = await repo.hybrid_search(
            external_id="test-123",
            query_text="test query",
            query_embedding=[0.1] * 384,
            limit=10,
            start_date=start_date,
            end_date=end_date,
        )

        assert len(results) == 1
        # Verify temporal filters were passed to SQL query
        call_args = mock_conn.execute.call_args[0]
        assert call_args[1][-4] == start_date  # start_date parameter
        assert call_args[1][-3] == end_date  # end_date parameter

    @pytest.mark.asyncio
    async def test_hybrid_search_with_similarity_thresholds(self):
        """Test hybrid_search filters by min_similarity_score and min_bm25_score"""
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_conn, mock_result = MagicMock(), MagicMock()

        now = datetime.now(timezone.utc)
        rows = [
            (
                1,
                "test-123",
                None,
                "High scoring chunk",
                0.8,
                now,
                now,
                5,
                now,
                {},
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

        repo = LongtermMemoryRepository(mock_pg, mock_neo4j)
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
            (1, "test-123", None, "Test chunk", 0.8, now, now, 15, now, {}, now, 0.9, 0.8, 0.7),
        ]
        mock_result.result.return_value = rows
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg.connection.return_value = mock_conn

        repo = LongtermMemoryRepository(mock_pg, mock_neo4j)
        results = await repo.hybrid_search(
            external_id="test-123", query_text="test query", query_embedding=[0.1] * 384, limit=10
        )

        assert len(results) == 1
        assert results[0].access_count == 15
        assert results[0].last_access is not None

    @pytest.mark.asyncio
    async def test_hybrid_search_combined_filters(self):
        """Test hybrid_search with multiple filters combined"""
        mock_pg, mock_neo4j = MagicMock(), MagicMock()
        mock_conn, mock_result = MagicMock(), MagicMock()

        now = datetime.now(timezone.utc)
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rows = [
            (
                1,
                "test-123",
                2,
                "Filtered chunk",
                0.9,
                start_date,
                now,
                7,
                now,
                {},
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

        repo = LongtermMemoryRepository(mock_pg, mock_neo4j)
        results = await repo.hybrid_search(
            external_id="test-123",
            query_text="test query",
            query_embedding=[0.1] * 384,
            limit=5,
            shortterm_memory_id=2,
            start_date=start_date,
            min_similarity_score=0.8,
            min_bm25_score=0.7,
        )

        assert len(results) == 1
        assert results[0].shortterm_memory_id == 2
        # Verify all filters were passed
        call_args = mock_conn.execute.call_args[0]
        params = call_args[1]
        assert params[-5] == 2  # shortterm_memory_id
        assert params[-4] == start_date  # start_date
        assert params[-2] == 0.8  # min_similarity_score
        assert params[-1] == 0.7  # min_bm25_score
