"""
Tests for Neo4j manager (database/neo4j_manager.py).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_mem.config.settings import Config
from agent_mem.database.neo4j_manager import Neo4jManager


class TestNeo4jManager:
    """Test Neo4jManager class."""

    def test_initialization(self, test_config: Config):
        """Test Neo4jManager initialization."""
        manager = Neo4jManager(test_config)

        assert manager.config == test_config
        assert manager._driver is None
        assert manager._initialized is False

    @pytest.mark.asyncio
    async def test_initialize(self, test_config: Config):
        """Test initialize method creates driver."""
        with patch("agent_mem.database.neo4j_manager.AsyncGraphDatabase") as mock_graph:
            mock_driver = MagicMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_graph.driver.return_value = mock_driver

            manager = Neo4jManager(test_config)
            await manager.initialize()

            assert manager._driver is not None
            assert manager._initialized is True
            mock_graph.driver.assert_called_once()
            mock_driver.verify_connectivity.assert_called_once()

    @pytest.mark.asyncio
    async def test_close(self, test_config: Config):
        """Test closing the driver."""
        with patch("agent_mem.database.neo4j_manager.AsyncGraphDatabase") as mock_graph:
            mock_driver = MagicMock()
            mock_driver.close = AsyncMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_graph.driver.return_value = mock_driver

            manager = Neo4jManager(test_config)
            await manager.initialize()
            await manager.close()

            mock_driver.close.assert_called_once()
            assert manager._driver is None
            assert manager._initialized is False

    @pytest.mark.asyncio
    async def test_execute_read(self, test_config: Config):
        """Test execute_read method."""
        with patch("agent_mem.database.neo4j_manager.AsyncGraphDatabase") as mock_graph:
            mock_result = [{"name": "test", "value": 123}]

            # Mock the session and its run method
            mock_run_result = MagicMock()
            mock_run_result.data = AsyncMock(return_value=mock_result)

            mock_session = MagicMock()
            mock_session.run = AsyncMock(return_value=mock_run_result)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_driver = MagicMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver.session = MagicMock(return_value=mock_session)
            mock_graph.driver.return_value = mock_driver

            manager = Neo4jManager(test_config)
            await manager.initialize()

            result = await manager.execute_read("MATCH (n:Node) RETURN n", {"param": "value"})

            assert result == mock_result

    @pytest.mark.asyncio
    async def test_execute_write(self, test_config: Config):
        """Test execute_write method."""
        with patch("agent_mem.database.neo4j_manager.AsyncGraphDatabase") as mock_graph:
            mock_result = [{"created": 1}]

            # Mock the session and its run method
            mock_run_result = MagicMock()
            mock_run_result.data = AsyncMock(return_value=mock_result)

            mock_session = MagicMock()
            mock_session.run = AsyncMock(return_value=mock_run_result)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_driver = MagicMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver.session = MagicMock(return_value=mock_session)
            mock_graph.driver.return_value = mock_driver

            manager = Neo4jManager(test_config)
            await manager.initialize()

            result = await manager.execute_write(
                "CREATE (n:Node {name: $name}) RETURN n", {"name": "test"}
            )

            assert result == mock_result

    @pytest.mark.asyncio
    async def test_session_context_manager(self, test_config: Config):
        """Test using session as context manager."""
        with patch("agent_mem.database.neo4j_manager.AsyncGraphDatabase") as mock_graph:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()

            mock_driver = MagicMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver.session = MagicMock(return_value=mock_session)
            mock_graph.driver.return_value = mock_driver

            manager = Neo4jManager(test_config)
            await manager.initialize()

            async with manager.session() as session:
                assert session is not None

    @pytest.mark.asyncio
    async def test_error_handling_read(self, test_config: Config):
        """Test error handling in execute_read."""
        with patch("agent_mem.database.neo4j_manager.AsyncGraphDatabase") as mock_graph:
            mock_run_result = MagicMock()
            mock_run_result.data = AsyncMock(side_effect=Exception("Neo4j error"))

            mock_session = MagicMock()
            mock_session.run = AsyncMock(return_value=mock_run_result)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_driver = MagicMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver.session = MagicMock(return_value=mock_session)
            mock_graph.driver.return_value = mock_driver

            manager = Neo4jManager(test_config)
            await manager.initialize()

            with pytest.raises(Exception, match="Neo4j error"):
                await manager.execute_read("MATCH (n) RETURN n")

    @pytest.mark.asyncio
    async def test_error_handling_write(self, test_config: Config):
        """Test error handling in execute_write."""
        with patch("agent_mem.database.neo4j_manager.AsyncGraphDatabase") as mock_graph:
            mock_run_result = MagicMock()
            mock_run_result.data = AsyncMock(side_effect=Exception("Write error"))

            mock_session = MagicMock()
            mock_session.run = AsyncMock(return_value=mock_run_result)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_driver = MagicMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver.session = MagicMock(return_value=mock_session)
            mock_graph.driver.return_value = mock_driver

            manager = Neo4jManager(test_config)
            await manager.initialize()

            with pytest.raises(Exception, match="Write error"):
                await manager.execute_write("CREATE (n:Node)")

    @pytest.mark.asyncio
    async def test_parameterized_queries(self, test_config: Config):
        """Test parameterized queries with various data types."""
        with patch("agent_mem.database.neo4j_manager.AsyncGraphDatabase") as mock_graph:
            mock_result = [{"success": True}]

            mock_run_result = MagicMock()
            mock_run_result.data = AsyncMock(return_value=mock_result)

            mock_session = MagicMock()
            mock_session.run = AsyncMock(return_value=mock_run_result)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_driver = MagicMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver.session = MagicMock(return_value=mock_session)
            mock_graph.driver.return_value = mock_driver

            manager = Neo4jManager(test_config)
            await manager.initialize()

            params = {
                "name": "test",
                "count": 42,
                "active": True,
                "tags": ["tag1", "tag2"],
            }

            result = await manager.execute_write(
                "CREATE (n:Node {name: $name, count: $count, active: $active, tags: $tags})", params
            )

            assert result == mock_result

    @pytest.mark.asyncio
    async def test_empty_result(self, test_config: Config):
        """Test handling empty query results."""
        with patch("agent_mem.database.neo4j_manager.AsyncGraphDatabase") as mock_graph:
            mock_result = []

            mock_run_result = MagicMock()
            mock_run_result.data = AsyncMock(return_value=mock_result)

            mock_session = MagicMock()
            mock_session.run = AsyncMock(return_value=mock_run_result)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_driver = MagicMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver.session = MagicMock(return_value=mock_session)
            mock_graph.driver.return_value = mock_driver

            manager = Neo4jManager(test_config)
            await manager.initialize()

            result = await manager.execute_read("MATCH (n:NonExistent) RETURN n")

            assert result == []
