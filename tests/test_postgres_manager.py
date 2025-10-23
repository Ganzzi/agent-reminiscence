"""
Tests for PostgreSQL manager (database/postgres_manager.py).
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_reminiscence.config.settings import Config
from agent_reminiscence.database.postgres_manager import PostgreSQLManager


class TestPostgreSQLManager:
    """Test PostgreSQLManager class."""

    @pytest.mark.asyncio
    async def test_initialization(self, test_config: Config):
        """Test PostgreSQLManager initialization."""
        manager = PostgreSQLManager(test_config)
        assert manager.config == test_config
        assert manager._pool is None
        assert not manager._initialized

    @pytest.mark.asyncio
    async def test_initialize_creates_pool(self, test_config: Config):
        """Test that initialize creates connection pool."""
        manager = PostgreSQLManager(test_config)

        with patch("agent_reminiscence.database.postgres_manager.ConnectionPool") as mock_pool:
            mock_instance = MagicMock()
            mock_pool.return_value = mock_instance

            await manager.initialize()

            mock_pool.assert_called_once()
            assert manager._pool is not None
            assert manager._initialized

    @pytest.mark.asyncio
    async def test_close(self, test_config: Config):
        """Test closing the connection pool."""
        manager = PostgreSQLManager(test_config)
        
        # Mock the pool
        mock_pool = MagicMock()
        mock_pool.close = MagicMock()  # close() is synchronous
        manager._pool = mock_pool
        manager._initialized = True

        await manager.close()

        mock_pool.close.assert_called_once()
        assert manager._pool is None
        assert not manager._initialized

    @pytest.mark.asyncio
    async def test_close_when_no_pool(self, test_config: Config):
        """Test close when pool doesn't exist."""
        manager = PostgreSQLManager(test_config)
        await manager.close()  # Should not raise error
        assert manager._pool is None

    @pytest.mark.asyncio
    async def test_execute_query(self, test_config: Config):
        """Test execute method."""
        manager = PostgreSQLManager(test_config)
        
        # Mock connection and pool
        mock_connection = MagicMock()
        mock_result = MagicMock()
        mock_connection.execute = AsyncMock(return_value=mock_result)

        mock_pool = MagicMock()
        mock_pool.connection = AsyncMock(return_value=mock_connection)
        
        manager._pool = mock_pool
        manager._initialized = True

        # Create async context manager
        mock_connection.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_connection.__aexit__ = AsyncMock(return_value=None)

        result = await manager.execute("INSERT INTO test VALUES ($1)", ["value"])

        mock_connection.execute.assert_called_once_with("INSERT INTO test VALUES ($1)", ["value"])

    @pytest.mark.asyncio
    async def test_execute_many(self, test_config: Config):
        """Test execute_many method."""
        manager = PostgreSQLManager(test_config)
        
        # Mock connection
        mock_connection = MagicMock()
        mock_result1 = MagicMock()
        mock_result2 = MagicMock()
        mock_connection.execute = AsyncMock(side_effect=[mock_result1, mock_result2])

        mock_pool = MagicMock()
        mock_pool.connection = AsyncMock(return_value=mock_connection)
        
        manager._pool = mock_pool
        manager._initialized = True

        mock_connection.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_connection.__aexit__ = AsyncMock(return_value=None)

        results = await manager.execute_many(
            "INSERT INTO test (col1, col2) VALUES ($1, $2)",
            [["val1", "val2"], ["val3", "val4"]]
        )

        assert len(results) == 2
        assert mock_connection.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_context_manager(self, test_config: Config):
        """Test using PostgreSQLManager as context manager."""
        with patch("agent_reminiscence.database.postgres_manager.ConnectionPool") as mock_pool:
            mock_instance = MagicMock()
            mock_instance.close = MagicMock()
            mock_pool.return_value = mock_instance

            async with PostgreSQLManager(test_config) as manager:
                assert manager._initialized
                assert manager._pool is not None

            # Pool should be closed after exiting context
            mock_instance.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling(self, test_config: Config):
        """Test error handling when manager not initialized."""
        manager = PostgreSQLManager(test_config)
        
        # Don't initialize - should raise error
        with pytest.raises(RuntimeError, match="PostgreSQL manager not initialized"):
            await manager.execute("SELECT 1")

    @pytest.mark.asyncio
    async def test_verify_connection(self, test_config: Config):
        """Test connection verification."""
        manager = PostgreSQLManager(test_config)
        
        # Mock successful connection
        mock_connection = MagicMock()
        mock_result = MagicMock()
        mock_connection.execute = AsyncMock(return_value=mock_result)

        mock_pool = MagicMock()
        mock_pool.connection = AsyncMock(return_value=mock_connection)
        
        manager._pool = mock_pool
        manager._initialized = True

        mock_connection.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_connection.__aexit__ = AsyncMock(return_value=None)

        result = await manager.verify_connection()
        
        assert result is True
        mock_connection.execute.assert_called_once_with("SELECT 1")


