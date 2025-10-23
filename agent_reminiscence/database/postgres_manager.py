"""
PostgreSQL connection management using PSQLPy.

Provides connection pool management for the agent_mem package.
"""

import logging
from typing import Optional
from contextlib import asynccontextmanager

from psqlpy import ConnectionPool, Connection

from agent_reminiscence.config import Config

logger = logging.getLogger(__name__)


class PostgreSQLManager:
    """
    PostgreSQL connection manager for agent_mem.

    Manages connection pool lifecycle and provides connections
    for repository operations.
    """

    def __init__(self, config: Config):
        """
        Initialize PostgreSQL manager.

        Args:
            config: Configuration object with database settings
        """
        self.config = config
        self._pool: Optional[ConnectionPool] = None
        self._initialized = False

        # Build DSN from config
        self.dsn = (
            f"postgresql://{config.postgres_user}:{config.postgres_password}"
            f"@{config.postgres_host}:{config.postgres_port}/{config.postgres_db}"
        )

        logger.info(
            f"PostgreSQL manager configured for {config.postgres_host}:{config.postgres_port}/{config.postgres_db}"
        )

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        if self._initialized:
            logger.warning("PostgreSQL manager already initialized")
            return

        if self._pool is None:
            self._pool = ConnectionPool(
                dsn=self.dsn,
                max_db_pool_size=10,  # Reasonable default for agent operations
            )
            self._initialized = True
            logger.info("PostgreSQL connection pool initialized (max_size=10)")

    async def close(self) -> None:
        """Close the connection pool and clean up resources."""
        if self._pool:
            self._pool.close()
            self._pool = None
            self._initialized = False
            logger.info("PostgreSQL connection pool closed")

    @asynccontextmanager
    async def connection(self):
        """
        Get a connection from the pool (context manager).

        Usage:
            async with manager.connection() as conn:
                result = await conn.execute("SELECT * FROM table")

        Yields:
            Connection from the pool
        """
        if not self._initialized or not self._pool:
            raise RuntimeError("PostgreSQL manager not initialized. Call initialize() first.")

        conn: Connection = await self._pool.connection()
        try:
            yield conn
        finally:
            # Connection is automatically returned to pool when context exits
            pass

    async def get_connection(self) -> Connection:
        """
        Get a connection from the pool (manual management).

        Note: Prefer using the connection() context manager for automatic cleanup.

        Returns:
            Connection from the pool
        """
        if not self._initialized or not self._pool:
            raise RuntimeError("PostgreSQL manager not initialized. Call initialize() first.")

        return await self._pool.connection()

    async def execute(self, query: str, parameters: Optional[list] = None):
        """
        Execute a query using a connection from the pool.

        Args:
            query: SQL query to execute
            parameters: Query parameters (optional)

        Returns:
            Query result

        Example:
            result = await manager.execute(
                "SELECT * FROM active_memory WHERE external_id = $1",
                [external_id]
            )
        """
        async with self.connection() as conn:
            return await conn.execute(query, parameters or [])

    async def execute_many(self, query: str, parameters_list: list[list]):
        """
        Execute a query multiple times with different parameters.

        Args:
            query: SQL query to execute
            parameters_list: List of parameter lists

        Returns:
            List of query results

        Example:
            results = await manager.execute_many(
                "INSERT INTO table (col1, col2) VALUES ($1, $2)",
                [["val1", "val2"], ["val3", "val4"]]
            )
        """
        async with self.connection() as conn:
            results = []
            for parameters in parameters_list:
                result = await conn.execute(query, parameters)
                results.append(result)
            return results

    async def verify_connection(self) -> bool:
        """
        Verify that the database connection is working.

        Returns:
            True if connection is working, False otherwise
        """
        try:
            async with self.connection() as conn:
                result = await conn.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Failed to verify PostgreSQL connection: {e}")
            return False

    async def ensure_extensions(self) -> None:
        """
        Ensure required PostgreSQL extensions are installed.

        Required extensions:
        - vector: For vector similarity search
        - pg_tokenizer: For text tokenization
        - vchord_bm25: For BM25 full-text search
        """
        extensions = ["vector", "pg_tokenizer", "vchord_bm25"]

        for extension in extensions:
            try:
                query = f"CREATE EXTENSION IF NOT EXISTS {extension} CASCADE"
                await self.execute(query)
                logger.info(f"Ensured extension exists: {extension}")
            except Exception as e:
                logger.warning(
                    f"Could not create extension {extension}: {e}. "
                    f"You may need superuser privileges."
                )

    async def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            table_name: Name of the table to check

        Returns:
            True if table exists, False otherwise
        """
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = $1
            )
        """
        async with self.connection() as conn:
            result = await conn.execute(query, [table_name])
            row = result.result()[0]
            return row[0] if row else False

    # Context manager support
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


