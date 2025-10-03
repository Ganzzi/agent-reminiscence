"""
Neo4j connection management.

Provides driver and session management for the agent_mem package.
"""

import logging
from typing import Optional, Any, Dict, List
from contextlib import asynccontextmanager

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import ServiceUnavailable, AuthError

from agent_mem.config import Config

logger = logging.getLogger(__name__)


class Neo4jManager:
    """
    Neo4j connection manager for agent_mem.

    Manages driver lifecycle and provides sessions for graph operations.
    """

    def __init__(self, config: Config):
        """
        Initialize Neo4j manager.

        Args:
            config: Configuration object with Neo4j settings
        """
        self.config = config
        self._driver: Optional[AsyncDriver] = None
        self._initialized = False

        logger.info(f"Neo4j manager configured for {config.neo4j_uri}")

    async def initialize(self) -> None:
        """Initialize the Neo4j driver."""
        if self._initialized:
            logger.warning("Neo4j manager already initialized")
            return

        if self._driver is None:
            auth = (self.config.neo4j_user, self.config.neo4j_password)

            self._driver = AsyncGraphDatabase.driver(
                self.config.neo4j_uri,
                auth=auth,
                max_connection_pool_size=50,
                max_transaction_retry_time=30.0,
            )

            # Verify connectivity
            try:
                await self._driver.verify_connectivity()
                self._initialized = True
                logger.info(f"Neo4j driver initialized successfully: {self.config.neo4j_uri}")
            except ServiceUnavailable as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                await self.close()
                raise RuntimeError(
                    f"Could not connect to Neo4j at {self.config.neo4j_uri}. "
                    f"Ensure Neo4j is running and credentials are correct."
                )
            except AuthError as e:
                logger.error(f"Neo4j authentication failed: {e}")
                await self.close()
                raise RuntimeError(f"Neo4j authentication failed. Check username/password.")

    async def close(self) -> None:
        """Close the Neo4j driver and clean up resources."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            self._initialized = False
            logger.info("Neo4j driver closed")

    @asynccontextmanager
    async def session(
        self,
        database: Optional[str] = None,
        default_access_mode: str = "WRITE",
    ):
        """
        Get a session (context manager).

        Usage:
            async with manager.session() as session:
                result = await session.run("CREATE (n:Node {name: $name})", name="test")

        Args:
            database: Database name (uses config default if not specified)
            default_access_mode: "READ" or "WRITE"

        Yields:
            Neo4j session
        """
        if not self._initialized or not self._driver:
            raise RuntimeError("Neo4j manager not initialized. Call initialize() first.")

        db = database or self.config.neo4j_database

        async with self._driver.session(
            database=db,
            default_access_mode=default_access_mode,
        ) as session:
            yield session

    async def execute_write(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a write query.

        Args:
            query: Cypher query to execute
            parameters: Query parameters (optional)
            database: Database name (optional, uses config default)

        Returns:
            List of result records as dictionaries

        Example:
            result = await manager.execute_write(
                "CREATE (n:Node {name: $name}) RETURN n",
                {"name": "test"}
            )
        """
        async with self.session(database=database, default_access_mode="WRITE") as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records

    async def execute_read(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a read query.

        Args:
            query: Cypher query to execute
            parameters: Query parameters (optional)
            database: Database name (optional, uses config default)

        Returns:
            List of result records as dictionaries

        Example:
            result = await manager.execute_read(
                "MATCH (n:Node {name: $name}) RETURN n",
                {"name": "test"}
            )
        """
        async with self.session(database=database, default_access_mode="READ") as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records

    async def verify_connection(self) -> bool:
        """
        Verify that the Neo4j connection is working.

        Returns:
            True if connection is working, False otherwise
        """
        try:
            await self._driver.verify_connectivity()
            return True
        except Exception as e:
            logger.error(f"Failed to verify Neo4j connection: {e}")
            return False

    async def ensure_constraints(self) -> None:
        """
        Ensure required constraints exist in Neo4j.

        Creates unique constraints for entity IDs to prevent duplicates.
        """
        constraints = [
            # Shortterm entity constraints
            """
            CREATE CONSTRAINT shortterm_entity_id_unique IF NOT EXISTS
            FOR (e:ShorttermEntity) REQUIRE e.id IS UNIQUE
            """,
            # Longterm entity constraints
            """
            CREATE CONSTRAINT longterm_entity_id_unique IF NOT EXISTS
            FOR (e:LongtermEntity) REQUIRE e.id IS UNIQUE
            """,
        ]

        for constraint in constraints:
            try:
                await self.execute_write(constraint)
                logger.info(f"Ensured constraint exists")
            except Exception as e:
                logger.warning(f"Could not create constraint: {e}")

    async def create_indexes(self) -> None:
        """
        Create indexes on commonly queried properties.

        Indexes improve query performance for searches by external_id, name, type, etc.
        """
        indexes = [
            # Shortterm entity indexes
            "CREATE INDEX shortterm_entity_external_id IF NOT EXISTS FOR (e:ShorttermEntity) ON (e.external_id)",
            "CREATE INDEX shortterm_entity_name IF NOT EXISTS FOR (e:ShorttermEntity) ON (e.name)",
            "CREATE INDEX shortterm_entity_type IF NOT EXISTS FOR (e:ShorttermEntity) ON (e.type)",
            # Longterm entity indexes
            "CREATE INDEX longterm_entity_external_id IF NOT EXISTS FOR (e:LongtermEntity) ON (e.external_id)",
            "CREATE INDEX longterm_entity_name IF NOT EXISTS FOR (e:LongtermEntity) ON (e.name)",
            "CREATE INDEX longterm_entity_type IF NOT EXISTS FOR (e:LongtermEntity) ON (e.type)",
        ]

        for index in indexes:
            try:
                await self.execute_write(index)
                logger.info(f"Ensured index exists")
            except Exception as e:
                logger.warning(f"Could not create index: {e}")

    async def clear_all_data(self, confirm: bool = False) -> None:
        """
        Delete all nodes and relationships in the database.

        WARNING: This is destructive and irreversible!

        Args:
            confirm: Must be True to proceed (safety check)
        """
        if not confirm:
            raise ValueError("Must pass confirm=True to clear all data. This is irreversible!")

        logger.warning("Clearing all data from Neo4j database")
        await self.execute_write("MATCH (n) DETACH DELETE n")
        logger.info("All data cleared from Neo4j")

    # Context manager support
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
