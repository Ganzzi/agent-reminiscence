"""
Central Storage Service - Singleton storage for memory retrieval data.

This service provides a singleton storage that maintains memory data per external_id.
Used by the Memory Retriever Agent to store raw data during search operations,
allowing efficient pointer-based references instead of including full content.
"""

import logging
from typing import Dict, Optional
from threading import Lock

from agent_mem.database.models import (
    ShorttermMemoryChunk,
    LongtermMemoryChunk,
    ShorttermEntity,
    LongtermEntity,
    ShorttermRelationship,
    LongtermRelationship,
)

logger = logging.getLogger(__name__)


class CentralStorage:
    """
    Singleton central storage for raw memory data retrieved during search.

    Stores chunks, entities, and relationships indexed by external_id and pointer_id.
    Allows agent to reference data without including full content in responses.

    This is a singleton to ensure all retrieval operations share the same storage
    and can access data across multiple concurrent requests.
    """

    _instance: Optional["CentralStorage"] = None
    _lock: Lock = Lock()

    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize storage dictionaries (only once)."""
        if self._initialized:
            return

        # Storage format: {external_id: {pointer_id: data}}
        self._chunks: Dict[str, Dict[str, ShorttermMemoryChunk | LongtermMemoryChunk]] = {}
        self._entities: Dict[str, Dict[str, ShorttermEntity | LongtermEntity]] = {}
        self._relationships: Dict[str, Dict[str, ShorttermRelationship | LongtermRelationship]] = {}

        self._initialized = True
        logger.debug("CentralStorage singleton initialized")

    def _ensure_external_id_exists(self, external_id: str) -> None:
        """Ensure storage dictionaries exist for an external_id."""
        if external_id not in self._chunks:
            self._chunks[external_id] = {}
        if external_id not in self._entities:
            self._entities[external_id] = {}
        if external_id not in self._relationships:
            self._relationships[external_id] = {}

    def store_chunk(
        self,
        external_id: str,
        tool_call_id: str,
        chunk: ShorttermMemoryChunk | LongtermMemoryChunk,
    ) -> str:
        """
        Store a chunk and return its pointer ID.

        Args:
            external_id: Agent identifier
            tool_call_id: Unique tool call identifier
            chunk: Memory chunk to store

        Returns:
            Pointer ID for referencing the chunk
        """
        self._ensure_external_id_exists(external_id)
        pointer_id = f"{tool_call_id}:chunk:{chunk.id}"
        self._chunks[external_id][pointer_id] = chunk
        return pointer_id

    def store_entity(
        self, external_id: str, tool_call_id: str, entity: ShorttermEntity | LongtermEntity
    ) -> str:
        """
        Store an entity and return its pointer ID.

        Args:
            external_id: Agent identifier
            tool_call_id: Unique tool call identifier
            entity: Entity to store

        Returns:
            Pointer ID for referencing the entity
        """
        self._ensure_external_id_exists(external_id)
        pointer_id = f"{tool_call_id}:entity:{entity.id}"
        self._entities[external_id][pointer_id] = entity
        return pointer_id

    def store_relationship(
        self,
        external_id: str,
        tool_call_id: str,
        relationship: ShorttermRelationship | LongtermRelationship,
    ) -> str:
        """
        Store a relationship and return its pointer ID.

        Args:
            external_id: Agent identifier
            tool_call_id: Unique tool call identifier
            relationship: Relationship to store

        Returns:
            Pointer ID for referencing the relationship
        """
        self._ensure_external_id_exists(external_id)
        pointer_id = f"{tool_call_id}:relationship:{relationship.id}"
        self._relationships[external_id][pointer_id] = relationship
        return pointer_id

    def get_chunk(
        self, external_id: str, pointer_id: str
    ) -> Optional[ShorttermMemoryChunk | LongtermMemoryChunk]:
        """
        Retrieve a chunk by pointer ID.

        Args:
            external_id: Agent identifier
            pointer_id: Pointer ID to retrieve

        Returns:
            Memory chunk if found, None otherwise
        """
        return self._chunks.get(external_id, {}).get(pointer_id)

    def get_entity(
        self, external_id: str, pointer_id: str
    ) -> Optional[ShorttermEntity | LongtermEntity]:
        """
        Retrieve an entity by pointer ID.

        Args:
            external_id: Agent identifier
            pointer_id: Pointer ID to retrieve

        Returns:
            Entity if found, None otherwise
        """
        return self._entities.get(external_id, {}).get(pointer_id)

    def get_relationship(
        self, external_id: str, pointer_id: str
    ) -> Optional[ShorttermRelationship | LongtermRelationship]:
        """
        Retrieve a relationship by pointer ID.

        Args:
            external_id: Agent identifier
            pointer_id: Pointer ID to retrieve

        Returns:
            Relationship if found, None otherwise
        """
        return self._relationships.get(external_id, {}).get(pointer_id)

    def clear_external_id(self, external_id: str) -> None:
        """
        Clear all stored data for a specific external_id.

        Args:
            external_id: Agent identifier to clear
        """
        if external_id in self._chunks:
            del self._chunks[external_id]
        if external_id in self._entities:
            del self._entities[external_id]
        if external_id in self._relationships:
            del self._relationships[external_id]
        logger.debug(f"Cleared storage for external_id={external_id}")

    def clear_all(self) -> None:
        """Clear all stored data for all external_ids."""
        self._chunks.clear()
        self._entities.clear()
        self._relationships.clear()
        logger.debug("Cleared all central storage")

    def get_stats(self, external_id: Optional[str] = None) -> Dict[str, int]:
        """
        Get storage statistics.

        Args:
            external_id: Optional external_id to get stats for. If None, returns total stats.

        Returns:
            Dictionary with counts of stored items
        """
        if external_id:
            return {
                "chunks": len(self._chunks.get(external_id, {})),
                "entities": len(self._entities.get(external_id, {})),
                "relationships": len(self._relationships.get(external_id, {})),
            }
        else:
            return {
                "total_external_ids": len(
                    set(
                        list(self._chunks.keys())
                        + list(self._entities.keys())
                        + list(self._relationships.keys())
                    )
                ),
                "total_chunks": sum(len(chunks) for chunks in self._chunks.values()),
                "total_entities": sum(len(entities) for entities in self._entities.values()),
                "total_relationships": sum(len(rels) for rels in self._relationships.values()),
            }


def get_central_storage() -> CentralStorage:
    """
    Get the singleton CentralStorage instance.

    Returns:
        CentralStorage singleton instance
    """
    return CentralStorage()
