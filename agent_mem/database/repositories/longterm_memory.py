"""
Longterm Memory Repository.

Handles CRUD operations for longterm memory tier:
- LongtermMemoryChunk CRUD with temporal tracking
- Vector and BM25 search with confidence/importance filtering
- Entity and Relationship management in Neo4j
"""

import logging
import json
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from psqlpy.extra_types import PgVector

from agent_mem.database.postgres_manager import PostgreSQLManager
from agent_mem.database.neo4j_manager import Neo4jManager
from agent_mem.database.models import (
    LongtermMemoryChunk,
    LongtermEntity,
    LongtermRelationship,
)

logger = logging.getLogger(__name__)


def _convert_neo4j_datetime(dt):
    """Convert Neo4j DateTime to Python datetime."""
    if dt is None:
        return None
    if hasattr(dt, "to_native"):
        return dt.to_native()
    return dt


class LongtermMemoryRepository:
    """
    Repository for longterm memory operations.

    Longterm memory stores consolidated knowledge with temporal validity tracking,
    confidence scores, and importance rankings.
    """

    def __init__(self, postgres_manager: PostgreSQLManager, neo4j_manager: Neo4jManager):
        """
        Initialize repository.

        Args:
            postgres_manager: PostgreSQL connection manager
            neo4j_manager: Neo4j connection manager
        """
        self.postgres = postgres_manager
        self.neo4j = neo4j_manager

    # =========================================================================
    # LONGTERM MEMORY CHUNK CRUD
    # =========================================================================

    async def create_chunk(
        self,
        external_id: str,
        content: str,
        embedding: Optional[List[float]] = None,
        shortterm_memory_id: Optional[int] = None,
        importance: float = 0.5,
        start_date: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LongtermMemoryChunk:
        """
        Create a new longterm memory chunk.

        The BM25 vector is auto-populated by database trigger.

        Args:
            external_id: Agent identifier
            content: Chunk content
            embedding: Optional embedding vector
            shortterm_memory_id: Source shortterm memory ID (optional)
            importance: Importance for prioritization (0-1)
            start_date: When information became valid (defaults to now)
            metadata: Optional metadata

        Returns:
            Created LongtermMemoryChunk object
        """
        query = """
            INSERT INTO longterm_memory_chunk 
            (external_id, shortterm_memory_id, content, embedding, 
             importance, start_date, last_updated, access_count, last_access, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING id, external_id, shortterm_memory_id, content, 
                      importance, start_date, last_updated, access_count, last_access, metadata
        """

        # Convert embedding to PostgreSQL vector format if provided
        pg_vector = None
        if embedding:
            pg_vector = PgVector(embedding)

        current_time = datetime.now(timezone.utc)

        async with self.postgres.connection() as conn:
            result = await conn.execute(
                query,
                [
                    external_id,
                    shortterm_memory_id,
                    content,
                    pg_vector,
                    importance,
                    start_date or current_time,
                    current_time,  # last_updated
                    0,  # access_count
                    None,  # last_access
                    metadata or {},
                ],
            )

            row = result.result()[0]
            chunk = self._chunk_row_to_model(row)

            logger.info(f"Created longterm chunk {chunk.id} for {external_id}")
            return chunk

    async def get_chunk_by_id(self, chunk_id: int) -> Optional[LongtermMemoryChunk]:
        """
        Get a chunk by ID.

        Args:
            chunk_id: Chunk ID

        Returns:
            LongtermMemoryChunk or None if not found
        """
        query = """
            SELECT id, external_id, shortterm_memory_id, content, 
                   importance, start_date, last_updated, access_count, last_access, metadata
            FROM longterm_memory_chunk
            WHERE id = $1
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(query, [chunk_id])
            rows = result.result()

            if not rows:
                return None

            return self._chunk_row_to_model(rows[0])

    async def get_valid_chunks_by_external_id(
        self,
        external_id: str,
        limit: int = 100,
        min_importance: float = 0.0,
    ) -> List[LongtermMemoryChunk]:
        """
        Get all currently valid chunks for an external_id.

        Args:
            external_id: Agent identifier
            limit: Maximum number of chunks
            min_importance: Minimum importance score

        Returns:
            List of LongtermMemoryChunk objects
        """
        query = """
            SELECT id, external_id, shortterm_memory_id, content, 
                   importance, start_date, last_updated, access_count, last_access, metadata
            FROM longterm_memory_chunk
            WHERE external_id = $1 
              AND importance >= $2
            ORDER BY importance DESC, start_date DESC
            LIMIT $3
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(query, [external_id, min_importance, limit])
            rows = result.result()

            chunks = [self._chunk_row_to_model(row) for row in rows]

            logger.debug(f"Retrieved {len(chunks)} valid longterm chunks for {external_id}")
            return chunks

    async def get_chunks_by_temporal_range(
        self,
        external_id: str,
        start_date: datetime,
        end_date: datetime,
        limit: int = 100,
    ) -> List[LongtermMemoryChunk]:
        """
        Get chunks valid during a specific time range.

        Args:
            external_id: Agent identifier
            start_date: Range start
            end_date: Range end
            limit: Maximum chunks

        Returns:
            List of LongtermMemoryChunk objects
        """
        query = """
            SELECT id, external_id, shortterm_memory_id, content, 
                   importance, start_date, last_updated, access_count, last_access, metadata
            FROM longterm_memory_chunk
            WHERE external_id = $1 
              AND start_date >= $2
              AND start_date <= $3
            ORDER BY start_date DESC
            LIMIT $4
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(query, [external_id, start_date, end_date, limit])
            rows = result.result()

            chunks = [self._chunk_row_to_model(row) for row in rows]

            logger.debug(
                f"Retrieved {len(chunks)} temporal chunks for {external_id} "
                f"({start_date} to {end_date})"
            )
            return chunks

    async def update_chunk(
        self,
        chunk_id: int,
        content: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[LongtermMemoryChunk]:
        """
        Update a chunk.

        Args:
            chunk_id: Chunk ID
            content: New content (optional)
            embedding: New embedding (optional)
            importance: New importance (optional)
            metadata: New metadata (optional)

        Returns:
            Updated LongtermMemoryChunk or None if not found
        """
        updates = []
        params = []
        param_idx = 1

        if content is not None:
            updates.append(f"content = ${param_idx}")
            params.append(content)
            param_idx += 1

        if embedding is not None:
            embedding_str = f"[{','.join(map(str, embedding))}]"
            updates.append(f"embedding = ${param_idx}")
            params.append(embedding_str)
            param_idx += 1

        if importance is not None:
            updates.append(f"importance = ${param_idx}")
            params.append(importance)
            param_idx += 1

        if metadata is not None:
            updates.append(f"metadata = ${param_idx}")
            params.append(metadata)
            param_idx += 1

        if not updates:
            return await self.get_chunk_by_id(chunk_id)

        # Always update last_updated timestamp
        updates.append(f"last_updated = ${param_idx}")
        params.append(datetime.now(timezone.utc))
        param_idx += 1

        params.append(chunk_id)

        query = f"""
            UPDATE longterm_memory_chunk
            SET {", ".join(updates)}
            WHERE id = ${param_idx}
            RETURNING id, external_id, shortterm_memory_id, content, 
                      importance, start_date, last_updated, access_count, last_access, metadata
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(query, params)
            rows = result.result()

            if not rows:
                return None

            chunk = self._chunk_row_to_model(rows[0])
            logger.info(f"Updated longterm chunk {chunk_id}")
            return chunk

    async def supersede_chunk(self, chunk_id: int, end_date: Optional[datetime] = None) -> bool:
        """
        Mark a chunk as superseded by deleting it.
        (Note: end_date parameter is kept for backwards compatibility but is not used)

        Args:
            chunk_id: Chunk ID
            end_date: Deprecated parameter (not used)

        Returns:
            True if deleted
        """
        return await self.delete_chunk(chunk_id)

    async def delete_chunk(self, chunk_id: int) -> bool:
        """
        Delete a chunk.

        Args:
            chunk_id: Chunk ID

        Returns:
            True if deleted
        """
        query = "DELETE FROM longterm_memory_chunk WHERE id = $1"

        async with self.postgres.connection() as conn:
            await conn.execute(query, [chunk_id])
            logger.info(f"Deleted longterm chunk {chunk_id}")
            return True

    async def update_chunk_timestamps(self, shortterm_memory_id: int) -> int:
        """
        Update last_updated timestamp for all chunks from a shortterm memory.

        Only updates chunks where last_updated is NULL (first-time promotion).

        Args:
            shortterm_memory_id: Shortterm memory ID

        Returns:
            Number of chunks updated
        """
        query = """
            UPDATE longterm_memory_chunk
            SET last_updated = CURRENT_TIMESTAMP
            WHERE shortterm_memory_id = $1 AND last_updated IS NULL
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(query, [shortterm_memory_id])
            updated_count = len(result.result()) if result.result() else 0
            logger.info(
                f"Updated last_updated for {updated_count} chunks "
                f"from shortterm memory {shortterm_memory_id}"
            )
            return updated_count

    # =========================================================================
    # SEARCH OPERATIONS
    # =========================================================================
    async def hybrid_search(
        self,
        external_id: str,
        query_text: str,
        query_embedding: List[float],
        limit: int = 10,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        min_importance: float = 0.0,
        only_valid: bool = True,
    ) -> List[LongtermMemoryChunk]:
        """
        Hybrid search combining vector similarity and BM25.

        Args:
            external_id: Agent identifier
            query_text: Query text for BM25
            query_embedding: Query embedding for vector search
            limit: Maximum results
            vector_weight: Weight for vector similarity (0-1)
            bm25_weight: Weight for BM25 score (0-1)
            min_importance: Minimum importance score
            only_valid: Kept for backwards compatibility (not used)

        Returns:
            List of LongtermMemoryChunk with combined scores
        """
        pg_vector = PgVector(query_embedding)

        query = f"""
            WITH vector_results AS (
                SELECT 
                    id,
                    1 - (embedding <=> $1) AS vector_score
                FROM longterm_memory_chunk
                WHERE external_id = $2 AND embedding IS NOT NULL
            ),
            bm25_results AS (
                SELECT 
                    id,
                    content_bm25 <&> to_bm25query('idx_longterm_chunk_bm25', tokenize($3, 'bert')) AS bm25_score
                FROM longterm_memory_chunk
                WHERE external_id = $2 AND content_bm25 IS NOT NULL
            )
            SELECT 
                c.id, c.external_id, c.shortterm_memory_id, c.content, 
                c.importance, c.start_date, c.last_updated, c.access_count, c.last_access,
                c.metadata, c.created_at,
                COALESCE(v.vector_score, 0) * $4 + COALESCE(b.bm25_score, 0) * $5 AS combined_score
            FROM longterm_memory_chunk c
            LEFT JOIN vector_results v ON c.id = v.id
            LEFT JOIN bm25_results b ON c.id = b.id
            WHERE c.external_id = $2
              AND (v.vector_score IS NOT NULL OR b.bm25_score IS NOT NULL)
              AND c.importance >= $6
            ORDER BY combined_score DESC
            LIMIT $7
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(
                query,
                [
                    pg_vector,
                    external_id,
                    query_text,
                    vector_weight,
                    bm25_weight,
                    min_importance,
                    limit,
                ],
            )
            rows = result.result()

            chunks = []
            for row in rows:
                chunk = self._chunk_row_to_model(row[:11])
                chunk.similarity_score = float(row[11])  # Store combined score
                chunks.append(chunk)

            logger.debug(f"Hybrid search found {len(chunks)} longterm chunks for {external_id}")
            return chunks

    # =========================================================================
    # NEO4J ENTITY OPERATIONS
    # =========================================================================

    async def create_entity(
        self,
        external_id: str,
        name: str,
        types: List[str],
        description: Optional[str] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LongtermEntity:
        """
        Create a new longterm entity in Neo4j.

        Args:
            external_id: Agent identifier
            name: Entity name
            types: Entity types (e.g., ['Person', 'Developer'], ['Technology', 'Tool'])
            description: Optional description
            importance: Importance score (0-1)
            metadata: Optional metadata

        Returns:
            Created LongtermEntity object
        """
        now = datetime.now(timezone.utc)

        query = """
        CREATE (e:LongtermEntity {
            external_id: $external_id,
            name: $name,
            types: $types,
            description: $description,
            importance: $importance,
            access_count: $access_count,
            start_date: $now,
            last_updated: $now,
            metadata: $metadata_json
        })
        RETURN elementId(e) AS id, e.external_id AS external_id, e.name AS name, 
               e.types AS types, e.description AS description,
               e.importance AS importance, e.access_count AS access_count,
               e.start_date AS start_date, e.last_updated AS last_updated,
               e.metadata AS metadata
        """

        async with self.neo4j.session() as session:
            result = await session.run(
                query,
                external_id=external_id,
                name=name,
                types=types,
                description=description,
                importance=importance,
                access_count=0,
                now=now,
                metadata_json=json.dumps(metadata or {}),
            )
            record = await result.single()

            entity = LongtermEntity(
                id=record["id"],
                external_id=record["external_id"],
                name=record["name"],
                types=record["types"] or [],
                description=record["description"],
                importance=record["importance"],
                access_count=record["access_count"] or 0,
                start_date=_convert_neo4j_datetime(record["start_date"]),
                last_updated=_convert_neo4j_datetime(record["last_updated"]),
                metadata=json.loads(record["metadata"]) if record["metadata"] else {},
            )

            logger.info(f"Created longterm entity {entity.id}: {name}")
            return entity

    async def get_entity(self, entity_id: int) -> Optional[LongtermEntity]:
        """
        Get a longterm entity by ID.

        Args:
            entity_id: Entity node ID

        Returns:
            LongtermEntity or None if not found
        """
        query = """
        MATCH (e:LongtermEntity)
        WHERE id(e) = $entity_id
        RETURN elementId(e) AS id, e.external_id AS external_id, e.name AS name, 
               e.types AS types, e.description AS description,
               e.importance AS importance, e.access_count AS access_count, e.last_access AS last_access,
               e.metadata AS metadata
        """

        async with self.neo4j.session() as session:
            result = await session.run(query, entity_id=entity_id)
            record = await result.single()

            if not record:
                return None

            return LongtermEntity(
                id=record["id"],
                external_id=record["external_id"],
                name=record["name"],
                types=record["types"] or [],
                description=record["description"],
                importance=record["importance"],
                access_count=record["access_count"] or 0,
                last_access=_convert_neo4j_datetime(record["last_access"]),
                metadata=json.loads(record["metadata"]) if record["metadata"] else {},
            )

    async def get_entities_by_external_id(
        self,
        external_id: str,
        min_importance: float = 0.0,
    ) -> List[LongtermEntity]:
        """
        Get all longterm entities for an external_id.

        Args:
            external_id: Agent identifier
            min_importance: Minimum importance score filter

        Returns:
            List of LongtermEntity objects
        """
        query = """
        MATCH (e:LongtermEntity {external_id: $external_id})
        WHERE e.importance >= $min_importance
        RETURN elementId(e) AS id, e.external_id AS external_id, e.name AS name, 
               e.types AS types, e.description AS description,
               e.importance AS importance, e.access_count AS access_count, e.last_access AS last_access,
               e.metadata AS metadata
        ORDER BY e.importance DESC
        """

        async with self.neo4j.session() as session:
            result = await session.run(
                query,
                external_id=external_id,
                min_importance=min_importance,
            )
            records = [record async for record in result]

            entities = [
                LongtermEntity(
                    id=record["id"],
                    external_id=record["external_id"],
                    name=record["name"],
                    types=record["types"] or [],
                    description=record["description"],
                    importance=record["importance"],
                    access_count=record["access_count"] or 0,
                    last_access=_convert_neo4j_datetime(record["last_access"]),
                    metadata=json.loads(record["metadata"]) if record["metadata"] else {},
                )
                for record in records
            ]

            logger.debug(f"Retrieved {len(entities)} longterm entities for {external_id}")
            return entities

    async def update_entity(
        self,
        entity_id: int,
        types: Optional[List[str]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        confidence: Optional[float] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[LongtermEntity]:
        """
        Update a longterm entity.

        Args:
            entity_id: Entity node ID
            types: New types (optional)
            name: New name (optional)
            description: New description (optional)
            confidence: New confidence (optional)
            importance: New importance (optional)
            metadata: New metadata (optional)

        Returns:
            Updated LongtermEntity or None if not found
        """
        updates = []
        params = {"entity_id": entity_id, "now": datetime.now(timezone.utc)}

        if types is not None:
            updates.append("e.types = $types")
            params["types"] = types

        if name is not None:
            updates.append("e.name = $name")
            params["name"] = name

        if description is not None:
            updates.append("e.description = $description")
            params["description"] = description

        if confidence is not None:
            updates.append("e.importance = $confidence")
            params["confidence"] = confidence

        if importance is not None:
            updates.append("e.importance = $importance")
            params["importance"] = importance

        if metadata is not None:
            updates.append("e.metadata = $metadata")
            params["metadata"] = metadata

        if not updates:
            return await self.get_entity(entity_id)

        updates.append("e.last_updated = $now")

        query = f"""
        MATCH (e:LongtermEntity)
        WHERE id(e) = $entity_id
        SET {", ".join(updates)}
        RETURN elementId(e) AS id, e.external_id AS external_id, e.name AS name, 
               e.types AS types, e.description AS description,
               e.importance AS importance, e.access_count AS access_count, e.last_access AS last_access,
               e.metadata AS metadata
        """

        async with self.neo4j.session() as session:
            result = await session.run(query, **params)
            record = await result.single()

            if not record:
                return None

            entity = LongtermEntity(
                id=record["id"],
                external_id=record["external_id"],
                name=record["name"],
                types=record["types"] or [],
                description=record["description"],
                importance=record["importance"],
                access_count=record["access_count"] or 0,
                last_access=_convert_neo4j_datetime(record["last_access"]),
                metadata=json.loads(record["metadata"]) if record["metadata"] else {},
            )

            logger.info(f"Updated longterm entity {entity_id}")
            return entity

    async def update_entity_with_metadata(
        self,
        entity_id: int,
        importance: Optional[float] = None,
        last_access: Optional[datetime] = None,
        metadata_update: Optional[Dict[str, Any]] = None,
    ) -> Optional[LongtermEntity]:
        """
        Update entity and merge metadata updates array.

        Used during promotion to track importance changes over time.

        Args:
            entity_id: Entity node ID
            importance: New importance score
            last_access: Last access timestamp
            metadata_update: Dict with updates to merge (e.g., {"updates": [...]})

        Returns:
            Updated LongtermEntity or None if not found
        """
        updates = []
        params = {"entity_id": entity_id, "now": datetime.now(timezone.utc)}

        if importance is not None:
            updates.append("e.importance = $importance")
            params["importance"] = importance

        if last_access is not None:
            updates.append("e.last_access = $last_access")
            params["last_access"] = last_access

        updates.append("e.last_updated = $now")

        # Handle metadata merging
        if metadata_update:
            # Merge metadata - preserve existing fields and add/update new ones
            for key, value in metadata_update.items():
                updates.append(f"e.metadata = coalesce(e.metadata, {{}}) + ${key}_value")
                params[f"{key}_value"] = {key: value}

        query = f"""
        MATCH (e:LongtermEntity)
        WHERE id(e) = $entity_id
        SET {", ".join(updates)}
        RETURN elementId(e) AS id, e.external_id AS external_id, e.name AS name, 
               e.types AS types, e.description AS description,
               e.importance AS importance, e.access_count AS access_count,
               e.last_access AS last_access, e.metadata AS metadata
        """

        async with self.neo4j.session() as session:
            result = await session.run(query, **params)
            record = await result.single()

            if not record:
                return None

            entity = LongtermEntity(
                id=record["id"],
                external_id=record["external_id"],
                name=record["name"],
                types=record["types"] or [],
                description=record["description"],
                importance=record["importance"],
                access_count=record["access_count"] or 0,
                last_access=_convert_neo4j_datetime(record["last_access"]),
                metadata=json.loads(record["metadata"]) if record["metadata"] else {},
            )

            logger.info(f"Updated longterm entity {entity_id} with metadata tracking")
            return entity

    async def delete_entity(self, entity_id: int) -> bool:
        """
        Delete a longterm entity and all its relationships.

        Args:
            entity_id: Entity node ID

        Returns:
            True if deleted
        """
        query = """
        MATCH (e:LongtermEntity)
        WHERE id(e) = $entity_id
        DETACH DELETE e
        """

        async with self.neo4j.session() as session:
            await session.run(query, entity_id=entity_id)
            logger.info(f"Deleted longterm entity {entity_id}")
            return True

    # =========================================================================
    # NEO4J RELATIONSHIP OPERATIONS
    # =========================================================================

    async def create_relationship(
        self,
        external_id: str,
        from_entity_id: int,
        to_entity_id: int,
        types: List[str],
        description: Optional[str] = None,
        strength: float = 0.5,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LongtermRelationship:
        """
        Create a new longterm relationship between entities in Neo4j.

        Args:
            external_id: Agent identifier
            from_entity_id: Source entity node ID
            to_entity_id: Target entity node ID
            types: Relationship types (e.g., ['USES', 'DEPENDS_ON'])
            description: Optional description
            strength: Relationship strength (0-1)
            importance: Importance score (0-1)
            metadata: Optional metadata

        Returns:
            Created LongtermRelationship object
        """
        now = datetime.now(timezone.utc)

        # Convert metadata to JSON string for Neo4j storage
        metadata_json = json.dumps(metadata or {})

        query = """
        MATCH (from:LongtermEntity)
        WHERE elementId(from) = $from_entity_id
        MATCH (to:LongtermEntity)
        WHERE elementId(to) = $to_entity_id
        CREATE (from)-[r:LONGTERM_RELATES {
            external_id: $external_id,
            types: $types,
            description: $description,
            strength: $strength,
            importance: $importance,
            access_count: $access_count,
            start_date: $now,
            last_updated: $now,
            metadata: $metadata_json
        }]->(to)
        RETURN elementId(r) AS id, r.external_id AS external_id,
               elementId(from) AS from_entity_id, elementId(to) AS to_entity_id,
               from.name AS from_entity_name, to.name AS to_entity_name,
               r.types AS types, r.description AS description,
               r.importance AS strength, r.importance AS importance, r.access_count AS access_count,
               r.start_date AS start_date, r.last_updated AS last_updated,
               r.metadata AS metadata
        """

        async with self.neo4j.session() as session:
            result = await session.run(
                query,
                external_id=external_id,
                from_entity_id=from_entity_id,
                to_entity_id=to_entity_id,
                types=types,
                description=description,
                strength=strength,
                importance=importance,
                access_count=0,
                now=now,
                metadata_json=json.dumps(metadata or {}),
            )
            record = await result.single()

            relationship = LongtermRelationship(
                id=record["id"],
                external_id=record["external_id"],
                from_entity_id=record["from_entity_id"],
                to_entity_id=record["to_entity_id"],
                from_entity_name=record["from_entity_name"],
                to_entity_name=record["to_entity_name"],
                types=record["types"] or [],
                description=record["description"],
                strength=record["strength"],
                importance=record["importance"],
                access_count=record["access_count"] or 0,
                start_date=_convert_neo4j_datetime(record["start_date"]),
                last_updated=_convert_neo4j_datetime(record["last_updated"]),
                metadata=json.loads(record["metadata"]) if record["metadata"] else {},
            )

            logger.info(
                f"Created longterm relationship {relationship.id}: "
                f"{record['from_entity_name']} -> {record['to_entity_name']}"
            )
            return relationship

    async def get_relationship(self, relationship_id: int) -> Optional[LongtermRelationship]:
        """
        Get a longterm relationship by ID.

        Args:
            relationship_id: Relationship ID

        Returns:
            LongtermRelationship or None if not found
        """
        query = """
        MATCH (from:LongtermEntity)-[r:LONGTERM_RELATES]->(to:LongtermEntity)
        WHERE elementId(r) = $relationship_id
        RETURN elementId(r) AS id, r.external_id AS external_id,
               elementId(from) AS from_entity_id, elementId(to) AS to_entity_id,
               from.name AS from_entity_name, to.name AS to_entity_name,
               r.types AS types, r.description AS description,
               r.importance AS importance, r.access_count AS access_count, r.last_access AS last_access,
               r.metadata AS metadata
        """

        async with self.neo4j.session() as session:
            result = await session.run(query, relationship_id=relationship_id)
            record = await result.single()

            if not record:
                return None

            return LongtermRelationship(
                id=record["id"],
                external_id=record["external_id"],
                from_entity_id=record["from_entity_id"],
                to_entity_id=record["to_entity_id"],
                from_entity_name=record["from_entity_name"],
                to_entity_name=record["to_entity_name"],
                types=record["types"] or [],
                description=record["description"],
                importance=record["importance"],
                access_count=record["access_count"] or 0,
                last_access=_convert_neo4j_datetime(record["last_access"]),
                metadata=json.loads(record["metadata"]) if record["metadata"] else {},
            )

    async def get_relationships_by_external_id(
        self,
        external_id: str,
        min_importance: float = 0.0,
    ) -> List[LongtermRelationship]:
        """
        Get all longterm relationships for an external_id.

        Args:
            external_id: Agent identifier
            min_importance: Minimum importance score filter

        Returns:
            List of LongtermRelationship objects
        """
        query = """
        MATCH (from:LongtermEntity)-[r:LONGTERM_RELATES {external_id: $external_id}]->(to:LongtermEntity)
        WHERE r.importance >= $min_importance
        RETURN elementId(r) AS id, r.external_id AS external_id,
               elementId(from) AS from_entity_id, elementId(to) AS to_entity_id,
               from.name AS from_entity_name, to.name AS to_entity_name,
               r.types AS types, r.description AS description,
               r.importance AS importance, r.access_count AS access_count, r.last_access AS last_access,
               r.metadata AS metadata
        ORDER BY r.importance DESC
        """

        async with self.neo4j.session() as session:
            result = await session.run(
                query,
                external_id=external_id,
                min_importance=min_importance,
            )
            records = [record async for record in result]

            relationships = [
                LongtermRelationship(
                    id=record["id"],
                    external_id=record["external_id"],
                    from_entity_id=record["from_entity_id"],
                    to_entity_id=record["to_entity_id"],
                    from_entity_name=record["from_entity_name"],
                    to_entity_name=record["to_entity_name"],
                    types=record["types"] or [],
                    description=record["description"],
                    importance=record["importance"],
                    access_count=record["access_count"] or 0,
                    last_access=_convert_neo4j_datetime(record["last_access"]),
                    metadata=json.loads(record["metadata"]) if record["metadata"] else {},
                )
                for record in records
            ]

            logger.debug(f"Retrieved {len(relationships)} longterm relationships for {external_id}")
            return relationships

    async def update_relationship(
        self,
        relationship_id: int,
        types: Optional[List[str]] = None,
        description: Optional[str] = None,
        confidence: Optional[float] = None,
        strength: Optional[float] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[LongtermRelationship]:
        """
        Update a longterm relationship.

        Args:
            relationship_id: Relationship ID
            types: New types (optional)
            description: New description (optional)
            confidence: New confidence (optional)
            strength: New strength (optional)
            importance: New importance (optional)
            metadata: New metadata (optional)

        Returns:
            Updated LongtermRelationship or None if not found
        """
        updates = []
        params = {"relationship_id": relationship_id, "now": datetime.now(timezone.utc)}

        if types is not None:
            updates.append("r.types = $types")
            params["types"] = types

        if description is not None:
            updates.append("r.description = $description")
            params["description"] = description

        if confidence is not None:
            updates.append("r.importance = $confidence")
            params["confidence"] = confidence

        if strength is not None:
            updates.append("r.importance = $strength")
            params["strength"] = strength

        if importance is not None:
            updates.append("r.importance = $importance")
            params["importance"] = importance

        if metadata is not None:
            updates.append("r.metadata = $metadata")
            params["metadata"] = metadata

        if not updates:
            return await self.get_relationship(relationship_id)

        updates.append("r.last_updated = $now")

        query = f"""
        MATCH (from:LongtermEntity)-[r:LONGTERM_RELATES]->(to:LongtermEntity)
        WHERE elementId(r) = $relationship_id
        SET {", ".join(updates)}
        RETURN elementId(r) AS id, r.external_id AS external_id,
               elementId(from) AS from_entity_id, elementId(to) AS to_entity_id,
               from.name AS from_entity_name, to.name AS to_entity_name,
               r.types AS types, r.description AS description,
               r.importance AS importance, r.access_count AS access_count, r.last_access AS last_access,
               r.metadata AS metadata
        """

        async with self.neo4j.session() as session:
            result = await session.run(query, **params)
            record = await result.single()

            if not record:
                return None

            relationship = LongtermRelationship(
                id=record["id"],
                external_id=record["external_id"],
                from_entity_id=record["from_entity_id"],
                to_entity_id=record["to_entity_id"],
                from_entity_name=record["from_entity_name"],
                to_entity_name=record["to_entity_name"],
                types=record["types"] or [],
                description=record["description"],
                importance=record["importance"],
                access_count=record["access_count"] or 0,
                last_access=_convert_neo4j_datetime(record["last_access"]),
                metadata=json.loads(record["metadata"]) if record["metadata"] else {},
            )

            logger.info(f"Updated longterm relationship {relationship_id}")
            return relationship

    async def delete_relationship(self, relationship_id: int) -> bool:
        """
        Delete a longterm relationship.

        Args:
            relationship_id: Relationship ID

        Returns:
            True if deleted
        """
        query = """
        MATCH ()-[r:LONGTERM_RELATES]->()
        WHERE elementId(r) = $relationship_id
        DELETE r
        """

        async with self.neo4j.session() as session:
            await session.run(query, relationship_id=relationship_id)
            logger.info(f"Deleted longterm relationship {relationship_id}")
            return True

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _chunk_row_to_model(self, row) -> LongtermMemoryChunk:
        """Convert database row to LongtermMemoryChunk model."""
        return LongtermMemoryChunk(
            id=row[0],
            external_id=row[1],
            shortterm_memory_id=row[2],
            content=row[3],
            importance=float(row[4]),
            start_date=row[5],
            last_updated=row[6],
            access_count=int(row[7]) if row[7] is not None else 0,
            last_access=row[8],
            metadata=row[9] if isinstance(row[9], dict) else {},
        )
