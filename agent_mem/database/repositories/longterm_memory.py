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

from agent_mem.database.postgres_manager import PostgreSQLManager
from agent_mem.database.neo4j_manager import Neo4jManager
from agent_mem.database.models import (
    LongtermMemoryChunk,
    LongtermEntity,
    LongtermRelationship,
)

logger = logging.getLogger(__name__)


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
        chunk_order: int,
        embedding: Optional[List[float]] = None,
        shortterm_memory_id: Optional[int] = None,
        confidence_score: float = 0.5,
        importance_score: float = 0.5,
        start_date: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LongtermMemoryChunk:
        """
        Create a new longterm memory chunk.

        The BM25 vector is auto-populated by database trigger.

        Args:
            external_id: Agent identifier
            content: Chunk content
            chunk_order: Order of chunk
            embedding: Optional embedding vector
            shortterm_memory_id: Source shortterm memory ID (optional)
            confidence_score: Confidence in information accuracy (0-1)
            importance_score: Importance for prioritization (0-1)
            start_date: When information became valid (defaults to now)
            metadata: Optional metadata

        Returns:
            Created LongtermMemoryChunk object
        """
        query = """
            INSERT INTO longterm_memory_chunk 
            (external_id, shortterm_memory_id, chunk_order, content, embedding, 
             confidence_score, importance_score, start_date, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id, external_id, shortterm_memory_id, chunk_order, content, 
                      confidence_score, importance_score, start_date, end_date, 
                      metadata, created_at
        """

        # Convert embedding to PostgreSQL vector format if provided
        embedding_str = None
        if embedding:
            embedding_str = f"[{','.join(map(str, embedding))}]"

        async with self.postgres.connection() as conn:
            result = await conn.execute(
                query,
                [
                    external_id,
                    shortterm_memory_id,
                    chunk_order,
                    content,
                    embedding_str,
                    confidence_score,
                    importance_score,
                    start_date or datetime.now(timezone.utc),
                    json.dumps(metadata or {}),
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
            SELECT id, external_id, shortterm_memory_id, chunk_order, content, 
                   confidence_score, importance_score, start_date, end_date, 
                   metadata, created_at
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
        min_confidence: float = 0.0,
        min_importance: float = 0.0,
    ) -> List[LongtermMemoryChunk]:
        """
        Get all currently valid chunks for an external_id.

        Valid chunks have end_date = NULL.

        Args:
            external_id: Agent identifier
            limit: Maximum number of chunks
            min_confidence: Minimum confidence score
            min_importance: Minimum importance score

        Returns:
            List of LongtermMemoryChunk objects
        """
        query = """
            SELECT id, external_id, shortterm_memory_id, chunk_order, content, 
                   confidence_score, importance_score, start_date, end_date, 
                   metadata, created_at
            FROM longterm_memory_chunk
            WHERE external_id = $1 
              AND end_date IS NULL
              AND confidence_score >= $2
              AND importance_score >= $3
            ORDER BY importance_score DESC, confidence_score DESC, start_date DESC
            LIMIT $4
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(query, [external_id, min_confidence, min_importance, limit])
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
            SELECT id, external_id, shortterm_memory_id, chunk_order, content, 
                   confidence_score, importance_score, start_date, end_date, 
                   metadata, created_at
            FROM longterm_memory_chunk
            WHERE external_id = $1 
              AND start_date <= $3
              AND (end_date IS NULL OR end_date >= $2)
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
        confidence_score: Optional[float] = None,
        importance_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[LongtermMemoryChunk]:
        """
        Update a chunk.

        Args:
            chunk_id: Chunk ID
            content: New content (optional)
            embedding: New embedding (optional)
            confidence_score: New confidence (optional)
            importance_score: New importance (optional)
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

        if confidence_score is not None:
            updates.append(f"confidence_score = ${param_idx}")
            params.append(confidence_score)
            param_idx += 1

        if importance_score is not None:
            updates.append(f"importance_score = ${param_idx}")
            params.append(importance_score)
            param_idx += 1

        if metadata is not None:
            updates.append(f"metadata = ${param_idx}")
            params.append(json.dumps(metadata))
            param_idx += 1

        if not updates:
            return await self.get_chunk_by_id(chunk_id)

        params.append(chunk_id)

        query = f"""
            UPDATE longterm_memory_chunk
            SET {", ".join(updates)}
            WHERE id = ${param_idx}
            RETURNING id, external_id, shortterm_memory_id, chunk_order, content, 
                      confidence_score, importance_score, start_date, end_date, 
                      metadata, created_at
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
        Mark a chunk as superseded by setting its end_date.

        Args:
            chunk_id: Chunk ID
            end_date: End date (defaults to now)

        Returns:
            True if updated, False if not found
        """
        query = """
            UPDATE longterm_memory_chunk
            SET end_date = $2
            WHERE id = $1 AND end_date IS NULL
        """

        async with self.postgres.connection() as conn:
            await conn.execute(query, [chunk_id, end_date or datetime.now(timezone.utc)])
            logger.info(f"Superseded longterm chunk {chunk_id}")
            return True

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

    # =========================================================================
    # SEARCH OPERATIONS
    # =========================================================================

    async def vector_search(
        self,
        external_id: str,
        query_embedding: List[float],
        limit: int = 10,
        min_similarity: float = 0.0,
        min_confidence: float = 0.0,
        min_importance: float = 0.0,
        only_valid: bool = True,
    ) -> List[LongtermMemoryChunk]:
        """
        Search chunks by vector similarity.

        Args:
            external_id: Agent identifier
            query_embedding: Query embedding vector
            limit: Maximum results
            min_similarity: Minimum cosine similarity (0-1)
            min_confidence: Minimum confidence score
            min_importance: Minimum importance score
            only_valid: Only return chunks with end_date = NULL

        Returns:
            List of LongtermMemoryChunk with similarity_score set
        """
        embedding_str = f"[{','.join(map(str, query_embedding))}]"

        valid_clause = "AND end_date IS NULL" if only_valid else ""

        query = f"""
            SELECT 
                id, external_id, shortterm_memory_id, chunk_order, content, 
                confidence_score, importance_score, start_date, end_date, 
                metadata, created_at,
                1 - (embedding <=> $1::vector) AS similarity
            FROM longterm_memory_chunk
            WHERE external_id = $2 
              AND embedding IS NOT NULL
              AND (1 - (embedding <=> $1::vector)) >= $3
              AND confidence_score >= $4
              AND importance_score >= $5
              {valid_clause}
            ORDER BY embedding <=> $1::vector
            LIMIT $6
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(
                query,
                [
                    embedding_str,
                    external_id,
                    min_similarity,
                    min_confidence,
                    min_importance,
                    limit,
                ],
            )
            rows = result.result()

            chunks = []
            for row in rows:
                chunk = self._chunk_row_to_model(row[:11])
                chunk.similarity_score = float(row[11])
                chunks.append(chunk)

            logger.debug(f"Vector search found {len(chunks)} longterm chunks for {external_id}")
            return chunks

    async def bm25_search(
        self,
        external_id: str,
        query_text: str,
        limit: int = 10,
        min_confidence: float = 0.0,
        min_importance: float = 0.0,
        only_valid: bool = True,
    ) -> List[LongtermMemoryChunk]:
        """
        Search chunks by BM25 keyword matching.

        Args:
            external_id: Agent identifier
            query_text: Query text
            limit: Maximum results
            min_confidence: Minimum confidence score
            min_importance: Minimum importance score
            only_valid: Only return chunks with end_date = NULL

        Returns:
            List of LongtermMemoryChunk with bm25_score set
        """
        valid_clause = "AND end_date IS NULL" if only_valid else ""

        query = f"""
            SELECT 
                id, external_id, shortterm_memory_id, chunk_order, content, 
                confidence_score, importance_score, start_date, end_date, 
                metadata, created_at,
                bm25_score(content_bm25, tokenize($1, 'bert')) AS score
            FROM longterm_memory_chunk
            WHERE external_id = $2
              AND content_bm25 IS NOT NULL
              AND confidence_score >= $3
              AND importance_score >= $4
              {valid_clause}
            ORDER BY score DESC
            LIMIT $5
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(
                query, [query_text, external_id, min_confidence, min_importance, limit]
            )
            rows = result.result()

            chunks = []
            for row in rows:
                chunk = self._chunk_row_to_model(row[:11])
                chunk.bm25_score = float(row[11]) if row[11] is not None else 0.0
                chunks.append(chunk)

            logger.debug(f"BM25 search found {len(chunks)} longterm chunks for {external_id}")
            return chunks

    async def hybrid_search(
        self,
        external_id: str,
        query_text: str,
        query_embedding: List[float],
        limit: int = 10,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        min_confidence: float = 0.0,
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
            min_confidence: Minimum confidence score
            min_importance: Minimum importance score
            only_valid: Only return chunks with end_date = NULL

        Returns:
            List of LongtermMemoryChunk with combined scores
        """
        embedding_str = f"[{','.join(map(str, query_embedding))}]"
        valid_clause = "AND c.end_date IS NULL" if only_valid else ""

        query = f"""
            WITH vector_results AS (
                SELECT 
                    id,
                    1 - (embedding <=> $1::vector) AS vector_score
                FROM longterm_memory_chunk
                WHERE external_id = $2 AND embedding IS NOT NULL
            ),
            bm25_results AS (
                SELECT 
                    id,
                    bm25_score(content_bm25, tokenize($3, 'bert')) AS bm25_score
                FROM longterm_memory_chunk
                WHERE external_id = $2 AND content_bm25 IS NOT NULL
            )
            SELECT 
                c.id, c.external_id, c.shortterm_memory_id, c.chunk_order, c.content, 
                c.confidence_score, c.importance_score, c.start_date, c.end_date, 
                c.metadata, c.created_at,
                COALESCE(v.vector_score, 0) * $4 + COALESCE(b.bm25_score, 0) * $5 AS combined_score
            FROM longterm_memory_chunk c
            LEFT JOIN vector_results v ON c.id = v.id
            LEFT JOIN bm25_results b ON c.id = b.id
            WHERE c.external_id = $2
              AND (v.vector_score IS NOT NULL OR b.bm25_score IS NOT NULL)
              AND c.confidence_score >= $6
              AND c.importance_score >= $7
              {valid_clause}
            ORDER BY combined_score DESC
            LIMIT $8
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(
                query,
                [
                    embedding_str,
                    external_id,
                    query_text,
                    vector_weight,
                    bm25_weight,
                    min_confidence,
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
        entity_type: str,
        description: Optional[str] = None,
        confidence: float = 0.5,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LongtermEntity:
        """
        Create a new longterm entity in Neo4j.

        Args:
            external_id: Agent identifier
            name: Entity name
            entity_type: Entity type (e.g., 'Person', 'Technology', 'Concept')
            description: Optional description
            confidence: Confidence score (0-1)
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
            type: $entity_type,
            description: $description,
            confidence: $confidence,
            importance: $importance,
            start_date: $now,
            last_updated: $now,
            metadata: $metadata
        })
        RETURN id(e) AS id, e.external_id AS external_id, e.name AS name, 
               e.type AS type, e.description AS description,
               e.confidence AS confidence, e.importance AS importance,
               e.start_date AS start_date, e.last_updated AS last_updated,
               e.metadata AS metadata
        """

        async with self.neo4j.session() as session:
            result = await session.run(
                query,
                external_id=external_id,
                name=name,
                entity_type=entity_type,
                description=description,
                confidence=confidence,
                importance=importance,
                now=now,
                metadata=metadata or {},
            )
            record = await result.single()

            entity = LongtermEntity(
                id=record["id"],
                external_id=record["external_id"],
                name=record["name"],
                type=record["type"],
                description=record["description"],
                confidence=record["confidence"],
                importance=record["importance"],
                start_date=record["start_date"],
                last_updated=record["last_updated"],
                metadata=record["metadata"],
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
        RETURN id(e) AS id, e.external_id AS external_id, e.name AS name, 
               e.type AS type, e.description AS description,
               e.confidence AS confidence, e.importance AS importance,
               e.start_date AS start_date, e.last_updated AS last_updated,
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
                type=record["type"],
                description=record["description"],
                confidence=record["confidence"],
                importance=record["importance"],
                start_date=record["start_date"],
                last_updated=record["last_updated"],
                metadata=record["metadata"],
            )

    async def get_entities_by_external_id(
        self,
        external_id: str,
        min_confidence: float = 0.0,
        min_importance: float = 0.0,
    ) -> List[LongtermEntity]:
        """
        Get all longterm entities for an external_id.

        Args:
            external_id: Agent identifier
            min_confidence: Minimum confidence score filter
            min_importance: Minimum importance score filter

        Returns:
            List of LongtermEntity objects
        """
        query = """
        MATCH (e:LongtermEntity {external_id: $external_id})
        WHERE e.confidence >= $min_confidence AND e.importance >= $min_importance
        RETURN id(e) AS id, e.external_id AS external_id, e.name AS name, 
               e.type AS type, e.description AS description,
               e.confidence AS confidence, e.importance AS importance,
               e.start_date AS start_date, e.last_updated AS last_updated,
               e.metadata AS metadata
        ORDER BY e.importance DESC, e.confidence DESC
        """

        async with self.neo4j.session() as session:
            result = await session.run(
                query,
                external_id=external_id,
                min_confidence=min_confidence,
                min_importance=min_importance,
            )
            records = await result.list()

            entities = [
                LongtermEntity(
                    id=record["id"],
                    external_id=record["external_id"],
                    name=record["name"],
                    type=record["type"],
                    description=record["description"],
                    confidence=record["confidence"],
                    importance=record["importance"],
                    start_date=record["start_date"],
                    last_updated=record["last_updated"],
                    metadata=record["metadata"],
                )
                for record in records
            ]

            logger.debug(f"Retrieved {len(entities)} longterm entities for {external_id}")
            return entities

    async def update_entity(
        self,
        entity_id: int,
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

        if name is not None:
            updates.append("e.name = $name")
            params["name"] = name

        if description is not None:
            updates.append("e.description = $description")
            params["description"] = description

        if confidence is not None:
            updates.append("e.confidence = $confidence")
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
        RETURN id(e) AS id, e.external_id AS external_id, e.name AS name, 
               e.type AS type, e.description AS description,
               e.confidence AS confidence, e.importance AS importance,
               e.start_date AS start_date, e.last_updated AS last_updated,
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
                type=record["type"],
                description=record["description"],
                confidence=record["confidence"],
                importance=record["importance"],
                start_date=record["start_date"],
                last_updated=record["last_updated"],
                metadata=record["metadata"],
            )

            logger.info(f"Updated longterm entity {entity_id}")
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
        relationship_type: str,
        description: Optional[str] = None,
        confidence: float = 0.5,
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
            relationship_type: Relationship type (e.g., 'USES', 'DEPENDS_ON')
            description: Optional description
            confidence: Confidence score (0-1)
            strength: Relationship strength (0-1)
            importance: Importance score (0-1)
            metadata: Optional metadata

        Returns:
            Created LongtermRelationship object
        """
        now = datetime.now(timezone.utc)

        query = """
        MATCH (from:LongtermEntity), (to:LongtermEntity)
        WHERE id(from) = $from_entity_id AND id(to) = $to_entity_id
        CREATE (from)-[r:RELATES_TO {
            external_id: $external_id,
            type: $relationship_type,
            description: $description,
            confidence: $confidence,
            strength: $strength,
            importance: $importance,
            start_date: $now,
            last_updated: $now,
            metadata: $metadata
        }]->(to)
        RETURN id(r) AS id, r.external_id AS external_id,
               id(from) AS from_entity_id, id(to) AS to_entity_id,
               from.name AS from_entity_name, to.name AS to_entity_name,
               r.type AS type, r.description AS description,
               r.confidence AS confidence, r.strength AS strength,
               r.importance AS importance,
               r.start_date AS start_date, r.last_updated AS last_updated,
               r.metadata AS metadata
        """

        async with self.neo4j.session() as session:
            result = await session.run(
                query,
                external_id=external_id,
                from_entity_id=from_entity_id,
                to_entity_id=to_entity_id,
                relationship_type=relationship_type,
                description=description,
                confidence=confidence,
                strength=strength,
                importance=importance,
                now=now,
                metadata=metadata or {},
            )
            record = await result.single()

            relationship = LongtermRelationship(
                id=record["id"],
                external_id=record["external_id"],
                from_entity_id=record["from_entity_id"],
                to_entity_id=record["to_entity_id"],
                from_entity_name=record["from_entity_name"],
                to_entity_name=record["to_entity_name"],
                type=record["type"],
                description=record["description"],
                confidence=record["confidence"],
                strength=record["strength"],
                importance=record["importance"],
                start_date=record["start_date"],
                last_updated=record["last_updated"],
                metadata=record["metadata"],
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
        MATCH (from:LongtermEntity)-[r:RELATES_TO]->(to:LongtermEntity)
        WHERE id(r) = $relationship_id
        RETURN id(r) AS id, r.external_id AS external_id,
               id(from) AS from_entity_id, id(to) AS to_entity_id,
               from.name AS from_entity_name, to.name AS to_entity_name,
               r.type AS type, r.description AS description,
               r.confidence AS confidence, r.strength AS strength,
               r.importance AS importance,
               r.start_date AS start_date, r.last_updated AS last_updated,
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
                type=record["type"],
                description=record["description"],
                confidence=record["confidence"],
                strength=record["strength"],
                importance=record["importance"],
                start_date=record["start_date"],
                last_updated=record["last_updated"],
                metadata=record["metadata"],
            )

    async def get_relationships_by_external_id(
        self,
        external_id: str,
        min_confidence: float = 0.0,
        min_importance: float = 0.0,
    ) -> List[LongtermRelationship]:
        """
        Get all longterm relationships for an external_id.

        Args:
            external_id: Agent identifier
            min_confidence: Minimum confidence score filter
            min_importance: Minimum importance score filter

        Returns:
            List of LongtermRelationship objects
        """
        query = """
        MATCH (from:LongtermEntity)-[r:RELATES_TO {external_id: $external_id}]->(to:LongtermEntity)
        WHERE r.confidence >= $min_confidence AND r.importance >= $min_importance
        RETURN id(r) AS id, r.external_id AS external_id,
               id(from) AS from_entity_id, id(to) AS to_entity_id,
               from.name AS from_entity_name, to.name AS to_entity_name,
               r.type AS type, r.description AS description,
               r.confidence AS confidence, r.strength AS strength,
               r.importance AS importance,
               r.start_date AS start_date, r.last_updated AS last_updated,
               r.metadata AS metadata
        ORDER BY r.importance DESC, r.strength DESC, r.confidence DESC
        """

        async with self.neo4j.session() as session:
            result = await session.run(
                query,
                external_id=external_id,
                min_confidence=min_confidence,
                min_importance=min_importance,
            )
            records = await result.list()

            relationships = [
                LongtermRelationship(
                    id=record["id"],
                    external_id=record["external_id"],
                    from_entity_id=record["from_entity_id"],
                    to_entity_id=record["to_entity_id"],
                    from_entity_name=record["from_entity_name"],
                    to_entity_name=record["to_entity_name"],
                    type=record["type"],
                    description=record["description"],
                    confidence=record["confidence"],
                    strength=record["strength"],
                    importance=record["importance"],
                    start_date=record["start_date"],
                    last_updated=record["last_updated"],
                    metadata=record["metadata"],
                )
                for record in records
            ]

            logger.debug(f"Retrieved {len(relationships)} longterm relationships for {external_id}")
            return relationships

    async def update_relationship(
        self,
        relationship_id: int,
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

        if description is not None:
            updates.append("r.description = $description")
            params["description"] = description

        if confidence is not None:
            updates.append("r.confidence = $confidence")
            params["confidence"] = confidence

        if strength is not None:
            updates.append("r.strength = $strength")
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
        MATCH (from:LongtermEntity)-[r:RELATES_TO]->(to:LongtermEntity)
        WHERE id(r) = $relationship_id
        SET {", ".join(updates)}
        RETURN id(r) AS id, r.external_id AS external_id,
               id(from) AS from_entity_id, id(to) AS to_entity_id,
               from.name AS from_entity_name, to.name AS to_entity_name,
               r.type AS type, r.description AS description,
               r.confidence AS confidence, r.strength AS strength,
               r.importance AS importance,
               r.start_date AS start_date, r.last_updated AS last_updated,
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
                type=record["type"],
                description=record["description"],
                confidence=record["confidence"],
                strength=record["strength"],
                importance=record["importance"],
                start_date=record["start_date"],
                last_updated=record["last_updated"],
                metadata=record["metadata"],
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
        MATCH ()-[r:RELATES_TO]->()
        WHERE id(r) = $relationship_id
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
            chunk_order=row[3],
            content=row[4],
            confidence_score=float(row[5]),
            start_date=row[7],
            end_date=row[8],
            metadata=row[9] if isinstance(row[9], dict) else {},
        )
