"""
Shortterm Memory Repository.

Handles CRUD operations for shortterm memory tier:
- ShorttermMemory CRUD
- ShorttermMemoryChunk CRUD with vector and BM25 search
- Entity and Relationship management in Neo4j
"""

import logging
import json
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone

from agent_mem.database.postgres_manager import PostgreSQLManager
from agent_mem.database.neo4j_manager import Neo4jManager
from agent_mem.database.models import (
    ShorttermMemory,
    ShorttermMemoryChunk,
    ShorttermEntity,
    ShorttermRelationship,
)

logger = logging.getLogger(__name__)


class ShorttermMemoryRepository:
    """
    Repository for shortterm memory operations.

    Shortterm memory stores recent consolidated knowledge from active memory,
    with vector embeddings for semantic search and graph entities/relationships.
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
    # SHORTTERM MEMORY CRUD
    # =========================================================================

    async def create_memory(
        self,
        external_id: str,
        title: str,
        summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ShorttermMemory:
        """
        Create a new shortterm memory.

        Args:
            external_id: Agent identifier
            title: Memory title
            summary: Optional summary
            metadata: Optional metadata

        Returns:
            Created ShorttermMemory object
        """
        query = """
            INSERT INTO shortterm_memory 
            (external_id, title, summary, metadata)
            VALUES ($1, $2, $3, $4)
            RETURNING id, external_id, title, summary, metadata, 
                      created_at, last_updated
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(
                query,
                [external_id, title, summary, metadata or {}],
            )

            row = result.result()[0]
            memory = self._memory_row_to_model(row)

            logger.info(f"Created shortterm memory {memory.id} for {external_id}")
            return memory

    async def get_memory_by_id(self, memory_id: int) -> Optional[ShorttermMemory]:
        """
        Get a shortterm memory by ID.

        Args:
            memory_id: Memory ID

        Returns:
            ShorttermMemory object or None if not found
        """
        query = """
            SELECT id, external_id, title, summary, metadata, 
                   created_at, last_updated
            FROM shortterm_memory
            WHERE id = $1
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(query, [memory_id])
            rows = result.result()

            if not rows:
                return None

            return self._memory_row_to_model(rows[0])

    async def get_memories_by_external_id(
        self, external_id: str, limit: int = 100
    ) -> List[ShorttermMemory]:
        """
        Get all shortterm memories for an external_id.

        Args:
            external_id: Agent identifier
            limit: Maximum number of memories to return

        Returns:
            List of ShorttermMemory objects
        """
        query = """
            SELECT id, external_id, title, summary, metadata, 
                   created_at, last_updated
            FROM shortterm_memory
            WHERE external_id = $1
            ORDER BY last_updated DESC
            LIMIT $2
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(query, [external_id, limit])
            rows = result.result()

            memories = [self._memory_row_to_model(row) for row in rows]

            logger.debug(f"Retrieved {len(memories)} shortterm memories for {external_id}")
            return memories

    async def update_memory(
        self,
        memory_id: int,
        title: Optional[str] = None,
        summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ShorttermMemory]:
        """
        Update a shortterm memory.

        Args:
            memory_id: Memory ID
            title: New title (optional)
            summary: New summary (optional)
            metadata: New metadata (optional)

        Returns:
            Updated ShorttermMemory or None if not found
        """
        updates = []
        params = []
        param_idx = 1

        if title is not None:
            updates.append(f"title = ${param_idx}")
            params.append(title)
            param_idx += 1

        if summary is not None:
            updates.append(f"summary = ${param_idx}")
            params.append(summary)
            param_idx += 1

        if metadata is not None:
            updates.append(f"metadata = ${param_idx}")
            params.append(metadata)
            param_idx += 1

        if not updates:
            return await self.get_memory_by_id(memory_id)

        updates.append(f"last_updated = CURRENT_TIMESTAMP")
        params.append(memory_id)

        query = f"""
            UPDATE shortterm_memory
            SET {", ".join(updates)}
            WHERE id = ${param_idx}
            RETURNING id, external_id, title, summary, metadata, 
                      created_at, last_updated
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(query, params)
            rows = result.result()

            if not rows:
                return None

            memory = self._memory_row_to_model(rows[0])
            logger.info(f"Updated shortterm memory {memory_id}")
            return memory

    async def delete_memory(self, memory_id: int) -> bool:
        """
        Delete a shortterm memory and all its chunks.

        Args:
            memory_id: Memory ID

        Returns:
            True if deleted, False if not found
        """
        query = "DELETE FROM shortterm_memory WHERE id = $1"

        async with self.postgres.connection() as conn:
            await conn.execute(query, [memory_id])
            logger.info(f"Deleted shortterm memory {memory_id}")
            return True

    # =========================================================================
    # SHORTTERM MEMORY CHUNK CRUD
    # =========================================================================

    async def create_chunk(
        self,
        shortterm_memory_id: int,
        external_id: str,
        content: str,
        chunk_order: int,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ShorttermMemoryChunk:
        """
        Create a new chunk for a shortterm memory.

        The BM25 vector is auto-populated by database trigger.

        Args:
            shortterm_memory_id: Parent memory ID
            external_id: Agent identifier
            content: Chunk content
            chunk_order: Order of chunk in memory
            embedding: Optional embedding vector
            metadata: Optional metadata

        Returns:
            Created ShorttermMemoryChunk object
        """
        query = """
            INSERT INTO shortterm_memory_chunk 
            (shortterm_memory_id, external_id, content, chunk_order, embedding, metadata)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id, shortterm_memory_id, external_id, content, chunk_order, 
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
                    shortterm_memory_id,
                    external_id,
                    content,
                    chunk_order,
                    embedding_str,
                    metadata or {},
                ],
            )

            row = result.result()[0]
            chunk = self._chunk_row_to_model(row)

            logger.info(f"Created shortterm chunk {chunk.id} for memory {shortterm_memory_id}")
            return chunk

    async def get_chunk_by_id(self, chunk_id: int) -> Optional[ShorttermMemoryChunk]:
        """
        Get a chunk by ID.

        Args:
            chunk_id: Chunk ID

        Returns:
            ShorttermMemoryChunk or None if not found
        """
        query = """
            SELECT id, shortterm_memory_id, external_id, content, chunk_order, 
                   metadata, created_at
            FROM shortterm_memory_chunk
            WHERE id = $1
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(query, [chunk_id])
            rows = result.result()

            if not rows:
                return None

            return self._chunk_row_to_model(rows[0])

    async def get_chunks_by_memory_id(self, shortterm_memory_id: int) -> List[ShorttermMemoryChunk]:
        """
        Get all chunks for a memory, ordered by chunk_order.

        Args:
            shortterm_memory_id: Memory ID

        Returns:
            List of ShorttermMemoryChunk objects
        """
        query = """
            SELECT id, shortterm_memory_id, external_id, content, chunk_order, 
                   metadata, created_at
            FROM shortterm_memory_chunk
            WHERE shortterm_memory_id = $1
            ORDER BY chunk_order ASC
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(query, [shortterm_memory_id])
            rows = result.result()

            chunks = [self._chunk_row_to_model(row) for row in rows]

            logger.debug(
                f"Retrieved {len(chunks)} chunks for shortterm memory {shortterm_memory_id}"
            )
            return chunks

    async def update_chunk(
        self,
        chunk_id: int,
        content: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ShorttermMemoryChunk]:
        """
        Update a chunk.

        Args:
            chunk_id: Chunk ID
            content: New content (optional)
            embedding: New embedding (optional)
            metadata: New metadata (optional)

        Returns:
            Updated ShorttermMemoryChunk or None if not found
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

        if metadata is not None:
            updates.append(f"metadata = ${param_idx}")
            params.append(metadata)
            param_idx += 1

        if not updates:
            return await self.get_chunk_by_id(chunk_id)

        params.append(chunk_id)

        query = f"""
            UPDATE shortterm_memory_chunk
            SET {", ".join(updates)}
            WHERE id = ${param_idx}
            RETURNING id, shortterm_memory_id, external_id, content, chunk_order, 
                      metadata, created_at
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(query, params)
            rows = result.result()

            if not rows:
                return None

            chunk = self._chunk_row_to_model(rows[0])
            logger.info(f"Updated shortterm chunk {chunk_id}")
            return chunk

    async def delete_chunk(self, chunk_id: int) -> bool:
        """
        Delete a chunk.

        Args:
            chunk_id: Chunk ID

        Returns:
            True if deleted, False if not found
        """
        query = "DELETE FROM shortterm_memory_chunk WHERE id = $1"

        async with self.postgres.connection() as conn:
            await conn.execute(query, [chunk_id])
            logger.info(f"Deleted shortterm chunk {chunk_id}")
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
    ) -> List[ShorttermMemoryChunk]:
        """
        Search chunks by vector similarity.

        Args:
            external_id: Agent identifier
            query_embedding: Query embedding vector
            limit: Maximum results
            min_similarity: Minimum cosine similarity (0-1)

        Returns:
            List of ShorttermMemoryChunk with similarity_score set
        """
        embedding_str = f"[{','.join(map(str, query_embedding))}]"

        query = """
            SELECT 
                id, shortterm_memory_id, external_id, content, chunk_order, 
                metadata, created_at,
                1 - (embedding <=> $1::vector) AS similarity
            FROM shortterm_memory_chunk
            WHERE external_id = $2 
              AND embedding IS NOT NULL
              AND (1 - (embedding <=> $1::vector)) >= $3
            ORDER BY embedding <=> $1::vector
            LIMIT $4
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(query, [embedding_str, external_id, min_similarity, limit])
            rows = result.result()

            chunks = []
            for row in rows:
                chunk = self._chunk_row_to_model(row[:7])
                chunk.similarity_score = float(row[7])
                chunks.append(chunk)

            logger.debug(f"Vector search found {len(chunks)} chunks for {external_id}")
            return chunks

    async def bm25_search(
        self,
        external_id: str,
        query_text: str,
        limit: int = 10,
    ) -> List[ShorttermMemoryChunk]:
        """
        Search chunks by BM25 keyword matching.

        Args:
            external_id: Agent identifier
            query_text: Query text
            limit: Maximum results

        Returns:
            List of ShorttermMemoryChunk with bm25_score set
        """
        query = """
            SELECT 
                id, shortterm_memory_id, external_id, content, chunk_order, 
                metadata, created_at,
                bm25_score(content_bm25, tokenize($1, 'bert')) AS score
            FROM shortterm_memory_chunk
            WHERE external_id = $2
              AND content_bm25 IS NOT NULL
            ORDER BY score DESC
            LIMIT $3
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(query, [query_text, external_id, limit])
            rows = result.result()

            chunks = []
            for row in rows:
                chunk = self._chunk_row_to_model(row[:7])
                chunk.bm25_score = float(row[7]) if row[7] is not None else 0.0
                chunks.append(chunk)

            logger.debug(f"BM25 search found {len(chunks)} chunks for {external_id}")
            return chunks

    async def hybrid_search(
        self,
        external_id: str,
        query_text: str,
        query_embedding: List[float],
        limit: int = 10,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ) -> List[ShorttermMemoryChunk]:
        """
        Hybrid search combining vector similarity and BM25.

        Args:
            external_id: Agent identifier
            query_text: Query text for BM25
            query_embedding: Query embedding for vector search
            limit: Maximum results
            vector_weight: Weight for vector similarity (0-1)
            bm25_weight: Weight for BM25 score (0-1)

        Returns:
            List of ShorttermMemoryChunk with combined scores
        """
        embedding_str = f"[{','.join(map(str, query_embedding))}]"

        query = """
            WITH vector_results AS (
                SELECT 
                    id,
                    1 - (embedding <=> $1::vector) AS vector_score
                FROM shortterm_memory_chunk
                WHERE external_id = $2 AND embedding IS NOT NULL
            ),
            bm25_results AS (
                SELECT 
                    id,
                    bm25_score(content_bm25, tokenize($3, 'bert')) AS bm25_score
                FROM shortterm_memory_chunk
                WHERE external_id = $2 AND content_bm25 IS NOT NULL
            )
            SELECT 
                c.id, c.shortterm_memory_id, c.external_id, c.content, c.chunk_order, 
                c.metadata, c.created_at,
                COALESCE(v.vector_score, 0) * $4 + COALESCE(b.bm25_score, 0) * $5 AS combined_score
            FROM shortterm_memory_chunk c
            LEFT JOIN vector_results v ON c.id = v.id
            LEFT JOIN bm25_results b ON c.id = b.id
            WHERE c.external_id = $2
              AND (v.vector_score IS NOT NULL OR b.bm25_score IS NOT NULL)
            ORDER BY combined_score DESC
            LIMIT $6
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
                    limit,
                ],
            )
            rows = result.result()

            chunks = []
            for row in rows:
                chunk = self._chunk_row_to_model(row[:7])
                chunk.similarity_score = float(row[7])  # Store combined score
                chunks.append(chunk)

            logger.debug(f"Hybrid search found {len(chunks)} chunks for {external_id}")
            return chunks

    # =========================================================================
    # NEO4J ENTITY OPERATIONS
    # =========================================================================

    async def create_entity(
        self,
        external_id: str,
        shortterm_memory_id: int,
        name: str,
        entity_type: str,
        description: Optional[str] = None,
        confidence: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ShorttermEntity:
        """
        Create a new entity in Neo4j.

        Args:
            external_id: Agent identifier
            shortterm_memory_id: Parent memory ID
            name: Entity name
            entity_type: Entity type (e.g., 'Person', 'Technology', 'Concept')
            description: Optional description
            confidence: Confidence score (0-1)
            metadata: Optional metadata

        Returns:
            Created ShorttermEntity object
        """
        now = datetime.now(timezone.utc)

        query = """
        CREATE (e:ShorttermEntity {
            external_id: $external_id,
            shortterm_memory_id: $shortterm_memory_id,
            name: $name,
            type: $entity_type,
            description: $description,
            confidence: $confidence,
            first_seen: $now,
            last_seen: $now,
            metadata: $metadata
        })
        RETURN id(e) AS id, e.external_id AS external_id, e.shortterm_memory_id AS shortterm_memory_id,
               e.name AS name, e.type AS type, e.description AS description,
               e.confidence AS confidence, e.first_seen AS first_seen, 
               e.last_seen AS last_seen, e.metadata AS metadata
        """

        async with self.neo4j.session() as session:
            result = await session.run(
                query,
                external_id=external_id,
                shortterm_memory_id=shortterm_memory_id,
                name=name,
                entity_type=entity_type,
                description=description,
                confidence=confidence,
                now=now,
                metadata=metadata or {},
            )
            record = await result.single()

            entity = ShorttermEntity(
                id=record["id"],
                external_id=record["external_id"],
                shortterm_memory_id=record["shortterm_memory_id"],
                name=record["name"],
                type=record["type"],
                description=record["description"],
                confidence=record["confidence"],
                first_seen=record["first_seen"],
                last_seen=record["last_seen"],
                metadata=record["metadata"],
            )

            logger.info(f"Created shortterm entity {entity.id}: {name}")
            return entity

    async def get_entity(self, entity_id: int) -> Optional[ShorttermEntity]:
        """
        Get an entity by ID.

        Args:
            entity_id: Entity node ID

        Returns:
            ShorttermEntity or None if not found
        """
        query = """
        MATCH (e:ShorttermEntity)
        WHERE id(e) = $entity_id
        RETURN id(e) AS id, e.external_id AS external_id, e.shortterm_memory_id AS shortterm_memory_id,
               e.name AS name, e.type AS type, e.description AS description,
               e.confidence AS confidence, e.first_seen AS first_seen, 
               e.last_seen AS last_seen, e.metadata AS metadata
        """

        async with self.neo4j.session() as session:
            result = await session.run(query, entity_id=entity_id)
            record = await result.single()

            if not record:
                return None

            return ShorttermEntity(
                id=record["id"],
                external_id=record["external_id"],
                shortterm_memory_id=record["shortterm_memory_id"],
                name=record["name"],
                type=record["type"],
                description=record["description"],
                confidence=record["confidence"],
                first_seen=record["first_seen"],
                last_seen=record["last_seen"],
                metadata=record["metadata"],
            )

    async def get_entities_by_memory(self, shortterm_memory_id: int) -> List[ShorttermEntity]:
        """
        Get all entities for a shortterm memory.

        Args:
            shortterm_memory_id: Memory ID

        Returns:
            List of ShorttermEntity objects
        """
        query = """
        MATCH (e:ShorttermEntity {shortterm_memory_id: $shortterm_memory_id})
        RETURN id(e) AS id, e.external_id AS external_id, e.shortterm_memory_id AS shortterm_memory_id,
               e.name AS name, e.type AS type, e.description AS description,
               e.confidence AS confidence, e.first_seen AS first_seen, 
               e.last_seen AS last_seen, e.metadata AS metadata
        ORDER BY e.confidence DESC
        """

        async with self.neo4j.session() as session:
            result = await session.run(query, shortterm_memory_id=shortterm_memory_id)
            records = await result.list()

            entities = [
                ShorttermEntity(
                    id=record["id"],
                    external_id=record["external_id"],
                    shortterm_memory_id=record["shortterm_memory_id"],
                    name=record["name"],
                    type=record["type"],
                    description=record["description"],
                    confidence=record["confidence"],
                    first_seen=record["first_seen"],
                    last_seen=record["last_seen"],
                    metadata=record["metadata"],
                )
                for record in records
            ]

            logger.debug(
                f"Retrieved {len(entities)} entities for shortterm memory {shortterm_memory_id}"
            )
            return entities

    async def update_entity(
        self,
        entity_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ShorttermEntity]:
        """
        Update an entity.

        Args:
            entity_id: Entity node ID
            name: New name (optional)
            description: New description (optional)
            confidence: New confidence (optional)
            metadata: New metadata (optional)

        Returns:
            Updated ShorttermEntity or None if not found
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

        if metadata is not None:
            updates.append("e.metadata = $metadata")
            params["metadata"] = metadata

        if not updates:
            return await self.get_entity(entity_id)

        updates.append("e.last_seen = $now")

        query = f"""
        MATCH (e:ShorttermEntity)
        WHERE id(e) = $entity_id
        SET {", ".join(updates)}
        RETURN id(e) AS id, e.external_id AS external_id, e.shortterm_memory_id AS shortterm_memory_id,
               e.name AS name, e.type AS type, e.description AS description,
               e.confidence AS confidence, e.first_seen AS first_seen, 
               e.last_seen AS last_seen, e.metadata AS metadata
        """

        async with self.neo4j.session() as session:
            result = await session.run(query, **params)
            record = await result.single()

            if not record:
                return None

            entity = ShorttermEntity(
                id=record["id"],
                external_id=record["external_id"],
                shortterm_memory_id=record["shortterm_memory_id"],
                name=record["name"],
                type=record["type"],
                description=record["description"],
                confidence=record["confidence"],
                first_seen=record["first_seen"],
                last_seen=record["last_seen"],
                metadata=record["metadata"],
            )

            logger.info(f"Updated shortterm entity {entity_id}")
            return entity

    async def delete_entity(self, entity_id: int) -> bool:
        """
        Delete an entity and all its relationships.

        Args:
            entity_id: Entity node ID

        Returns:
            True if deleted
        """
        query = """
        MATCH (e:ShorttermEntity)
        WHERE id(e) = $entity_id
        DETACH DELETE e
        """

        async with self.neo4j.session() as session:
            await session.run(query, entity_id=entity_id)
            logger.info(f"Deleted shortterm entity {entity_id}")
            return True

    # =========================================================================
    # NEO4J RELATIONSHIP OPERATIONS
    # =========================================================================

    async def create_relationship(
        self,
        external_id: str,
        shortterm_memory_id: int,
        from_entity_id: int,
        to_entity_id: int,
        relationship_type: str,
        description: Optional[str] = None,
        confidence: float = 0.5,
        strength: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ShorttermRelationship:
        """
        Create a new relationship between entities in Neo4j.

        Args:
            external_id: Agent identifier
            shortterm_memory_id: Parent memory ID
            from_entity_id: Source entity node ID
            to_entity_id: Target entity node ID
            relationship_type: Relationship type (e.g., 'USES', 'DEPENDS_ON')
            description: Optional description
            confidence: Confidence score (0-1)
            strength: Relationship strength (0-1)
            metadata: Optional metadata

        Returns:
            Created ShorttermRelationship object
        """
        now = datetime.now(timezone.utc)

        query = """
        MATCH (from:ShorttermEntity), (to:ShorttermEntity)
        WHERE id(from) = $from_entity_id AND id(to) = $to_entity_id
        CREATE (from)-[r:RELATES_TO {
            external_id: $external_id,
            shortterm_memory_id: $shortterm_memory_id,
            type: $relationship_type,
            description: $description,
            confidence: $confidence,
            strength: $strength,
            first_observed: $now,
            last_observed: $now,
            metadata: $metadata
        }]->(to)
        RETURN id(r) AS id, r.external_id AS external_id, r.shortterm_memory_id AS shortterm_memory_id,
               id(from) AS from_entity_id, id(to) AS to_entity_id,
               from.name AS from_entity_name, to.name AS to_entity_name,
               r.type AS type, r.description AS description,
               r.confidence AS confidence, r.strength AS strength,
               r.first_observed AS first_observed, r.last_observed AS last_observed,
               r.metadata AS metadata
        """

        async with self.neo4j.session() as session:
            result = await session.run(
                query,
                external_id=external_id,
                shortterm_memory_id=shortterm_memory_id,
                from_entity_id=from_entity_id,
                to_entity_id=to_entity_id,
                relationship_type=relationship_type,
                description=description,
                confidence=confidence,
                strength=strength,
                now=now,
                metadata=metadata or {},
            )
            record = await result.single()

            relationship = ShorttermRelationship(
                id=record["id"],
                external_id=record["external_id"],
                shortterm_memory_id=record["shortterm_memory_id"],
                from_entity_id=record["from_entity_id"],
                to_entity_id=record["to_entity_id"],
                from_entity_name=record["from_entity_name"],
                to_entity_name=record["to_entity_name"],
                type=record["type"],
                description=record["description"],
                confidence=record["confidence"],
                strength=record["strength"],
                first_observed=record["first_observed"],
                last_observed=record["last_observed"],
                metadata=record["metadata"],
            )

            logger.info(
                f"Created shortterm relationship {relationship.id}: "
                f"{record['from_entity_name']} -> {record['to_entity_name']}"
            )
            return relationship

    async def get_relationship(self, relationship_id: int) -> Optional[ShorttermRelationship]:
        """
        Get a relationship by ID.

        Args:
            relationship_id: Relationship ID

        Returns:
            ShorttermRelationship or None if not found
        """
        query = """
        MATCH (from:ShorttermEntity)-[r:RELATES_TO]->(to:ShorttermEntity)
        WHERE id(r) = $relationship_id
        RETURN id(r) AS id, r.external_id AS external_id, r.shortterm_memory_id AS shortterm_memory_id,
               id(from) AS from_entity_id, id(to) AS to_entity_id,
               from.name AS from_entity_name, to.name AS to_entity_name,
               r.type AS type, r.description AS description,
               r.confidence AS confidence, r.strength AS strength,
               r.first_observed AS first_observed, r.last_observed AS last_observed,
               r.metadata AS metadata
        """

        async with self.neo4j.session() as session:
            result = await session.run(query, relationship_id=relationship_id)
            record = await result.single()

            if not record:
                return None

            return ShorttermRelationship(
                id=record["id"],
                external_id=record["external_id"],
                shortterm_memory_id=record["shortterm_memory_id"],
                from_entity_id=record["from_entity_id"],
                to_entity_id=record["to_entity_id"],
                from_entity_name=record["from_entity_name"],
                to_entity_name=record["to_entity_name"],
                type=record["type"],
                description=record["description"],
                confidence=record["confidence"],
                strength=record["strength"],
                first_observed=record["first_observed"],
                last_observed=record["last_observed"],
                metadata=record["metadata"],
            )

    async def get_relationships_by_memory(
        self, shortterm_memory_id: int
    ) -> List[ShorttermRelationship]:
        """
        Get all relationships for a shortterm memory.

        Args:
            shortterm_memory_id: Memory ID

        Returns:
            List of ShorttermRelationship objects
        """
        query = """
        MATCH (from:ShorttermEntity)-[r:RELATES_TO {shortterm_memory_id: $shortterm_memory_id}]->(to:ShorttermEntity)
        RETURN id(r) AS id, r.external_id AS external_id, r.shortterm_memory_id AS shortterm_memory_id,
               id(from) AS from_entity_id, id(to) AS to_entity_id,
               from.name AS from_entity_name, to.name AS to_entity_name,
               r.type AS type, r.description AS description,
               r.confidence AS confidence, r.strength AS strength,
               r.first_observed AS first_observed, r.last_observed AS last_observed,
               r.metadata AS metadata
        ORDER BY r.strength DESC, r.confidence DESC
        """

        async with self.neo4j.session() as session:
            result = await session.run(query, shortterm_memory_id=shortterm_memory_id)
            records = await result.list()

            relationships = [
                ShorttermRelationship(
                    id=record["id"],
                    external_id=record["external_id"],
                    shortterm_memory_id=record["shortterm_memory_id"],
                    from_entity_id=record["from_entity_id"],
                    to_entity_id=record["to_entity_id"],
                    from_entity_name=record["from_entity_name"],
                    to_entity_name=record["to_entity_name"],
                    type=record["type"],
                    description=record["description"],
                    confidence=record["confidence"],
                    strength=record["strength"],
                    first_observed=record["first_observed"],
                    last_observed=record["last_observed"],
                    metadata=record["metadata"],
                )
                for record in records
            ]

            logger.debug(
                f"Retrieved {len(relationships)} relationships for shortterm memory {shortterm_memory_id}"
            )
            return relationships

    async def update_relationship(
        self,
        relationship_id: int,
        description: Optional[str] = None,
        confidence: Optional[float] = None,
        strength: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ShorttermRelationship]:
        """
        Update a relationship.

        Args:
            relationship_id: Relationship ID
            description: New description (optional)
            confidence: New confidence (optional)
            strength: New strength (optional)
            metadata: New metadata (optional)

        Returns:
            Updated ShorttermRelationship or None if not found
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

        if metadata is not None:
            updates.append("r.metadata = $metadata")
            params["metadata"] = metadata

        if not updates:
            return await self.get_relationship(relationship_id)

        updates.append("r.last_observed = $now")

        query = f"""
        MATCH (from:ShorttermEntity)-[r:RELATES_TO]->(to:ShorttermEntity)
        WHERE id(r) = $relationship_id
        SET {", ".join(updates)}
        RETURN id(r) AS id, r.external_id AS external_id, r.shortterm_memory_id AS shortterm_memory_id,
               id(from) AS from_entity_id, id(to) AS to_entity_id,
               from.name AS from_entity_name, to.name AS to_entity_name,
               r.type AS type, r.description AS description,
               r.confidence AS confidence, r.strength AS strength,
               r.first_observed AS first_observed, r.last_observed AS last_observed,
               r.metadata AS metadata
        """

        async with self.neo4j.session() as session:
            result = await session.run(query, **params)
            record = await result.single()

            if not record:
                return None

            relationship = ShorttermRelationship(
                id=record["id"],
                external_id=record["external_id"],
                shortterm_memory_id=record["shortterm_memory_id"],
                from_entity_id=record["from_entity_id"],
                to_entity_id=record["to_entity_id"],
                from_entity_name=record["from_entity_name"],
                to_entity_name=record["to_entity_name"],
                type=record["type"],
                description=record["description"],
                confidence=record["confidence"],
                strength=record["strength"],
                first_observed=record["first_observed"],
                last_observed=record["last_observed"],
                metadata=record["metadata"],
            )

            logger.info(f"Updated shortterm relationship {relationship_id}")
            return relationship

    async def delete_relationship(self, relationship_id: int) -> bool:
        """
        Delete a relationship.

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
            logger.info(f"Deleted shortterm relationship {relationship_id}")
            return True

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _memory_row_to_model(self, row) -> ShorttermMemory:
        """Convert database row to ShorttermMemory model."""
        return ShorttermMemory(
            id=row[0],
            external_id=row[1],
            title=row[2],
            summary=row[3],
            metadata=row[4] if isinstance(row[4], dict) else {},
            created_at=row[5],
            last_updated=row[6],
            chunks=[],  # Populated separately if needed
        )

    def _chunk_row_to_model(self, row) -> ShorttermMemoryChunk:
        """Convert database row to ShorttermMemoryChunk model."""
        return ShorttermMemoryChunk(
            id=row[0],
            shortterm_memory_id=row[1],
            content=row[3],
            chunk_order=row[4],
            metadata=row[5] if isinstance(row[5], dict) else {},
        )
