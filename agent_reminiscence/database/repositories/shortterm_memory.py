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

from psqlpy.extra_types import PgVector

from agent_reminiscence.database.postgres_manager import PostgreSQLManager
from agent_reminiscence.database.neo4j_manager import Neo4jManager
from agent_reminiscence.database.models import (
    ShorttermMemory,
    ShorttermMemoryChunk,
    ShorttermEntity,
    ShorttermRelationship,
    ShorttermEntityRelationshipSearchResult,
    ShorttermKnowledgeTriplet,
)

logger = logging.getLogger(__name__)


def _convert_neo4j_datetime(dt):
    """Convert Neo4j DateTime to Python datetime."""
    if dt is None:
        return None
    if hasattr(dt, "to_native"):
        return dt.to_native()
    return dt


def _relationship_to_triplet(
    relationship: ShorttermRelationship,
    source_entity_name: str,
    target_entity_name: str,
    shortterm_memory_id: Optional[int] = None,
) -> ShorttermKnowledgeTriplet:
    """
    Convert Neo4j relationship to RDF-style shortterm triplet.

    Args:
        relationship: ShorttermRelationship from Neo4j
        source_entity_name: Subject entity name
        target_entity_name: Object entity name
        shortterm_memory_id: Source shortterm memory ID (optional)

    Returns:
        ShorttermKnowledgeTriplet with subject-predicate-object format
    """
    # Use first relationship type as predicate
    predicate = relationship.types[0] if relationship.types else "RELATED_TO"

    # Store additional types in metadata if they exist
    additional_types = relationship.types[1:] if len(relationship.types) > 1 else []

    return ShorttermKnowledgeTriplet(
        subject=source_entity_name,
        predicate=predicate,
        object=target_entity_name,
        importance=relationship.importance,
        shortterm_memory_id=shortterm_memory_id,
        access_count=relationship.access_count,
        description=relationship.description,
        metadata={
            "relationship_id": relationship.id,
            "additional_types": additional_types,
            "last_access": (
                relationship.last_access.isoformat() if relationship.last_access else None
            ),
            "confidence": (
                relationship.metadata.get("confidence", 1.0) if relationship.metadata else 1.0
            ),
            **(relationship.metadata or {}),  # Include all custom metadata
        },
    )


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
            RETURNING id, external_id, title, summary, update_count, metadata, 
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
            RETURNING id, external_id, title, summary, update_count, metadata, 
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
        embedding: Optional[List[float]] = None,
        section_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ShorttermMemoryChunk:
        """
        Create a new chunk for a shortterm memory.

        The BM25 vector is auto-populated by database trigger.

        Args:
            shortterm_memory_id: Parent memory ID
            external_id: Agent identifier
            content: Chunk content
            embedding: Optional embedding vector
            section_id: Optional section reference from active memory
            metadata: Optional metadata

        Returns:
            Created ShorttermMemoryChunk object
        """
        query = """
            INSERT INTO shortterm_memory_chunk 
            (shortterm_memory_id, external_id, content, embedding, section_id, metadata, access_count, last_access)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id, shortterm_memory_id, content, section_id, metadata, access_count, last_access
        """

        # Convert embedding to PostgreSQL vector format if provided
        pg_vector = None
        if embedding:
            pg_vector = PgVector(embedding)

        async with self.postgres.connection() as conn:
            result = await conn.execute(
                query,
                [
                    shortterm_memory_id,
                    external_id,
                    content,
                    pg_vector,
                    section_id,
                    metadata or {},
                    0,  # access_count
                    None,  # last_access
                ],
            )

            result_data = result.result()
            logger.debug(f"create_chunk result: {result_data}")
            if not result_data:
                raise ValueError("No data returned from INSERT RETURNING")

            row = result_data[0]
            chunk = self._chunk_row_to_model(row)

            logger.info(
                f"Created shortterm chunk {chunk.id} for memory {shortterm_memory_id}"
                + (f" (section: {section_id})" if section_id else "")
            )
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
            SELECT id, shortterm_memory_id, external_id, content, 
                   section_id, metadata, access_count, last_access, created_at
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
        Get all chunks for a memory.

        Args:
            shortterm_memory_id: Memory ID

        Returns:
            List of ShorttermMemoryChunk objects
        """
        query = """
            SELECT id, shortterm_memory_id, external_id, content, 
                   section_id, metadata, access_count, last_access, created_at
            FROM shortterm_memory_chunk
            WHERE shortterm_memory_id = $1
            ORDER BY id ASC
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
            RETURNING id, shortterm_memory_id, external_id, content, 
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

    async def get_chunks_by_section_id(
        self,
        shortterm_memory_id: int,
        section_id: str,
    ) -> List[ShorttermMemoryChunk]:
        """
        Get all chunks that reference a specific section.

        Args:
            shortterm_memory_id: Memory ID
            section_id: Section ID from active memory

        Returns:
            List of chunks referencing this section
        """
        query = """
            SELECT id, shortterm_memory_id, external_id, content,
                   section_id, metadata, created_at
            FROM shortterm_memory_chunk
            WHERE shortterm_memory_id = $1 AND section_id = $2
            ORDER BY id
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(query, [shortterm_memory_id, section_id])
            rows = result.result()

            chunks = [self._chunk_row_to_model(row) for row in rows]
            logger.debug(
                f"Retrieved {len(chunks)} chunks for section '{section_id}' "
                f"in memory {shortterm_memory_id}"
            )
            return chunks

    async def increment_update_count(self, memory_id: int) -> Optional[ShorttermMemory]:
        """
        Increment the update_count for a shortterm memory.

        Called after consolidation from active memory.

        Args:
            memory_id: Memory ID

        Returns:
            Updated ShorttermMemory or None if not found
        """
        query = """
            UPDATE shortterm_memory
            SET update_count = update_count + 1,
                last_updated = CURRENT_TIMESTAMP
            WHERE id = $1
            RETURNING id, external_id, title, summary, update_count, metadata,
                      created_at
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(query, [memory_id])
            rows = result.result()

            if not rows:
                logger.warning(f"Shortterm memory {memory_id} not found")
                return None

            memory = self._memory_row_to_model(rows[0])
            logger.info(
                f"Incremented update_count for shortterm memory {memory_id} "
                f"to {memory.update_count}"
            )
            return memory

    async def reset_update_count(self, memory_id: int) -> Optional[ShorttermMemory]:
        """
        Reset update_count to 0 for a shortterm memory.

        Called after promotion to longterm memory.

        Args:
            memory_id: Memory ID

        Returns:
            Updated ShorttermMemory or None if not found
        """
        query = """
            UPDATE shortterm_memory
            SET update_count = 0,
                last_updated = CURRENT_TIMESTAMP
            WHERE id = $1
            RETURNING id, external_id, title, summary, update_count, metadata,
                      created_at
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(query, [memory_id])
            rows = result.result()

            if not rows:
                logger.warning(f"Shortterm memory {memory_id} not found")
                return None

            memory = self._memory_row_to_model(rows[0])
            logger.info(f"Reset update_count for shortterm memory {memory_id}")
            return memory

    async def delete_all_chunks(self, memory_id: int) -> int:
        """
        Delete all chunks for a shortterm memory.

        Called after promotion to longterm memory.

        Args:
            memory_id: Memory ID

        Returns:
            Number of chunks deleted
        """
        query = """
            DELETE FROM shortterm_memory_chunk
            WHERE shortterm_memory_id = $1
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(query, [memory_id])
            # Get the number of deleted rows
            deleted_count = len(result.result()) if result.result() else 0
            logger.info(f"Deleted {deleted_count} chunks from shortterm memory {memory_id}")
            return deleted_count

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
        shortterm_memory_id: Optional[int] = None,
        min_similarity_score: Optional[float] = None,
        min_bm25_score: Optional[float] = None,
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
            shortterm_memory_id: Optional filter by specific memory ID
            min_similarity_score: Optional minimum vector similarity threshold
            min_bm25_score: Optional minimum BM25 score threshold

        Returns:
            List of ShorttermMemoryChunk with combined scores
        """
        pg_vector = PgVector(query_embedding)

        query = """
            WITH vector_results AS (
                SELECT 
                    id,
                    1 - (embedding <=> $1) AS vector_score
                FROM shortterm_memory_chunk
                WHERE external_id = $2 AND embedding IS NOT NULL
                AND ($7::int IS NULL OR shortterm_memory_id = $7)
            ),
            bm25_results AS (
                SELECT 
                    id,
                    content_bm25 <&> to_bm25query('idx_shortterm_chunk_bm25', tokenize($3, 'bert')) AS bm25_score
                FROM shortterm_memory_chunk
                WHERE external_id = $2 AND content_bm25 IS NOT NULL
                AND ($7::int IS NULL OR shortterm_memory_id = $7)
            )
            SELECT 
                c.id, c.shortterm_memory_id, c.external_id, c.content, c.section_id, c.metadata, c.access_count, c.last_access, c.created_at,
                COALESCE(v.vector_score, 0) * $4 + COALESCE(b.bm25_score, 0) * $5 AS combined_score,
                COALESCE(v.vector_score, 0) AS vector_score,
                COALESCE(b.bm25_score, 0) AS bm25_score
            FROM shortterm_memory_chunk c
            LEFT JOIN vector_results v ON c.id = v.id
            LEFT JOIN bm25_results b ON c.id = b.id
            WHERE c.external_id = $2
              AND ($7::int IS NULL OR c.shortterm_memory_id = $7)
              AND (v.vector_score IS NOT NULL OR b.bm25_score IS NOT NULL)
              AND ($8::float IS NULL OR COALESCE(v.vector_score, 0) >= $8)
              AND ($9::float IS NULL OR COALESCE(b.bm25_score, 0) >= $9)
            ORDER BY combined_score DESC
            LIMIT $6
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
                    limit,
                    shortterm_memory_id,
                    min_similarity_score,
                    min_bm25_score,
                ],
            )
            rows = result.result()
            logger.debug(f"hybrid_search got {len(rows)} rows")

            chunks = []
            for row in rows:
                logger.debug(f"Processing row: {row}")
                if isinstance(row, dict):
                    # Handle dict format
                    if "id" not in row:
                        logger.error(f"Invalid row format in hybrid_search: {row}")
                        continue
                    chunk = self._chunk_row_to_model(
                        {
                            "id": row["id"],
                            "shortterm_memory_id": row["shortterm_memory_id"],
                            "external_id": row["external_id"],
                            "content": row["content"],
                            "section_id": row.get("section_id"),
                            "metadata": row["metadata"],
                            "created_at": row["created_at"],
                        }
                    )
                    chunk.similarity_score = float(row["combined_score"])
                elif isinstance(row, (list, tuple)) and len(row) >= 12:
                    # Handle tuple format: id, shortterm_memory_id, external_id, content, section_id, metadata, access_count, last_access, created_at, combined_score, vector_score, bm25_score
                    chunk = self._chunk_row_to_model(row[:9])  # First 9 columns for the model
                    chunk.similarity_score = float(row[9])  # combined_score
                    chunk.bm25_score = float(row[11]) if row[11] is not None else None  # bm25_score
                else:
                    logger.error(f"Invalid row format in hybrid_search: {row}")
                    continue
                chunks.append(chunk)

            logger.debug(f"Hybrid search found {len(chunks)} chunks for {external_id}")
            return chunks

    async def increment_chunk_access(self, chunk_id: int) -> Optional[ShorttermMemoryChunk]:
        """
        Increment access count and update last access timestamp for a chunk.

        Args:
            chunk_id: Chunk ID

        Returns:
            Updated ShorttermMemoryChunk or None if not found
        """
        query = """
            UPDATE shortterm_memory_chunk
            SET access_count = access_count + 1,
                last_access = CURRENT_TIMESTAMP
            WHERE id = $1
            RETURNING id, shortterm_memory_id, external_id, content, section_id, metadata, access_count, last_access, created_at
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(query, [chunk_id])
            rows = result.result()

            if not rows:
                return None

            return self._chunk_row_to_model(rows[0])

    async def search_entities_with_relationships(
        self,
        entity_names: List[str],
        external_id: str,
        limit: int = 10,
        shortterm_memory_id: Optional[int] = None,
        min_importance: Optional[float] = None,
    ) -> ShorttermEntityRelationshipSearchResult:
        """
        Search for entities by name and return them with their relationships.

        This performs a graph search starting from entities matching the query names,
        then includes all incoming and outgoing relationships.

        Args:
            entity_names: List of entity names to search for (case-insensitive partial match)
            external_id: Agent identifier
            limit: Maximum number of matched entities to return
            shortterm_memory_id: Optional filter by specific memory ID
            min_importance: Optional minimum importance threshold

        Returns:
            ShorttermEntityRelationshipSearchResult containing matched entities,
            related entities, and all connecting relationships
        """
        # Build Cypher query
        query = """
        MATCH (e:ShorttermEntity)
        WHERE e.external_id = $external_id
        """

        # Add name matching conditions
        name_conditions = []
        for i, name in enumerate(entity_names):
            name_conditions.append(f"toLower(e.name) CONTAINS toLower($name_{i})")
        if name_conditions:
            query += f" AND ({' OR '.join(name_conditions)})"

        # Add optional filters
        if shortterm_memory_id is not None:
            query += f" AND e.shortterm_memory_id = $shortterm_memory_id"
        if min_importance is not None:
            query += f" AND e.importance >= $min_importance"

        query += f"""
        WITH e
        LIMIT $limit

        // Get incoming relationships with their nodes
        OPTIONAL MATCH (other)-[r_in:SHORTTERM_RELATES]->(e)
        WHERE other.external_id = $external_id
        """
        if shortterm_memory_id is not None:
            query += f" AND r_in.shortterm_memory_id = $shortterm_memory_id"

        query += """
        // Get outgoing relationships with their nodes
        OPTIONAL MATCH (e)-[r_out:SHORTTERM_RELATES]->(other2)
        WHERE other2.external_id = $external_id
        """
        if shortterm_memory_id is not None:
            query += f" AND r_out.shortterm_memory_id = $shortterm_memory_id"

        query += """
        RETURN e,
               collect(DISTINCT other) as related_incoming,
               collect(DISTINCT other2) as related_outgoing,
               collect(DISTINCT {rel: r_in, from_id: elementId(other), to_id: elementId(e)}) as relationships_in,
               collect(DISTINCT {rel: r_out, from_id: elementId(e), to_id: elementId(other2)}) as relationships_out
        """

        # Build parameters as dictionary
        params = {"external_id": external_id}
        for i, name in enumerate(entity_names):
            params[f"name_{i}"] = name

        param_index = len(entity_names) + 2
        if shortterm_memory_id is not None:
            params["shortterm_memory_id"] = shortterm_memory_id
            param_index += 1
        if min_importance is not None:
            params["min_importance"] = min_importance
            param_index += 1
        params["limit"] = limit

        logger.info(
            f"Searching shortterm entities: names={entity_names}, external_id={external_id}, params={params}, query={query}"
        )

        async with self.neo4j.session() as session:
            result = await session.run(query, params)
            records = []
            async for record in result:
                records.append(record)

        logger.info(f"Found {len(records)} matching Neo4j entity records")  # Process results
        matched_entities = []
        related_entities = []
        relationships = []
        entity_ids_seen = set()
        relationship_ids_seen = set()

        for record in records:
            # Process matched entity
            entity_data = record["e"]
            entity = ShorttermEntity(
                id=entity_data.element_id,
                external_id=entity_data["external_id"],
                shortterm_memory_id=entity_data["shortterm_memory_id"],
                name=entity_data["name"],
                types=entity_data.get("types", []),
                description=entity_data.get("description"),
                importance=entity_data.get("importance", 0.5),
                access_count=entity_data.get("access_count", 0),
                last_access=_convert_neo4j_datetime(entity_data.get("last_access")),
                metadata=(
                    json.loads(entity_data.get("metadata", "{}"))
                    if entity_data.get("metadata")
                    else {}
                ),
            )
            matched_entities.append(entity)
            entity_ids_seen.add(entity.id)

            # Process incoming related entities
            for other_data in record["related_incoming"]:
                if other_data and other_data.element_id not in entity_ids_seen:
                    other_entity = ShorttermEntity(
                        id=other_data.element_id,
                        external_id=other_data["external_id"],
                        shortterm_memory_id=other_data["shortterm_memory_id"],
                        name=other_data["name"],
                        types=other_data.get("types", []),
                        description=other_data.get("description"),
                        importance=other_data.get("importance", 0.5),
                        access_count=other_data.get("access_count", 0),
                        last_access=_convert_neo4j_datetime(other_data.get("last_access")),
                        metadata=(
                            json.loads(other_data.get("metadata", "{}"))
                            if other_data.get("metadata")
                            else {}
                        ),
                    )
                    related_entities.append(other_entity)
                    entity_ids_seen.add(other_entity.id)

            # Process outgoing related entities
            for other_data in record["related_outgoing"]:
                if other_data and other_data.element_id not in entity_ids_seen:
                    other_entity = ShorttermEntity(
                        id=other_data.element_id,
                        external_id=other_data["external_id"],
                        shortterm_memory_id=other_data["shortterm_memory_id"],
                        name=other_data["name"],
                        types=other_data.get("types", []),
                        description=other_data.get("description"),
                        importance=other_data.get("importance", 0.5),
                        access_count=other_data.get("access_count", 0),
                        last_access=_convert_neo4j_datetime(other_data.get("last_access")),
                        metadata=(
                            json.loads(other_data.get("metadata", "{}"))
                            if other_data.get("metadata")
                            else {}
                        ),
                    )
                    related_entities.append(other_entity)
                    entity_ids_seen.add(other_entity.id)

            # Process incoming relationships
            for rel_map in record["relationships_in"]:
                # Skip if the map is null or the relationship is null
                if not rel_map or not rel_map.get("rel"):
                    continue

                rel_data = rel_map["rel"]
                if rel_data.element_id not in relationship_ids_seen:
                    relationship = ShorttermRelationship(
                        id=rel_data.element_id,
                        external_id=rel_data["external_id"],
                        shortterm_memory_id=rel_data["shortterm_memory_id"],
                        from_entity_id=rel_map["from_id"],  # Get from the map
                        to_entity_id=rel_map["to_id"],  # Get from the map
                        from_entity_name=rel_data.get("from_entity_name"),
                        to_entity_name=rel_data.get("to_entity_name"),
                        types=rel_data.get("types", []),
                        description=rel_data.get("description"),
                        importance=rel_data.get("importance", 0.5),
                        access_count=rel_data.get("access_count", 0),
                        last_access=_convert_neo4j_datetime(rel_data.get("last_access")),
                        metadata=(
                            json.loads(rel_data.get("metadata", "{}"))
                            if rel_data.get("metadata")
                            else {}
                        ),
                    )
                    relationships.append(relationship)
                    relationship_ids_seen.add(relationship.id)

            # Process outgoing relationships
            for rel_map in record["relationships_out"]:
                # Skip if the map is null or the relationship is null
                if not rel_map or not rel_map.get("rel"):
                    continue

                rel_data = rel_map["rel"]
                if rel_data.element_id not in relationship_ids_seen:
                    relationship = ShorttermRelationship(
                        id=rel_data.element_id,
                        external_id=rel_data["external_id"],
                        shortterm_memory_id=rel_data["shortterm_memory_id"],
                        from_entity_id=rel_map["from_id"],  # Get from the map
                        to_entity_id=rel_map["to_id"],  # Get from the map
                        from_entity_name=rel_data.get("from_entity_name"),
                        to_entity_name=rel_data.get("to_entity_name"),
                        types=rel_data.get("types", []),
                        description=rel_data.get("description"),
                        importance=rel_data.get("importance", 0.5),
                        access_count=rel_data.get("access_count", 0),
                        last_access=_convert_neo4j_datetime(rel_data.get("last_access")),
                        metadata=(
                            json.loads(rel_data.get("metadata", "{}"))
                            if rel_data.get("metadata")
                            else {}
                        ),
                    )
                    relationships.append(relationship)
                    relationship_ids_seen.add(relationship.id)

        return ShorttermEntityRelationshipSearchResult(
            query_entity_names=entity_names,
            external_id=external_id,
            shortterm_memory_id=shortterm_memory_id,
            matched_entities=matched_entities,
            related_entities=related_entities,
            relationships=relationships,
            metadata={
                "total_matched_entities": len(matched_entities),
                "total_related_entities": len(related_entities),
                "total_relationships": len(relationships),
                "min_importance": min_importance,
            },
        )

    async def search_entity_triplets(
        self,
        entity_names: List[str],
        external_id: str,
        limit: int = 10,
        shortterm_memory_id: Optional[int] = None,
        min_importance: Optional[float] = None,
    ) -> Tuple[List[ShorttermEntity], List[ShorttermKnowledgeTriplet]]:
        """
        Search for entities and return their relationships as triplets.

        This is the v0.2.0 optimized version of search_entities_with_relationships().
        Returns entities and knowledge triplets instead of raw relationships.

        Args:
            entity_names: List of entity names to search for
            external_id: Agent identifier
            limit: Maximum number of matched entities to return
            shortterm_memory_id: Optional filter by specific memory ID
            min_importance: Optional minimum importance threshold

        Returns:
            Tuple of (matched_entities, triplets)
            - matched_entities: List of ShorttermEntity objects
            - triplets: List of ShorttermKnowledgeTriplet objects
        """
        # Use existing search method to get relationships
        result = await self.search_entities_with_relationships(
            entity_names=entity_names,
            external_id=external_id,
            limit=limit,
            shortterm_memory_id=shortterm_memory_id,
            min_importance=min_importance,
        )

        # Convert relationships to triplets
        triplets = []
        for rel in result.relationships:
            triplet = _relationship_to_triplet(
                relationship=rel,
                source_entity_name=rel.from_entity_name or "Unknown",
                target_entity_name=rel.to_entity_name or "Unknown",
                shortterm_memory_id=shortterm_memory_id,
            )
            triplets.append(triplet)

        logger.info(
            f"Converted {len(result.relationships)} relationships to {len(triplets)} triplets"
        )

        return result.matched_entities, triplets

    async def increment_entity_access(self, entity_id: str) -> Optional[ShorttermEntity]:
        """
        Increment access count and update last access timestamp for an entity.

        Args:
            entity_id: Neo4j elementId of the entity

        Returns:
            Updated ShorttermEntity or None if not found
        """
        query = """
        MATCH (e:ShorttermEntity)
        WHERE elementId(e) = $entity_id
        SET e.access_count = e.access_count + 1,
            e.last_access = datetime()
        RETURN e
        """

        async with self.neo4j.session() as session:
            result = await session.run(query, {"entity_id": entity_id})
            record = await result.single()

            if not record:
                return None

            entity_data = record["e"]
            return ShorttermEntity(
                id=entity_data.element_id,
                external_id=entity_data["external_id"],
                shortterm_memory_id=entity_data["shortterm_memory_id"],
                name=entity_data["name"],
                types=entity_data.get("types", []),
                description=entity_data.get("description"),
                importance=entity_data.get("importance", 0.5),
                access_count=entity_data.get("access_count", 0),
                last_access=_convert_neo4j_datetime(entity_data.get("last_access")),
                metadata=entity_data.get("metadata", {}),
            )

    async def increment_relationship_access(
        self, relationship_id: str
    ) -> Optional[ShorttermRelationship]:
        """
        Increment access count and update last access timestamp for a relationship.

        Args:
            relationship_id: Neo4j elementId of the relationship

        Returns:
            Updated ShorttermRelationship or None if not found
        """
        query = """
        MATCH ()-[r:SHORTTERM_RELATES]->()
        WHERE elementId(r) = $relationship_id
        SET r.access_count = r.access_count + 1,
            r.last_access = datetime()
        RETURN r
        """

        async with self.neo4j.session() as session:
            result = await session.run(query, {"relationship_id": relationship_id})
            record = await result.single()

            if not record:
                return None

            rel_data = record["r"]
            return ShorttermRelationship(
                id=rel_data.element_id,
                external_id=rel_data["external_id"],
                shortterm_memory_id=rel_data["shortterm_memory_id"],
                from_entity_id=rel_data["from_entity_id"],
                to_entity_id=rel_data["to_entity_id"],
                from_entity_name=rel_data.get("from_entity_name"),
                to_entity_name=rel_data.get("to_entity_name"),
                types=rel_data.get("types", []),
                description=rel_data.get("description"),
                importance=rel_data.get("importance", 0.5),
                access_count=rel_data.get("access_count", 0),
                last_access=_convert_neo4j_datetime(rel_data.get("last_access")),
                metadata=rel_data.get("metadata", {}),
            )

    # =========================================================================
    # NEO4J ENTITY OPERATIONS
    # =========================================================================

    async def create_entity(
        self,
        external_id: str,
        shortterm_memory_id: int,
        name: str,
        types: List[str],  # Changed from entity_type: str to types: List[str]
        description: Optional[str] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ShorttermEntity:
        """
        Create a new entity in Neo4j.

        Args:
            external_id: Agent identifier
            shortterm_memory_id: Parent memory ID
            name: Entity name
            types: Entity types (e.g., ['Person', 'Developer'])
            description: Optional description
            importance: Importance score (0-1)
            metadata: Optional metadata

        Returns:
            Created ShorttermEntity object
        """
        now = datetime.now(timezone.utc)

        # Convert metadata to JSON string for Neo4j storage
        metadata_json = json.dumps(metadata or {})

        logger.info(
            f"Creating shortterm entity: name='{name}', external_id='{external_id}', shortterm_memory_id={shortterm_memory_id}"
        )

        query = """
        CREATE (e:ShorttermEntity {
            external_id: $external_id,
            shortterm_memory_id: $shortterm_memory_id,
            name: $name,
            types: $types,
            description: $description,
            importance: $importance,
            access_count: $access_count,
            last_access: $last_access,
            metadata: $metadata_json
        })
        RETURN elementId(e) AS id, e.external_id AS external_id, e.shortterm_memory_id AS shortterm_memory_id,
               e.name AS name, e.types AS types, e.description AS description,
               e.importance AS importance, e.access_count AS access_count, e.last_access AS last_access, 
               e.metadata AS metadata
        """

        async with self.neo4j.session() as session:
            result = await session.run(
                query,
                external_id=external_id,
                shortterm_memory_id=shortterm_memory_id,
                name=name,
                types=types,
                description=description,
                importance=importance,
                access_count=0,
                last_access=None,
                metadata_json=metadata_json,
            )
            record = await result.single()

            # Parse metadata JSON back to dict
            metadata_dict = json.loads(record["metadata"]) if record["metadata"] else {}

            entity = ShorttermEntity(
                id=record["id"],
                external_id=record["external_id"],
                shortterm_memory_id=record["shortterm_memory_id"],
                name=record["name"],
                types=record["types"] or [],
                description=record["description"],
                importance=record["importance"],
                access_count=record["access_count"] or 0,
                last_access=_convert_neo4j_datetime(record["last_access"]),
                metadata=metadata_dict,
            )

            logger.info(f"Created entity with Neo4j element_id={entity.id}")
            return entity

            logger.info(f"Created shortterm entity {entity.id}: {name} with types {types}")
            return entity

    async def get_entity(self, entity_id: int) -> Optional[ShorttermEntity]:
        """
        Get an entity by ID.

        Args:
            entity_id: Entity node ID (elementId)

        Returns:
            ShorttermEntity or None if not found
        """
        query = """
        MATCH (e:ShorttermEntity)
        WHERE elementId(e) = $entity_id
        RETURN elementId(e) AS id, e.external_id AS external_id, e.shortterm_memory_id AS shortterm_memory_id,
               e.name AS name, e.types AS types, e.description AS description,
               e.importance AS importance, e.access_count AS access_count, e.last_access AS last_access, 
               e.metadata AS metadata
        """

        async with self.neo4j.session() as session:
            result = await session.run(query, entity_id=entity_id)
            record = await result.single()

            if not record:
                return None

            # Parse metadata JSON back to dict
            metadata_dict = json.loads(record["metadata"]) if record["metadata"] else {}

            return ShorttermEntity(
                id=record["id"],
                external_id=record["external_id"],
                shortterm_memory_id=record["shortterm_memory_id"],
                name=record["name"],
                types=record["types"] or [],
                description=record["description"],
                importance=record["importance"] if record["importance"] is not None else 0.5,
                access_count=record["access_count"] or 0,
                last_access=_convert_neo4j_datetime(record["last_access"]),
                metadata=metadata_dict,
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
        RETURN elementId(e) AS id, e.external_id AS external_id, e.shortterm_memory_id AS shortterm_memory_id,
               e.name AS name, e.types AS types, e.description AS description,
               e.importance AS importance, e.access_count AS access_count, e.last_access AS last_access, 
               e.metadata AS metadata
        ORDER BY e.importance DESC
        """

        async with self.neo4j.session() as session:
            result = await session.run(query, shortterm_memory_id=shortterm_memory_id)
            # Use list comprehension with async iteration instead of .list()
            records = [record async for record in result]

            entities = [
                ShorttermEntity(
                    id=record["id"],
                    external_id=record["external_id"],
                    shortterm_memory_id=record["shortterm_memory_id"],
                    name=record["name"],
                    types=record["types"] or [],
                    description=record["description"],
                    importance=record["importance"] if record["importance"] is not None else 0.5,
                    access_count=record["access_count"] or 0,
                    last_access=_convert_neo4j_datetime(record["last_access"]),
                    metadata=json.loads(record["metadata"]) if record["metadata"] else {},
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
        types: Optional[List[str]] = None,  # Added types parameter
        description: Optional[str] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ShorttermEntity]:
        """
        Update an entity.

        Args:
            entity_id: Entity node ID (elementId)
            name: New name (optional)
            types: New types list (optional)
            description: New description (optional)
            importance: New importance (optional)
            metadata: New metadata (optional)

        Returns:
            Updated ShorttermEntity or None if not found
        """
        updates = []
        params = {"entity_id": entity_id, "now": datetime.now(timezone.utc)}

        if name is not None:
            updates.append("e.name = $name")
            params["name"] = name

        if types is not None:
            updates.append("e.types = $types")
            params["types"] = types

        if description is not None:
            updates.append("e.description = $description")
            params["description"] = description

        if importance is not None:
            updates.append("e.importance = $importance")
            params["importance"] = importance

        if metadata is not None:
            updates.append("e.metadata = $metadata_json")
            params["metadata_json"] = json.dumps(metadata)

        if not updates:
            return await self.get_entity(entity_id)

        # Update last access time and increment access count
        updates.append("e.last_access = $now")
        updates.append("e.access_count = COALESCE(e.access_count, 0) + 1")

        query = f"""
        MATCH (e:ShorttermEntity)
        WHERE elementId(e) = $entity_id
        SET {", ".join(updates)}
        RETURN elementId(e) AS id, e.external_id AS external_id, e.shortterm_memory_id AS shortterm_memory_id,
               e.name AS name, e.types AS types, e.description AS description,
               e.importance AS importance, e.access_count AS access_count, e.last_access AS last_access, 
               e.metadata AS metadata
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
                types=record["types"] or [],
                description=record["description"],
                importance=record["importance"] if record["importance"] is not None else 0.5,
                access_count=record["access_count"] or 0,
                last_access=_convert_neo4j_datetime(record["last_access"]),
                metadata=json.loads(record["metadata"]) if record["metadata"] else {},
            )

            logger.info(f"Updated shortterm entity {entity_id}")
            return entity

    async def delete_entity(self, entity_id: int) -> bool:
        """
        Delete an entity and all its relationships.

        Args:
            entity_id: Entity node ID (elementId)

        Returns:
            True if deleted
        """
        query = """
        MATCH (e:ShorttermEntity)
        WHERE elementId(e) = $entity_id
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
        types: List[str],  # Changed from relationship_type: str to types: List[str]
        description: Optional[str] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ShorttermRelationship:
        """
        Create a new relationship between entities in Neo4j.

        Args:
            external_id: Agent identifier
            shortterm_memory_id: Parent memory ID
            from_entity_id: Source entity node ID (elementId)
            to_entity_id: Target entity node ID (elementId)
            types: Relationship types (e.g., ['USES', 'DEPENDS_ON'])
            description: Optional description
            importance: Importance score (0-1)
            metadata: Optional metadata

        Returns:
            Created ShorttermRelationship object
        """
        now = datetime.now(timezone.utc)

        # Convert metadata to JSON string for Neo4j storage
        metadata_json = json.dumps(metadata or {})

        query = """
        MATCH (from:ShorttermEntity)
        WHERE elementId(from) = $from_entity_id
        MATCH (to:ShorttermEntity)
        WHERE elementId(to) = $to_entity_id
        CREATE (from)-[r:SHORTTERM_RELATES {
            external_id: $external_id,
            shortterm_memory_id: $shortterm_memory_id,
            types: $types,
            description: $description,
            importance: $importance,
            access_count: $access_count,
            last_access: $last_access,
            metadata: $metadata_json
        }]->(to)
        RETURN elementId(r) AS id, r.external_id AS external_id, r.shortterm_memory_id AS shortterm_memory_id,
               elementId(from) AS from_entity_id, elementId(to) AS to_entity_id,
               from.name AS from_entity_name, to.name AS to_entity_name,
               r.types AS types, r.description AS description,
               r.importance AS importance, r.access_count AS access_count, r.last_access AS last_access,
               r.metadata AS metadata
        """

        async with self.neo4j.session() as session:
            result = await session.run(
                query,
                external_id=external_id,
                shortterm_memory_id=shortterm_memory_id,
                from_entity_id=from_entity_id,
                to_entity_id=to_entity_id,
                types=types,
                description=description,
                importance=importance,
                access_count=0,
                last_access=None,
                metadata_json=metadata_json,
            )
            record = await result.single()

            # Parse metadata JSON back to dict
            metadata_dict = json.loads(record["metadata"]) if record["metadata"] else {}

            relationship = ShorttermRelationship(
                id=record["id"],
                external_id=record["external_id"],
                shortterm_memory_id=record["shortterm_memory_id"],
                from_entity_id=record["from_entity_id"],
                to_entity_id=record["to_entity_id"],
                from_entity_name=record["from_entity_name"],
                to_entity_name=record["to_entity_name"],
                types=record["types"] or [],
                description=record["description"],
                importance=record["importance"],
                access_count=record["access_count"] or 0,
                last_access=_convert_neo4j_datetime(record["last_access"]),
                metadata=metadata_dict,
            )

            logger.info(
                f"Created shortterm relationship {relationship.id}: "
                f"{record['from_entity_name']} -> {record['to_entity_name']} with types {types}"
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
        MATCH (from:ShorttermEntity)-[r:SHORTTERM_RELATES]->(to:ShorttermEntity)
        WHERE elementId(r) = $relationship_id
        RETURN elementId(r) AS id, r.external_id AS external_id, r.shortterm_memory_id AS shortterm_memory_id,
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

            return ShorttermRelationship(
                id=record["id"],
                external_id=record["external_id"],
                shortterm_memory_id=record["shortterm_memory_id"],
                from_entity_id=record["from_entity_id"],
                to_entity_id=record["to_entity_id"],
                from_entity_name=record["from_entity_name"],
                to_entity_name=record["to_entity_name"],
                types=record["types"] or [],
                description=record["description"],
                importance=record["importance"] if record["importance"] is not None else 0.5,
                access_count=record["access_count"] or 0,
                last_access=_convert_neo4j_datetime(record["last_access"]),
                metadata=json.loads(record["metadata"]) if record["metadata"] else {},
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
        MATCH (from:ShorttermEntity)-[r:SHORTTERM_RELATES {shortterm_memory_id: $shortterm_memory_id}]->(to:ShorttermEntity)
        RETURN elementId(r) AS id, r.external_id AS external_id, r.shortterm_memory_id AS shortterm_memory_id,
               elementId(from) AS from_entity_id, elementId(to) AS to_entity_id,
               from.name AS from_entity_name, to.name AS to_entity_name,
               r.types AS types, r.description AS description,
               r.importance AS importance, r.access_count AS access_count, r.last_access AS last_access,
               r.metadata AS metadata
        ORDER BY r.importance DESC
        """

        async with self.neo4j.session() as session:
            result = await session.run(query, shortterm_memory_id=shortterm_memory_id)
            records = [record async for record in result]

            relationships = [
                ShorttermRelationship(
                    id=record["id"],
                    external_id=record["external_id"],
                    shortterm_memory_id=record["shortterm_memory_id"],
                    from_entity_id=record["from_entity_id"],
                    to_entity_id=record["to_entity_id"],
                    from_entity_name=record["from_entity_name"],
                    to_entity_name=record["to_entity_name"],
                    types=record["types"] or [],
                    description=record["description"],
                    importance=record["importance"] if record["importance"] is not None else 0.5,
                    access_count=record["access_count"] or 0,
                    last_access=_convert_neo4j_datetime(record["last_access"]),
                    metadata=json.loads(record["metadata"]) if record["metadata"] else {},
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
        types: Optional[List[str]] = None,
        description: Optional[str] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ShorttermRelationship]:
        """
        Update a relationship.

        Args:
            relationship_id: Relationship ID
            types: New types (optional)
            description: New description (optional)
            importance: New importance (optional)
            metadata: New metadata (optional)

        Returns:
            Updated ShorttermRelationship or None if not found
        """
        updates = []
        params = {"relationship_id": relationship_id, "now": datetime.now(timezone.utc)}

        if types is not None:
            updates.append("r.types = $types")
            params["types"] = types

        if description is not None:
            updates.append("r.description = $description")
            params["description"] = description

        if importance is not None:
            updates.append("r.importance = $importance")
            params["importance"] = importance

        if metadata is not None:
            updates.append("r.metadata = $metadata")
            params["metadata"] = metadata

        if not updates:
            return await self.get_relationship(relationship_id)

        updates.append("r.last_access = $now")

        query = f"""
        MATCH (from:ShorttermEntity)-[r:SHORTTERM_RELATES]->(to:ShorttermEntity)
        WHERE elementId(r) = $relationship_id
        SET {", ".join(updates)}
        RETURN elementId(r) AS id, r.external_id AS external_id, r.shortterm_memory_id AS shortterm_memory_id,
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

            relationship = ShorttermRelationship(
                id=record["id"],
                external_id=record["external_id"],
                shortterm_memory_id=record["shortterm_memory_id"],
                from_entity_id=record["from_entity_id"],
                to_entity_id=record["to_entity_id"],
                from_entity_name=record["from_entity_name"],
                to_entity_name=record["to_entity_name"],
                types=record["types"] or [],
                description=record["description"],
                importance=record["importance"] if record["importance"] is not None else 0.5,
                access_count=record["access_count"] or 0,
                last_access=_convert_neo4j_datetime(record["last_access"]),
                metadata=json.loads(record["metadata"]) if record["metadata"] else {},
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
        MATCH ()-[r:SHORTTERM_RELATES]->()
        WHERE elementId(r) = $relationship_id
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
        # Handle both dict and list/tuple row formats
        if isinstance(row, dict):
            return ShorttermMemory(
                id=row.get("id"),
                external_id=row.get("external_id"),
                title=row.get("title"),
                summary=row.get("summary"),
                update_count=row.get("update_count", 0),
                metadata=row.get("metadata", {}),
                created_at=row.get("created_at", datetime.now(timezone.utc)),
                last_updated=row.get("last_updated", datetime.now(timezone.utc)),
                chunks=[],  # Populated separately if needed
            )
        else:
            # Handle list/tuple format
            # Expected order: id, external_id, title, summary, update_count, metadata, created_at, last_updated
            return ShorttermMemory(
                id=row[0],
                external_id=row[1],
                title=row[2],
                summary=row[3],
                update_count=int(row[4]) if row[4] is not None else 0,
                metadata=row[5] if isinstance(row[5], dict) else {},
                created_at=row[6] if len(row) > 6 else datetime.now(timezone.utc),
                last_updated=row[7] if len(row) > 7 else datetime.now(timezone.utc),
                chunks=[],  # Populated separately if needed
            )

    def _chunk_row_to_model(self, row) -> ShorttermMemoryChunk:
        """Convert database row to ShorttermMemoryChunk model."""
        # Handle both dict (psqlpy) and tuple/list formats
        if isinstance(row, dict):
            return ShorttermMemoryChunk(
                id=row["id"],
                shortterm_memory_id=row["shortterm_memory_id"],
                content=row["content"],
                section_id=row.get("section_id"),
                access_count=row.get("access_count", 0),
                last_access=row.get("last_access"),
                metadata=row.get("metadata", {}),
            )
        elif isinstance(row, (list, tuple)):
            # Handle different tuple formats
            if len(row) >= 9:
                # Full format with external_id and created_at:
                # id, shortterm_memory_id, external_id, content, section_id, metadata, access_count, last_access, created_at
                return ShorttermMemoryChunk(
                    id=row[0],
                    shortterm_memory_id=row[1],
                    content=row[3],
                    section_id=row[4] if isinstance(row[4], str) else None,
                    metadata=row[5] if isinstance(row[5], dict) else {},
                    access_count=int(row[6]) if row[6] is not None else 0,
                    last_access=row[7] if len(row) > 7 else None,
                )
            else:
                # Legacy format: id, shortterm_memory_id, content, section_id, metadata, access_count, last_access
                return ShorttermMemoryChunk(
                    id=row[0],
                    shortterm_memory_id=row[1],
                    content=row[2],
                    section_id=row[3] if isinstance(row[3], str) else None,
                    metadata=row[4] if len(row) > 4 and isinstance(row[4], dict) else {},
                    access_count=int(row[5]) if len(row) > 5 and row[5] is not None else 0,
                    last_access=row[6] if len(row) > 6 else None,
                )
        else:
            raise ValueError(f"Unsupported row format: {type(row)}")
