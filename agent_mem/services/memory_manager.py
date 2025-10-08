"""
Memory Manager - Core orchestration for memory operations.

Handles memory lifecycle: Active → Shortterm → Longterm
Coordinates consolidation, promotion, and retrieval across memory tiers.
"""

import logging
import asyncio
from typing import List, Optional, Dict, Any, Set
from datetime import datetime, timezone

from agent_mem.config import Config
from agent_mem.database import (
    PostgreSQLManager,
    Neo4jManager,
    ActiveMemoryRepository,
)
from agent_mem.database.repositories import (
    ShorttermMemoryRepository,
    LongtermMemoryRepository,
)
from agent_mem.database.models import (
    ActiveMemory,
    RetrievalResult,
    RetrievedChunk,
    RetrievedEntity,
    RetrievedRelationship,
    ShorttermMemory,
    ShorttermMemoryChunk,
    LongtermMemoryChunk,
    ConflictSection,
    ConsolidationConflicts,
    ConflictEntityDetail,
    ConflictRelationshipDetail,
)
from agent_mem.services.embedding import EmbeddingService
from agent_mem.utils.helpers import chunk_text
from agent_mem.agents import (
    MemoryRetrieveAgent,
    retrieve_memory,
    extract_entities_and_relationships,
)

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Core memory management orchestrator.

    STATELESS - Can serve multiple agents/workers.
    Coordinates database operations, embedding generation, and agent workflows.
    """

    def __init__(self, config: Config):
        """
        Initialize memory manager (stateless).

        Args:
            config: Configuration object
        """
        self.config = config

        # Database managers
        self.postgres_manager = PostgreSQLManager(config)
        self.neo4j_manager = Neo4jManager(config)

        # Services
        self.embedding_service = EmbeddingService(config)

        # AI Agents
        self.retriever_agent: Optional[MemoryRetrieveAgent] = None

        # Repositories (initialized after database connection)
        self.active_repo: Optional[ActiveMemoryRepository] = None
        self.shortterm_repo: Optional[ShorttermMemoryRepository] = None
        self.longterm_repo: Optional[LongtermMemoryRepository] = None

        # Consolidation locks to prevent concurrent consolidation
        self._consolidation_locks: Dict[int, asyncio.Lock] = {}

        # Track background consolidation tasks
        self._background_tasks: Set[asyncio.Task] = set()

        self._initialized = False

        logger.info("MemoryManager created (stateless)")

    async def initialize(self) -> None:
        """Initialize database connections and repositories."""
        if self._initialized:
            logger.warning("MemoryManager already initialized")
            return

        logger.info("Initializing MemoryManager...")

        # Initialize database managers
        await self.postgres_manager.initialize()
        await self.neo4j_manager.initialize()

        # Initialize repositories
        self.active_repo = ActiveMemoryRepository(self.postgres_manager)
        self.shortterm_repo = ShorttermMemoryRepository(self.postgres_manager, self.neo4j_manager)
        self.longterm_repo = LongtermMemoryRepository(self.postgres_manager, self.neo4j_manager)

        # Verify embedding service
        embedding_ok = await self.embedding_service.verify_connection()
        if not embedding_ok:
            logger.warning(
                "Ollama embedding service not available. " "Embeddings will return zero vectors."
            )

        # Initialize AI agents
        self.retriever_agent = MemoryRetrieveAgent(self.config)
        logger.info("AI agents initialized")

        self._initialized = True
        logger.info("MemoryManager initialization complete")

    async def close(self) -> None:
        """Close all database connections."""
        if not self._initialized:
            return

        logger.info("Closing MemoryManager connections...")

        # Wait for all background consolidation tasks to complete
        if self._background_tasks:
            logger.info(
                f"Waiting for {len(self._background_tasks)} background tasks to complete..."
            )
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
            logger.info("All background tasks completed")

        await self.postgres_manager.close()
        await self.neo4j_manager.close()

        self._initialized = False
        logger.info("MemoryManager closed")

    async def create_active_memory(
        self,
        external_id: str,
        title: str,
        template_content: str,
        initial_sections: Dict[str, Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> ActiveMemory:
        """
        Create a new active memory with template and sections.

        Args:
            external_id: Agent identifier
            title: Memory title
            template_content: YAML template
            initial_sections: Initial sections {section_id: {content, update_count}}
            metadata: Metadata dictionary

        Returns:
            Created ActiveMemory object
        """
        self._ensure_initialized()

        logger.info(f"Creating active memory for {external_id}: {title}")

        memory = await self.active_repo.create(
            external_id=external_id,
            title=title,
            template_content=template_content,
            sections=initial_sections,
            metadata=metadata,
        )

        logger.info(f"Created active memory {memory.id} for {external_id}")
        return memory

    async def get_active_memories(self, external_id: str) -> List[ActiveMemory]:
        """
        Get all active memories for a specific agent.

        Args:
            external_id: Agent identifier

        Returns:
            List of ActiveMemory objects
        """
        self._ensure_initialized()

        logger.info(f"Retrieving all active memories for {external_id}")

        memories = await self.active_repo.get_all_by_external_id(external_id)

        logger.info(f"Retrieved {len(memories)} active memories for {external_id}")
        return memories

    async def update_active_memory_sections(
        self,
        external_id: str,
        memory_id: int,
        sections: List[Dict[str, str]],
    ) -> ActiveMemory:
        """
        Update multiple sections in an active memory (batch update).

        After updating all sections, checks if total update count across all sections
        exceeds threshold for consolidation. If threshold is met, triggers
        consolidation in the background without blocking.

        Args:
            external_id: Agent identifier
            memory_id: Memory ID
            sections: List of section updates with section_id and new_content
                     Example: [{"section_id": "progress", "new_content": "..."}]

        Returns:
            Updated ActiveMemory object

        Raises:
            ValueError: If memory not found or section invalid

        Example:
            >>> sections = [
            ...     {"section_id": "progress", "new_content": "Updated progress..."},
            ...     {"section_id": "notes", "new_content": "New notes..."}
            ... ]
            >>> memory = await manager.update_active_memory_sections(
            ...     external_id="agent-123",
            ...     memory_id=1,
            ...     sections=sections
            ... )
        """
        self._ensure_initialized()

        logger.info(f"Updating {len(sections)} sections in memory {memory_id} for {external_id}")

        # Update all sections in repository (transactional)
        memory = await self.active_repo.update_sections(
            memory_id=memory_id,
            section_updates=sections,
        )

        if not memory:
            raise ValueError(f"Active memory {memory_id} not found")

        logger.info(f"Updated {len(sections)} sections in memory {memory_id}")

        # Calculate threshold based on total number of sections
        num_sections = len(memory.sections)
        threshold = self.config.avg_section_update_count_for_consolidation * num_sections

        # Calculate total update count across all sections
        total_update_count = sum(
            section.get("update_count", 0) for section in memory.sections.values()
        )

        logger.debug(
            f"Total update count: {total_update_count}, "
            f"Threshold: {threshold} ({num_sections} sections * "
            f"{self.config.avg_section_update_count_for_consolidation})"
        )

        # Check if consolidation threshold is met
        if total_update_count >= threshold:
            logger.info(
                f"Total update count ({total_update_count}) >= threshold ({threshold}). "
                f"Triggering consolidation in background..."
            )

            # Start consolidation in background (non-blocking)
            task = asyncio.create_task(self._consolidate_with_lock(external_id, memory.id))
            self._background_tasks.add(task)
            # Remove task from set when done
            task.add_done_callback(self._background_tasks.discard)

        return memory

    async def delete_active_memory(
        self,
        external_id: str,
        memory_id: int,
    ) -> bool:
        """
        Delete an active memory and all its sections.

        Args:
            external_id: Agent identifier
            memory_id: Memory ID to delete

        Returns:
            True if deleted, False if not found
        """
        self._ensure_initialized()

        logger.info(f"Deleting active memory {memory_id} for {external_id}")

        # Verify memory exists and belongs to this agent
        memory = await self.active_repo.get_by_id(memory_id)
        if not memory or memory.external_id != external_id:
            logger.warning(f"Memory {memory_id} not found or does not belong to {external_id}")
            return False

        # Delete the memory (repository will handle cascade delete of sections)
        success = await self.active_repo.delete(memory_id)

        if success:
            logger.info(f"Deleted active memory {memory_id} for {external_id}")
        else:
            logger.warning(f"Failed to delete memory {memory_id}")

        return success

    async def retrieve_memories(
        self,
        external_id: str,
        query: str,
        limit: int = 10,
        synthesis: bool = False,
    ) -> RetrievalResult:
        """
        Retrieve memories across all tiers using MemoryRetrieveAgent.

        Uses intelligent retrieval agent to:
        - Analyze query and determine optimal search strategy
        - Search across shortterm and longterm tiers
        - Extract and rank relevant entities and relationships
        - Optionally synthesize results into natural language

        Args:
            external_id: Agent identifier
            query: Search query
            limit: Maximum results per tier (can be overridden by agent)
            synthesis: Force AI synthesis regardless of query complexity (default: False)

        Returns:
            RetrievalResult object with aggregated results and optional AI synthesis
        """
        self._ensure_initialized()

        logger.info(f"Retrieving memories for {external_id}, query: {query[:50]}...")

        try:
            # Use the retrieve_memory function from memory_retriever agent
            result = await retrieve_memory(
                query=query,
                external_id=external_id,
                shortterm_repo=self.shortterm_repo,
                longterm_repo=self.longterm_repo,
                active_repo=self.active_repo,
                embedding_service=self.embedding_service,
                synthesis=synthesis,
            )

            logger.info(
                f"Memory retrieval completed: mode={result.mode}, "
                f"confidence={result.confidence:.2f}, "
                f"{len(result.chunks)} chunks, {len(result.entities)} entities, "
                f"{len(result.relationships)} relationships"
            )

            return result

        except Exception as e:
            logger.error(f"Agent-based retrieval failed: {e}", exc_info=True)
            logger.warning("Falling back to basic retrieval...")

            # Fallback to basic retrieval (always search both tiers)
            return await self._retrieve_memories_basic(external_id, query, limit)

    async def _retrieve_memories_basic(
        self,
        external_id: str,
        query: str,
        limit: int = 10,
    ) -> RetrievalResult:
        """
        Fallback: Basic retrieval without AI agent.

        Used when MemoryRetrieveAgent fails or is unavailable.
        Performs simple hybrid search across both tiers without intelligent ranking or synthesis.

        Args:
            external_id: Agent identifier
            query: Search query
            limit: Maximum results per tier

        Returns:
            RetrievalResult with basic search results (no synthesis)
        """
        self._ensure_initialized()

        logger.info(f"Basic retrieval (no agent) for {external_id}, query: {query[:50]}...")

        # Generate embedding for query
        query_embedding = await self.embedding_service.generate_embedding(query)

        chunks = []
        entities = []
        relationships = []

        # Search shortterm memory
        # Search shortterm memory
        try:
            st_chunks = await self.shortterm_repo.hybrid_search_chunks(
                external_id=external_id,
                query_text=query,
                query_embedding=query_embedding,
                limit=limit,
                vector_weight=0.5,
                bm25_weight=0.5,
            )

            for chunk in st_chunks:
                # Calculate a combined score (average of available scores)
                scores = []
                if chunk.similarity_score is not None:
                    scores.append(chunk.similarity_score)
                if chunk.bm25_score is not None:
                    scores.append(chunk.bm25_score)
                combined_score = sum(scores) / len(scores) if scores else 0.0

                chunks.append(
                    RetrievedChunk(
                        id=chunk.id,
                        content=chunk.content,
                        tier="shortterm",
                        score=combined_score,
                        importance=None,
                        start_date=None,
                    )
                )

            logger.info(f"Found {len(st_chunks)} shortterm chunks")
        except Exception as e:
            logger.error(f"Error searching shortterm chunks: {e}", exc_info=True)

        # Search longterm memory
        try:
            lt_chunks = await self.longterm_repo.hybrid_search_chunks(
                external_id=external_id,
                query_text=query,
                query_embedding=query_embedding,
                limit=limit,
                vector_weight=0.5,
                bm25_weight=0.5,
            )

            for chunk in lt_chunks:
                # Calculate a combined score (average of available scores)
                scores = []
                if chunk.similarity_score is not None:
                    scores.append(chunk.similarity_score)
                if chunk.bm25_score is not None:
                    scores.append(chunk.bm25_score)
                combined_score = sum(scores) / len(scores) if scores else 0.0

                chunks.append(
                    RetrievedChunk(
                        id=chunk.id,
                        content=chunk.content,
                        tier="longterm",
                        score=combined_score,
                        importance=chunk.importance,
                        start_date=chunk.start_date,
                    )
                )

            logger.info(f"Found {len(lt_chunks)} longterm chunks")
        except Exception as e:
            logger.error(f"Error searching longterm chunks: {e}", exc_info=True)

        # Sort chunks by score (descending)
        chunks.sort(key=lambda x: x.score, reverse=True)

        # Limit total chunks
        chunks = chunks[: limit * 2]  # Allow more results since we're combining tiers

        return RetrievalResult(
            mode="pointer",
            chunks=chunks,
            entities=entities,
            relationships=relationships,
            synthesis=None,
            search_strategy="Basic hybrid search (vector + BM25) across both tiers without AI agent",
            confidence=0.5,  # Lower confidence for basic search
            metadata={
                "limit": limit,
                "total_chunks": len(chunks),
            },
        )

    def _ensure_initialized(self) -> None:
        """Ensure manager is initialized."""
        if not self._initialized:
            raise RuntimeError("MemoryManager not initialized. Call initialize() first.")

    # =========================================================================
    # CONSOLIDATION WORKFLOWS
    # =========================================================================

    async def _consolidate_with_lock(
        self,
        external_id: str,
        memory_id: int,
    ) -> None:
        """
        Consolidate with lock to prevent concurrent consolidation.

        Ensures only one consolidation process runs for a given memory at a time.
        If a consolidation is already in progress, logs a message and returns.

        Args:
            external_id: Agent identifier
            memory_id: Memory ID to consolidate
        """
        # Get or create lock for this memory
        if memory_id not in self._consolidation_locks:
            self._consolidation_locks[memory_id] = asyncio.Lock()

        lock = self._consolidation_locks[memory_id]

        # Try to acquire lock (non-blocking)
        if lock.locked():
            logger.info(f"Consolidation already in progress for memory {memory_id}, skipping")
            return

        async with lock:
            try:
                logger.info(f"Starting consolidation for memory {memory_id}")
                await self._consolidate_to_shortterm(external_id, memory_id)
                logger.info(f"Consolidation completed for memory {memory_id}")
            except Exception as e:
                logger.error(
                    f"Consolidation failed for memory {memory_id}: {e}",
                    exc_info=True,
                )
            finally:
                # Clean up lock if no longer needed
                if memory_id in self._consolidation_locks:
                    del self._consolidation_locks[memory_id]
                    logger.debug(f"Removed consolidation lock for memory {memory_id}")

    async def _consolidate_to_shortterm(
        self, external_id: str, active_memory_id: int
    ) -> Optional[ShorttermMemory]:
        """
        Consolidate active memory to shortterm using new 13-step algorithm.

        Algorithm:
        1. Get active memory
        2. Find or create shortterm memory
        3. Extract sections with update_count > 0
        4. For each extracted section, find existing chunks
        5. For sections without chunks, create new chunks
        6. For sections with chunks, add to conflict list
        7. Extract entities/relationships from active memory
        8. Get all shortterm entities/relationships
        9. Merge active entities with shortterm (detect conflicts)
        10. Merge active relationships with shortterm (detect conflicts)
        11. Send conflicts to memorizer agent for resolution
        12. Reset active memory section update counts
        13. Increment shortterm memory update_count
        14. Check promotion threshold and promote if needed

        Args:
            external_id: Agent identifier
            active_memory_id: Active memory to consolidate

        Returns:
            Updated ShorttermMemory or None if consolidation failed
        """
        logger.info(
            f"Starting consolidation for external_id={external_id}, "
            f"active_memory_id={active_memory_id}"
        )

        try:
            # STEP 1: Get active memory
            active_memory = await self.active_repo.get_by_id(active_memory_id)
            if not active_memory:
                logger.error(f"Active memory {active_memory_id} not found")
                return None

            logger.info(f"Retrieved active memory: {active_memory.title}")

            # STEP 2: Find or create shortterm memory
            shortterm_memory = await self._find_or_create_shortterm_memory(
                external_id=external_id,
                title=active_memory.title,
                metadata=active_memory.metadata,
            )

            logger.info(f"Using shortterm memory ID: {shortterm_memory.id}")

            # STEP 3: Extract sections with update_count > 0
            updated_sections = {
                section_id: section_data
                for section_id, section_data in active_memory.sections.items()
                if section_data.get("update_count", 0) > 0
            }

            if not updated_sections:
                logger.info("No updated sections found, skipping consolidation")
                return shortterm_memory

            logger.info(
                f"Found {len(updated_sections)} updated sections: "
                f"{list(updated_sections.keys())}"
            )

            # Initialize conflict tracker
            conflicts = ConsolidationConflicts(
                external_id=external_id,
                active_memory_id=active_memory_id,
                shortterm_memory_id=shortterm_memory.id,
                created_at=datetime.now(timezone.utc),
            )

            # STEP 4: Find chunks for each extracted section
            section_chunks_map: Dict[str, List[ShorttermMemoryChunk]] = {}

            for section_id in updated_sections.keys():
                existing_chunks = await self.shortterm_repo.get_chunks_by_section_id(
                    shortterm_memory_id=shortterm_memory.id,
                    section_id=section_id,
                )
                section_chunks_map[section_id] = existing_chunks
                logger.info(
                    f"Section '{section_id}': found {len(existing_chunks)} " f"existing chunks"
                )

            # STEP 5: Create chunks for sections without existing chunks
            new_chunks_created = 0

            for section_id, section_data in updated_sections.items():
                existing_chunks = section_chunks_map[section_id]

                if len(existing_chunks) == 0:
                    # No existing chunks - create new ones
                    content = section_data.get("content", "")
                    if not content:
                        logger.warning(f"Section '{section_id}' has no content, skipping")
                        continue

                    # Chunk the content
                    content_chunks = chunk_text(
                        text=content,
                        chunk_size=self.config.chunk_size,
                        overlap=self.config.chunk_overlap,
                    )

                    logger.info(
                        f"Creating {len(content_chunks)} new chunks for " f"section '{section_id}'"
                    )

                    # Create chunks with embeddings
                    for i, chunk_content in enumerate(content_chunks):
                        try:
                            # Generate embedding
                            embedding = await self.embedding_service.get_embedding(chunk_content)

                            # Create chunk
                            chunk = await self.shortterm_repo.create_chunk(
                                shortterm_memory_id=shortterm_memory.id,
                                external_id=external_id,
                                content=chunk_content,
                                embedding=embedding,
                                section_id=section_id,
                                metadata={
                                    "source": "consolidation",
                                    "section_id": section_id,
                                },
                            )
                            new_chunks_created += 1
                        except Exception as e:
                            logger.error(
                                f"Failed to create chunk {i} for section " f"'{section_id}': {e}",
                                exc_info=True,
                            )

            logger.info(f"Created {new_chunks_created} new chunks")

            # STEP 6: Add conflicts for sections with existing chunks
            for section_id, section_data in updated_sections.items():
                existing_chunks = section_chunks_map[section_id]

                if len(existing_chunks) > 0:
                    # Existing chunks found - this is a conflict
                    conflicts.sections.append(
                        ConflictSection(
                            section_id=section_id,
                            section_content=section_data.get("content", ""),
                            update_count=section_data.get("update_count", 0),
                            existing_chunks=existing_chunks,
                            metadata={"has_conflicts": len(existing_chunks) > 0},
                        )
                    )
                    logger.info(
                        f"Conflict detected for section '{section_id}': "
                        f"{len(existing_chunks)} existing chunks"
                    )

            # STEP 7: Extract entities and relationships from active memory
            combined_content = self._extract_content_from_sections(active_memory)

            logger.info("Extracting entities and relationships from active memory...")
            active_extraction = await extract_entities_and_relationships(combined_content)

            active_entities = active_extraction.entities if active_extraction else []
            active_relationships = active_extraction.relationships if active_extraction else []

            logger.info(
                f"Extracted {len(active_entities)} entities and "
                f"{len(active_relationships)} relationships from active memory"
            )

            # STEP 8: Get all shortterm entities and relationships
            try:
                shortterm_entities = await self.shortterm_repo.get_entities_by_memory(
                    shortterm_memory_id=shortterm_memory.id
                )

                shortterm_relationships = await self.shortterm_repo.get_relationships_by_memory(
                    shortterm_memory_id=shortterm_memory.id
                )

                logger.info(
                    f"Retrieved {len(shortterm_entities)} entities and "
                    f"{len(shortterm_relationships)} relationships from shortterm"
                )
            except Exception as e:
                logger.warning(f"Failed to retrieve entities/relationships from Neo4j: {e}")
                logger.warning("Continuing consolidation without entity/relationship merging")
                shortterm_entities = []
                shortterm_relationships = []

            # STEP 9-12: Process entities and relationships (optional - skip if Neo4j unavailable)
            # Create lookup maps
            shortterm_entities_map = {entity.name: entity for entity in shortterm_entities}
            shortterm_rel_map = {
                (rel.from_entity_name, rel.to_entity_name): rel
                for rel in shortterm_relationships
                if rel.from_entity_name and rel.to_entity_name
            }

            # STEP 9: Merge active entities with shortterm
            logger.info("Merging entities...")

            for active_entity in active_entities:
                entity_name = active_entity.name
                if not entity_name:
                    continue

                # Get types as list
                active_types = (
                    active_entity.types
                    if hasattr(active_entity, "types") and active_entity.types
                    else [active_entity.type.value] if hasattr(active_entity, "type") else []
                )

                active_confidence = (
                    active_entity.confidence if hasattr(active_entity, "confidence") else 0.5
                )
                active_description = (
                    active_entity.description if hasattr(active_entity, "description") else ""
                )

                if entity_name in shortterm_entities_map:
                    # Entity exists in shortterm - MERGE
                    shortterm_entity = shortterm_entities_map[entity_name]

                    # Merge types (union of both sets)
                    merged_types = list(set(shortterm_entity.types + active_types))

                    # Recalculate confidence (weighted average, favor higher)
                    merged_confidence = max(
                        (shortterm_entity.importance + active_confidence) / 2,
                        shortterm_entity.importance,
                        active_confidence,
                    )

                    # Use more detailed description
                    merged_description = (
                        active_description
                        if len(active_description) > len(shortterm_entity.description or "")
                        else shortterm_entity.description
                    )

                    # Update entity in database
                    try:
                        await self.shortterm_repo.update_entity(
                            entity_id=shortterm_entity.id,
                            types=merged_types,
                            description=merged_description,
                            confidence=merged_confidence,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update entity {entity_name}: {e}")
                        # Continue without updating

                    # Track conflict
                    conflicts.entity_conflicts.append(
                        ConflictEntityDetail(
                            name=entity_name,
                            shortterm_types=shortterm_entity.types,
                            active_types=active_types,
                            merged_types=merged_types,
                            shortterm_importance=shortterm_entity.importance,
                            active_confidence=active_confidence,
                            merged_confidence=merged_confidence,
                            shortterm_description=shortterm_entity.description,
                            active_description=active_description,
                            merged_description=merged_description,
                        )
                    )

                    logger.debug(
                        f"Merged entity '{entity_name}': "
                        f"types {shortterm_entity.types} + {active_types} = "
                        f"{merged_types}"
                    )
                else:
                    # New entity - CREATE
                    try:
                        await self.shortterm_repo.create_entity(
                            external_id=external_id,
                            shortterm_memory_id=shortterm_memory.id,
                            name=entity_name,
                            types=active_types,
                            description=active_description,
                            importance=active_confidence,  # Using confidence as importance
                            metadata={},
                        )

                        logger.debug(
                            f"Created new entity '{entity_name}' with types {active_types}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create entity {entity_name}: {e}")
                        # Continue without creating

            # STEP 10: Merge active relationships with shortterm
            logger.info("Merging relationships...")

            # Refresh entity map after new entity creation
            try:
                shortterm_entities = await self.shortterm_repo.get_entities_by_memory(
                    shortterm_memory_id=shortterm_memory.id
                )
                shortterm_entities_map = {entity.name: entity for entity in shortterm_entities}
            except Exception as e:
                logger.warning(f"Failed to refresh entities after creation: {e}")
                # Continue with existing entity map

            for active_rel in active_relationships:
                from_entity = active_rel.source if hasattr(active_rel, "source") else None
                to_entity = active_rel.target if hasattr(active_rel, "target") else None

                if not from_entity or not to_entity:
                    continue

                # Get types as list
                active_types = (
                    active_rel.types
                    if hasattr(active_rel, "types") and active_rel.types
                    else [active_rel.type.value] if hasattr(active_rel, "type") else []
                )

                active_confidence = (
                    active_rel.importance if hasattr(active_rel, "confidence") else 0.5
                )
                active_description = (
                    active_rel.description if hasattr(active_rel, "description") else ""
                )

                rel_key = (from_entity, to_entity)

                if rel_key in shortterm_rel_map:
                    # Relationship exists - MERGE
                    shortterm_rel = shortterm_rel_map[rel_key]

                    # Merge types
                    merged_types = list(set(shortterm_rel.types + active_types))

                    # Recalculate confidence
                    merged_confidence = max(
                        (shortterm_rel.importance + active_confidence) / 2,
                        shortterm_rel.importance,
                        active_confidence,
                    )

                    # Use more detailed description
                    merged_description = (
                        active_description
                        if len(active_description) > len(shortterm_rel.description or "")
                        else shortterm_rel.description
                    )

                    # Update relationship in database
                    try:
                        await self.shortterm_repo.update_relationship(
                            relationship_id=shortterm_rel.id,
                            types=merged_types,
                            description=merged_description,
                            confidence=merged_confidence,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to update relationship {from_entity}-{to_entity}: {e}"
                        )
                        # Continue without updating

                    # Track conflict
                    conflicts.relationship_conflicts.append(
                        ConflictRelationshipDetail(
                            from_entity=from_entity,
                            to_entity=to_entity,
                            shortterm_types=shortterm_rel.types,
                            active_types=active_types,
                            merged_types=merged_types,
                            shortterm_importance=shortterm_rel.importance,
                            active_confidence=active_confidence,
                            merged_confidence=merged_confidence,
                        )
                    )

                    logger.debug(
                        f"Merged relationship '{from_entity}'-'{to_entity}': "
                        f"types {shortterm_rel.types} + {active_types} = "
                        f"{merged_types}"
                    )
                else:
                    # New relationship - CREATE
                    # First, get entity IDs
                    from_entity_obj = shortterm_entities_map.get(from_entity)
                    to_entity_obj = shortterm_entities_map.get(to_entity)

                    if not from_entity_obj or not to_entity_obj:
                        logger.warning(
                            f"Cannot create relationship '{from_entity}'-'{to_entity}': "
                            f"entities not found"
                        )
                        continue

                    try:
                        await self.shortterm_repo.create_relationship(
                            external_id=external_id,
                            shortterm_memory_id=shortterm_memory.id,
                            from_entity_id=from_entity_obj.id,
                            to_entity_id=to_entity_obj.id,
                            types=active_types,
                            description=active_description,
                            importance=active_confidence,  # Using confidence as importance
                            metadata={},
                        )

                        logger.debug(
                            f"Created new relationship '{from_entity}'-'{to_entity}' "
                            f"with types {active_types}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to create relationship {from_entity}-{to_entity}: {e}"
                        )
                        # Continue without creating

            # STEP 11: Reset active memory section update counts
            await self.active_repo.reset_all_update_counts(active_memory.id)
            logger.info(f"Reset section update counts for active memory {active_memory.id}")

            # STEP 12: Send conflicts to memorizer agent for resolution
            # Count total existing chunks across all sections
            total_existing_chunks = sum(
                len(section.existing_chunks) for section in conflicts.sections
            )

            conflicts.total_conflicts = (
                total_existing_chunks
                + len(conflicts.entity_conflicts)
                + len(conflicts.relationship_conflicts)
            )

            if conflicts.total_conflicts > 0:
                logger.info(f"Detected {conflicts.total_conflicts} total conflicts")

                # Call the memorizer agent for intelligent conflict resolution
                try:
                    from agent_mem.agents.memorizer import resolve_conflicts

                    resolution = await resolve_conflicts(conflicts, self.shortterm_repo)

                    logger.info(f"Memorizer agent resolution: {resolution.summary}")
                    logger.info(
                        f"Resolution actions: "
                        f"{len(resolution.chunk_updates)} chunk updates, "
                        f"{len(resolution.chunk_creates)} chunk creates, "
                        f"{len(resolution.entity_updates)} entity updates, "
                        f"{len(resolution.relationship_updates)} relationship updates"
                    )

                    # Note: The resolution actions were already applied by the agent tools
                    # The auto-merge in steps 9-10 already handled basic merging
                    # The agent provides additional intelligent refinements

                except Exception as e:
                    logger.error(f"Error in memorizer agent resolution: {e}", exc_info=True)
                    logger.info(
                        "Continuing with automatic conflict resolution from merge steps 9-10"
                    )
            else:
                logger.info("No conflicts detected, skipping memorizer agent")

            # STEP 13: Increment shortterm memory update_count
            updated_shortterm = await self.shortterm_repo.increment_update_count(
                memory_id=shortterm_memory.id
            )

            if not updated_shortterm:
                logger.error(
                    f"Failed to increment update_count for shortterm " f"{shortterm_memory.id}"
                )
                return shortterm_memory

            logger.info(
                f"Incremented shortterm update_count to " f"{updated_shortterm.update_count}"
            )

            # STEP 14: Check promotion threshold
            if updated_shortterm.update_count >= self.config.shortterm_update_count_threshold:
                logger.info(
                    f"Shortterm update_count ({updated_shortterm.update_count}) >= "
                    f"threshold ({self.config.shortterm_update_count_threshold}). "
                    f"Triggering promotion..."
                )

                # Promote to longterm
                await self._promote_to_longterm(external_id, updated_shortterm.id)

                # Delete all chunks
                deleted_count = await self.shortterm_repo.delete_all_chunks(updated_shortterm.id)
                logger.info(f"Deleted {deleted_count} chunks after promotion")

                # Reset update count
                await self.shortterm_repo.reset_update_count(updated_shortterm.id)
                logger.info("Reset shortterm update_count to 0")

            logger.info(
                f"Consolidation complete: created {new_chunks_created} chunks, "
                f"merged {len(conflicts.entity_conflicts)} entities, "
                f"merged {len(conflicts.relationship_conflicts)} relationships"
            )

            return updated_shortterm

        except Exception as e:
            logger.error(f"Consolidation failed: {e}", exc_info=True)
            return None

    async def _find_or_create_shortterm_memory(
        self,
        external_id: str,
        title: str,
        metadata: Dict[str, Any],
    ) -> ShorttermMemory:
        """
        Find existing shortterm memory by title or create new one.

        Args:
            external_id: Agent identifier
            title: Memory title
            metadata: Metadata dictionary

        Returns:
            ShorttermMemory object
        """
        # Try to find existing shortterm memory with same title
        existing_memories = await self.shortterm_repo.get_memories_by_external_id(external_id)

        for memory in existing_memories:
            if memory.title == title:
                logger.info(f"Found existing shortterm memory {memory.id} for title: {title}")
                return memory

        # Create new shortterm memory
        logger.info(f"Creating new shortterm memory for title: {title}")
        new_memory = await self.shortterm_repo.create_memory(
            external_id=external_id,
            title=title,
            summary=f"Consolidated from active memory: {title}",
            metadata=metadata,
        )

        return new_memory

    def _extract_content_from_sections(self, active_memory: ActiveMemory) -> str:
        """
        Extract and concatenate content from all sections.

        Args:
            active_memory: ActiveMemory object

        Returns:
            Concatenated content string
        """
        parts = [f"# {active_memory.title}\n"]

        if active_memory.sections:
            for section_id, section_data in active_memory.sections.items():
                content = section_data.get("content", "")
                if content and content.strip():
                    parts.append(f"\n## {section_id}\n{content}")

        return "\n".join(parts)

    # =========================================================================
    # PROMOTION WORKFLOWS
    # =========================================================================

    async def _promote_to_longterm(
        self, external_id: str, shortterm_memory_id: int
    ) -> List[LongtermMemoryChunk]:
        """
        Enhanced promotion with type merging, confidence recalculation, and state history tracking.

        Workflow:
        1. Update last_updated timestamps for longterm chunks from this shortterm memory
        2. Get shortterm chunks
        3. Copy ALL chunks to longterm (no filtering by importance)
        4. Get all shortterm and longterm entities
        5. Process each shortterm entity:
           - If no entity with same name exists in longterm: create new entity
           - If exists: merge types, recalculate confidence, add state history to metadata
        6. Get all shortterm and longterm relationships
        7. Process each shortterm relationship:
           - If no relationship with same source+target exists: create new relationship
           - If exists: merge types, recalculate confidence, add state history to metadata
        8. Update shortterm memory metadata with promotion history

        Args:
            external_id: Agent identifier
            shortterm_memory_id: Shortterm memory ID to promote

        Returns:
            List of created LongtermMemoryChunk objects
        """
        self._ensure_initialized()

        logger.info(
            f"Promoting shortterm memory {shortterm_memory_id} to longterm for {external_id}"
        )

        # Track counts for promotion history
        chunks_added = 0
        entities_added = 0
        entities_modified = 0
        relationships_added = 0
        relationships_modified = 0

        try:
            # STEP 1: Update timestamps for existing chunks from this shortterm memory
            updated_count = await self.longterm_repo.update_chunk_timestamps(shortterm_memory_id)
            logger.info(f"Updated timestamps for {updated_count} existing chunks")

            # STEP 2: Get shortterm chunks
            shortterm_chunks = await self.shortterm_repo.get_chunks_by_memory_id(
                shortterm_memory_id
            )

            if not shortterm_chunks:
                logger.warning(f"No chunks found in shortterm memory {shortterm_memory_id}")
            else:

                # STEP 3: Copy ALL chunks to longterm (no filtering)
                longterm_chunks = []
                for chunk in shortterm_chunks:
                    try:
                        # Get embedding from chunk
                        embedding = await self.embedding_service.get_embedding(chunk.content)

                        # Calculate scores
                        importance_score = chunk.metadata.get("importance_score", 0.75)

                        # Create longterm chunk with temporal tracking
                        longterm_chunk = await self.longterm_repo.create_chunk(
                            external_id=external_id,
                            shortterm_memory_id=shortterm_memory_id,
                            content=chunk.content,
                            embedding=embedding,
                            importance=importance_score,
                            start_date=datetime.now(timezone.utc),
                            metadata={
                                **chunk.metadata,
                                "promoted_from_shortterm": shortterm_memory_id,
                                "promoted_at": datetime.now(timezone.utc).isoformat(),
                            },
                        )

                        longterm_chunks.append(longterm_chunk)
                        chunks_added += 1

                    except Exception as e:
                        logger.error(f"Failed to promote chunk {chunk.id}: {e}", exc_info=True)
                        continue

                logger.info(
                    f"Successfully promoted {len(longterm_chunks)} chunks "
                    f"from shortterm memory {shortterm_memory_id} to longterm"
                )

            # STEP 4: Get all shortterm and longterm entities
            shortterm_entities = await self.shortterm_repo.get_entities_by_memory(
                shortterm_memory_id
            )

            if not shortterm_entities:
                logger.info("No entities to promote")
            else:
                logger.info(f"Processing {len(shortterm_entities)} entities for promotion...")

                # Get existing longterm entities for comparison
                longterm_entities = await self.longterm_repo.get_entities_by_external_id(
                    external_id
                )

                # Create lookup dictionary by name (case-insensitive)
                longterm_entity_map = {entity.name.lower(): entity for entity in longterm_entities}

                # STEP 5: Process each shortterm entity
                entity_id_map = {}  # Map shortterm entity ID to longterm entity ID

                for st_entity in shortterm_entities:
                    lt_match = longterm_entity_map.get(st_entity.name.lower())

                    if lt_match:
                        # Entity exists - merge types and update
                        # Merge types (union of both lists, preserving order)
                        merged_types = list(lt_match.types)  # Start with longterm types
                        for st_type in st_entity.types:
                            if st_type not in merged_types:
                                merged_types.append(st_type)

                        # Recalculate confidence (weighted average, favoring more confident value)
                        # Give 60% weight to higher confidence, 40% to lower
                        max_conf = max(lt_match.importance, st_entity.importance)
                        min_conf = min(lt_match.importance, st_entity.importance)
                        new_confidence = 0.6 * max_conf + 0.4 * min_conf

                        # Calculate importance
                        importance = self._calculate_importance(st_entity)

                        # Add state history entry to metadata
                        now = datetime.now(timezone.utc)
                        state_history_entry = {
                            "timestamp": now.isoformat(),
                            "source": "shortterm_promotion",
                            "shortterm_memory_id": shortterm_memory_id,
                            "old_types": lt_match.types,
                            "new_types": merged_types,
                            "old_importance": lt_match.importance,
                            "new_confidence": new_confidence,
                        }

                        # Get existing state_history array or create new
                        existing_metadata = lt_match.metadata or {}
                        state_history = existing_metadata.get("state_history", [])
                        state_history.append(state_history_entry)

                        # Update entity with merged values
                        logger.debug(
                            f"Updating longterm entity: {st_entity.name} "
                            f"(types {lt_match.types} -> {merged_types}, "
                            f"importance {lt_match.importance:.2f} -> {new_confidence:.2f})"
                        )

                        # Prepare metadata update
                        updated_metadata = {
                            **existing_metadata,
                            "state_history": state_history,
                            "last_promoted_from_shortterm": shortterm_memory_id,
                            "last_promotion_date": now.isoformat(),
                        }

                        # Update entity using existing update method
                        updated_entity = await self.longterm_repo.update_entity(
                            entity_id=lt_match.id,
                            types=merged_types,
                            confidence=new_confidence,
                            importance=importance,
                            last_access=now,
                            metadata=updated_metadata,
                        )

                        if updated_entity:
                            entity_id_map[st_entity.id] = updated_entity.id
                            entities_modified += 1
                            # Update lookup map
                            longterm_entity_map[st_entity.name.lower()] = updated_entity
                        else:
                            logger.error(f"Failed to update entity {lt_match.id}")

                    else:
                        # Entity doesn't exist - create new
                        importance = self._calculate_importance(st_entity)
                        now = datetime.now(timezone.utc)

                        logger.debug(f"Creating new longterm entity: {st_entity.name}")

                        # Initialize state_history with creation entry
                        initial_state_history = [
                            {
                                "timestamp": now.isoformat(),
                                "source": "shortterm_promotion",
                                "shortterm_memory_id": shortterm_memory_id,
                                "types": st_entity.types,
                                "confidence": st_entity.importance,
                                "action": "created",
                            }
                        ]

                        created_entity = await self.longterm_repo.create_entity(
                            external_id=external_id,
                            name=st_entity.name,
                            types=st_entity.types,
                            description=st_entity.description or "",
                            importance=importance,
                            metadata={
                                **st_entity.metadata,
                                "promoted_from_shortterm": shortterm_memory_id,
                                "promoted_at": now.isoformat(),
                                "state_history": initial_state_history,
                            },
                        )

                        entity_id_map[st_entity.id] = created_entity.id
                        entities_added += 1
                        # Add to lookup map
                        longterm_entity_map[st_entity.name.lower()] = created_entity

                logger.info(
                    f"Entity promotion complete: {entities_added} created, "
                    f"{entities_modified} updated"
                )

            # STEP 6: Get all shortterm and longterm relationships
            shortterm_relationships = await self.shortterm_repo.get_relationships_by_memory(
                shortterm_memory_id
            )

            if not shortterm_relationships:
                logger.info("No relationships to promote")
            else:
                logger.info(
                    f"Processing {len(shortterm_relationships)} relationships for promotion..."
                )

                # Get existing longterm relationships
                longterm_relationships = await self.longterm_repo.get_relationships_by_external_id(
                    external_id
                )

                # Create lookup dictionary by (from_name, to_name) tuple (case-insensitive)
                longterm_rel_map = {}
                for rel in longterm_relationships:
                    key = (
                        rel.from_entity_name.lower() if rel.from_entity_name else "",
                        rel.to_entity_name.lower() if rel.to_entity_name else "",
                    )
                    longterm_rel_map[key] = rel

                # STEP 7: Process each shortterm relationship
                for st_rel in shortterm_relationships:
                    try:
                        # Get longterm entity IDs from map
                        from_lt_id = entity_id_map.get(st_rel.from_entity_id)
                        to_lt_id = entity_id_map.get(st_rel.to_entity_id)

                        if not from_lt_id or not to_lt_id:
                            logger.warning(
                                f"Skipping relationship: entities not found in entity_id_map "
                                f"(from: {st_rel.from_entity_id}, to: {st_rel.to_entity_id})"
                            )
                            continue

                        # Get entity names for lookup
                        from_name = st_rel.from_entity_name or ""
                        to_name = st_rel.to_entity_name or ""
                        rel_key = (from_name.lower(), to_name.lower())

                        lt_match = longterm_rel_map.get(rel_key)

                        if lt_match:
                            # Relationship exists - merge types and update
                            # Merge types (union of both lists, preserving order)
                            merged_types = list(lt_match.types)  # Start with longterm types
                            for st_type in st_rel.types:
                                if st_type not in merged_types:
                                    merged_types.append(st_type)

                            # Recalculate confidence (weighted average, favoring more confident value)
                            max_conf = max(lt_match.importance, st_rel.importance)
                            min_conf = min(lt_match.importance, st_rel.importance)
                            new_confidence = 0.6 * max_conf + 0.4 * min_conf

                            # Add state history entry to metadata
                            now = datetime.now(timezone.utc)
                            state_history_entry = {
                                "timestamp": now.isoformat(),
                                "source": "shortterm_promotion",
                                "shortterm_memory_id": shortterm_memory_id,
                                "old_types": lt_match.types,
                                "new_types": merged_types,
                                "old_importance": lt_match.importance,
                                "new_confidence": new_confidence,
                            }

                            # Get existing state_history array or create new
                            existing_metadata = lt_match.metadata or {}
                            state_history = existing_metadata.get("state_history", [])
                            state_history.append(state_history_entry)

                            # Update relationship
                            logger.debug(
                                f"Updating longterm relationship: {from_name} -> {to_name} "
                                f"(types {lt_match.types} -> {merged_types}, "
                                f"importance {lt_match.importance:.2f} -> {new_confidence:.2f}, "
                            )

                            # Prepare metadata update
                            updated_metadata = {
                                **existing_metadata,
                                "state_history": state_history,
                                "last_promoted_from_shortterm": shortterm_memory_id,
                                "last_promotion_date": now.isoformat(),
                            }

                            # Update relationship
                            updated_rel = await self.longterm_repo.update_relationship(
                                relationship_id=lt_match.id,
                                types=merged_types,
                                confidence=new_confidence,
                                metadata=updated_metadata,
                            )

                            if updated_rel:
                                relationships_modified += 1
                                # Update lookup map
                                longterm_rel_map[rel_key] = updated_rel
                            else:
                                logger.error(f"Failed to update relationship {lt_match.id}")

                        else:
                            # Relationship doesn't exist - create new
                            now = datetime.now(timezone.utc)

                            logger.debug(
                                f"Creating new longterm relationship: {from_name} -> {to_name}"
                            )

                            # Initialize state_history with creation entry
                            initial_state_history = [
                                {
                                    "timestamp": now.isoformat(),
                                    "source": "shortterm_promotion",
                                    "shortterm_memory_id": shortterm_memory_id,
                                    "types": st_rel.types,
                                    "confidence": st_rel.importance,
                                    "action": "created",
                                }
                            ]

                            # Calculate importance based on entity and relationship importance
                            importance = st_rel.importance * 0.8 + st_rel.importance * 0.2

                            created_rel = await self.longterm_repo.create_relationship(
                                external_id=external_id,
                                from_entity_id=from_lt_id,
                                to_entity_id=to_lt_id,
                                types=st_rel.types,
                                description=st_rel.description or "",
                                importance=importance,
                                metadata={
                                    **st_rel.metadata,
                                    "promoted_from_shortterm": shortterm_memory_id,
                                    "promoted_at": now.isoformat(),
                                    "state_history": initial_state_history,
                                },
                            )

                            relationships_added += 1
                            # Add to lookup map
                            longterm_rel_map[rel_key] = created_rel

                    except Exception as e:
                        logger.error(
                            f"Failed to promote relationship from {st_rel.from_entity_name} "
                            f"to {st_rel.to_entity_name}: {e}",
                            exc_info=True,
                        )

                logger.info(
                    f"Relationship promotion complete: {relationships_added} created, "
                    f"{relationships_modified} updated"
                )

            # STEP 8: Update shortterm memory metadata with promotion history
            try:
                # Get current shortterm memory
                st_memory = await self.shortterm_repo.get_memory_by_id(shortterm_memory_id)
                if st_memory:
                    # Get existing promotion_history or create new
                    existing_metadata = st_memory.metadata or {}
                    promotion_history = existing_metadata.get("promotion_history", [])

                    # Add new promotion entry
                    promotion_entry = {
                        "date": datetime.now(timezone.utc).isoformat(),
                        "chunks_added": chunks_added,
                        "entities_added": entities_added,
                        "entities_modified": entities_modified,
                        "relationships_added": relationships_added,
                        "relationships_modified": relationships_modified,
                    }
                    promotion_history.append(promotion_entry)

                    # Update shortterm memory metadata
                    updated_metadata = {
                        **existing_metadata,
                        "promotion_history": promotion_history,
                        "last_promotion_date": promotion_entry["date"],
                    }

                    await self.shortterm_repo.update_memory(
                        memory_id=shortterm_memory_id,
                        metadata=updated_metadata,
                    )

                    logger.info(
                        f"Updated shortterm memory {shortterm_memory_id} with promotion history"
                    )

            except Exception as e:
                logger.error(f"Failed to update shortterm memory metadata: {e}", exc_info=True)

            logger.info(
                f"Promotion complete - Chunks: {chunks_added} added, "
                f"Entities: {entities_added} added/{entities_modified} modified, "
                f"Relationships: {relationships_added} added/{relationships_modified} modified"
            )

            return longterm_chunks if "longterm_chunks" in locals() else []

        except Exception as e:
            logger.error(f"Promotion failed: {e}", exc_info=True)
            return []

    # =========================================================================
    # HELPER FUNCTIONS FOR ENTITY/RELATIONSHIP PROCESSING
    # =========================================================================

    def _calculate_importance(self, entity) -> float:
        """
        Calculate importance score for entity promotion.

        Factors considered:
        - Entity confidence
        - Entity type (some types are more important)

        Args:
            entity: Entity to calculate importance for

        Returns:
            Importance score (0.0-1.0)
        """
        # Start with entity confidence
        base_score = getattr(entity, "confidence", 0.5)

        # Get entity type
        entity_type = getattr(entity, "type", None)
        if hasattr(entity_type, "value"):
            entity_type = entity_type.value

        # Adjust based on entity type
        type_multipliers = {
            "PERSON": 1.2,
            "ORGANIZATION": 1.2,
            "TECHNOLOGY": 1.15,
            "CONCEPT": 1.1,
            "PROJECT": 1.1,
            "FRAMEWORK": 1.1,
            "LIBRARY": 1.05,
            "TOOL": 1.05,
            "DATABASE": 1.05,
        }

        multiplier = type_multipliers.get(entity_type, 1.0)
        importance = base_score * multiplier

        # Cap at 1.0
        return min(importance, 1.0)
