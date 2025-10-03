"""
Memory Manager - Core orchestration for memory operations.

Handles memory lifecycle: Active → Shortterm → Longterm
Coordinates consolidation, promotion, and retrieval across memory tiers.
"""

import logging
from typing import List, Optional, Dict, Any
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
    ShorttermMemory,
    ShorttermMemoryChunk,
    LongtermMemoryChunk,
)
from agent_mem.services.embedding import EmbeddingService
from agent_mem.utils.helpers import chunk_text
from agent_mem.agents import (
    MemoryRetrieveAgent,
    extract_entities_and_relationships,
    consolidate_memory,
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

    async def update_active_memory_section(
        self,
        external_id: str,
        memory_id: int,
        section_id: str,
        new_content: str,
    ) -> ActiveMemory:
        """
        Update a specific section in an active memory.

        Args:
            external_id: Agent identifier
            memory_id: Memory ID
            section_id: Section ID to update
            new_content: New content for the section

        Returns:
            Updated ActiveMemory object

        Raises:
            ValueError: If memory not found or section invalid
        """
        self._ensure_initialized()

        logger.info(f"Updating section '{section_id}' in memory {memory_id} for {external_id}")

        memory = await self.active_repo.update_section(
            memory_id=memory_id,
            section_id=section_id,
            new_content=new_content,
        )

        if not memory:
            raise ValueError(f"Active memory {memory_id} not found")

        logger.info(f"Updated section '{section_id}' in memory {memory_id}")

        # Check consolidation threshold and trigger consolidation if needed
        section = memory.sections.get(section_id)
        if section and section["update_count"] >= self.config.consolidation_threshold:
            logger.info(
                f"Section '{section_id}' reached consolidation threshold "
                f"({section['update_count']} >= {self.config.consolidation_threshold}). "
                f"Triggering consolidation..."
            )
            try:
                await self._consolidate_to_shortterm(external_id, memory.id)
            except Exception as e:
                logger.error(f"Consolidation failed: {e}", exc_info=True)
                # Don't fail the update if consolidation fails

        return memory

    async def retrieve_memories(
        self,
        external_id: str,
        query: str,
        search_shortterm: bool = True,
        search_longterm: bool = True,
        limit: int = 10,
    ) -> RetrievalResult:
        """
        Retrieve memories across all tiers using MemoryRetrieveAgent.

        Workflow:
        1. Use MemoryRetrieveAgent to determine optimal search strategy
        2. Get active memories
        3. Search shortterm/longterm based on strategy
        4. Use MemoryRetrieveAgent to synthesize results

        Args:
            external_id: Agent identifier
            query: Search query
            search_shortterm: Search shortterm memory (can be overridden by agent)
            search_longterm: Search longterm memory (can be overridden by agent)
            limit: Maximum results per tier (can be overridden by agent)

        Returns:
            RetrievalResult object with aggregated results and AI synthesis
        """
        self._ensure_initialized()

        logger.info(f"Retrieving memories for {external_id}, query: {query[:50]}...")

        try:
            # 1. Use agent to determine search strategy
            logger.info("Using MemoryRetrieveAgent to determine search strategy...")
            active_memories = await self.get_active_memories(external_id)

            search_strategy = await self.retriever_agent.determine_strategy(
                external_id=external_id,
                query=query,
                active_memories=active_memories,
            )

            logger.info(
                f"Search strategy: active={search_strategy.search_active}, "
                f"shortterm={search_strategy.search_shortterm}, "
                f"longterm={search_strategy.search_longterm}, "
                f"weights=({search_strategy.vector_weight:.2f}/{search_strategy.bm25_weight:.2f})"
            )

            # 2. Initialize result containers
            shortterm_chunks = []
            longterm_chunks = []
            entities = []
            relationships = []

            # Generate query embedding for vector search
            query_embedding = await self.embedding_service.get_embedding(query)

            # 3. Search shortterm memory if strategy enables it
            if search_strategy.search_shortterm and search_shortterm and self.shortterm_repo:
                try:
                    shortterm_chunks = await self.shortterm_repo.hybrid_search(
                        external_id=external_id,
                        query_text=query,
                        query_embedding=query_embedding,
                        vector_weight=search_strategy.vector_weight,
                        bm25_weight=search_strategy.bm25_weight,
                        limit=search_strategy.limit_per_tier,
                    )
                    logger.info(f"Found {len(shortterm_chunks)} shortterm chunks")
                except Exception as e:
                    logger.error(f"Shortterm search failed: {e}", exc_info=True)

            # 4. Search longterm memory if strategy enables it
            if search_strategy.search_longterm and search_longterm and self.longterm_repo:
                try:
                    longterm_chunks = await self.longterm_repo.hybrid_search(
                        external_id=external_id,
                        query_text=query,
                        query_embedding=query_embedding,
                        vector_weight=search_strategy.vector_weight,
                        bm25_weight=search_strategy.bm25_weight,
                        only_valid=True,  # Only get currently valid chunks
                        limit=search_strategy.limit_per_tier,
                    )
                    logger.info(f"Found {len(longterm_chunks)} longterm chunks")
                except Exception as e:
                    logger.error(f"Longterm search failed: {e}", exc_info=True)

            # 5. Use agent to synthesize results
            logger.info("Using MemoryRetrieveAgent to synthesize results...")
            synthesis = await self.retriever_agent.synthesize_results(
                external_id=external_id,
                query=query,
                active_memories=active_memories if search_strategy.search_active else [],
                shortterm_chunks=shortterm_chunks,
                longterm_chunks=longterm_chunks,
            )

            # Format synthesized response
            synthesized_response = f"{synthesis.summary}\n\n" f"Key Points:\n" + "\n".join(
                f"- {point}" for point in synthesis.key_points
            )
            if synthesis.gaps:
                synthesized_response += f"\n\nGaps: {', '.join(synthesis.gaps)}"

            # Note: Entity/relationship retrieval from Neo4j is not currently implemented
            # in the retrieve_memories flow. Entities are extracted and stored during
            # consolidation (see consolidate_to_shortterm), but direct entity-based
            # search is not yet available. To enable entity search:
            # 1. Add entity name/type filters to search parameters
            # 2. Query Neo4j for matching entities
            # 3. Use entity relationships to expand search context
            # For now, entities are populated during consolidation only.
            result = RetrievalResult(
                query=query,
                active_memories=active_memories if search_strategy.search_active else [],
                shortterm_chunks=shortterm_chunks,
                longterm_chunks=longterm_chunks,
                entities=entities,  # Currently empty - not retrieved in this flow
                relationships=relationships,  # Currently empty - not retrieved in this flow
                synthesized_response=synthesized_response,
            )

            logger.info(
                f"Memory retrieval complete (confidence: {synthesis.confidence:.2f}): "
                f"{len(result.active_memories)} active, "
                f"{len(shortterm_chunks)} shortterm, {len(longterm_chunks)} longterm"
            )
            return result

        except Exception as e:
            logger.error(f"Agent-based retrieval failed: {e}", exc_info=True)
            logger.warning("Falling back to basic retrieval...")

            # Fallback to basic retrieval
            return await self._retrieve_memories_basic(
                external_id, query, search_shortterm, search_longterm, limit
            )

    async def _retrieve_memories_basic(
        self,
        external_id: str,
        query: str,
        search_shortterm: bool = True,
        search_longterm: bool = True,
        limit: int = 10,
    ) -> RetrievalResult:
        """
        Fallback: Basic retrieval without AI agent.

        Used when MemoryRetrieveAgent fails or is unavailable.
        """
        self._ensure_initialized()

        logger.info(f"Basic retrieval (no agent) for {external_id}, query: {query[:50]}...")

        # Get active memories
        active_memories = await self.get_active_memories(external_id)

        # Initialize result containers
        shortterm_chunks = []
        longterm_chunks = []
        entities = []
        relationships = []

        # Generate query embedding for vector search
        query_embedding = await self.embedding_service.get_embedding(query)

        # Search shortterm memory if enabled
        if search_shortterm and self.shortterm_repo:
            try:
                shortterm_chunks = await self.shortterm_repo.hybrid_search(
                    external_id=external_id,
                    query_text=query,
                    query_embedding=query_embedding,
                    vector_weight=self.config.vector_weight,
                    bm25_weight=self.config.bm25_weight,
                    limit=limit,
                )
                logger.info(f"Found {len(shortterm_chunks)} shortterm chunks")
            except Exception as e:
                logger.error(f"Shortterm search failed: {e}", exc_info=True)

        # Search longterm memory if enabled
        if search_longterm and self.longterm_repo:
            try:
                longterm_chunks = await self.longterm_repo.hybrid_search(
                    external_id=external_id,
                    query_text=query,
                    query_embedding=query_embedding,
                    vector_weight=self.config.vector_weight,
                    bm25_weight=self.config.bm25_weight,
                    only_valid=True,
                    limit=limit,
                )
                logger.info(f"Found {len(longterm_chunks)} longterm chunks")
            except Exception as e:
                logger.error(f"Longterm search failed: {e}", exc_info=True)

        # Build synthesized response (simple version)
        synthesized_response = self._synthesize_retrieval_response(
            active_memories=active_memories,
            shortterm_chunks=shortterm_chunks,
            longterm_chunks=longterm_chunks,
            query=query,
        )

        result = RetrievalResult(
            query=query,
            active_memories=active_memories,
            shortterm_chunks=shortterm_chunks,
            longterm_chunks=longterm_chunks,
            entities=entities,
            relationships=relationships,
            synthesized_response=synthesized_response,
        )

        logger.info(
            f"Basic memory retrieval complete: {len(active_memories)} active, "
            f"{len(shortterm_chunks)} shortterm, {len(longterm_chunks)} longterm"
        )
        return result

    def _synthesize_retrieval_response(
        self,
        active_memories: List[ActiveMemory],
        shortterm_chunks: List[ShorttermMemoryChunk],
        longterm_chunks: List[LongtermMemoryChunk],
        query: str,
    ) -> str:
        """
        Synthesize a human-readable response from retrieved memories.

        Args:
            active_memories: Active memories
            shortterm_chunks: Shortterm chunks
            longterm_chunks: Longterm chunks
            query: Original query

        Returns:
            Synthesized response string
        """
        parts = []

        if active_memories:
            parts.append(f"Found {len(active_memories)} active working memories.")

        if shortterm_chunks:
            top_chunk = shortterm_chunks[0]
            score = getattr(top_chunk, "similarity_score", 0.0) or 0.0
            parts.append(
                f"Found {len(shortterm_chunks)} recent memory chunks "
                f"(top relevance: {score:.2f})."
            )

        if longterm_chunks:
            top_chunk = longterm_chunks[0]
            score = getattr(top_chunk, "similarity_score", 0.0) or 0.0
            parts.append(
                f"Found {len(longterm_chunks)} consolidated knowledge chunks "
                f"(top relevance: {score:.2f})."
            )

        if not parts:
            return f"No memories found for query: {query}"

        return " ".join(parts)

    def _ensure_initialized(self) -> None:
        """Ensure manager is initialized."""
        if not self._initialized:
            raise RuntimeError("MemoryManager not initialized. Call initialize() first.")

    # =========================================================================
    # CONSOLIDATION WORKFLOWS
    # =========================================================================

    async def _consolidate_to_shortterm(
        self, external_id: str, active_memory_id: int
    ) -> Optional[ShorttermMemory]:
        """
        Consolidate active memory to shortterm memory with entity/relationship extraction.

        Workflow:
        1. Get active memory
        2. Find or create matching shortterm memory
        3. Extract content from all sections
        4. Chunk the content and store with embeddings
        5. Extract entities and relationships using ER Extractor Agent
        6. Compare and merge/add entities using auto-resolution
        7. Create relationships in Neo4j

        Args:
            external_id: Agent identifier
            active_memory_id: Active memory ID to consolidate

        Returns:
            Created/updated ShorttermMemory or None if failed
        """
        self._ensure_initialized()

        logger.info(
            f"Consolidating active memory {active_memory_id} to shortterm for {external_id}"
        )

        try:
            # 1. Get active memory
            active_memory = await self.active_repo.get_by_id(active_memory_id)
            if not active_memory:
                logger.error(f"Active memory {active_memory_id} not found")
                return None

            # 2. Find or create shortterm memory
            shortterm_memory = await self._find_or_create_shortterm_memory(
                external_id=external_id,
                title=active_memory.title,
                metadata=active_memory.metadata,
            )

            # 3. Extract content from all sections
            all_content = self._extract_content_from_sections(active_memory)
            if not all_content:
                logger.warning(f"No content to consolidate from active memory {active_memory_id}")
                return shortterm_memory

            # 4. Chunk the content
            chunks = chunk_text(
                text=all_content,
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
            )
            logger.info(f"Created {len(chunks)} chunks from active memory")

            # Store chunks with embeddings
            for i, chunk_content in enumerate(chunks):
                try:
                    embedding = await self.embedding_service.get_embedding(chunk_content)
                    await self.shortterm_repo.create_chunk(
                        shortterm_memory_id=shortterm_memory.id,
                        external_id=external_id,
                        content=chunk_content,
                        chunk_order=i,
                        embedding=embedding,
                        metadata={
                            "source": "active_memory",
                            "active_memory_id": active_memory_id,
                            "consolidated_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                except Exception as e:
                    logger.error(f"Failed to create chunk {i}: {e}", exc_info=True)
                    continue

            # 5. Extract entities and relationships using ER Extractor Agent
            logger.info("Extracting entities and relationships using ER Extractor Agent...")
            try:
                extraction_result = await extract_entities_and_relationships(all_content)
                logger.info(
                    f"Extracted {len(extraction_result.entities)} entities and "
                    f"{len(extraction_result.relationships)} relationships"
                )
            except Exception as e:
                logger.error(f"ER extraction failed: {e}", exc_info=True)
                # Continue without entities/relationships
                return shortterm_memory

            # 6. Get existing shortterm entities for comparison
            existing_entities = await self.shortterm_repo.get_entities_by_memory_id(
                shortterm_memory.id
            )

            # 7. Process and store entities with auto-resolution
            entity_map = {}  # Map entity names to IDs for relationship creation
            for extracted_entity in extraction_result.entities:
                # Check if entity already exists
                existing_match = None
                for existing in existing_entities:
                    if existing.name.lower() == extracted_entity.name.lower():
                        existing_match = existing
                        break

                if existing_match:
                    # Calculate similarity and overlap for auto-resolution
                    similarity = await self._calculate_semantic_similarity(
                        extracted_entity.name, existing_match.name
                    )
                    overlap = self._calculate_entity_overlap(extracted_entity, existing_match)

                    # Auto-resolution: Merge if similarity >= 0.85 AND overlap >= 0.7
                    if similarity >= 0.85 and overlap >= 0.7:
                        logger.debug(
                            f"Auto-merging entity: {extracted_entity.name} "
                            f"(similarity={similarity:.2f}, overlap={overlap:.2f})"
                        )
                        # Update existing entity
                        updated_entity = await self.shortterm_repo.update_entity(
                            entity_id=existing_match.id,
                            confidence=max(existing_match.confidence, extracted_entity.confidence),
                            metadata={
                                **existing_match.metadata,
                                "last_updated": datetime.now(timezone.utc).isoformat(),
                                "merged_count": existing_match.metadata.get("merged_count", 0) + 1,
                            },
                        )
                        entity_map[extracted_entity.name] = updated_entity.id
                    else:
                        # Conflict detected: Create new entity (manual merge required)
                        logger.debug(
                            f"Conflict detected for entity: {extracted_entity.name} "
                            f"(similarity={similarity:.2f}, overlap={overlap:.2f}). Creating new."
                        )
                        created_entity = await self.shortterm_repo.create_entity(
                            external_id=external_id,
                            shortterm_memory_id=shortterm_memory.id,
                            name=extracted_entity.name,
                            entity_type=extracted_entity.type.value,
                            description=f"Extracted from active memory {active_memory_id}",
                            confidence=extracted_entity.confidence,
                            metadata={
                                "extracted_at": datetime.now(timezone.utc).isoformat(),
                                "source": "er_extractor_agent",
                                "conflict_with": existing_match.id,
                            },
                        )
                        entity_map[extracted_entity.name] = created_entity.id
                else:
                    # No existing entity: Create new
                    logger.debug(f"Creating new entity: {extracted_entity.name}")
                    created_entity = await self.shortterm_repo.create_entity(
                        external_id=external_id,
                        shortterm_memory_id=shortterm_memory.id,
                        name=extracted_entity.name,
                        entity_type=extracted_entity.type.value,
                        description=f"Extracted from active memory {active_memory_id}",
                        confidence=extracted_entity.confidence,
                        metadata={
                            "extracted_at": datetime.now(timezone.utc).isoformat(),
                            "source": "er_extractor_agent",
                        },
                    )
                    entity_map[extracted_entity.name] = created_entity.id

            # 8. Create relationships
            relationships_created = 0
            for extracted_rel in extraction_result.relationships:
                try:
                    from_id = entity_map.get(extracted_rel.source)
                    to_id = entity_map.get(extracted_rel.target)

                    if not from_id or not to_id:
                        logger.warning(
                            f"Skipping relationship {extracted_rel.source} -> {extracted_rel.target}: "
                            f"entities not found in entity_map"
                        )
                        continue

                    await self.shortterm_repo.create_relationship(
                        external_id=external_id,
                        shortterm_memory_id=shortterm_memory.id,
                        from_entity_id=from_id,
                        to_entity_id=to_id,
                        relationship_type=extracted_rel.type.value,
                        description=f"Extracted from active memory {active_memory_id}",
                        confidence=extracted_rel.confidence,
                        strength=0.5,  # Default strength, can be enhanced later
                        metadata={
                            "extracted_at": datetime.now(timezone.utc).isoformat(),
                            "source": "er_extractor_agent",
                        },
                    )
                    relationships_created += 1
                except Exception as e:
                    logger.error(
                        f"Failed to create relationship "
                        f"{extracted_rel.source} -> {extracted_rel.target}: {e}"
                    )

            logger.info(
                f"Successfully consolidated active memory {active_memory_id} "
                f"to shortterm memory {shortterm_memory.id}: "
                f"{len(chunks)} chunks, {len(entity_map)} entities, "
                f"{relationships_created} relationships"
            )

            return shortterm_memory

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
    # HELPER FUNCTIONS FOR ENTITY/RELATIONSHIP PROCESSING
    # =========================================================================

    async def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts using embeddings.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Get embeddings
            emb1 = await self.embedding_service.get_embedding(text1)
            emb2 = await self.embedding_service.get_embedding(text2)

            # Calculate cosine similarity
            import numpy as np

            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            logger.error(f"Failed to calculate semantic similarity: {e}")
            # Fallback to string comparison
            return 1.0 if text1.lower() == text2.lower() else 0.0

    def _calculate_entity_overlap(self, entity1, entity2) -> float:
        """
        Calculate overlap between two entities based on name and type.

        Args:
            entity1: First entity (extracted or existing)
            entity2: Second entity (extracted or existing)

        Returns:
            Overlap score (0.0-1.0)
        """
        # Extract names and types
        name1 = getattr(entity1, "name", "").lower()
        name2 = getattr(entity2, "name", "").lower()

        # Handle entity type (might be enum or string)
        type1 = getattr(entity1, "type", None)
        type2 = getattr(entity2, "type", None)

        # Convert enum to string if needed
        if hasattr(type1, "value"):
            type1 = type1.value
        if hasattr(type2, "value"):
            type2 = type2.value

        # Check name similarity
        name_match = name1 == name2

        # Check type match
        type_match = type1 == type2

        # Return overlap score
        if name_match and type_match:
            return 1.0
        elif name_match or type_match:
            return 0.5
        else:
            return 0.0

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

    # =========================================================================
    # PROMOTION WORKFLOWS
    # =========================================================================

    async def _promote_to_longterm(
        self, external_id: str, shortterm_memory_id: int
    ) -> List[LongtermMemoryChunk]:
        """
        Promote shortterm memory to longterm memory with entity/relationship handling.

        Workflow:
        1. Get shortterm memory chunks and filter by importance
        2. Copy chunks to longterm with temporal tracking
        3. Get shortterm entities and compare with longterm entities
        4. Update existing longterm entities or create new ones
        5. Promote relationships with temporal tracking

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

        try:
            # 1. Get shortterm chunks
            shortterm_chunks = await self.shortterm_repo.get_chunks_by_memory_id(
                shortterm_memory_id
            )

            if not shortterm_chunks:
                logger.warning(f"No chunks found in shortterm memory {shortterm_memory_id}")
                return []

            longterm_chunks = []

            # 2. Filter and copy chunks to longterm with temporal tracking
            for chunk in shortterm_chunks:
                try:
                    # Calculate importance score
                    importance_score = chunk.metadata.get("importance_score", 0.75)
                    confidence_score = 0.85

                    # Skip if below threshold
                    if importance_score < self.config.shortterm_promotion_threshold:
                        logger.debug(
                            f"Skipping chunk {chunk.id} - "
                            f"importance {importance_score} below threshold"
                        )
                        continue

                    # Get embedding from chunk
                    embedding = await self.embedding_service.get_embedding(chunk.content)

                    # Create longterm chunk with temporal tracking
                    longterm_chunk = await self.longterm_repo.create_chunk(
                        external_id=external_id,
                        shortterm_memory_id=shortterm_memory_id,
                        content=chunk.content,
                        chunk_order=chunk.chunk_order,
                        embedding=embedding,
                        confidence_score=confidence_score,
                        importance_score=importance_score,
                        start_date=datetime.now(timezone.utc),
                        end_date=None,  # Currently valid
                        metadata={
                            **chunk.metadata,
                            "promoted_from_shortterm": shortterm_memory_id,
                            "promoted_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )

                    longterm_chunks.append(longterm_chunk)

                except Exception as e:
                    logger.error(f"Failed to promote chunk {chunk.id}: {e}", exc_info=True)
                    continue

            logger.info(
                f"Successfully promoted {len(longterm_chunks)} chunks "
                f"from shortterm memory {shortterm_memory_id} to longterm"
            )

            # 3. Get shortterm entities for promotion
            shortterm_entities = await self.shortterm_repo.get_entities_by_memory_id(
                shortterm_memory_id
            )

            if shortterm_entities:
                logger.info(f"Promoting {len(shortterm_entities)} entities to longterm...")

                # Get existing longterm entities for comparison
                longterm_entities = await self.longterm_repo.get_entities_by_external_id(
                    external_id
                )

                # 4. Process entities with confidence update
                entities_created = 0
                entities_updated = 0
                entity_id_map = {}  # Map shortterm entity ID to longterm entity ID

                for st_entity in shortterm_entities:
                    # Find matching longterm entity
                    lt_match = None
                    for lt_entity in longterm_entities:
                        if (
                            lt_entity.name.lower() == st_entity.name.lower()
                            and lt_entity.type == st_entity.type
                        ):
                            lt_match = lt_entity
                            break

                    if lt_match:
                        # Update existing entity confidence using weighted formula
                        weight = 0.7
                        new_confidence = (
                            weight * lt_match.confidence + (1 - weight) * st_entity.confidence
                        )

                        logger.debug(
                            f"Updating longterm entity: {st_entity.name} "
                            f"(confidence {lt_match.confidence:.2f} -> {new_confidence:.2f})"
                        )

                        updated_entity = await self.longterm_repo.update_entity(
                            entity_id=lt_match.id,
                            confidence=new_confidence,
                            last_seen=datetime.now(timezone.utc),
                            metadata={
                                **lt_match.metadata,
                                "last_updated_from_shortterm": shortterm_memory_id,
                                "last_updated": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                        entity_id_map[st_entity.id] = updated_entity.id
                        entities_updated += 1
                    else:
                        # Create new longterm entity with temporal tracking
                        importance = self._calculate_importance(st_entity)

                        logger.debug(f"Creating new longterm entity: {st_entity.name}")

                        created_entity = await self.longterm_repo.create_entity(
                            external_id=external_id,
                            name=st_entity.name,
                            entity_type=st_entity.type,
                            description=st_entity.description or "",
                            confidence=st_entity.confidence,
                            importance=importance,
                            first_seen=datetime.now(timezone.utc),
                            last_seen=datetime.now(timezone.utc),
                            metadata={
                                **st_entity.metadata,
                                "promoted_from_shortterm": shortterm_memory_id,
                                "promoted_at": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                        entity_id_map[st_entity.id] = created_entity.id
                        entities_created += 1

                logger.info(
                    f"Entity promotion complete: {entities_created} created, "
                    f"{entities_updated} updated"
                )

                # 5. Promote relationships with temporal tracking
                shortterm_relationships = await self.shortterm_repo.get_relationships_by_memory_id(
                    shortterm_memory_id
                )

                if shortterm_relationships:
                    logger.info(
                        f"Promoting {len(shortterm_relationships)} relationships to longterm..."
                    )

                    relationships_created = 0
                    relationships_updated = 0

                    for st_rel in shortterm_relationships:
                        try:
                            # Get longterm entity IDs
                            from_lt_id = entity_id_map.get(st_rel.from_entity_id)
                            to_lt_id = entity_id_map.get(st_rel.to_entity_id)

                            if not from_lt_id or not to_lt_id:
                                logger.warning(
                                    f"Skipping relationship: entities not found in entity_id_map"
                                )
                                continue

                            # Check if relationship already exists in longterm
                            # (This would require a method to check existing relationships)
                            # For now, we'll create new relationships

                            await self.longterm_repo.create_relationship(
                                external_id=external_id,
                                from_entity_id=from_lt_id,
                                to_entity_id=to_lt_id,
                                relationship_type=st_rel.type,
                                description=st_rel.description or "",
                                confidence=st_rel.confidence,
                                strength=st_rel.strength,
                                start_date=datetime.now(timezone.utc),
                                last_updated=datetime.now(timezone.utc),
                                metadata={
                                    **st_rel.metadata,
                                    "promoted_from_shortterm": shortterm_memory_id,
                                    "promoted_at": datetime.now(timezone.utc).isoformat(),
                                },
                            )
                            relationships_created += 1

                        except Exception as e:
                            logger.error(f"Failed to promote relationship: {e}", exc_info=True)

                    logger.info(
                        f"Relationship promotion complete: {relationships_created} created, "
                        f"{relationships_updated} updated"
                    )

            return longterm_chunks

        except Exception as e:
            logger.error(f"Promotion failed: {e}", exc_info=True)
            return []
