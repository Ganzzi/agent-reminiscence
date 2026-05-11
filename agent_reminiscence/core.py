"""
Core AgentMem class - Main interface for memory management.
"""

from typing import List, Optional, Dict, Any, Callable, Union
from uuid import UUID
from datetime import datetime, timezone
import logging

from pydantic_ai.usage import RunUsage

from agent_reminiscence.config import Config, get_config, set_config
from agent_reminiscence.database.models import (
    ActiveMemory,
    RetrievalResultV2,
    ConsolidationConflicts,
)
from agent_reminiscence.services.memory_manager import MemoryManager
from agent_reminiscence.services.llm_model_provider import model_provider, LLMExecutor

logger = logging.getLogger(__name__)


class AgentMem:
    """
    Main interface for hierarchical memory management.

    AgentMem is STATELESS and can serve multiple agents/workers.
    Pass external_id to each method call to specify which agent's memory to access.

    Provides 6 simple methods to manage all memory tiers:
    1. create_active_memory(external_id, ...) - Create new working memory
    2. get_active_memories(external_id) - Get all working memories
    3. update_active_memory_sections(external_id, memory_id, sections) - Update multiple sections
    3. update_active_memory(external_id, memory_id, sections) - Update multiple sections (alias: update_active_memory_sections)
    4. delete_active_memory(external_id, memory_id) - Delete working memory
    5. search_memories(external_id, query, ...) - Fast search (< 200ms, no agent)
    6. deep_search_memories(external_id, query, ...) - Comprehensive search with AI synthesis

    Example:
        ```python
        agent_mem = AgentMem()
        await agent_mem.initialize()

        # Create memory with template
        memory = await agent_mem.create_active_memory(
            external_id="agent-123",
            title="Task Memory",
            template_content=TEMPLATE_YAML,
            initial_sections={"current_task": {"content": "...", "update_count": 0}}
        )

        # Update sections (supports single or multiple sections)
        await agent_mem.update_active_memory_sections(
            external_id="agent-123",
            memory_id=1,
            sections=[
                {"section_id": "progress", "new_content": "Updated progress..."}
            ]
        )

        # Retrieve information
        results = await agent_mem.search_memories(
            external_id="agent-123",
            query="How do I implement authentication?"
        )

        await agent_mem.close()
        ```
    """

    def __init__(
        self,
        config: Optional[Config] = None,
    ):
        """
        Initialize AgentMem (stateless - serves multiple agents).

        Args:
            config: Optional configuration object (uses environment variables by default)
        """
        # Set or get configuration
        if config:
            set_config(config)
        self.config = get_config()

        # Initialize memory manager (lazy initialization)
        self._memory_manager: Optional[MemoryManager] = None
        self._initialized = False

        logger.info("AgentMem initialized (stateless)")

    async def initialize(self) -> None:
        """
        Initialize database connections and ensure schema exists.

        This must be called before using any other methods.
        """
        if self._initialized:
            logger.warning("AgentMem already initialized")
            return

        logger.info("Initializing AgentMem...")

        # Initialize memory manager (stateless)
        self._memory_manager = MemoryManager(
            config=self.config,
        )
        await self._memory_manager.initialize()

        self._initialized = True
        logger.info("AgentMem initialization complete")

    def set_usage_processor(self, processor: Callable[[str, RunUsage], Any]) -> None:
        """
        Register a callback to process LLM token usage from agent operations.

        The processor will be called after agent runs with the external_id and RunUsage.

        Args:
            processor: Callable that accepts (external_id: str, usage: RunUsage)

        Example:
            ```python
            def log_usage(external_id: str, usage):
                print(f"{external_id}: {usage.total_tokens} tokens")

            agent_mem.set_usage_processor(log_usage)
            ```
        """
        self._ensure_initialized()
        if self._memory_manager and hasattr(self._memory_manager, "usage_processor"):
            self._memory_manager.usage_processor = processor
            logger.info("Usage processor registered")
        else:
            logger.warning("Memory manager not available for usage processor registration")

    def set_llm_executor(self, executor: Optional[LLMExecutor]) -> None:
        """
        Register an injected LLM executor for backend queue execution.

        When executor is set, backend mode is enabled and model provider
        initialization without executor is blocked.
        """
        model_provider.set_executor(executor)
        model_provider.set_backend_mode(executor is not None)
        if executor is None:
            logger.info("LLM executor cleared; backend mode disabled")
        else:
            logger.info("LLM executor registered; backend mode enabled")

    async def create_active_memory(
        self,
        external_id: str | UUID | int,
        title: str,
        template_content: str | Dict[str, Any],  # Support both
        initial_sections: Optional[Dict[str, Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ActiveMemory:
        """
                Create a new active memory with template-driven structure.

                Args:
                    external_id: Unique identifier for the agent (UUID, string, or int)
                    title: Memory title
                    template_content: Template defining section structure. Can be:
                        - Dict (JSON): {"template": {...}, "sections": [...]}
                        - Str (YAML): Parsed to dict automatically
                    initial_sections: Optional initial sections that override template defaults
                        {"section_id": {"content": "...", "update_count": 0, ...}}
                    metadata: Optional metadata dictionary

                Returns:
                    Created ActiveMemory object

                Example (JSON):
                    ```python
                    memory = await agent_mem.create_active_memory(
                        external_id="agent-123",
                        title="Task Memory",
                        template_content={
                            "template": {
                                "id": "task_memory_v1",
                                "name": "Task Memory"
                            },
                            "sections": [
                                {
                                    "id": "current_task",
                                    "description": "What is being worked on now"
                                }
                            ]
                        },
                        initial_sections={
                            "current_task": {"content": "# Task\nImplement feature"}
                        }
                    )
                    ```

                Example (YAML - backward compatible):
                    ```python
                    memory = await agent_mem.create_active_memory(
                        external_id="agent-123",
                        title="Task Memory",
                        template_content='''
        template:
          id: "task_memory_v1"
          name: "Task Memory"
        sections:
          - id: "current_task"
            title: "Current Task"
        ''',
                        initial_sections={"current_task": {"content": "..."}}
                    )
                    ```
        """
        self._ensure_initialized()

        # Validate inputs
        if not title or not title.strip():
            raise ValueError("title cannot be empty")
        if not template_content:
            raise ValueError("template_content cannot be empty")

        external_id_str = str(external_id)

        # Parse template if string (YAML)
        if isinstance(template_content, str):
            import yaml

            try:
                template_dict = yaml.safe_load(template_content)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML template: {e}")
        else:
            template_dict = template_content

        logger.info(f"Creating active memory for {external_id_str}: {title}")
        return await self._memory_manager.create_active_memory(
            external_id=external_id_str,
            title=title,
            template_content=template_dict,
            initial_sections=initial_sections or {},
            metadata=metadata or {},
        )

    async def get_active_memories(
        self,
        external_id: str | UUID | int,
    ) -> List[ActiveMemory]:
        """
        Get all active memories for a specific agent.

        Args:
            external_id: Unique identifier for the agent

        Returns:
            List of ActiveMemory objects

        Raises:
            RuntimeError: If not initialized

        Example:
            ```python
            memories = await agent_mem.get_active_memories(external_id="agent-123")
            for memory in memories:
                print(f"{memory.title}")
                for section_id, section_data in memory.sections.items():
                    print(f"  {section_id}: {section_data['update_count']} updates")
            ```
        """
        self._ensure_initialized()
        external_id_str = str(external_id)

        logger.info(f"Retrieving all active memories for {external_id_str}")
        return await self._memory_manager.get_active_memories(external_id=external_id_str)

    async def update_active_memory_sections(
        self,
        external_id: str | UUID | int,
        memory_id: int,
        sections: List[Dict[str, Any]],
    ) -> ActiveMemory:
        """
        Upsert multiple sections in an active memory (batch operation).

        Supports:
        - Creating new sections (automatically added to template)
        - Updating existing sections
        - Content replacement with pattern matching
        - Content insertion/appending

        Args:
            external_id: Unique identifier for the agent
            memory_id: ID of the memory to update
            sections: List of section updates:
                [
                    {
                        "section_id": "progress",
                        "old_content": "# Old header",  # Optional: pattern to find
                        "new_content": "# New content",
                        "action": "replace"  # "replace" or "insert", default "replace"
                    }
                ]

        Returns:
            Updated ActiveMemory object

        Action Behaviors:
            **replace**:
            - If old_content is null/empty: Replaces entire section content
            - If old_content is provided: Replaces that substring with new_content

            **insert**:
            - If old_content is null/empty: Appends new_content at end
            - If old_content is provided: Inserts new_content right after old_content

            **New Section** (section doesn't exist):
            - Creates section with new_content
            - Adds section definition to template_content

        Examples:
            ```python
            # Replace entire section
            await agent_mem.update_active_memory_sections(
                external_id="agent-123",
                memory_id=1,
                sections=[
                    {
                        "section_id": "progress",
                        "new_content": "# Progress\n- All done!",
                        "action": "replace"
                    }
                ]
            )

            # Replace specific part
            await agent_mem.update_active_memory_sections(
                external_id="agent-123",
                memory_id=1,
                sections=[
                    {
                        "section_id": "progress",
                        "old_content": "- Step 1: In progress",
                        "new_content": "- Step 1: Complete",
                        "action": "replace"
                    }
                ]
            )

            # Append new content
            await agent_mem.update_active_memory_sections(
                external_id="agent-123",
                memory_id=1,
                sections=[
                    {
                        "section_id": "progress",
                        "new_content": "\n- Step 4: Started",
                        "action": "insert"
                    }
                ]
            )

            # Insert new section
            await agent_mem.update_active_memory_sections(
                external_id="agent-123",
                memory_id=1,
                sections=[
                    {
                        "section_id": "new_section",
                        "new_content": "# New Section\nContent...",
                        "action": "replace"
                    }
                ]
            )
            ```
        """
        self._ensure_initialized()
        external_id_str = str(external_id)

        logger.info(
            f"Upserting {len(sections)} sections in memory {memory_id} for {external_id_str}"
        )
        return await self._memory_manager.update_active_memory_sections(
            external_id=external_id_str,
            memory_id=memory_id,
            sections=sections,
        )

    async def update_active_memory(
        self,
        external_id: str | UUID | int,
        memory_id: int,
        sections: List[Dict[str, Any]],
    ) -> ActiveMemory:
        """
        Alias for update_active_memory_sections.

        Upsert multiple sections in an active memory.

        Args:
            external_id: Unique identifier for the agent
            memory_id: ID of the memory to update
            sections: List of section updates

        Returns:
            Updated ActiveMemory object
        """
        return await self.update_active_memory_sections(external_id, memory_id, sections)

    async def delete_active_memory(
        self,
        external_id: str | UUID | int,
        memory_id: int,
    ) -> bool:
        """
        Delete an active memory and all its sections.

        This permanently removes the memory from the database. This action cannot be undone.

        Args:
            external_id: Unique identifier for the agent
            memory_id: ID of the memory to delete

        Returns:
            True if memory was deleted, False if not found

        Raises:
            RuntimeError: If not initialized

        Example:
            ```python
            success = await agent_mem.delete_active_memory(
                external_id="agent-123",
                memory_id=42
            )
            if success:
                print("Memory deleted successfully")
            else:
                print("Memory not found")
            ```
        """
        self._ensure_initialized()
        external_id_str = str(external_id)

        logger.info(f"Deleting memory {memory_id} for {external_id_str}")
        return await self._memory_manager.delete_active_memory(
            external_id=external_id_str,
            memory_id=memory_id,
        )

    async def search_memories(
        self,
        external_id: str | UUID | int,
        query: str,
        limit: int = 10,
    ) -> RetrievalResultV2:
        """
        Fast pointer-based memory retrieval (< 200ms target).

        Searches directly across memory tiers without agent overhead,
        returning pointer references optimized for quick lookups.

        This is the preferred method for:
        - Simple fact lookups
        - Rapid-response applications
        - Latency-critical scenarios
        - Batch retrieval operations

        No LLM tokens used in this method.

        Args:
            external_id: Unique identifier for the agent (UUID, string, or int)
            query: Search query describing what information is needed
            limit: Maximum results per tier (default: 10)

        Returns:
            RetrievalResult with:
                - mode: "pointer" (pointer-based references)
                - chunks: RetrievedChunk objects with content and relevance scores
                - entities: Empty list (pointers only)
                - relationships: Empty list (pointers only)
                - synthesis: None (no AI synthesis)
                - search_strategy: Explanation of fast search approach
                - confidence: Heuristic confidence based on search results
                - metadata: Search counts and performance metrics

        Raises:
            RuntimeError: If not initialized

        Example:
            ```python
            result = await agent_mem.search_memories(
                external_id="agent-123",
                query="authentication requirements",
                limit=5
            )

            print(f"Mode: {result.mode}")
            print(f"Strategy: {result.search_strategy}")

            for chunk in result.chunks:
                print(f"- {chunk.text[:100]}...")
                print(f"  Score: {chunk.relevance_score:.2f}, Tier: {chunk.source}")
            ```
        """
        self._ensure_initialized()
        external_id_str = str(external_id)

        logger.info(f"Fast search for {external_id_str}, query: {query[:50]}...")
        return await self._memory_manager.search_memories(
            external_id=external_id_str,
            query=query,
            limit=limit,
        )

    async def deep_search_memories(
        self,
        external_id: str | UUID | int,
        query: str,
        limit: int = 10,
    ) -> RetrievalResultV2:
        """
        Deep memory retrieval with AI synthesis (500ms-2s target).

        Uses MemoryRetrieveAgent for intelligent query understanding, cross-tier
        optimization, entity extraction, relationship inference, and natural language synthesis.

        This is the comprehensive retrieval mode for:
        - Complex query interpretation
        - Multi-tier relationship analysis
        - Natural language summaries
        - Deep knowledge exploration
        - Understanding context and implications

        ⚠️ WARNING: This method uses LLM tokens (significant cost).
        Monitor token usage with your LLM provider.

        Args:
            external_id: Unique identifier for the agent (UUID, string, or int)
            query: Search query describing what information is needed
            limit: Maximum results per tier (default: 10)

        Returns:
            RetrievalResult with:
                - mode: "synthesis" (AI-synthesized)
                - chunks: RetrievedChunk objects with full content
                - entities: RetrievedEntity objects from graph (extracted)
                - relationships: RetrievedRelationship objects (inferred)
                - synthesis: AI-generated interpretation and summary
                - search_strategy: Explanation of deep search approach
                - confidence: Confidence score (0.0-1.0) in result relevance
                - metadata: Token usage, search counts, performance metrics

        Raises:
            RuntimeError: If not initialized

        Example:
            ```python
            result = await agent_mem.deep_search_memories(
                external_id="agent-123",
                query="Summarize the system architecture decisions",
                limit=5
            )

            print(f"Mode: {result.mode}")
            print(f"Confidence: {result.confidence:.2f}")

            if result.synthesis:
                print(f"Summary:\\n{result.synthesis}")

            for entity in result.entities:
                print(f"- {entity.name}: {entity.types}")

            for rel in result.relationships:
                print(f"- {rel.from_entity} → {rel.type} → {rel.to_entity}")
            ```
        """
        self._ensure_initialized()
        external_id_str = str(external_id)

        logger.info(f"Deep search for {external_id_str}, query: {query[:50]}...")
        logger.warning(f"Deep search uses LLM tokens - monitor token usage")
        return await self._memory_manager.deep_search_memories(
            external_id=external_id_str,
            query=query,
            limit=limit,
        )

    async def consolidate_and_return_conflicts(
        self,
        external_id: str | UUID | int,
        memory_id: int,
    ) -> ConsolidationConflicts:
        """
        Consolidate active memory to shortterm and return detected conflicts.

        This triggers the full consolidation workflow:
        1. Extract updated sections from active memory
        2. Create or find shortterm memory
        3. Chunk content and generate embeddings
        4. Detect section conflicts (when existing chunks match sections being updated)
        5. Extract and merge entities/relationships
        6. Return comprehensive conflict report for downstream processing

        The consolidation process does NOT automatically resolve conflicts.
        Conflicts are returned for a resolution agent to decide on merging/updating.

        Workflow:
        - Sections with update_count > 0 are processed
        - New chunks are created for sections without existing chunks
        - Existing chunks in same section trigger conflict tracking
        - Entities and relationships are extracted and merged (with conflict detection)
        - All conflicts are aggregated and returned

        Args:
            external_id: Unique identifier for the agent (UUID, string, or int)
            memory_id: ID of the active memory to consolidate

        Returns:
            ConsolidationConflicts object containing:
                - external_id: Agent identifier
                - active_memory_id: Source active memory ID
                - shortterm_memory_id: Target shortterm memory ID
                - sections: List of ConflictSection (section conflicts)
                - entity_conflicts: List of ConflictEntityDetail (entity merges)
                - relationship_conflicts: List of ConflictRelationshipDetail (relationship merges)
                - total_conflicts: Count of all conflicts
                - created_at: Timestamp of consolidation run

        Raises:
            RuntimeError: If not initialized
            ValueError: If memory_id not found

        Example:
            ```python
            # Trigger consolidation and get conflicts
            conflicts = await agent_mem.consolidate_and_return_conflicts(
                external_id="agent-123",
                memory_id=42
            )

            print(f"Consolidation completed")
            print(f"Total conflicts: {len(conflicts.sections) + len(conflicts.entity_conflicts)}")

            for section_conflict in conflicts.sections:
                print(f"Section '{section_conflict.section_id}' has {len(section_conflict.existing_chunks)} existing chunks")
                # This section has both new content AND existing chunks to reconcile

            for entity_conflict in conflicts.entity_conflicts:
                print(f"Entity '{entity_conflict.name}': merged types {entity_conflict.shortterm_types} + {entity_conflict.active_types}")

            # Send conflicts to resolution agent for decision-making
            resolved = await resolution_agent.resolve(conflicts)
            ```
        """
        self._ensure_initialized()
        external_id_str = str(external_id)

        logger.info(f"Starting consolidation for {external_id_str}, memory {memory_id}")
        conflicts = await self._memory_manager._consolidate_to_shortterm(
            external_id=external_id_str,
            active_memory_id=memory_id,
        )

        if conflicts is None:
            # Consolidation failed or found no updates
            logger.warning(f"Consolidation returned None for memory {memory_id}")
            # Return empty conflicts object
            conflicts = ConsolidationConflicts(
                external_id=external_id_str,
                active_memory_id=memory_id,
                shortterm_memory_id=0,
                created_at=datetime.now(timezone.utc),
            )

        logger.info(
            f"Consolidation complete: {len(conflicts.sections)} section conflicts, "
            f"{len(conflicts.entity_conflicts)} entity conflicts, "
            f"{len(conflicts.relationship_conflicts)} relationship conflicts"
        )

        return conflicts

    async def update_shortterm_chunk(
        self,
        external_id: str | UUID | int,
        chunk_id: int,
        new_content: str,
    ) -> dict:
        """
        Update a shortterm memory chunk's content.

        Validates external_id ownership before modifying. This ensures tenant
        isolation by checking the chunk belongs to the specified agent.

        Args:
            external_id: Unique identifier for the agent (UUID, string, or int)
            chunk_id: ID of the chunk to update
            new_content: New content for the chunk

        Returns:
            Dict with updated chunk fields (id, content, metadata)

        Raises:
            RuntimeError: If not initialized
            ValueError: If chunk not found or ownership validation fails

        Example:
            ```python
            result = await agent_mem.update_shortterm_chunk(
                external_id="agent-123",
                chunk_id=42,
                new_content="Updated content here"
            )
            print(f"Updated chunk {result['id']}")
            ```
        """
        self._ensure_initialized()
        external_id_str = str(external_id)

        logger.info(f"Updating shortterm chunk {chunk_id} for {external_id_str}")

        # Get chunk first to validate ownership
        chunk = await self._memory_manager.shortterm_repo.get_chunk_by_id(chunk_id)
        if chunk is None:
            raise ValueError(f"Chunk {chunk_id} not found")

        if chunk.external_id != external_id_str:
            raise ValueError(
                f"Ownership validation failed: chunk {chunk_id} belongs to {chunk.external_id}, not {external_id_str}"
            )

        # Update the chunk
        updated = await self._memory_manager.shortterm_repo.update_chunk(
            chunk_id=chunk_id,
            content=new_content,
        )

        if updated is None:
            raise ValueError(f"Failed to update chunk {chunk_id}")

        return {
            "id": updated.id,
            "shortterm_memory_id": updated.shortterm_memory_id,
            "content": updated.content,
            "metadata": updated.metadata if hasattr(updated, "metadata") else {},
        }

    async def create_shortterm_chunk(
        self,
        external_id: str | UUID | int,
        memory_id: int,
        content: str,
        chunk_order: int,
        section_id: str | None = None,
    ) -> dict:
        """
        Create a new shortterm memory chunk.

        Validates external_id ownership before creating. The chunk is added
        to the specified shortterm memory.

        Args:
            external_id: Unique identifier for the agent (UUID, string, or int)
            memory_id: ID of the shortterm memory to add chunk to
            content: Content for the new chunk
            chunk_order: Order position of the chunk
            section_id: Optional section reference from active memory

        Returns:
            Dict with created chunk fields (id, content, metadata)

        Raises:
            RuntimeError: If not initialized
            ValueError: If memory not found or ownership validation fails

        Example:
            ```python
            result = await agent_mem.create_shortterm_chunk(
                external_id="agent-123",
                memory_id=5,
                content="New chunk content",
                chunk_order=1
            )
            print(f"Created chunk {result['id']}")
            ```
        """
        self._ensure_initialized()
        external_id_str = str(external_id)

        logger.info(f"Creating shortterm chunk for memory {memory_id}, external_id {external_id_str}")

        # Validate memory ownership
        memory = await self._memory_manager.shortterm_repo.get_memory_by_id(memory_id)
        if memory is None:
            raise ValueError(f"Shortterm memory {memory_id} not found")

        if memory.external_id != external_id_str:
            raise ValueError(
                f"Ownership validation failed: memory {memory_id} belongs to {memory.external_id}, not {external_id_str}"
            )

        # Create the chunk
        created = await self._memory_manager.shortterm_repo.create_chunk(
            shortterm_memory_id=memory_id,
            external_id=external_id_str,
            content=content,
            section_id=section_id,
            metadata={"chunk_order": chunk_order},
        )

        return {
            "id": created.id,
            "shortterm_memory_id": created.shortterm_memory_id,
            "content": created.content,
            "section_id": created.section_id,
            "metadata": created.metadata if hasattr(created, "metadata") else {},
        }

    async def update_shortterm_entity(
        self,
        external_id: str | UUID | int,
        entity_id: int,
        name: str | None = None,
        types: list[str] | None = None,
        description: str | None = None,
        importance: float | None = None,
    ) -> dict:
        """
        Update a shortterm entity's attributes.

        Validates external_id ownership before modifying. Only updates fields
        that are provided (None values are ignored).

        Args:
            external_id: Unique identifier for the agent (UUID, string, or int)
            entity_id: Neo4j elementId of the entity
            name: New name (optional)
            types: New types list (optional)
            description: New description (optional)
            importance: New importance score (optional)

        Returns:
            Dict with updated entity fields

        Raises:
            RuntimeError: If not initialized
            ValueError: If entity not found or ownership validation fails

        Example:
            ```python
            result = await agent_mem.update_shortterm_entity(
                external_id="agent-123",
                entity_id=42,
                types=["Person", "Developer"],
                importance=0.9
            )
            print(f"Updated entity {result['id']}")
            ```
        """
        self._ensure_initialized()
        external_id_str = str(external_id)

        logger.info(f"Updating shortterm entity {entity_id} for {external_id_str}")

        # Get entity first to validate ownership
        entity = await self._memory_manager.shortterm_repo.get_entity(entity_id)
        if entity is None:
            raise ValueError(f"Entity {entity_id} not found")

        if entity.external_id != external_id_str:
            raise ValueError(
                f"Ownership validation failed: entity {entity_id} belongs to {entity.external_id}, not {external_id_str}"
            )

        # Update the entity
        updated = await self._memory_manager.shortterm_repo.update_entity(
            entity_id=entity_id,
            name=name,
            types=types,
            description=description,
            importance=importance,
        )

        if updated is None:
            raise ValueError(f"Failed to update entity {entity_id}")

        return {
            "id": str(updated.id),
            "name": updated.name,
            "types": updated.types,
            "description": updated.description,
            "importance": updated.importance,
            "external_id": updated.external_id,
        }

    async def update_shortterm_relationship(
        self,
        external_id: str | UUID | int,
        relationship_id: int,
        types: list[str] | None = None,
        description: str | None = None,
        importance: float | None = None,
        strength: float | None = None,
    ) -> dict:
        """
        Update a shortterm relationship's attributes.

        Validates external_id ownership before modifying. Only updates fields
        that are provided (None values are ignored).

        Args:
            external_id: Unique identifier for the agent (UUID, string, or int)
            relationship_id: Neo4j elementId of the relationship
            types: New types list (optional)
            description: New description (optional)
            importance: New importance score (optional)
            strength: New strength score (optional)

        Returns:
            Dict with updated relationship fields

        Raises:
            RuntimeError: If not initialized
            ValueError: If relationship not found or ownership validation fails

        Example:
            ```python
            result = await agent_mem.update_shortterm_relationship(
                external_id="agent-123",
                relationship_id=99,
                types=["WORKS_WITH"],
                strength=0.8
            )
            print(f"Updated relationship {result['id']}")
            ```
        """
        self._ensure_initialized()
        external_id_str = str(external_id)

        logger.info(f"Updating shortterm relationship {relationship_id} for {external_id_str}")

        # Get relationship first to validate ownership
        relationship = await self._memory_manager.shortterm_repo.get_relationship(relationship_id)
        if relationship is None:
            raise ValueError(f"Relationship {relationship_id} not found")

        if relationship.external_id != external_id_str:
            raise ValueError(
                f"Ownership validation failed: relationship {relationship_id} belongs to {relationship.external_id}, not {external_id_str}"
            )

        # Build metadata with strength if provided
        metadata = None
        if strength is not None:
            existing_metadata = relationship.metadata or {}
            metadata = {**existing_metadata, "strength": strength}

        # Update the relationship
        updated = await self._memory_manager.shortterm_repo.update_relationship(
            relationship_id=relationship_id,
            types=types,
            description=description,
            importance=importance,
            metadata=metadata,
        )

        if updated is None:
            raise ValueError(f"Failed to update relationship {relationship_id}")

        return {
            "id": str(updated.id),
            "types": updated.types,
            "description": updated.description,
            "importance": updated.importance,
            "external_id": updated.external_id,
            "strength": updated.metadata.get("strength") if updated.metadata else None,
        }

    async def close(self) -> None:
        """
        Close all database connections and clean up resources.

        Should be called when done using AgentMem.

        Example:
            ```python
            try:
                agent_mem = AgentMem(external_id="agent-123")
                await agent_mem.initialize()
                # ... use agent_mem ...
            finally:
                await agent_mem.close()
            ```
        """
        if not self._initialized:
            return

        logger.info("Closing AgentMem connections")

        if self._memory_manager:
            await self._memory_manager.close()

        self._initialized = False
        logger.info("AgentMem closed")

    def _ensure_initialized(self) -> None:
        """Ensure AgentMem is initialized before use."""
        if not self._initialized:
            raise RuntimeError(
                "AgentMem not initialized. Call `await agent_mem.initialize()` first."
            )

    # Context manager support
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
