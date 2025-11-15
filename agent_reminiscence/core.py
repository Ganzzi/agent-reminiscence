"""
Core AgentMem class - Main interface for memory management.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
import logging

from agent_reminiscence.config import Config, get_config, set_config
from agent_reminiscence.database.models import (
    ActiveMemory,
    RetrievalResultV2,
)
from agent_reminiscence.services.memory_manager import MemoryManager

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
        results = await agent_mem.retrieve_memories(
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

    def set_usage_processor(self, processor) -> None:
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
