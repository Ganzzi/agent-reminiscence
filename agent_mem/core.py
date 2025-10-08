"""
Core AgentMem class - Main interface for memory management.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
import logging

from agent_mem.config import Config, get_config, set_config
from agent_mem.database.models import (
    ActiveMemory,
    RetrievalResult,
)
from agent_mem.services.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class AgentMem:
    """
    Main interface for hierarchical memory management.

    AgentMem is STATELESS and can serve multiple agents/workers.
    Pass external_id to each method call to specify which agent's memory to access.

    Provides 4 simple methods to manage all memory tiers:
    1. create_active_memory(external_id, ...) - Create new working memory
    2. get_active_memories(external_id) - Get all working memories
    3. update_active_memory_sections(external_id, memory_id, sections) - Update multiple sections
    4. retrieve_memories(external_id, query, ...) - Search shortterm and longterm memories

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

    async def create_active_memory(
        self,
        external_id: str | UUID | int,
        title: str,
        template_content: str,
        initial_sections: Optional[Dict[str, Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ActiveMemory:
        """
                Create a new active memory with template-driven structure.

                Active memories represent the agent's working memory - current tasks,
                recent decisions, and ongoing work context.

                Args:
                    external_id: Unique identifier for the agent (UUID, string, or int)
                    title: Memory title
                    template_content: YAML template defining section structure
                    initial_sections: Optional initial sections {section_id: {content: str, update_count: int}}
                    metadata: Optional metadata dictionary

                Returns:
                    Created ActiveMemory object

                Raises:
                    RuntimeError: If not initialized

                Example:
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
          - id: "progress"
            title: "Progress"
        ''',
                        initial_sections={
                            "current_task": {"content": "# Task\nImplement feature", "update_count": 0},
                            "progress": {"content": "# Progress\n- Started", "update_count": 0}
                        },
                        metadata={"priority": "high"}
                    )
                    ```
        """
        self._ensure_initialized()

        # Validate inputs
        if not title or not title.strip():
            raise ValueError("title cannot be empty")
        if not template_content or not template_content.strip():
            raise ValueError("template_content cannot be empty")

        external_id_str = str(external_id)

        logger.info(f"Creating active memory for {external_id_str}: {title}")
        return await self._memory_manager.create_active_memory(
            external_id=external_id_str,
            title=title,
            template_content=template_content,
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
        sections: List[Dict[str, str]],
    ) -> ActiveMemory:
        """
        Update multiple sections in an active memory (batch update).

        After updating all sections, checks if total update count across all sections
        exceeds threshold for consolidation. Consolidation runs in background if triggered.

        Args:
            external_id: Unique identifier for the agent
            memory_id: ID of the memory to update
            sections: List of section updates, each dict with 'section_id' and 'new_content'
                     Example: [{"section_id": "progress", "new_content": "..."}, ...]

        Returns:
            Updated ActiveMemory object

        Raises:
            RuntimeError: If not initialized
            ValueError: If memory not found or sections invalid

        Example:
            ```python
            memory = await agent_mem.update_active_memory_sections(
                external_id="agent-123",
                memory_id=1,
                sections=[
                    {"section_id": "progress", "new_content": "Updated progress..."},
                    {"section_id": "notes", "new_content": "New notes..."}
                ]
            )
            ```
        """
        self._ensure_initialized()
        external_id_str = str(external_id)

        logger.info(
            f"Updating {len(sections)} sections in memory {memory_id} for {external_id_str}"
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

    async def retrieve_memories(
        self,
        external_id: str | UUID | int,
        query: str,
        limit: int = 10,
        synthesis: bool = False,
    ) -> RetrievalResult:
        """
        Search and retrieve relevant memories across shortterm and longterm tiers.

        This method uses the Memory Retrieve Agent to intelligently search across
        memory tiers, returning matched chunks, entities, and relationships along
        with optional synthesized response.

        Args:
            external_id: Unique identifier for the agent
            query: Search query describing what information is needed
            limit: Maximum results per tier (default: 10)
            synthesis: Force AI synthesis of results regardless of query complexity (default: False)

        Returns:
            RetrievalResult containing:
                - mode: "pointer" or "synthesis" (determines if synthesis is included)
                - chunks: List of RetrievedChunk objects with content, tier, and scores
                - entities: List of RetrievedEntity objects from the graph
                - relationships: List of RetrievedRelationship objects from the graph
                - synthesis: Optional AI-generated summary (only in synthesis mode)
                - search_strategy: Explanation of the search approach used
                - confidence: Confidence score (0.0-1.0) in result relevance
                - metadata: Additional search metadata (counts, timing, etc.)

        Raises:
            RuntimeError: If not initialized

        Example:
            ```python
            result = await agent_mem.retrieve_memories(
                external_id="agent-123",
                query="How did I implement authentication?",
                limit=5,
                synthesis=True  # Request AI summary
            )

            print(f"Mode: {result.mode}")
            print(f"Strategy: {result.search_strategy}")
            print(f"Confidence: {result.confidence}")

            if result.synthesis:
                print(f"Summary: {result.synthesis}")

            for chunk in result.chunks:
                print(f"Chunk: {chunk.content[:100]}...")
                print(f"Tier: {chunk.tier}, Score: {chunk.score}")

            for entity in result.entities:
                print(f"Entity: {entity.name} (types: {entity.types})")
            ```
        """
        self._ensure_initialized()
        external_id_str = str(external_id)

        logger.info(f"Retrieving memories for {external_id_str}, query: {query[:50]}...")
        return await self._memory_manager.retrieve_memories(
            external_id=external_id_str,
            query=query,
            limit=limit,
            synthesis=synthesis,
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
