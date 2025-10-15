"""
Memory Service - Wrapper around AgentMem for UI operations
"""

from typing import Dict, List, Optional, Any
from uuid import UUID
import streamlit as st
import logging
import sys
from pathlib import Path

# Add parent directory to path to import agent_mem
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from agent_mem import AgentMem
    from agent_mem.database.models import ActiveMemory
    from agent_mem.config.settings import Config
except ImportError as e:
    logging.error(f"Failed to import AgentMem: {e}")
    AgentMem = None
    ActiveMemory = None
    Config = None

logger = logging.getLogger(__name__)


class MemoryService:
    """High-level service for memory operations"""

    def __init__(self, db_config: Dict[str, Any]):
        """
        Initialize memory service

        Args:
            db_config: Database configuration dictionary
        """
        self.db_config = db_config
        self._agent_mem: Optional[AgentMem] = None
        self._initialized = False

    async def _get_agent_mem(self) -> AgentMem:
        """
        Get or create AgentMem instance with initialization

        Returns:
            Initialized AgentMem instance
        """
        if self._agent_mem is None:
            if AgentMem is None or Config is None:
                raise ImportError("AgentMem not available. Please install agent_mem package.")

            # Create Config object from dictionary
            config = Config(
                postgres_host=self.db_config["postgres_host"],
                postgres_port=self.db_config["postgres_port"],
                postgres_db=self.db_config["postgres_db"],
                postgres_user=self.db_config["postgres_user"],
                postgres_password=self.db_config["postgres_password"],
                neo4j_uri=self.db_config["neo4j_uri"],
                neo4j_user=self.db_config["neo4j_user"],
                neo4j_password=self.db_config["neo4j_password"],
            )

            # Initialize AgentMem with config
            self._agent_mem = AgentMem(config=config)

        # Ensure initialization is complete
        if not self._initialized:
            await self._agent_mem.initialize()
            self._initialized = True
            logger.info("AgentMem initialized successfully")

        return self._agent_mem

    async def create_active_memory(
        self,
        external_id: str | UUID | int,
        title: str,
        template_content: Dict,
        initial_sections: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[ActiveMemory]:
        """
        Create a new active memory

        Args:
            external_id: Agent identifier
            title: Memory title
            template_content: Template dictionary
            initial_sections: Initial section content
            metadata: Additional metadata

        Returns:
            Created ActiveMemory or None if error
        """
        try:
            agent_mem = await self._get_agent_mem()

            memory = await agent_mem.create_active_memory(
                external_id=external_id,
                title=title,
                template_content=template_content,
                initial_sections=initial_sections,
                metadata=metadata,
            )

            logger.info(f"Created memory {memory.id} for agent {external_id}")
            return memory

        except Exception as e:
            logger.error(f"Error creating memory: {e}", exc_info=True)
            return None

    async def get_active_memories(self, external_id: str | UUID | int) -> List[ActiveMemory]:
        """
        Get all active memories for an agent

        Args:
            external_id: Agent identifier

        Returns:
            List of ActiveMemory objects
        """
        try:
            agent_mem = await self._get_agent_mem()

            memories = await agent_mem.get_active_memories(external_id=external_id)

            logger.info(f"Retrieved {len(memories)} memories for agent {external_id}")
            return memories

        except Exception as e:
            logger.error(f"Error retrieving memories: {e}", exc_info=True)
            return []

    async def update_active_memory_section(
        self, external_id: str | UUID | int, memory_id: int, section_id: str, new_content: str
    ) -> Optional[ActiveMemory]:
        """
        Update a section in an active memory

        Args:
            external_id: Agent identifier
            memory_id: Memory ID
            section_id: Section ID to update
            new_content: New content for the section

        Returns:
            Updated ActiveMemory or None if error
        """
        try:
            agent_mem = await self._get_agent_mem()

            memory = await agent_mem.update_active_memory_section(
                external_id=external_id,
                memory_id=memory_id,
                section_id=section_id,
                new_content=new_content,
            )

            logger.info(f"Updated memory {memory_id} section {section_id}")
            return memory

        except Exception as e:
            logger.error(f"Error updating memory section: {e}", exc_info=True)
            return None

    async def retrieve_memories(
        self, external_id: str | UUID | int, query: str, limit: int = 5
    ) -> List[Dict]:
        """
        Retrieve relevant memories based on query

        Args:
            external_id: Agent identifier
            query: Search query
            limit: Maximum number of results

        Returns:
            List of retrieved memory dictionaries
        """
        try:
            agent_mem = await self._get_agent_mem()

            results = await agent_mem.retrieve_memories(
                external_id=external_id, query=query, limit=limit
            )

            logger.info(f"Retrieved {len(results)} memories for query: {query[:50]}...")
            return results

        except Exception as e:
            logger.error(f"Error retrieving memories: {e}", exc_info=True)
            return []

    def format_memory_for_display(self, memory: ActiveMemory) -> Dict[str, Any]:
        """
        Format memory object for display

        Args:
            memory: ActiveMemory object

        Returns:
            Dictionary with formatted data
        """
        # Parse template_content as dict (new JSON format)
        template_data = memory.template_content if isinstance(memory.template_content, dict) else {}

        template_info = template_data.get("template", {})
        template_id = template_info.get("id", "unknown")
        template_name = template_info.get("name", "Unknown Template")

        # Format sections with new fields
        formatted_sections = {}
        if memory.sections:
            for section_id, section_data in memory.sections.items():
                formatted_sections[section_id] = {
                    "content": section_data.get("content", ""),
                    "update_count": section_data.get("update_count", 0),
                    "awake_update_count": section_data.get("awake_update_count", 0),
                    "last_updated": section_data.get("last_updated"),
                }

        return {
            "id": memory.id,
            "title": memory.title,
            "template_id": template_id,
            "template_name": template_name,
            "created_at": memory.created_at,
            "updated_at": memory.updated_at,
            "sections": formatted_sections,
            "metadata": memory.metadata,
            "section_count": len(memory.sections) if memory.sections else 0,
        }

    async def get_memory_by_id(
        self, external_id: str | UUID | int, memory_id: int
    ) -> Optional[ActiveMemory]:
        """
        Get a specific memory by ID

        Args:
            external_id: Agent identifier
            memory_id: Memory ID

        Returns:
            ActiveMemory object or None
        """
        try:
            memories = await self.get_active_memories(external_id)

            for memory in memories:
                if memory.id == memory_id:
                    return memory

            return None

        except Exception as e:
            logger.error(f"Error getting memory by ID: {e}", exc_info=True)
            return None

    def check_consolidation_needed(self, section: Dict, threshold: int = 8) -> tuple[bool, str]:
        """
        Check if a section needs consolidation

        Args:
            section: Section dictionary
            threshold: Update count threshold

        Returns:
            Tuple of (needs_consolidation, warning_message)
        """
        update_count = section.get("update_count", 0)

        if update_count >= threshold + 2:
            return True, f"⚠️ Critical: {update_count} updates! Consolidation strongly recommended."
        elif update_count >= threshold:
            return True, f"⚡ Warning: {update_count} updates. Consider consolidation soon."
        else:
            return False, ""

    async def delete_active_memory(
        self, external_id: str | UUID | int, memory_id: int
    ) -> tuple[bool, str]:
        """
        Delete an active memory

        Args:
            external_id: Agent identifier
            memory_id: Memory ID to delete

        Returns:
            Tuple of (success, message)
        """
        try:
            agent_mem = await self._get_agent_mem()

            # Check if the delete method exists (Phase 8.6 implementation)
            if not hasattr(agent_mem, "delete_active_memory"):
                # Temporary implementation for Phase 8.1-8.5
                # Will be replaced when Phase 8.6 adds the actual API
                logger.warning(
                    f"delete_active_memory API not yet implemented. Memory {memory_id} deletion skipped."
                )
                return False, "Delete API not yet implemented. Please complete Phase 8.6."

            success = await agent_mem.delete_active_memory(
                external_id=external_id, memory_id=memory_id
            )

            if success:
                logger.info(f"Deleted memory {memory_id} for agent {external_id}")
                return True, "Memory deleted successfully"
            else:
                logger.warning(f"Memory {memory_id} not found for deletion")
                return False, "Memory not found"

        except Exception as e:
            logger.error(f"Error deleting memory: {e}", exc_info=True)
            return False, f"Error: {str(e)}"

    async def test_connection(self) -> tuple[bool, str]:
        """
        Test database connection

        Returns:
            Tuple of (is_connected, message)
        """
        try:
            agent_mem = await self._get_agent_mem()
            # Try a simple operation to verify connection
            await agent_mem.get_active_memories(external_id="test_connection")
            return True, "Successfully connected to database"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
