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
except ImportError as e:
    logging.error(f"Failed to import AgentMem: {e}")
    AgentMem = None
    ActiveMemory = None

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
        
    def _get_agent_mem(self) -> AgentMem:
        """
        Get or create AgentMem instance
        
        Returns:
            AgentMem instance
        """
        if self._agent_mem is None:
            if AgentMem is None:
                raise ImportError("AgentMem not available. Please install agent_mem package.")
                
            # Initialize AgentMem with database configuration
            self._agent_mem = AgentMem(
                postgres_host=self.db_config['postgres_host'],
                postgres_port=self.db_config['postgres_port'],
                postgres_db=self.db_config['postgres_db'],
                postgres_user=self.db_config['postgres_user'],
                postgres_password=self.db_config['postgres_password'],
                neo4j_uri=self.db_config['neo4j_uri'],
                neo4j_user=self.db_config['neo4j_user'],
                neo4j_password=self.db_config['neo4j_password'],
            )
            
        return self._agent_mem
        
    async def create_active_memory(
        self,
        external_id: str | UUID | int,
        title: str,
        template_content: Dict,
        initial_sections: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None
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
            agent_mem = self._get_agent_mem()
            
            memory = await agent_mem.create_active_memory(
                external_id=external_id,
                title=title,
                template_content=template_content,
                initial_sections=initial_sections,
                metadata=metadata
            )
            
            logger.info(f"Created memory {memory.id} for agent {external_id}")
            return memory
            
        except Exception as e:
            logger.error(f"Error creating memory: {e}", exc_info=True)
            return None
            
    async def get_active_memories(
        self,
        external_id: str | UUID | int
    ) -> List[ActiveMemory]:
        """
        Get all active memories for an agent
        
        Args:
            external_id: Agent identifier
            
        Returns:
            List of ActiveMemory objects
        """
        try:
            agent_mem = self._get_agent_mem()
            
            memories = await agent_mem.get_active_memories(external_id=external_id)
            
            logger.info(f"Retrieved {len(memories)} memories for agent {external_id}")
            return memories
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}", exc_info=True)
            return []
            
    async def update_active_memory_section(
        self,
        external_id: str | UUID | int,
        memory_id: int,
        section_id: str,
        new_content: str
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
            agent_mem = self._get_agent_mem()
            
            memory = await agent_mem.update_active_memory_section(
                external_id=external_id,
                memory_id=memory_id,
                section_id=section_id,
                new_content=new_content
            )
            
            logger.info(f"Updated memory {memory_id} section {section_id}")
            return memory
            
        except Exception as e:
            logger.error(f"Error updating memory section: {e}", exc_info=True)
            return None
            
    async def retrieve_memories(
        self,
        external_id: str | UUID | int,
        query: str,
        limit: int = 5
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
            agent_mem = self._get_agent_mem()
            
            results = await agent_mem.retrieve_memories(
                external_id=external_id,
                query=query,
                limit=limit
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
        return {
            'id': memory.id,
            'title': memory.title,
            'template_id': memory.template_id,
            'template_name': memory.template_name,
            'created_at': memory.created_at,
            'updated_at': memory.updated_at,
            'sections': memory.sections,
            'metadata': memory.metadata,
            'section_count': len(memory.sections) if memory.sections else 0,
        }
        
    async def get_memory_by_id(
        self,
        external_id: str | UUID | int,
        memory_id: int
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
        update_count = section.get('update_count', 0)
        
        if update_count >= threshold + 2:
            return True, f"⚠️ Critical: {update_count} updates! Consolidation strongly recommended."
        elif update_count >= threshold:
            return True, f"⚡ Warning: {update_count} updates. Consider consolidation soon."
        else:
            return False, ""
            
    async def test_connection(self) -> tuple[bool, str]:
        """
        Test database connection
        
        Returns:
            Tuple of (is_connected, message)
        """
        try:
            agent_mem = self._get_agent_mem()
            # Try a simple operation to verify connection
            await agent_mem.get_active_memories(external_id="test_connection")
            return True, "Successfully connected to database"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
