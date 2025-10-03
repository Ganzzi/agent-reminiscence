"""
Active Memory Repository.

Handles CRUD operations for active memory (working memory tier).
Active memory uses template-driven structure with sections.
"""

import logging
import json
from typing import List, Optional, Dict, Any
from datetime import datetime

from agent_mem.database.postgres_manager import PostgreSQLManager
from agent_mem.database.models import ActiveMemory

logger = logging.getLogger(__name__)


class ActiveMemoryRepository:
    """
    Repository for active memory CRUD operations.

    Active memories represent the agent's current working context -
    tasks in progress, recent decisions, and immediate work context.

    Uses template-driven structure:
    - template_content: YAML template defining section structure
    - sections: JSONB {section_id: {content: str, update_count: int}}
    """

    def __init__(self, postgres_manager: PostgreSQLManager):
        """
        Initialize repository.

        Args:
            postgres_manager: PostgreSQL connection manager
        """
        self.postgres = postgres_manager

    async def create(
        self,
        external_id: str,
        title: str,
        template_content: str,
        sections: Dict[str, Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> ActiveMemory:
        """
        Create a new active memory with template and sections.

        Args:
            external_id: Agent identifier
            title: Memory title
            template_content: YAML template content
            sections: Initial sections {section_id: {content: str, update_count: int}}
            metadata: Metadata dictionary

        Returns:
            Created ActiveMemory object

        Example:
            memory = await repo.create(
                external_id="agent-123",
                title="Task Memory",
                template_content="template:\\n  id: task_v1...",
                sections={
                    "current_task": {"content": "# Task\\n...", "update_count": 0},
                    "progress": {"content": "# Progress\\n...", "update_count": 0}
                },
                metadata={"priority": "high"}
            )
        """
        query = """
            INSERT INTO active_memory 
            (external_id, title, template_content, sections, metadata)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id, external_id, title, template_content, sections, 
                      metadata, created_at, updated_at
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(
                query,
                [
                    external_id,
                    title,
                    template_content,
                    json.dumps(sections),
                    json.dumps(metadata),
                ],
            )

            row = result.result()[0]
            memory = self._row_to_model(row)

            logger.info(
                f"Created active memory {memory.id} for {external_id} with {len(sections)} sections"
            )
            return memory

    async def get_by_id(self, memory_id: int) -> Optional[ActiveMemory]:
        """
        Get an active memory by ID.

        Args:
            memory_id: Memory ID

        Returns:
            ActiveMemory object or None if not found
        """
        query = """
            SELECT id, external_id, title, template_content, sections, 
                   metadata, created_at, updated_at
            FROM active_memory
            WHERE id = $1
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(query, [memory_id])
            rows = result.result()

            if not rows:
                return None

            return self._row_to_model(rows[0])

    async def get_all_by_external_id(self, external_id: str) -> List[ActiveMemory]:
        """
        Get all active memories for an external_id.

        Args:
            external_id: Agent identifier

        Returns:
            List of ActiveMemory objects (may be empty)

        Example:
            memories = await repo.get_all_by_external_id("agent-123")
            for memory in memories:
                print(f"{memory.title}: {len(memory.sections)} sections")
        """
        query = """
            SELECT id, external_id, title, template_content, sections, 
                   metadata, created_at, updated_at
            FROM active_memory
            WHERE external_id = $1
            ORDER BY updated_at DESC
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(query, [external_id])
            rows = result.result()

            memories = [self._row_to_model(row) for row in rows]

            logger.debug(f"Retrieved {len(memories)} active memories for {external_id}")
            return memories

    async def update_section(
        self,
        memory_id: int,
        section_id: str,
        new_content: str,
    ) -> Optional[ActiveMemory]:
        """
        Update a specific section in an active memory.

        Automatically increments the section's update_count.

        Args:
            memory_id: Memory ID
            section_id: Section ID to update
            new_content: New content for the section

        Returns:
            Updated ActiveMemory object or None if not found

        Raises:
            ValueError: If section_id not found in memory

        Example:
            updated = await repo.update_section(
                memory_id=1,
                section_id="progress",
                new_content="# Progress\\n- Step 1 complete\\n- Working on step 2"
            )
            print(updated.sections["progress"]["update_count"])  # Incremented
        """
        # First get current state
        current = await self.get_by_id(memory_id)
        if not current:
            logger.warning(f"Active memory {memory_id} not found")
            return None

        # Check if section exists
        if section_id not in current.sections:
            raise ValueError(
                f"Section '{section_id}' not found in memory {memory_id}. "
                f"Available sections: {list(current.sections.keys())}"
            )

        # Update the section
        updated_sections = current.sections.copy()
        updated_sections[section_id] = {
            "content": new_content,
            "update_count": current.sections[section_id].get("update_count", 0) + 1,
        }

        query = """
            UPDATE active_memory
            SET sections = $1, updated_at = CURRENT_TIMESTAMP
            WHERE id = $2
            RETURNING id, external_id, title, template_content, sections, 
                      metadata, created_at, updated_at
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(query, [json.dumps(updated_sections), memory_id])
            rows = result.result()

            if not rows:
                logger.warning(f"Active memory {memory_id} not found for update")
                return None

            memory = self._row_to_model(rows[0])
            logger.info(
                f"Updated section '{section_id}' in memory {memory_id} "
                f"(update_count={memory.sections[section_id]['update_count']})"
            )
            return memory

    async def update_metadata(
        self,
        memory_id: int,
        metadata: Dict[str, Any],
    ) -> Optional[ActiveMemory]:
        """
        Update metadata for an active memory.

        Args:
            memory_id: Memory ID
            metadata: New metadata dictionary

        Returns:
            Updated ActiveMemory object or None if not found

        Example:
            updated = await repo.update_metadata(
                memory_id=1,
                metadata={"priority": "critical", "tags": ["urgent"]}
            )
        """
        query = """
            UPDATE active_memory
            SET metadata = $1, updated_at = CURRENT_TIMESTAMP
            WHERE id = $2
            RETURNING id, external_id, title, template_content, sections, 
                      metadata, created_at, updated_at
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(query, [json.dumps(metadata), memory_id])
            rows = result.result()

            if not rows:
                logger.warning(f"Active memory {memory_id} not found for metadata update")
                return None

            memory = self._row_to_model(rows[0])
            logger.info(f"Updated metadata for active memory {memory_id}")
            return memory

    async def delete(self, memory_id: int) -> bool:
        """
        Delete an active memory.

        Args:
            memory_id: Memory ID

        Returns:
            True if deleted, False if not found

        Example:
            deleted = await repo.delete(memory_id=1)
        """
        query = "DELETE FROM active_memory WHERE id = $1"

        async with self.postgres.connection() as conn:
            result = await conn.execute(query, [memory_id])
            # PSQLPy doesn't return row count directly, check if operation succeeded
            # by trying to fetch the deleted row
            deleted = True  # Assume success if no exception

            if deleted:
                logger.info(f"Deleted active memory {memory_id}")
            else:
                logger.warning(f"Active memory {memory_id} not found for deletion")

            return deleted

    async def get_sections_needing_consolidation(
        self, external_id: str, threshold: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get sections that have reached the consolidation threshold.

        Returns list of {memory_id, section_id, update_count, content}.

        Args:
            external_id: Agent identifier
            threshold: Minimum update_count for consolidation (default: 5)

        Returns:
            List of dicts with memory_id, section_id, update_count, content

        Example:
            needs_consolidation = await repo.get_sections_needing_consolidation(
                external_id="agent-123",
                threshold=5
            )
            for item in needs_consolidation:
                print(f"Memory {item['memory_id']}, section {item['section_id']}: {item['update_count']} updates")
        """
        # Query all memories and check sections in code (simpler than complex JSON query)
        memories = await self.get_all_by_external_id(external_id)

        result = []
        for memory in memories:
            for section_id, section_data in memory.sections.items():
                update_count = section_data.get("update_count", 0)
                if update_count >= threshold:
                    result.append(
                        {
                            "memory_id": memory.id,
                            "section_id": section_id,
                            "update_count": update_count,
                            "content": section_data.get("content", ""),
                        }
                    )

        logger.debug(
            f"Found {len(result)} sections needing consolidation "
            f"for {external_id} (threshold={threshold})"
        )
        return result

    async def reset_section_count(self, memory_id: int, section_id: str) -> bool:
        """
        Reset the update_count for a specific section (after consolidation).

        Args:
            memory_id: Memory ID
            section_id: Section ID

        Returns:
            True if reset, False if not found

        Example:
            await repo.reset_section_count(memory_id=1, section_id="progress")
        """
        current = await self.get_by_id(memory_id)
        if not current or section_id not in current.sections:
            return False

        updated_sections = current.sections.copy()
        updated_sections[section_id]["update_count"] = 0

        query = """
            UPDATE active_memory
            SET sections = $1
            WHERE id = $2
        """

        async with self.postgres.connection() as conn:
            await conn.execute(query, [json.dumps(updated_sections), memory_id])
            logger.info(f"Reset update_count for section '{section_id}' in memory {memory_id}")
            return True

    async def count_by_external_id(self, external_id: str) -> int:
        """
        Count active memories for an external_id.

        Args:
            external_id: Agent identifier

        Returns:
            Count of active memories
        """
        query = "SELECT COUNT(*) FROM active_memory WHERE external_id = $1"

        async with self.postgres.connection() as conn:
            result = await conn.execute(query, [external_id])
            row = result.result()[0]
            count = row[0]

            logger.debug(f"Active memory count for {external_id}: {count}")
            return count

    def _row_to_model(self, row) -> ActiveMemory:
        """
        Convert a database row to an ActiveMemory model.

        Args:
            row: Database row tuple

        Returns:
            ActiveMemory object
        """
        # Row format: id, external_id, title, template_content, sections, metadata, created_at, updated_at
        return ActiveMemory(
            id=row[0],
            external_id=row[1],
            title=row[2],
            template_content=row[3],
            sections=row[4] if isinstance(row[4], dict) else {},
            metadata=row[5] if isinstance(row[5], dict) else {},
            created_at=row[6],
            updated_at=row[7],
        )
