"""
Active Memory Repository.

Handles CRUD operations for active memory (working memory tier).
Active memory uses template-driven structure with sections.
"""

import logging
import json
from typing import List, Optional, Dict, Any
from datetime import datetime

from agent_reminiscence.database import postgres_manager
from agent_reminiscence.database.postgres_manager import PostgreSQLManager
from agent_reminiscence.database.models import ActiveMemory

logger = logging.getLogger(__name__)


class ActiveMemoryRepository:
    """
    Repository for active memory CRUD operations.
    """

    def __init__(self, postgres: PostgreSQLManager):
        self.postgres = postgres
        self.logger = logging.getLogger(__name__)

    def _row_to_model(self, row) -> ActiveMemory:
        """Convert database row to ActiveMemory model.

        Args:
            row: Database row (dict or tuple)

        Returns:
            ActiveMemory object
        """
        # Handle both dict and list/tuple row formats
        if isinstance(row, dict):
            return ActiveMemory(
                id=row["id"],
                external_id=row["external_id"],
                title=row["title"],
                template_content=row["template_content"] if isinstance(row["template_content"], dict) else json.loads(row["template_content"]),
                sections=row["sections"] if isinstance(row["sections"], dict) else json.loads(row["sections"]),
                metadata=row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
        else:
            # Handle list/tuple format
            return ActiveMemory(
                id=row[0],
                external_id=row[1],
                title=row[2],
                template_content=row[3] if isinstance(row[3], dict) else json.loads(row[3]),
                sections=row[4] if isinstance(row[4], dict) else json.loads(row[4]),
                metadata=row[5] if isinstance(row[5], dict) else json.loads(row[5]),
                created_at=row[6],
                updated_at=row[7],
            )

    async def create(
        self,
        external_id: str,
        title: str,
        template_content: Dict[str, Any],  # Changed from str
        initial_sections: Dict[str, Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> ActiveMemory:
        """
        Create a new active memory with template and sections.
        
        Args:
            external_id: Agent identifier
            title: Memory title
            template_content: JSON template with structure:
                {
                    "template": {"id": "...", "name": "..."},
                    "sections": [
                        {
                            "id": "section_id",
                            "description": "Default content for the section"
                        }
                    ]
                }
            initial_sections: Initial sections that override template defaults
                {"section_id": {"content": "...", "update_count": 0, ...}}
            metadata: Metadata dictionary
        """
        from datetime import datetime, timezone
        
        # Extract default sections from template
        template_sections = template_content.get("sections", [])
        
        # Build sections dict with defaults from template
        sections = {}
        for tmpl_section in template_sections:
            section_id = tmpl_section["id"]
            sections[section_id] = {
                "content": tmpl_section.get("description", ""),  # description becomes default content
                "update_count": tmpl_section.get("update_count", 0),
                "awake_update_count": tmpl_section.get("awake_update_count", 0),
                "last_updated": tmpl_section.get("last_updated")
            }
        
        # Override with initial_sections if provided
        for section_id, section_data in initial_sections.items():
            if section_id in sections:
                # Override existing defaults
                sections[section_id].update(section_data)
            else:
                # Add new section (ensure all fields)
                sections[section_id] = {
                    "content": section_data.get("content", ""),
                    "update_count": section_data.get("update_count", 0),
                    "awake_update_count": section_data.get("awake_update_count", 0),
                    "last_updated": section_data.get("last_updated")
                }
        
        # Ensure last_updated is ISO string if datetime
        for section_id, section_data in sections.items():
            if isinstance(section_data.get("last_updated"), datetime):
                sections[section_id]["last_updated"] = section_data["last_updated"].isoformat()
        
        query = """
            INSERT INTO active_memory 
            (external_id, title, template_content, sections, metadata)
            VALUES ($1, $2, $3::jsonb, $4::jsonb, $5::jsonb)
            RETURNING id, external_id, title, template_content, sections, 
                      metadata, created_at, updated_at
        """
        
        async with self.postgres.connection() as conn:
            result = await conn.execute(
                query, 
                [external_id, title, template_content, sections, metadata]
            )
            rows = result.result()
            return self._row_to_model(rows[0])

    async def upsert_sections(
        self,
        memory_id: int,
        section_updates: List[Dict[str, Any]],
    ) -> Optional[ActiveMemory]:
        """
        Upsert multiple sections in an active memory (batch operation).
        
        Supports:
        - Inserting new sections (adds to template_content.sections)
        - Updating existing sections
        - Content replacement (with optional pattern matching)
        - Content insertion/appending
        
        Args:
            memory_id: Memory ID
            section_updates: List of section updates:
                [
                    {
                        "section_id": "progress",
                        "old_content": "# Old",  # Optional: pattern to find
                        "new_content": "# New",
                        "action": "replace"  # "replace" or "insert"
                    }
                ]
        
        Returns:
            Updated ActiveMemory or None if not found
        
        Action Behaviors:
            - replace + no old_content: Replace entire section content
            - replace + old_content: Replace substring matching old_content
            - insert + no old_content: Append new_content at end
            - insert + old_content: Insert new_content after old_content
        """
        from datetime import datetime, timezone
        
        # Get current state
        current = await self.get_by_id(memory_id)
        if not current:
            logger.warning(f"Active memory {memory_id} not found")
            return None
        
        updated_sections = current.sections.copy()
        updated_template = current.template_content.copy()
        template_sections = updated_template.get("sections", [])
        
        for update in section_updates:
            section_id = update["section_id"]
            old_content = update.get("old_content")
            new_content = update.get("new_content", "")
            action = update.get("action", "replace")
            
            # Check if section exists
            if section_id not in updated_sections:
                # INSERT NEW SECTION
                logger.info(f"Inserting new section '{section_id}' in memory {memory_id}")
                
                # Add to sections
                updated_sections[section_id] = {
                    "content": new_content,
                    "update_count": 1,
                    "awake_update_count": 1,
                    "last_updated": datetime.now(timezone.utc).isoformat()
                }
                
                # Add to template_content.sections
                template_sections.append({
                    "id": section_id,
                    "description": "Dynamically added section"
                })
                updated_template["sections"] = template_sections
                
            else:
                # UPDATE EXISTING SECTION
                current_content = updated_sections[section_id]["content"]
                current_update_count = updated_sections[section_id].get("update_count", 0)
                current_awake_count = updated_sections[section_id].get("awake_update_count", 0)
                
                if action == "replace":
                    if not old_content:
                        # Replace entire content
                        final_content = new_content
                    else:
                        # Replace pattern
                        if old_content in current_content:
                            final_content = current_content.replace(old_content, new_content)
                        else:
                            logger.warning(
                                f"Pattern '{old_content[:50]}...' not found in section '{section_id}'. "
                                f"Replacing entire content."
                            )
                            final_content = new_content
                
                elif action == "insert":
                    if not old_content:
                        # Append at end
                        final_content = current_content + "\n" + new_content
                    else:
                        # Insert after pattern
                        if old_content in current_content:
                            parts = current_content.split(old_content, 1)
                            final_content = parts[0] + old_content + "\n" + new_content + parts[1]
                        else:
                            logger.warning(
                                f"Pattern '{old_content[:50]}...' not found in section '{section_id}'. "
                                f"Appending at end."
                            )
                            final_content = current_content + "\n" + new_content
                
                else:
                    raise ValueError(f"Invalid action '{action}'. Must be 'replace' or 'insert'.")
                
                # Update section
                updated_sections[section_id] = {
                    "content": final_content,
                    "update_count": current_update_count + 1,
                    "awake_update_count": current_awake_count + 1,
                    "last_updated": datetime.now(timezone.utc).isoformat()
                }
        
        # Update database
        query = """
            UPDATE active_memory
            SET sections = $1::jsonb, template_content = $2::jsonb, updated_at = CURRENT_TIMESTAMP
            WHERE id = $3
            RETURNING id, external_id, title, template_content, sections, 
                      metadata, created_at, updated_at
        """
        
        async with self.postgres.connection() as conn:
            result = await conn.execute(
                query, 
                [updated_sections, updated_template, memory_id]
            )
            rows = result.result()
            
            if not rows:
                logger.warning(f"Active memory {memory_id} not found for upsert")
                return None
            
            memory = self._row_to_model(rows[0])
            logger.info(f"Upserted {len(section_updates)} sections in memory {memory_id}")
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

        DEPRECATED: Use upsert_sections() instead for new functionality.
        
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
        import warnings
        warnings.warn(
            "update_section() is deprecated. Use upsert_sections() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
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
            result = await conn.execute(query, [updated_sections, memory_id])
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

    async def update_sections(
        self,
        memory_id: int,
        section_updates: List[Dict[str, str]],
    ) -> Optional[ActiveMemory]:
        """
        Update multiple sections in an active memory (batch update).

        DEPRECATED: Use upsert_sections() instead for new functionality.
        
        All updates are done in a single transaction.
        Automatically increments update_count for each section.

        Args:
            memory_id: Memory ID
            section_updates: List of dicts with 'section_id' and 'new_content'
                           Example: [{"section_id": "progress", "new_content": "..."}]

        Returns:
            Updated ActiveMemory object or None if not found

        Raises:
            ValueError: If any section_id not found in memory

        Example:
            updated = await repo.update_sections(
                memory_id=1,
                section_updates=[
                    {"section_id": "progress", "new_content": "Step 1 done"},
                    {"section_id": "notes", "new_content": "New insights"}
                ]
            )
        """
        import warnings
        warnings.warn(
            "update_sections() is deprecated. Use upsert_sections() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # First get current state
        current = await self.get_by_id(memory_id)
        if not current:
            logger.warning(f"Active memory {memory_id} not found")
            return None

        # Validate all sections exist
        for update in section_updates:
            section_id = update.get("section_id")
            if not section_id:
                raise ValueError("Each section update must have 'section_id'")
            if section_id not in current.sections:
                raise ValueError(
                    f"Section '{section_id}' not found in memory {memory_id}. "
                    f"Available sections: {list(current.sections.keys())}"
                )

        # Update all sections
        updated_sections = current.sections.copy()
        for update in section_updates:
            section_id = update["section_id"]
            new_content = update.get("new_content", "")

            updated_sections[section_id] = {
                "content": new_content,
                "update_count": current.sections[section_id].get("update_count", 0) + 1,
            }

        # Single database update with all changes
        query = """
            UPDATE active_memory
            SET sections = $1, updated_at = CURRENT_TIMESTAMP
            WHERE id = $2
            RETURNING id, external_id, title, template_content, sections, 
                      metadata, created_at, updated_at
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(query, [updated_sections, memory_id])
            rows = result.result()

            if not rows:
                logger.warning(f"Active memory {memory_id} not found for batch update")
                return None

            memory = self._row_to_model(rows[0])
            logger.info(f"Batch updated {len(section_updates)} sections in memory {memory_id}")
            return memory

    async def reset_all_update_counts(self, memory_id: int) -> bool:
        """
        Reset update_count to 0 for all sections in a memory.
        
        Called after successful consolidation to shortterm memory.
        Does NOT reset awake_update_count (permanent counter).
        
        Args:
            memory_id: Memory ID
        
        Returns:
            True if successful, False if memory not found
        """
        current = await self.get_by_id(memory_id)
        if not current:
            logger.warning(f"Active memory {memory_id} not found")
            return False
        
        # Reset only update_count, preserve awake_update_count and last_updated
        reset_sections = {}
        for section_id, section_data in current.sections.items():
            reset_sections[section_id] = {
                "content": section_data.get("content", ""),
                "update_count": 0,  # RESET
                "awake_update_count": section_data.get("awake_update_count", 0),  # PRESERVE
                "last_updated": section_data.get("last_updated")  # PRESERVE
            }
        
        query = """
            UPDATE active_memory
            SET sections = $1, updated_at = CURRENT_TIMESTAMP
            WHERE id = $2
        """
        
        async with self.postgres.connection() as conn:
            await conn.execute(query, [reset_sections, memory_id])
            logger.info(f"Reset update_count for all sections in memory {memory_id}")
            return True

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
            await conn.execute(query, [updated_sections, memory_id])
            logger.info(f"Reset update_count for section '{section_id}' in memory {memory_id}")
            return True

    async def delete(self, memory_id: int) -> bool:
        """
        Delete an active memory.

        Args:
            memory_id: The memory ID to delete

        Returns:
            True if deleted successfully

        Raises:
            ValueError: If memory not found
        """
        query = """
            DELETE FROM active_memory
            WHERE id = $1
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(query, [memory_id])
            # Parse the result to check if any rows were affected
            if result and hasattr(result, "result"):
                result_str = str(result.result())
                if "DELETE 0" in result_str or result.result() == 0:
                    raise ValueError(f"Active memory with ID {memory_id} not found")
            else:
                # Alternative approach - check if memory exists first
                memory = await self.get_by_id(memory_id)
                if not memory:
                    raise ValueError(f"Active memory with ID {memory_id} not found")

        logger.info(f"Deleted active memory {memory_id}")
        return True

    async def get_all_templates_by_external_id(self, external_id: str) -> List[Dict[str, str]]:
        """
        Get template_content from all active memories for an external_id.

        Args:
            external_id: Agent identifier

        Returns:
            List of dicts with 'memory_id', 'title', and 'template_content'

        Example:
            templates = await repo.get_all_templates_by_external_id("agent-123")
            for tmpl in templates:
                print(f"Memory {tmpl['memory_id']}: {tmpl['title']}")
        """
        query = """
            SELECT id, title, template_content
            FROM active_memory
            WHERE external_id = $1
            ORDER BY updated_at DESC
        """

        async with self.postgres.connection() as conn:
            result = await conn.execute(query, [external_id])
            rows = result.result()

            templates = [
                {
                    "memory_id": row["id"],
                    "title": row["title"],
                    "template_content": row["template_content"],
                }
                for row in rows
            ]

            logger.debug(f"Retrieved {len(templates)} templates for {external_id}")
            return templates


