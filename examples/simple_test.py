"""
Simple example demonstrating basic agent-mem functionality.
This script shows how to:
1. Create active memories with templates
2. Retrieve memories
3. Update memory sections
"""

import asyncio
import logging
from agent_mem import AgentMem

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Simple template for task memory
SIMPLE_TEMPLATE = """
template:
  id: "simple_task"
  name: "Simple Task Memory"
  version: "1.0.0"
sections:
  - id: "task"
    title: "Task"
  - id: "status"
    title: "Status"
"""


async def main():
    """Simple demonstration of agent-mem functionality."""
    logger.info("=" * 60)
    logger.info("Agent Memory System - Simple Test")
    logger.info("=" * 60)

    # Initialize the memory system
    logger.info("\n1. Initializing Agent Memory System...")
    memory = AgentMem()
    await memory.initialize()
    logger.info("   ✓ Memory system initialized\n")

    agent_id = "test_agent_001"

    # Create an active memory
    logger.info("2. Creating active memory...")

    memory1 = await memory.create_active_memory(
        external_id=agent_id,
        title="User Preferences",
        template_content=SIMPLE_TEMPLATE,
        initial_sections={
            "task": {
                "content": "Understanding user preferences for backend development",
                "update_count": 0,
            },
            "status": {
                "content": "# Preferences\n- Prefers Python\n- Working on AI agent project",
                "update_count": 0,
            },
        },
        metadata={"topic": "programming"},
    )
    logger.info(f"   ✓ Created memory ID {memory1.id}: {memory1.title}")
    logger.info(f"   ✓ Sections: {list(memory1.sections.keys())}\n")

    # Create another memory
    logger.info("3. Creating another memory...")
    memory2 = await memory.create_active_memory(
        external_id=agent_id,
        title="Current Project",
        template_content=SIMPLE_TEMPLATE,
        initial_sections={
            "task": {
                "content": "Working on AI agent memory management system",
                "update_count": 0,
            },
            "status": {
                "content": "# Status\n- Completed setup\n- Testing functionality",
                "update_count": 0,
            },
        },
        metadata={"topic": "project"},
    )
    logger.info(f"   ✓ Created memory ID {memory2.id}: {memory2.title}\n")

    # Retrieve all memories
    logger.info("4. Retrieving all active memories...")
    all_memories = await memory.get_active_memories(external_id=agent_id)
    logger.info(f"   ✓ Found {len(all_memories)} active memories:")
    for i, mem in enumerate(all_memories, 1):
        logger.info(f"      {i}. [{mem.id}] {mem.title}")
        for section_id, section_data in mem.sections.items():
            update_count = section_data.get("update_count", 0)
            logger.info(f"         ↳ {section_id}: {update_count} updates")
    logger.info("")

    # Update a section
    logger.info("5. Updating 'status' section...")
    updated = await memory.update_active_memory_sections(
        external_id=agent_id,
        memory_id=memory1.id,
        sections=[
            {
                "section_id": "status",
                "new_content": "# Preferences\n- Strongly prefers Python and FastAPI\n"
                "- Working on AI agent project\n"
                "- Interested in memory management",
            }
        ],
    )
    logger.info(f"   ✓ Updated section 'status'")
    logger.info(f"   ✓ Update count: {updated.sections['status']['update_count']}\n")

    # Update again to see counter increment
    logger.info("6. Updating section again...")
    updated = await memory.update_active_memory_sections(
        external_id=agent_id,
        memory_id=memory1.id,
        sections=[
            {
                "section_id": "status",
                "new_content": "# Preferences\n- Strongly prefers Python and FastAPI\n"
                "- Working on AI agent project\n"
                "- Interested in memory management\n"
                "- Uses Docker for services",
            }
        ],
    )
    logger.info(f"   ✓ Update count incremented: {updated.sections['status']['update_count']}\n")

    # Get updated memories
    logger.info("7. Retrieving updated memories...")
    all_memories = await memory.get_active_memories(external_id=agent_id)
    for mem in all_memories:
        if mem.id == memory1.id:
            logger.info(f"   ✓ Memory [{mem.id}] {mem.title}:")
            for section_id, section_data in mem.sections.items():
                logger.info(f"      ↳ {section_id}: {section_data.get('update_count', 0)} updates")
    logger.info("")

    # Clean up
    logger.info("8. Closing connections...")
    await memory.close()
    logger.info("   ✓ Connections closed\n")

    logger.info("=" * 60)
    logger.info("✨ Simple test completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
