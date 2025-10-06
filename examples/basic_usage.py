"""
Basic usage example for Agent Mem.

This example demonstrates:
1. Creating a stateless agent memory manager
2. Creating active memories with template-driven sections
3. Retrieving active memories for specific agents
4. Updating specific sections in active memories

Prerequisites:
- PostgreSQL with pgvector, pg_tokenizer, vchord_bm25 extensions
- Neo4j running
- Ollama running with nomic-embed-text model
- .env file configured with database credentials
"""

import asyncio
import logging
from agent_mem import AgentMem

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Example YAML template for task memory
TASK_TEMPLATE = """
template:
  id: "task_memory_v1"
  name: "Task Memory"
  version: "1.0.0"
  description: "Template for tracking task progress"
sections:
  - id: "current_task"
    title: "Current Task"
    description: "What is being worked on now"
  - id: "progress"
    title: "Progress"
    description: "Steps completed and in progress"
  - id: "blockers"
    title: "Blockers"
    description: "Issues preventing progress"
"""


async def main():
    """Run basic usage example."""

    logger.info("=== Agent Mem Basic Usage Example ===\n")

    # Initialize memory manager (STATELESS - serves multiple agents)
    agent_mem = AgentMem()

    try:
        # Initialize connections
        logger.info("Initializing Agent Mem...")
        await agent_mem.initialize()
        logger.info("✓ Initialized successfully\n")

        # Agent identifiers
        agent1_id = "example-agent-001"
        agent2_id = "example-agent-002"

        # 1. Create an active memory with template for Agent 1
        logger.info(f"1. Creating active memory for {agent1_id}...")
        memory1 = await agent_mem.create_active_memory(
            external_id=agent1_id,
            title="Build User Dashboard",
            template_content=TASK_TEMPLATE,
            initial_sections={
                "current_task": {
                    "content": "# Current Task\nImplementing real-time analytics dashboard",
                    "update_count": 0,
                },
                "progress": {
                    "content": "# Progress\n- Designed UI mockups\n- Set up React project",
                    "update_count": 0,
                },
                "blockers": {"content": "# Blockers\nNone at the moment", "update_count": 0},
            },
            metadata={"priority": "high", "client": "Acme Corp"},
        )
        logger.info(f"✓ Created memory ID {memory1.id} with {len(memory1.sections)} sections\n")

        # 2. Create another memory for Agent 2
        logger.info(f"2. Creating active memory for {agent2_id}...")
        memory2 = await agent_mem.create_active_memory(
            external_id=agent2_id,
            title="API Integration",
            template_content=TASK_TEMPLATE,
            initial_sections={
                "current_task": {
                    "content": "# Current Task\nIntegrating payment gateway API",
                    "update_count": 0,
                },
                "progress": {
                    "content": "# Progress\n- Read API documentation\n- Obtained test credentials",
                    "update_count": 0,
                },
                "blockers": {
                    "content": "# Blockers\nWaiting for production API keys",
                    "update_count": 0,
                },
            },
            metadata={"priority": "medium", "team": "Backend"},
        )
        logger.info(f"✓ Created memory ID {memory2.id} for {agent2_id}\n")

        # 3. Get all active memories for Agent 1
        logger.info(f"3. Retrieving all active memories for {agent1_id}...")
        agent1_memories = await agent_mem.get_active_memories(external_id=agent1_id)
        logger.info(f"✓ Retrieved {len(agent1_memories)} active memories:")
        for mem in agent1_memories:
            logger.info(f"   - [{mem.id}] {mem.title}")
            for section_id, section_data in mem.sections.items():
                logger.info(f"      ↳ {section_id}: {section_data.get('update_count', 0)} updates")
        logger.info("")

        # 4. Update a specific section
        logger.info("4. Updating 'progress' section...")
        updated = await agent_mem.update_active_memory_sections(
            external_id=agent1_id,
            memory_id=memory1.id,
            sections=[
                {
                    "section_id": "progress",
                    "new_content": "# Progress\n- Designed UI mockups\n- Set up React project\n"
                    "- Implemented chart components\n- Connected to WebSocket API",
                }
            ],
        )
        logger.info(
            f"✓ Updated section 'progress', update_count: {updated.sections['progress']['update_count']}\n"
        )

        # 5. Update multiple times to see counter increment
        logger.info("5. Updating 'progress' section multiple times...")
        for i in range(3):
            updated = await agent_mem.update_active_memory_sections(
                external_id=agent1_id,
                memory_id=memory1.id,
                sections=[
                    {
                        "section_id": "progress",
                        "new_content": f"# Progress\nUpdate iteration {i+2}: More progress...",
                    }
                ],
            )
            logger.info(
                f"   - Update {i+1}: update_count = {updated.sections['progress']['update_count']}"
            )
        logger.info("")

        # 6. Update a different section
        logger.info("6. Updating 'blockers' section...")
        updated = await agent_mem.update_active_memory_sections(
            external_id=agent1_id,
            memory_id=memory1.id,
            sections=[
                {
                    "section_id": "blockers",
                    "new_content": "# Blockers\n- Need dark mode design assets\n- Waiting for PDF export library approval",
                }
            ],
        )
        logger.info(
            f"✓ Updated section 'blockers', update_count: {updated.sections['blockers']['update_count']}\n"
        )

        # 7. Show section-level tracking
        logger.info("7. Current state of all sections:")
        final_state = await agent_mem.get_active_memories(external_id=agent1_id)
        for mem in final_state:
            logger.info(f"   Memory [{mem.id}] {mem.title}:")
            for section_id, section_data in mem.sections.items():
                count = section_data.get("update_count", 0)
                status = "⚠️  NEEDS CONSOLIDATION" if count >= 5 else "✓"
                logger.info(f"      {status} {section_id}: {count} updates")
        logger.info("")

        # 8. Try retrieval (stub for now)
        logger.info("8. Searching memories for Agent 1...")
        result = await agent_mem.retrieve_memories(
            external_id=agent1_id, query="What is the current status of the dashboard?"
        )
        logger.info(f"✓ Search result: {result.synthesized_response}")
        logger.info(f"   Active memories found: {len(result.active_memories)}")
        logger.info("")

        logger.info("=== Example Complete ===")
        logger.info("\nKey features demonstrated:")
        logger.info("✓ Stateless AgentMem serving multiple agents")
        logger.info("✓ Template-driven active memory with sections")
        logger.info("✓ Section-level update tracking")
        logger.info("✓ Per-section consolidation thresholds")
        logger.info("\nNext steps:")
        logger.info("- Implement shortterm/longterm repositories")
        logger.info("- Add Pydantic AI agents for intelligent consolidation")
        logger.info("- Implement hybrid search (vector + BM25)")
        logger.info("- Add entity/relationship extraction\n")
        logger.info(f"   Active memories found: {len(result.active_memories)}")
        logger.info("")

        logger.info("=== Example Complete ===")
        logger.info("\nNext steps:")
        logger.info("- Implement shortterm/longterm repositories")
        logger.info("- Add Pydantic AI agents for intelligent consolidation")
        logger.info("- Implement hybrid search (vector + BM25)")
        logger.info("- Add entity/relationship extraction\n")

    except Exception as e:
        logger.error(f"Error during example: {e}", exc_info=True)

    finally:
        # Clean up
        logger.info("Closing connections...")
        await agent_mem.close()
        logger.info("✓ Closed successfully")


if __name__ == "__main__":
    asyncio.run(main())
