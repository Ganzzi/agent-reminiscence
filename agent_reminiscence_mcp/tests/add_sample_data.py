"""
Add sample data to AgentMem for testing MCP server
Creates test active memories with sections
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agent_reminiscence.core import AgentMem

# Template for authentication project memory
AUTH_TEMPLATE = """
template:
  id: "auth_project_v1"
  name: "Authentication Project"
  version: "1.0.0"
  description: "Template for tracking authentication implementation"
sections:
  - id: "current_task"
    title: "Current Task"
    description: "What is being worked on now"
  - id: "learning_points"
    title: "Learning Points"
    description: "Key insights and learnings"
  - id: "next_steps"
    title: "Next Steps"
    description: "Upcoming tasks and action items"
"""


async def add_sample_data():
    """Add sample memories for testing MCP server"""
    print("🚀 Adding sample data to AgentMem...\n")
    print("=" * 60)

    # Initialize AgentMem
    print("\n1️⃣  Initializing AgentMem...")
    agent_mem = AgentMem()
    await agent_mem.initialize()
    print("   ✅ AgentMem initialized!")

    try:
        # Test agent ID
        external_id = "test-agent-123"

        # Check for existing memories
        print(f"\n2️⃣  Checking for existing memories for agent '{external_id}'...")
        existing_memories = await agent_mem.get_active_memories(external_id)

        if existing_memories:
            print(f"   ℹ️  Found {len(existing_memories)} existing memories")
            for mem in existing_memories:
                print(f"      • ID {mem.id}: {mem.title} ({len(mem.sections)} sections)")
            memory = existing_memories[0]
        else:
            # Create active memory with sections
            print(f"\n3️⃣  Creating new active memory...")

            initial_sections = {
                "current_task": {
                    "content": "# Current Task\n\nImplementing user authentication system with JWT tokens.\n\n**Goals:**\n- Token generation and validation\n- Secure password hashing with bcrypt\n- Token refresh mechanism\n- Rate limiting on login attempts",
                    "update_count": 0,
                },
                "learning_points": {
                    "content": "# Learning Points\n\n1. **Password Security**: Always hash passwords with bcrypt and salt. Never store plaintext.\n2. **JWT Best Practices**: Use strong secrets, keep tokens short-lived (15 min), implement refresh tokens.\n3. **Rate Limiting**: Critical for preventing brute force attacks. Redis is great for this.\n4. **HTTPS Only**: Authentication tokens must only be transmitted over HTTPS.",
                    "update_count": 0,
                },
                "next_steps": {
                    "content": "# Next Steps\n\n1. ✅ Set up JWT library (PyJWT)\n2. ✅ Create login endpoint (/api/auth/login)\n3. ⏳ Implement token refresh endpoint\n4. ⏳ Add rate limiting middleware\n5. ⏳ Write integration tests\n6. ⏳ Deploy to staging environment",
                    "update_count": 0,
                },
            }

            memory = await agent_mem.create_active_memory(
                external_id=external_id,
                title="Authentication Project Memory",
                template_content=AUTH_TEMPLATE,
                initial_sections=initial_sections,
                metadata={"priority": "high", "project": "auth-system"},
            )
            print(f"   ✅ Created active memory (ID: {memory.id})")
            print(f"   ✅ Added {len(memory.sections)} sections")

        # Display summary
        print("\n" + "=" * 60)
        print("✅ Sample Data Ready!\n")
        print("📊 Summary:")
        print(f"   • Agent ID: {external_id}")
        print(f"   • Active Memory ID: {memory.id}")
        print(f"   • Title: {memory.title}")
        print(f"   • Sections: {len(memory.sections)}")

        print("\n📝 Sections:")
        for section_id, section_data in memory.sections.items():
            print(
                f"   • {section_id}: {len(section_data.get('content', ''))} chars, {section_data.get('update_count', 0)} updates"
            )

        print("\n🧪 Test MCP Server Commands:")
        print(f"\n   1. Get active memories:")
        print(f'      {{"external_id": "{external_id}"}}')

        print(f"\n   2. Update a section:")
        print(
            f'      {{"external_id": "{external_id}", "memory_id": {memory.id}, "section_id": "current_task", "new_content": "Updated: Now working on token refresh..."}}'
        )

        print(f"\n   3. Search memories:")
        print(
            f'      {{"external_id": "{external_id}", "query": "JWT authentication best practices", "limit": 5}}'
        )

        print("\n🚀 Run MCP client test:")
        print("   py test_mcp_client.py")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("\n4️⃣  Closing AgentMem...")
        await agent_mem.close()
        print("   ✅ Closed!")


if __name__ == "__main__":
    asyncio.run(add_sample_data())


