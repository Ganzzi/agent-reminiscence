"""
Test AgentMem MCP Server with agent-mem-copilot external_id
Tests all 3 tools: get_active_memories, update_memory_section, search_memories
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agent_reminiscence.core import AgentMem


# Test template for copilot memory
COPILOT_TEMPLATE = {
    "template": {
        "id": "copilot_session_v1",
        "name": "Copilot Session Memory",
        "version": "1.0.0",
        "description": "Template for tracking GitHub Copilot session context"
    },
    "sections": [
        {
            "id": "current_context",
            "title": "Current Context",
            "description": "What the developer is working on"
        },
        {
            "id": "code_patterns",
            "title": "Code Patterns",
            "description": "Patterns and conventions observed"
        },
        {
            "id": "session_notes",
            "title": "Session Notes",
            "description": "Important notes and decisions"
        }
    ]
}


async def test_mcp_server():
    """Test all MCP server tools with agent-mem-copilot"""
    print("🧪 Testing AgentMem MCP Server")
    print("=" * 70)

    # Initialize AgentMem
    print("\n1️⃣  Initializing AgentMem...")
    agent_mem = AgentMem()
    await agent_mem.initialize()
    print("   ✅ AgentMem initialized!")

    external_id = "agent-mem-copilot"

    try:
        # Test 1: Get active memories (should be empty initially)
        print(f"\n2️⃣  Testing get_active_memories for '{external_id}'...")
        memories = await agent_mem.get_active_memories(external_id)
        print(f"   ✅ Found {len(memories)} existing memories")

        if memories:
            for mem in memories:
                print(f"      • ID {mem.id}: {mem.title} ({len(mem.sections)} sections)")

        # Test 2: Create a new memory if none exists
        if not memories:
            print(f"\n3️⃣  Creating new memory for '{external_id}'...")

            initial_sections = {
                "current_context": {
                    "content": "# Current Context\n\nWorking on AgentMem MCP server implementation.\n\n**Current Task:**\n- Testing MCP server with agent-mem-copilot external ID\n- Implementing logging system\n- Preparing for batch section updates",
                    "update_count": 0,
                    "awake_update_count": 0,
                    "last_updated": "2024-01-01T00:00:00Z",
                },
                "code_patterns": {
                    "content": "# Code Patterns\n\n1. **Logging**: Using Python's logging module with both file and stderr handlers\n2. **Async/Await**: All database operations are async\n3. **Error Handling**: Try-except with detailed logging and stack traces\n4. **Type Hints**: Full type annotations for better IDE support",
                    "update_count": 0,
                    "awake_update_count": 0,
                    "last_updated": "2024-01-01T00:00:00Z",
                },
                "session_notes": {
                    "content": "# Session Notes\n\n- Successfully set up logging with timestamps\n- Server logs to `agent_mem_mcp/logs/` directory\n- Using `uv` for dependency management\n- All 3 Docker services running (PostgreSQL, Neo4j, Ollama)",
                    "update_count": 0,
                    "awake_update_count": 0,
                    "last_updated": "2024-01-01T00:00:00Z",
                },
            }

            memory = await agent_mem.create_active_memory(
                external_id=external_id,
                title="Copilot Session - MCP Server Testing",
                template_content=COPILOT_TEMPLATE,
                initial_sections=initial_sections,
                metadata={"session_type": "development", "project": "agent-mem-mcp"},
            )
            print(f"   ✅ Created memory ID: {memory.id}")
            print(f"      Title: {memory.title}")
            print(f"      Sections: {list(memory.sections.keys())}")
            memory_id = memory.id
        else:
            memory = memories[0]
            memory_id = memory.id
            print(f"   ℹ️  Using existing memory ID: {memory_id}")

        # Test 3: Update a memory section
        print(f"\n4️⃣  Testing update_memory_section...")

        # Get the first available section
        available_sections = list(memory.sections.keys())
        if not available_sections:
            print("   ❌ No sections available in memory!")
            return False

        section_to_update = available_sections[0]
        print(f"      Updating '{section_to_update}' section...")

        # Get current content
        current_content = memory.sections[section_to_update].get("content", "")
        current_count = memory.sections[section_to_update].get("update_count", 0)

        # Update with new content
        new_content = f"{current_content}\n\n**Update:** Testing MCP server with agent-mem-copilot - {current_count + 1} update(s)"

        updated_memory = await agent_mem.update_active_memory_sections(
            external_id=external_id,
            memory_id=memory_id,
            sections=[{"section_id": section_to_update, "new_content": new_content}],
        )

        new_count = updated_memory.sections[section_to_update].get("update_count", 0)
        print(f"   ✅ Section updated! Update count: {current_count} → {new_count}")

        # Test 4: Update another section
        print(f"\n5️⃣  Testing another section update...")
        if len(available_sections) > 1:
            section_to_update_2 = available_sections[1]
            print(f"      Updating '{section_to_update_2}' section...")

            current_content_2 = memory.sections[section_to_update_2].get("content", "")
            current_count_2 = memory.sections[section_to_update_2].get("update_count", 0)

            new_content_2 = f"{current_content_2}\n\n**Update:** Second section test - {current_count_2 + 1} update(s)"

            updated_memory = await agent_mem.update_active_memory_sections(
                external_id=external_id,
                memory_id=memory_id,
                sections=[{"section_id": section_to_update_2, "new_content": new_content_2}],
            )

            new_count = updated_memory.sections[section_to_update_2].get("update_count", 0)
            print(f"   ✅ Section updated! Update count: {current_count_2} → {new_count}")
        else:
            print(f"   ⏭️  Skipping (only one section available)")

        # Test 5: Search memories
        print(f"\n6️⃣  Testing search_memories...")
        print(f"      Query: 'logging implementation details'")

        result = await agent_mem.retrieve_memories(
            external_id=external_id,
            query="logging implementation details",
            limit=5,
        )

        print(f"   ✅ Search completed!")
        print(f"      Mode: {result.mode}")
        print(f"      Strategy: {result.search_strategy}")
        print(f"      Confidence: {result.confidence:.2f}")
        print(f"      Chunks: {len(result.chunks)}")
        print(f"      Entities: {len(result.entities)}")
        print(f"      Relationships: {len(result.relationships)}")

        if result.synthesis:
            print(f"\n      Synthesis:")
            print(f"      {result.synthesis[:200]}...")

        # Test 6: Create a new memory explicitly (test create_active_memory)
        print(f"\n7️⃣  Testing create_active_memory explicitly...")
        new_memory_title = "Test Memory - Explicit Create"
        
        new_template = {
            "template": {"id": "test_memory_v1", "name": "Test Memory"},
            "sections": [{"id": "test_section", "description": "A test section"}]
        }
        
        new_initial_sections = {
            "test_section": {
                "content": "# Test Section\n\nThis is a test section created via explicit create_active_memory call.",
                "update_count": 0,
                "awake_update_count": 0,
                "last_updated": None
            }
        }
        
        new_memory = await agent_mem.create_active_memory(
            external_id=external_id,
            title=new_memory_title,
            template_content=new_template,
            initial_sections=new_initial_sections,
            metadata={"test_type": "explicit_create", "created_by": "test_agent_mem_copilot.py"},
        )
        print(f"   ✅ Created new memory ID: {new_memory.id}")
        print(f"      Title: {new_memory.title}")
        print(f"      Sections: {list(new_memory.sections.keys())}")
        new_memory_id = new_memory.id

        # Test 7: Delete the newly created memory (test delete_active_memory)
        print(f"\n8️⃣  Testing delete_active_memory...")
        delete_success = await agent_mem.delete_active_memory(
            external_id=external_id,
            memory_id=new_memory_id,
        )
        
        if delete_success:
            print(f"   ✅ Successfully deleted memory {new_memory_id}")
        else:
            print(f"   ❌ Failed to delete memory {new_memory_id}")
            return False

        # Verify deletion
        memories_after_delete = await agent_mem.get_active_memories(external_id)
        deleted_memory = next((m for m in memories_after_delete if m.id == new_memory_id), None)
        if deleted_memory:
            print(f"   ❌ Memory {new_memory_id} still exists after deletion!")
            return False
        else:
            print(f"   ✅ Confirmed memory {new_memory_id} was deleted")

        # Final verification
        print(f"\n9️⃣  Final verification - Getting all memories...")
        final_memories = await agent_mem.get_active_memories(external_id)
        print(f"   ✅ Final memory count: {len(final_memories)}")

        for mem in final_memories:
            print(f"\n      Memory ID: {mem.id}")
            print(f"      Title: {mem.title}")
            print(f"      Sections:")
            for section_id, section_data in mem.sections.items():
                count = section_data.get("update_count", 0)
                awake_count = section_data.get("awake_update_count", 0)
                content_preview = section_data.get("content", "")[:50]
                print(f"         • {section_id}: {count} updates, {awake_count} awake updates - {content_preview}...")

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED! MCP Server is ready!")
        print("=" * 70)

        print("\n📋 Summary:")
        print(f"   • External ID: {external_id}")
        print(f"   • Memory ID: {memory_id}")
        print(f"   • Total sections: {len(memory.sections)}")
        print(f"   • get_active_memories: ✅ Working")
        print(f"   • create_active_memory: ✅ Working")
        print(f"   • update_memory_section: ✅ Working")
        print(f"   • delete_active_memory: ✅ Working")
        print(f"   • search_memories: ✅ Working")

        print("\n🎯 Next Step: Test batch update_memory_sections")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = asyncio.run(test_mcp_server())
    sys.exit(0 if success else 1)


