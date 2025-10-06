"""
Test the batch update_memory_sections functionality
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agent_mem.core import AgentMem


async def test_batch_update():
    """Test batch update functionality"""
    print("🧪 Testing Batch Update (update_memory_sections)")
    print("=" * 70)

    # Initialize AgentMem
    print("\n1️⃣  Initializing AgentMem...")
    agent_mem = AgentMem()
    await agent_mem.initialize()
    print("   ✅ AgentMem initialized!")

    external_id = "agent-mem-copilot"

    try:
        # Get existing memory
        print(f"\n2️⃣  Getting memory for '{external_id}'...")
        memories = await agent_mem.get_active_memories(external_id)

        if not memories:
            print("   ❌ No memories found! Run test_agent_mem_copilot.py first")
            return False

        memory = memories[0]
        memory_id = memory.id
        print(f"   ✅ Found memory ID: {memory_id}")
        print(f"      Title: {memory.title}")
        print(f"      Sections: {list(memory.sections.keys())}")

        # Get section names
        available_sections = list(memory.sections.keys())
        if len(available_sections) < 2:
            print("   ⚠️  Need at least 2 sections for batch test")
            return False

        # Prepare batch update (update first 3 sections)
        sections_to_update = available_sections[: min(3, len(available_sections))]

        print(f"\n3️⃣  Preparing batch update for {len(sections_to_update)} sections...")
        for section_id in sections_to_update:
            current_count = memory.sections[section_id].get("update_count", 0)
            print(f"      • {section_id}: current count = {current_count}")

        # Perform batch update
        print(f"\n4️⃣  Performing batch update...")

        updated_memory = memory
        update_results = []

        for section_id in sections_to_update:
            current_content = memory.sections[section_id].get("content", "")
            current_count = memory.sections[section_id].get("update_count", 0)

            new_content = (
                f"{current_content}\n\n**Batch Update Test:** Section updated via batch operation"
            )

            updated_memory = await agent_mem.update_active_memory_sections(
                external_id=external_id,
                memory_id=memory_id,
                sections=[{"section_id": section_id, "new_content": new_content}],
            )

            new_count = updated_memory.sections[section_id].get("update_count", 0)
            update_results.append(
                {
                    "section_id": section_id,
                    "previous_count": current_count,
                    "new_count": new_count,
                }
            )

        print(f"   ✅ Batch update completed!")
        print(f"\n   Update Results:")
        for result in update_results:
            print(
                f"      • {result['section_id']}: {result['previous_count']} → {result['new_count']}"
            )

        # Verify updates
        print(f"\n5️⃣  Verifying updates...")
        final_memories = await agent_mem.get_active_memories(external_id)
        final_memory = final_memories[0]

        print(f"   ✅ Verification complete!")
        for section_id in sections_to_update:
            count = final_memory.sections[section_id].get("update_count", 0)
            print(f"      • {section_id}: {count} updates")

        print("\n" + "=" * 70)
        print("✅ BATCH UPDATE TEST PASSED!")
        print("=" * 70)

        print("\n📋 Summary:")
        print(f"   • External ID: {external_id}")
        print(f"   • Memory ID: {memory_id}")
        print(f"   • Sections updated: {len(sections_to_update)}")
        print(f"   • Total update operations: {len(update_results)}")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_batch_update())
    sys.exit(0 if success else 1)
