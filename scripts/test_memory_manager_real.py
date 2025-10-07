"""
Real Memory Manager Integration Test Script

This script tests the MemoryManager's consolidation and promotion workflows
with real databases, AI agents, and API keys.

Tests:
1. update_active_memory_sections (basic & with consolidation trigger)
2. _consolidate_with_lock method
3. _consolidate_to_shortterm with real memorizer agent
4. _promote_to_longterm with real database o                for i, chunk in enumerate(result.shortterm_chunks[:5], 1):
                    print(f"   {i}. {chunk.content[:100]}...")
            else:
                print("⚠️ No shortterm chunks found")

            # 8. Clean up
            print("\n🗑️ Cleaning up...")
            deleted = await memory_manager.delete_active_memory(external_id, memory.id)
            print(f"✅ Deleted active memory: {deleted}")ns

Usage:
    python scripts/test_memory_manager_real.py                    # Run all tests
    python scripts/test_memory_manager_real.py basic             # Run basic operations test
    python scripts/test_memory_manager_real.py lock              # Run consolidate with lock test
    python scripts/test_memory_manager_real.py consolidation     # Run consolidation with memorizer test
    python scripts/test_memory_manager_real.py promotion         # Run promotion to longterm test
    python scripts/test_memory_manager_real.py basic lock        # Run multiple specific tests

Available test aliases:
    - basic, basic_operations: Basic Memory Operations
    - lock, consolidate_lock: Consolidate with Lock
    - consolidation, memorizer: Consolidation with Memorizer
    - promotion, longterm: Promotion to Longterm

Requirements:
    - Set GOOGLE_API_KEY (or other provider API keys) in your .env file
    - PostgreSQL and Neo4j databases running (or use test databases)
    - Ensure embedding service is configured and accessible
"""

import asyncio
import logging
import sys
import uuid
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_mem.services.memory_manager import MemoryManager
from agent_mem.config.settings import get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =========================================================================
# TEST DATA GENERATORS
# =========================================================================


def generate_conversation_template():
    """Generate a realistic conversation template."""
    return """
conversation_memory:
  description: "Tracks ongoing conversation context and progress"
  sections:
    - id: summary
      description: "High-level summary of conversation"
    - id: context
      description: "Important context and background"
    - id: decisions
      description: "Key decisions made"
    - id: next_steps
      description: "Planned next actions"
    - id: participants
      description: "People involved in conversation"
""".strip()


def generate_sample_sections(update_counts=None):
    """Generate realistic conversation sections with specified update counts."""
    if update_counts is None:
        update_counts = {
            "summary": 1,
            "context": 1,
            "decisions": 0,
            "next_steps": 1,
            "participants": 0,
        }

    return {
        "summary": {
            "content": "Discussing implementation of memory consolidation feature for AI agent system. Focus on ensuring data consistency and performance optimization.",
            "update_count": update_counts.get("summary", 1),
        },
        "context": {
            "content": "Working on agent_mem library that handles hierarchical memory (active → shortterm → longterm). Using PostgreSQL for structured data and Neo4j for relationships.",
            "update_count": update_counts.get("context", 1),
        },
        "decisions": {
            "content": "Decided to use Pydantic AI agents for conflict resolution during consolidation. Will use Gemini model for memorizer agent.",
            "update_count": update_counts.get("decisions", 0),
        },
        "next_steps": {
            "content": "1. Test consolidation workflow with real data 2. Verify memorizer agent performance 3. Add comprehensive error handling",
            "update_count": update_counts.get("next_steps", 1),
        },
        "participants": {
            "content": "Primary developer working on memory system implementation. Collaborating with team on API design.",
            "update_count": update_counts.get("participants", 0),
        },
    }


def generate_section_updates():
    """Generate realistic section updates to trigger consolidation."""
    return [
        {
            "section_id": "summary",
            "new_content": "Discussing memory consolidation implementation. Added real integration tests to verify AI agent functionality with Gemini model.",
        },
        {
            "section_id": "context",
            "new_content": "Working on agent_mem library with hierarchical memory system. Using PostgreSQL + Neo4j. Recently added _consolidate_with_lock for concurrent safety.",
        },
        {
            "section_id": "next_steps",
            "new_content": "1. Run integration tests with real databases 2. Test memorizer agent conflict resolution 3. Verify promotion to longterm memory 4. Add performance monitoring",
        },
    ]


# =========================================================================
# TEST 1: BASIC MEMORY OPERATIONS
# =========================================================================


async def test_basic_memory_operations():
    """Test basic active memory operations without triggering consolidation."""
    print("\n" + "=" * 70)
    print("TEST 1: BASIC MEMORY OPERATIONS")
    print("=" * 70)

    try:
        config = get_config()

        # Set high consolidation threshold to prevent triggering
        config.avg_section_update_count_for_consolidation = 10.0

        print(
            f"📊 Consolidation threshold: {config.avg_section_update_count_for_consolidation} per section"
        )

        memory_manager = MemoryManager(config=config)
        await memory_manager.initialize()

        try:
            external_id = f"test-basic-{uuid.uuid4().hex[:8]}"

            print(f"\n🆔 Agent ID: {external_id}")

            # 1. Create active memory
            print("\n🔄 Creating active memory...")
            memory = await memory_manager.create_active_memory(
                external_id=external_id,
                title="Test Conversation - Basic Operations",
                template_content=generate_conversation_template(),
                initial_sections=generate_sample_sections(),
                metadata={"test_type": "basic_operations", "created_by": "integration_test"},
            )

            print(f"✅ Created memory {memory.id}")
            print(f"   - Sections: {len(memory.sections)}")
            print(
                f"   - Total update count: {sum(s.get('update_count', 0) for s in memory.sections.values())}"
            )

            # 2. Update sections (should not trigger consolidation)
            print("\n🔄 Updating memory sections...")
            updated_memory = await memory_manager.update_active_memory_sections(
                external_id=external_id,
                memory_id=memory.id,
                sections=[
                    {
                        "section_id": "summary",
                        "new_content": "Updated summary content for basic test",
                    },
                    {
                        "section_id": "context",
                        "new_content": "Updated context with additional details",
                    },
                ],
            )

            print(f"✅ Updated memory {updated_memory.id}")
            print(
                f"   - Total update count: {sum(s.get('update_count', 0) for s in updated_memory.sections.values())}"
            )

            # 3. Get all memories
            print("\n🔄 Retrieving all memories...")
            all_memories = await memory_manager.get_active_memories(external_id)
            print(f"✅ Retrieved {len(all_memories)} memories")

            # 4. Clean up
            print("\n🗑️ Cleaning up...")
            deleted = await memory_manager.delete_active_memory(external_id, memory.id)
            print(f"✅ Deleted: {deleted}")

        finally:
            await memory_manager.close()

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Basic memory operations test failed")
        return False


# =========================================================================
# TEST 1.5: CONSOLIDATE WITH LOCK
# =========================================================================
async def test_consolidate_with_lock():
    """Test the _consolidate_with_lock method with concurrent calls."""
    print("\n" + "=" * 70)
    print("TEST 1.5: CONSOLIDATE WITH LOCK")
    print("=" * 70)

    try:
        config = get_config()
        config.avg_section_update_count_for_consolidation = 2.0

        memory_manager = MemoryManager(config=config)
        await memory_manager.initialize()

        try:
            external_id = f"test-lock-{uuid.uuid4().hex[:8]}"

            # Create active memory with high update counts
            initial_sections = generate_sample_sections(
                {"summary": 2, "context": 2, "decisions": 1, "next_steps": 2, "participants": 1}
            )
            memory = await memory_manager.create_active_memory(
                external_id=external_id,
                title="Test Conversation - Lock Test",
                template_content=generate_conversation_template(),
                initial_sections=initial_sections,
                metadata={"test_type": "lock_test", "created_by": "integration_test"},
            )

            # Concurrently call _consolidate_with_lock
            task1 = asyncio.create_task(
                memory_manager._consolidate_with_lock(external_id, memory.id)
            )
            task2 = asyncio.create_task(
                memory_manager._consolidate_with_lock(external_id, memory.id)
            )
            await asyncio.gather(task1, task2)

            # Clean up
            await memory_manager.delete_active_memory(external_id, memory.id)

        finally:
            await memory_manager.close()

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("_consolidate_with_lock test failed")
        return False


# =========================================================================
# TEST 2: CONSOLIDATION WITH MEMORIZER AGENT
async def test_consolidation_workflow():
    """Test consolidation workflow with real memorizer agent.

    This test directly calls _consolidate_to_shortterm twice to properly test
    the memorizer agent:
    1. First call creates initial chunks (no conflicts)
    2. Update sections with new content
    3. Second call finds existing chunks and triggers memorizer agent (with conflicts!)

    Note: We call _consolidate_to_shortterm directly instead of relying on
    automatic triggering because the lock mechanism would prevent the second
    consolidation from running while the first is in progress.
    """
    print("\n" + "=" * 70)
    print("TEST 2: CONSOLIDATION WITH MEMORIZER AGENT")
    print("=" * 70)

    try:
        config = get_config()

        print(f"📊 Memorizer Model: {config.memorizer_agent_model}")

        memory_manager = MemoryManager(config=config)
        await memory_manager.initialize()

        try:
            external_id = f"test-consolidation-{uuid.uuid4().hex[:8]}"

            print(f"\n🆔 Agent ID: {external_id}")

            # 1. Create active memory with high update counts
            print("\n🔄 Step 1: Creating active memory...")
            initial_sections = generate_sample_sections(
                {"summary": 2, "context": 2, "decisions": 1, "next_steps": 2, "participants": 1}
            )

            memory = await memory_manager.create_active_memory(
                external_id=external_id,
                title="Test Conversation - Consolidation Test",
                template_content=generate_conversation_template(),
                initial_sections=initial_sections,
                metadata={"test_type": "consolidation_test", "created_by": "integration_test"},
            )

            print(f"✅ Created memory {memory.id} with {len(memory.sections)} sections")
            print(f"   All sections have update_count > 0 (ready for consolidation)")

            # 2. FIRST consolidation - creates initial chunks (NO conflicts)
            print("\n🔄 Step 2: First consolidation (creates initial chunks)...")
            print("   - No existing chunks → No conflicts")
            print("   - Memorizer agent should NOT be called")

            await memory_manager._consolidate_to_shortterm(external_id, memory.id)

            print("✅ First consolidation complete")

            # 3. Verify shortterm chunks were created
            print("\n🔍 Step 3: Verifying initial shortterm chunks...")
            result = await memory_manager._retrieve_memories_basic(
                external_id=external_id,
                query="consolidation test conversation memory",
                search_shortterm=True,
                search_longterm=False,
                limit=10,
            )

            if result.shortterm_chunks:
                print(f"✅ Created {len(result.shortterm_chunks)} initial chunks")
                for i, chunk in enumerate(result.shortterm_chunks[:3], 1):
                    print(f"   {i}. [{chunk.section_id}] {chunk.content[:70]}...")
            else:
                print("❌ ERROR: No chunks created!")
                return False

            # 4. Update sections with NEW content (creates update_count > 0 again)
            print("\n🔄 Step 4: Updating sections with new content...")
            conflict_updates = [
                {
                    "section_id": "summary",
                    "new_content": "UPDATED VERSION: Memory consolidation now includes advanced conflict resolution. The memorizer agent intelligently merges conflicting chunks using Gemini AI for semantic understanding and deduplication.",
                },
                {
                    "section_id": "context",
                    "new_content": "UPDATED VERSION: agent_mem library uses PostgreSQL for structured data and Neo4j for knowledge graphs. Background consolidation tasks handle async processing. Conflicts are detected when sections are updated multiple times.",
                },
                {
                    "section_id": "next_steps",
                    "new_content": "UPDATED VERSION: 1. Verify memorizer agent is properly invoked 2. Test AI-based conflict resolution 3. Check entity/relationship deduplication 4. Validate performance with real Gemini API calls 5. Monitor memory usage",
                },
            ]

            updated_memory = await memory_manager.update_active_memory_sections(
                external_id=external_id, memory_id=memory.id, sections=conflict_updates
            )

            print(f"✅ Updated {len(conflict_updates)} sections")
            print("   - Sections now have update_count > 0 again")
            print("   - BUT chunks already exist from first consolidation → CONFLICTS!")

            # 5. SECOND consolidation - finds conflicts, calls memorizer agent!
            print("\n🔄 Step 5: Second consolidation (WITH CONFLICTS)...")
            print("   ⚠️  CRITICAL: This should trigger the memorizer agent!")
            print("   - Updated sections have existing chunks → CONFLICTS DETECTED")
            print("   - Memorizer agent should resolve conflicts using Gemini AI")
            print("   - This may take 15-30 seconds due to AI processing...")
            print("\n   Watch the logs for:")
            print("   • 'Conflict detected for section' messages")
            print("   • 'Detected X total conflicts'")
            print("   • 'Memorizer agent resolution' messages")

            await memory_manager._consolidate_to_shortterm(external_id, memory.id)

            print("\n✅ Second consolidation complete!")
            print("   Check logs above to verify memorizer agent was called")

            # 6. Verify final results
            print("\n🔍 Step 6: Checking final shortterm memory...")
            result = await memory_manager._retrieve_memories_basic(
                external_id=external_id,
                query="consolidation conflict resolution memorizer advanced",
                search_shortterm=True,
                search_longterm=False,
                limit=10,
            )

            if result.shortterm_chunks:
                print(f"\n📝 Final shortterm chunks ({len(result.shortterm_chunks)} total):")
                for i, chunk in enumerate(result.shortterm_chunks[:5], 1):
                    print(f"   {i}. [{chunk.section_id}] {chunk.content[:70]}...")
            else:
                print("⚠️ No shortterm chunks found")

            # 7. Clean up
            print("\n🗑️ Step 7: Cleaning up...")
            deleted = await memory_manager.delete_active_memory(external_id, memory.id)
            print(f"✅ Deleted active memory: {deleted}")

        finally:
            await memory_manager.close()

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Consolidation test failed")
        return False


async def test_promotion_workflow():
    """Test promotion to longterm memory."""
    print("\n" + "=" * 70)
    print("TEST 3: PROMOTION TO LONGTERM MEMORY")
    print("=" * 70)

    try:
        config = get_config()

        # Configure for faster promotion
        config.avg_section_update_count_for_consolidation = 2.0
        # Note: Promotion threshold is handled internally by shortterm memory update count

        print(
            f"📊 Consolidation threshold: {config.avg_section_update_count_for_consolidation} per section"
        )

        memory_manager = MemoryManager(config=config)
        await memory_manager.initialize()

        try:
            external_id = f"test-promotion-{uuid.uuid4().hex[:8]}"
            # This test is more complex as it requires:
            # 1. Creating active memory
            # 2. Triggering consolidation multiple times
            # 3. Eventually triggering promotion
            # For now, we'll test the promotion mechanism indirectly

            print("\n🔄 Creating memory for promotion test...")
            memory = await memory_manager.create_active_memory(
                external_id=external_id,
                title="Test Conversation - Promotion Test",
                template_content=generate_conversation_template(),
                initial_sections=generate_sample_sections(
                    {"summary": 2, "context": 2, "decisions": 2, "next_steps": 2, "participants": 2}
                ),
                metadata={"test_type": "promotion_test", "created_by": "integration_test"},
            )

            print(f"✅ Created memory {memory.id}")

            # Trigger multiple consolidations to eventually cause promotion
            for i in range(3):
                print(f"\n🔄 Consolidation round {i+1}/3...")
                await memory_manager.update_active_memory_sections(
                    external_id=external_id,
                    memory_id=memory.id,
                    sections=[
                        {
                            "section_id": "summary",
                            "new_content": f"Updated summary round {i+1} - testing promotion workflow",
                        }
                    ],
                )

                # Wait for consolidation
                await asyncio.sleep(5)

            # Wait for all background consolidations to complete
            await asyncio.sleep(20)

            # Check longterm memory
            print("\n🔍 Checking for longterm memory...")
            result = await memory_manager._retrieve_memories_basic(
                external_id=external_id,
                query="promotion test conversation",
                search_shortterm=False,
                search_longterm=True,
                limit=5,
            )
            if result.longterm_chunks:
                print("\n📝 Sample longterm chunks:")
                for i, chunk in enumerate(result.longterm_chunks[:2], 1):
                    print(f"   {i}. {chunk.content[:100]}...")

            # Clean up
            print("\n🗑️ Cleaning up...")
            deleted = await memory_manager.delete_active_memory(external_id, memory.id)
            print(f"✅ Deleted active memory: {deleted}")

        finally:
            await memory_manager.close()

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Promotion test failed")
        return False


# =========================================================================
# MAIN TEST RUNNER
# =========================================================================


async def main(test_names=None):
    """Run all memory manager integration tests or specific tests.

    Args:
        test_names: List of test names to run. If None, runs all tests.
                   Available tests: basic, lock, consolidation, promotion
    """
    print("🚀 MEMORY MANAGER INTEGRATION TESTS")
    print("=" * 70)

    # Verify configuration
    config = get_config()
    print(f"\n📋 Configuration:")
    print(f"   - Database: {config.postgres_db}")
    print(f"   - Neo4j URI: {config.neo4j_uri}")
    print(f"   - Embedding Model: {config.embedding_model}")
    print(f"   - Memorizer Model: {config.memorizer_agent_model}")

    # Define all available tests
    all_tests = [
        ("Basic Memory Operations", test_basic_memory_operations, ["basic", "basic_operations"]),
        ("Consolidate with Lock", test_consolidate_with_lock, ["lock", "consolidate_lock"]),
        (
            "Consolidation with Memorizer",
            test_consolidation_workflow,
            ["consolidation", "memorizer"],
        ),
        ("Promotion to Longterm", test_promotion_workflow, ["promotion", "longterm"]),
    ]

    # Filter tests based on test_names
    if test_names:
        print(f"\n🎯 Running specific tests: {', '.join(test_names)}")
        tests_to_run = []
        for test_name, test_func, aliases in all_tests:
            if any(alias in test_names for alias in aliases):
                tests_to_run.append((test_name, test_func))
    else:
        print(f"\n🎯 Running all tests")
        tests_to_run = [(name, func) for name, func, _ in all_tests]

    if not tests_to_run:
        print("❌ No matching tests found!")
        return False

    results = []

    # Run tests
    for test_name, test_func in tests_to_run:
        try:
            print(f"\n🧪 Running: {test_name}")
            success = await test_func()
            results.append((test_name, success))

            if success:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")

        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            logger.exception(f"Test '{test_name}' failed with exception")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {test_name}")

    print(f"\n🎯 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests PASSED!")
        return True
    else:
        print("⚠️  Some tests FAILED. Check logs for details.")
        return False


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    if len(sys.argv) > 1:
        # Run specific tests
        test_names = sys.argv[1:]
        print(f"Running specific tests: {', '.join(test_names)}")
        success = asyncio.run(main(test_names))
    else:
        # Run all tests
        print("Running all tests")
        success = asyncio.run(main())

    # Exit with appropriate code
    sys.exit(0 if success else 1)
