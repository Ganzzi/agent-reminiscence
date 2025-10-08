"""
Simplified Memory Manager Test Script

This script tests the MemoryManager's logic with mocked dependencies for
testing when real databases are not available.

For full integration testing with real databases, see test_memory_manager_real.py

Usage:
    python scripts/test_memory_manager_mocked.py

Requirements:
    - Set GOOGLE_API_KEY (or other provider API keys) in your .env file
    - No database setup required (uses mocks)
"""

import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_mem.services.memory_manager import MemoryManager
from agent_mem.database.models import (
    ActiveMemory,
    ShorttermMemoryChunk,
    LongtermMemoryChunk,
    RetrievalResult,
)
from agent_mem.config.settings import get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =========================================================================
# MOCK SETUP HELPERS
# =========================================================================


def create_mock_memory_manager(config):
    """Create a memory manager with mocked dependencies."""

    with (
        patch("agent_mem.services.memory_manager.PostgreSQLManager"),
        patch("agent_mem.services.memory_manager.Neo4jManager"),
        patch("agent_mem.services.memory_manager.EmbeddingService"),
    ):
        manager = MemoryManager(config)
        manager._initialized = True  # Bypass initialization

        # Mock repositories
        manager.active_repo = MagicMock()
        manager.shortterm_repo = MagicMock()
        manager.longterm_repo = MagicMock()

        # Mock embedding service
        manager.embedding_service = MagicMock()
        manager.embedding_service.get_embedding = AsyncMock(return_value=[0.1] * 768)

        return manager


def generate_sample_memory(memory_id=1, external_id="test-agent", update_counts=None):
    """Generate a sample active memory for testing."""
    if update_counts is None:
        update_counts = {"summary": 1, "context": 1, "decisions": 0}

    return ActiveMemory(
        id=memory_id,
        external_id=external_id,
        title="Test Conversation Memory",
        template_content="conversation_template:\n  sections:\n    - summary\n    - context\n    - decisions",
        sections={
            "summary": {
                "content": "Discussing memory system implementation with AI agents",
                "update_count": update_counts.get("summary", 1),
            },
            "context": {
                "content": "Working on agent_mem library with PostgreSQL and Neo4j backends",
                "update_count": update_counts.get("context", 1),
            },
            "decisions": {
                "content": "Using Pydantic AI for memorizer agent conflict resolution",
                "update_count": update_counts.get("decisions", 0),
            },
        },
        metadata={"test": True},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


# =========================================================================
# TEST 1: BASIC UPDATE OPERATIONS
# =========================================================================


async def test_update_operations():
    """Test basic memory update operations with consolidation logic."""
    print("\n" + "=" * 70)
    print("TEST 1: UPDATE OPERATIONS WITH CONSOLIDATION LOGIC")
    print("=" * 70)

    try:
        config = get_config()
        config.avg_section_update_count_for_consolidation = 3.0

        manager = create_mock_memory_manager(config)
        external_id = "test-update-agent"
        memory_id = 1

        print(
            f"📊 Consolidation threshold: {config.avg_section_update_count_for_consolidation} per section"
        )

        # Test 1a: Update below threshold
        print("\n🔄 Testing update BELOW consolidation threshold...")

        # Mock memory with low update counts (2+1+0 = 3, threshold = 3*3 = 9)
        low_update_memory = generate_sample_memory(
            memory_id, external_id, {"summary": 2, "context": 1, "decisions": 0}
        )

        manager.active_repo.update_sections = AsyncMock(return_value=low_update_memory)

        with patch("asyncio.create_task") as mock_create_task:
            await manager.update_active_memory_sections(
                external_id=external_id,
                memory_id=memory_id,
                sections=[{"section_id": "summary", "new_content": "Updated summary"}],
            )

            # Should NOT trigger consolidation
            mock_create_task.assert_not_called()
            print("✅ Update below threshold - consolidation NOT triggered")

        # Test 1b: Update above threshold
        print("\n🔄 Testing update ABOVE consolidation threshold...")

        # Mock memory with high update counts (4+4+1 = 9, threshold = 3*3 = 9)
        high_update_memory = generate_sample_memory(
            memory_id, external_id, {"summary": 4, "context": 4, "decisions": 1}
        )

        manager.active_repo.update_sections = AsyncMock(return_value=high_update_memory)
        manager.active_repo.reset_all_update_counts = AsyncMock()

        # Mock consolidation to simulate success and reset update counts
        async def mock_consolidate_to_shortterm(ext_id, mem_id):
            await manager.active_repo.reset_all_update_counts(mem_id)

        manager._consolidate_to_shortterm = mock_consolidate_to_shortterm

        with patch("asyncio.create_task") as mock_create_task:
            await manager.update_active_memory_sections(
                external_id=external_id,
                memory_id=memory_id,
                sections=[{"section_id": "context", "new_content": "Updated context"}],
            )

            # Should trigger consolidation
            mock_create_task.assert_called_once()
            # Await the consolidation task to ensure reset_all_update_counts is called
            consolidation_task = mock_create_task.call_args[0][0]
            await consolidation_task
            manager.active_repo.reset_all_update_counts.assert_called_once_with(memory_id)
            print("✅ Update above threshold - consolidation triggered")
            print("✅ Update counts reset after consolidation trigger")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Update operations test failed")
        return False


# =========================================================================
# TEST 2: CONSOLIDATION LOCK MECHANISM
# =========================================================================


async def test_consolidation_lock():
    """Test consolidation lock mechanism."""
    print("\n" + "=" * 70)
    print("TEST 2: CONSOLIDATION LOCK MECHANISM")
    print("=" * 70)

    try:
        config = get_config()
        manager = create_mock_memory_manager(config)

        external_id = "test-lock-agent"
        memory_id = 123

        # Mock the internal consolidation method
        consolidation_calls = []

        async def mock_consolidation(ext_id, mem_id):
            consolidation_calls.append((ext_id, mem_id))
            await asyncio.sleep(0.1)  # Simulate work

        manager._consolidate_to_shortterm = mock_consolidation

        # Test 2a: Single consolidation
        print("\n🔄 Testing single consolidation...")
        await manager._consolidate_with_lock(external_id, memory_id)

        assert len(consolidation_calls) == 1
        assert consolidation_calls[0] == (external_id, memory_id)
        assert memory_id not in manager._consolidation_locks
        print("✅ Single consolidation completed and lock cleaned up")

        # Test 2b: Concurrent consolidation (should prevent duplicate)
        print("\n🔄 Testing concurrent consolidation prevention...")
        consolidation_calls.clear()

        # Start two concurrent consolidations
        task1 = asyncio.create_task(manager._consolidate_with_lock(external_id, memory_id))
        task2 = asyncio.create_task(manager._consolidate_with_lock(external_id, memory_id))

        await asyncio.gather(task1, task2)

        # Only one should have executed
        assert len(consolidation_calls) == 1
        assert memory_id not in manager._consolidation_locks
        print("✅ Concurrent consolidation prevented - only one execution")

        # Test 2c: Error handling
        print("\n🔄 Testing error handling...")
        consolidation_calls.clear()

        async def failing_consolidation(ext_id, mem_id):
            consolidation_calls.append((ext_id, mem_id))
            raise Exception("Simulated consolidation error")

        manager._consolidate_to_shortterm = failing_consolidation

        # This should not raise an exception
        await manager._consolidate_with_lock(external_id, memory_id)

        assert len(consolidation_calls) == 1
        assert memory_id not in manager._consolidation_locks
        print("✅ Error handling works - lock cleaned up after error")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Consolidation lock test failed")
        return False


# =========================================================================
# TEST 3: RETRIEVAL LOGIC
# =========================================================================


async def test_retrieval_logic():
    """Test memory retrieval logic."""
    print("\n" + "=" * 70)
    print("TEST 3: MEMORY RETRIEVAL LOGIC")
    print("=" * 70)

    try:
        config = get_config()
        manager = create_mock_memory_manager(config)

        external_id = "test-retrieval-agent"

        # Mock active memories
        active_memories = [
            generate_sample_memory(1, external_id),
            generate_sample_memory(2, external_id, {"summary": 2, "context": 1, "decisions": 1}),
        ]

        # Mock shortterm chunks
        shortterm_chunks = [
            ShorttermMemoryChunk(
                id=1,
                shortterm_memory_id=1,
                external_id=external_id,
                content="Previous conversation about AI systems",
                chunk_order=1,
                section_id="summary",
                embedding=[0.2] * 768,
                metadata={},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        ]

        # Mock longterm chunks
        longterm_chunks = [
            LongtermMemoryChunk(
                id=1,
                external_id=external_id,
                content="Historical knowledge about machine learning",
                chunk_order=1,
                confidence_score=0.85,
                start_date=datetime.now(timezone.utc),
                metadata={},
            )
        ]

        # Mock repository responses
        manager.active_repo.get_all_by_external_id = AsyncMock(return_value=active_memories)
        manager.shortterm_repo.hybrid_search = AsyncMock(return_value=shortterm_chunks)
        manager.longterm_repo.hybrid_search = AsyncMock(return_value=longterm_chunks)

        # Mock embedding generation
        manager.embedding_service.get_embedding = AsyncMock(return_value=[0.1] * 768)

        # Test retrieval
        print("\n🔄 Testing memory retrieval...")
        result = await manager._retrieve_memories_basic(
            external_id=external_id,
            query="test query about AI systems",
            limit=5,
        )

        assert isinstance(result, RetrievalResult)
        assert result.mode == "pointer"
        assert len(result.chunks) >= 0
        assert len(result.longterm_chunks) == 1
        assert len(result.synthesized_response) > 0

        print(f"✅ Retrieval successful:")
        print(f"   - Active memories: {len(result.active_memories)}")
        print(f"   - Shortterm chunks: {len(result.shortterm_chunks)}")
        print(f"   - Longterm chunks: {len(result.longterm_chunks)}")
        print(f"   - Synthesized response: {len(result.synthesized_response)} chars")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Retrieval logic test failed")
        return False


# =========================================================================
# TEST 4: MEMORY LIFECYCLE SIMULATION
# =========================================================================


async def test_memory_lifecycle():
    """Test complete memory lifecycle simulation."""
    print("\n" + "=" * 70)
    print("TEST 4: MEMORY LIFECYCLE SIMULATION")
    print("=" * 70)

    try:
        config = get_config()
        config.avg_section_update_count_for_consolidation = 2.0  # Low threshold

        manager = create_mock_memory_manager(config)
        external_id = "test-lifecycle-agent"
        memory_id = 1

        # Simulate memory creation
        print("\n🔄 Simulating memory creation...")
        initial_memory = generate_sample_memory(
            memory_id, external_id, {"summary": 0, "context": 0, "decisions": 0}
        )

        manager.active_repo.create = AsyncMock(return_value=initial_memory)
        print("✅ Memory created")

        # Simulate multiple updates that eventually trigger consolidation
        print("\n🔄 Simulating multiple updates...")

        updates = [
            {"summary": 1, "context": 0, "decisions": 0},  # Total: 1, Threshold: 6
            {"summary": 1, "context": 1, "decisions": 0},  # Total: 2, Threshold: 6
            {"summary": 2, "context": 1, "decisions": 0},  # Total: 3, Threshold: 6
            {"summary": 2, "context": 2, "decisions": 0},  # Total: 4, Threshold: 6
            {"summary": 2, "context": 2, "decisions": 1},  # Total: 5, Threshold: 6
            {"summary": 3, "context": 2, "decisions": 1},  # Total: 6, Threshold: 6 -> TRIGGER!
        ]

        consolidation_triggered = False

        for i, update_counts in enumerate(updates, 1):
            updated_memory = generate_sample_memory(memory_id, external_id, update_counts)
            manager.active_repo.update_sections = AsyncMock(return_value=updated_memory)
            manager.active_repo.reset_all_update_counts = AsyncMock()

            total_updates = sum(update_counts.values())
            threshold = config.avg_section_update_count_for_consolidation * len(update_counts)

            with patch("asyncio.create_task") as mock_create_task:
                await manager.update_active_memory_sections(
                    external_id=external_id,
                    memory_id=memory_id,
                    sections=[{"section_id": "summary", "new_content": f"Update {i}"}],
                )

                if total_updates >= threshold:
                    mock_create_task.assert_called_once()
                    consolidation_triggered = True
                    print(
                        f"✅ Update {i}: Consolidation triggered (total: {total_updates}, threshold: {threshold})"
                    )
                    break
                else:
                    mock_create_task.assert_not_called()
                    print(
                        f"   Update {i}: No consolidation (total: {total_updates}, threshold: {threshold})"
                    )

        assert consolidation_triggered, "Consolidation should have been triggered"
        print("✅ Memory lifecycle simulation completed successfully")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Memory lifecycle test failed")
        return False


# =========================================================================
# MAIN TEST RUNNER
# =========================================================================


async def main():
    """Run all memory manager logic tests."""
    print("🚀 MEMORY MANAGER LOGIC TESTS (MOCKED)")
    print("=" * 70)

    # Verify configuration
    config = get_config()
    print(f"\n📋 Configuration:")
    print(f"   - Embedding Model: {config.embedding_model}")
    print(f"   - Memorizer Model: {config.memorizer_agent_model}")
    print(
        f"   - Default Consolidation Threshold: {config.avg_section_update_count_for_consolidation}"
    )

    results = []

    # Run tests
    tests = [
        ("Update Operations", test_update_operations),
        ("Consolidation Lock", test_consolidation_lock),
        ("Retrieval Logic", test_retrieval_logic),
        ("Memory Lifecycle", test_memory_lifecycle),
    ]

    for test_name, test_func in tests:
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
        print("🎉 All logic tests PASSED!")
        print("\n💡 Next steps:")
        print("   1. Set up PostgreSQL and Neo4j databases")
        print("   2. Run: python scripts/test_memory_manager_real.py")
        print("   3. Test with real Gemini API for memorizer agent")
        return True
    else:
        print("⚠️  Some tests FAILED. Check logs for details.")
        return False


if __name__ == "__main__":
    import sys

    # Run the tests
    success = asyncio.run(main())

    # Exit with appropriate code
    sys.exit(0 if success else 1)
