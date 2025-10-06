"""
Real Agent Test Script

This script tests the ER Extractor and Memorizer agents with real API keys.
Run this to verify that the agents work correctly with your configured models.

Usage:
    python scripts/test_agents_real.py

Requirements:
    - Set GOOGLE_API_KEY (or other provider API keys) in your .env file
    - Ensure agent model configurations are set in .env or using defaults
"""

import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_mem.agents.er_extractor import extract_entities_and_relationships
from agent_mem.agents.memorizer import resolve_conflicts, format_conflicts_as_text
from agent_mem.database.models import (
    ConsolidationConflicts,
    ConflictSection,
    ConflictEntityDetail,
    ConflictRelationshipDetail,
    ShorttermMemoryChunk,
)
from agent_mem.config.settings import get_config
from unittest.mock import AsyncMock

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =========================================================================
# TEST 1: ER EXTRACTOR
# =========================================================================


async def test_er_extractor():
    """Test the ER Extractor agent with real API."""
    print("\n" + "=" * 70)
    print("TEST 1: ER EXTRACTOR AGENT")
    print("=" * 70)

    test_text = """
    Sarah is a senior software engineer at Microsoft. She specializes in 
    machine learning and works extensively with TensorFlow and PyTorch. 
    Her team uses Azure for cloud infrastructure and deploys models using 
    Kubernetes. The project is written in Python and uses FastAPI for the 
    REST API. Sarah collaborates with the data science team led by Dr. Chen.
    """

    print("\n📝 Input Text:")
    print(test_text.strip())

    try:
        print("\n🔄 Running ER Extractor...")
        result = await extract_entities_and_relationships(test_text)

        print(f"\n✅ Extraction Successful!")
        print(f"\n📊 Results:")
        print(f"   - Entities: {len(result.entities)}")
        print(f"   - Relationships: {len(result.relationships)}")

        print("\n👤 Entities:")
        for i, entity in enumerate(result.entities, 1):
            print(f"   {i}. {entity.name} ({entity.type.value})")
            print(f"      Confidence: {entity.confidence:.2f}")
            print(f"      Description: {entity.description}")

        print("\n🔗 Relationships:")
        for i, rel in enumerate(result.relationships, 1):
            print(f"   {i}. {rel.source} → {rel.target} ({rel.type.value})")
            print(f"      Confidence: {rel.confidence:.2f}")
            print(f"      Description: {rel.description}")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("ER Extractor test failed")
        return False


# =========================================================================
# TEST 2: MEMORIZER AGENT
# =========================================================================


async def test_memorizer():
    """Test the Memorizer agent with real API."""
    print("\n" + "=" * 70)
    print("TEST 2: MEMORIZER AGENT")
    print("=" * 70)

    # Create mock shortterm repository
    mock_repo = AsyncMock()

    # Setup mock responses
    mock_repo.update_chunk.return_value = ShorttermMemoryChunk(
        id=1,
        shortterm_memory_id=100,
        content="Updated chunk content",
        chunk_order=0,
        section_id="overview",
        metadata={},
    )

    # Create sample conflicts
    conflicts = ConsolidationConflicts(
        external_id="test-agent",
        active_memory_id=1,
        shortterm_memory_id=100,
        created_at=datetime.now(timezone.utc),
        total_conflicts=2,
        sections=[
            ConflictSection(
                section_id="overview",
                section_content="Python is a high-level, interpreted programming language. "
                "It emphasizes code readability and has extensive standard libraries. "
                "Python is widely used in web development, data science, and automation.",
                update_count=3,
                existing_chunks=[
                    ShorttermMemoryChunk(
                        id=1,
                        shortterm_memory_id=100,
                        content="Python is a programming language.",
                        chunk_order=0,
                        section_id="overview",
                        metadata={},
                    ),
                    ShorttermMemoryChunk(
                        id=2,
                        shortterm_memory_id=100,
                        content="Python is easy to learn.",
                        chunk_order=1,
                        section_id="overview",
                        metadata={},
                    ),
                ],
                metadata={},
            )
        ],
        entity_conflicts=[
            ConflictEntityDetail(
                name="Python",
                shortterm_types=["programming_language"],
                active_types=["programming_language", "interpreted"],
                merged_types=["programming_language", "interpreted"],
                shortterm_confidence=0.8,
                active_confidence=0.9,
                merged_confidence=0.85,
                shortterm_description="A programming language",
                active_description="A high-level, interpreted programming language",
                merged_description="A high-level, interpreted programming language",
            )
        ],
    )

    print("\n📋 Conflict Summary:")
    print(f"   - Total Conflicts: {conflicts.total_conflicts}")
    print(f"   - Section Conflicts: {len(conflicts.sections)}")
    print(f"   - Entity Conflicts: {len(conflicts.entity_conflicts)}")
    print(f"   - Relationship Conflicts: {len(conflicts.relationship_conflicts)}")

    try:
        print("\n🔄 Running Memorizer Agent...")
        resolution = await resolve_conflicts(conflicts, mock_repo)

        print(f"\n✅ Resolution Successful!")
        print(f"\n📊 Resolution Summary:")
        print(f"   {resolution.summary}")

        print(f"\n📝 Actions Taken:")
        print(f"   - Chunk Updates: {len(resolution.chunk_updates)}")
        print(f"   - Chunk Creates: {len(resolution.chunk_creates)}")
        print(f"   - Entity Updates: {len(resolution.entity_updates)}")
        print(f"   - Relationship Updates: {len(resolution.relationship_updates)}")

        if resolution.chunk_updates:
            print("\n📄 Chunk Updates:")
            for i, action in enumerate(resolution.chunk_updates, 1):
                print(f"   {i}. Chunk ID: {action.chunk_id}")
                print(f"      Reason: {action.reason}")
                print(f"      New Content (preview): {action.new_content[:100]}...")

        if resolution.entity_updates:
            print("\n👤 Entity Updates:")
            for i, action in enumerate(resolution.entity_updates, 1):
                print(f"   {i}. Entity ID: {action.entity_id}")
                print(f"      Reason: {action.reason}")
                if action.types:
                    print(f"      New Types: {action.types}")
                if action.confidence:
                    print(f"      New Confidence: {action.confidence:.2f}")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Memorizer test failed")
        return False


# =========================================================================
# TEST 3: CONFIGURATION CHECK
# =========================================================================


def test_configuration():
    """Test that configuration is properly loaded."""
    print("\n" + "=" * 70)
    print("TEST 3: CONFIGURATION CHECK")
    print("=" * 70)

    try:
        config = get_config()

        print("\n🔧 Agent Model Configuration:")
        print(f"   - ER Extractor Model: {config.er_extractor_agent_model}")
        print(f"   - Memorizer Model: {config.memorizer_agent_model}")
        print(f"   - Memory Update Model: {config.memory_update_agent_model}")
        print(f"   - Memory Retrieve Model: {config.memory_retrieve_agent_model}")

        print("\n🔑 API Keys Status:")
        api_keys = {
            "OpenAI": config.openai_api_key,
            "Anthropic": config.anthropic_api_key,
            "Google": config.google_api_key,
            "Grok": config.grok_api_key,
        }

        for provider, key in api_keys.items():
            status = "✅ Configured" if key else "❌ Not Set"
            if key:
                print(f"   - {provider}: {status} ({key[:10]}...)")
            else:
                print(f"   - {provider}: {status}")

        print("\n⚙️ Agent Settings:")
        print(f"   - Temperature: {config.agent_temperature}")
        print(f"   - Retries: {config.agent_retries}")

        # Check if at least one API key is configured
        if not any(api_keys.values()):
            print("\n⚠️  WARNING: No API keys configured!")
            print("   Please set at least one API key in your .env file:")
            print("   - GOOGLE_API_KEY=your-key-here")
            print("   - OPENAI_API_KEY=your-key-here")
            print("   - ANTHROPIC_API_KEY=your-key-here")
            return False

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Configuration test failed")
        return False


# =========================================================================
# MAIN TEST RUNNER
# =========================================================================


async def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("AGENT MEMORY - REAL AGENT TESTS")
    print("=" * 70)
    print("\nThis script tests the agents with real API calls.")
    print("Make sure you have configured your API keys in .env file.\n")

    results = {}

    # Test 1: Configuration
    print("\n🚀 Starting Tests...\n")
    results["configuration"] = test_configuration()

    if not results["configuration"]:
        print("\n❌ Configuration test failed. Please fix configuration before proceeding.")
        return

    # Test 2: ER Extractor
    results["er_extractor"] = await test_er_extractor()

    # Test 3: Memorizer
    results["memorizer"] = await test_memorizer()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r)

    print(f"\n📊 Results: {passed_tests}/{total_tests} tests passed")

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   - {test_name.replace('_', ' ').title()}: {status}")

    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Your agents are working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        logger.exception("Unexpected error in test runner")
        sys.exit(1)
