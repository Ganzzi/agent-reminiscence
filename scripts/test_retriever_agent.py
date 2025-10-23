"""
Test script for Memory Retriever Agent.

Tests the retriever agent with real database data setup.
Creates active memories, shortterm chunks/entities, and tests retrieval.
"""

import asyncio
import logging
import uuid
from agent_reminiscence.config.settings import get_config
from agent_reminiscence.database.postgres_manager import PostgreSQLManager
from agent_reminiscence.database.neo4j_manager import Neo4jManager
from agent_reminiscence.database.repositories import (
    ActiveMemoryRepository,
    ShorttermMemoryRepository,
    LongtermMemoryRepository,
)
from agent_reminiscence.services.embedding import EmbeddingService
from agent_reminiscence.agents.memory_retriever import retrieve_memory
from agent_reminiscence.services.memory_manager import MemoryManager

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_test_template():
    """Generate a test template for active memory."""
    return """
project_memory:
  description: "Track AI agent development project"
  sections:
    - id: current_work
      description: "Current tasks and focus areas"
    - id: technical_decisions
      description: "Key technical decisions made"
    - id: blockers
      description: "Current blockers and issues"
    - id: next_steps
      description: "Planned next actions"
""".strip()


async def setup_test_data(
    memory_manager: MemoryManager,
    external_id: str,
    shortterm_repo: ShorttermMemoryRepository,
    longterm_repo: LongtermMemoryRepository,
    embedding_service: EmbeddingService,
):
    """
    Set up comprehensive test data for retrieval agent testing.

    Creates:
    - Active memory with project tracking
    - Shortterm memory with chunks and entities/relationships
    - Longterm memory with chunks and entities/relationships

    Args:
        memory_manager: MemoryManager instance
        external_id: Agent identifier
        shortterm_repo: Shortterm repository
        longterm_repo: Longterm repository
        embedding_service: Embedding service for vectors
    """
    logger.info("\n" + "=" * 60)
    logger.info("Setting up test data")
    logger.info("=" * 60)

    # 1. Create active memory
    logger.info("\n📝 Creating active memory...")
    memory = await memory_manager.create_active_memory(
        external_id=external_id,
        title="AI Agent Memory System Project",
        template_content=generate_test_template(),
        initial_sections={
            "current_work": {
                "content": "Working on Memory Retriever Agent implementation. Implementing pointer-based references to reduce token costs by 90%. Setting up two-stage retrieval pipeline with central storage.",
                "update_count": 0,
            },
            "technical_decisions": {
                "content": "Decided to use singleton CentralStorage pattern. Storage is indexed by external_id to support multiple agents. Using Pydantic AI for agent framework with Google Gemini model.",
                "update_count": 0,
            },
            "blockers": {
                "content": "None currently. Testing in progress.",
                "update_count": 0,
            },
            "next_steps": {
                "content": "1. Complete test script with data setup 2. Test pointer mode vs synthesis mode 3. Verify storage cleanup",
                "update_count": 0,
            },
        },
        metadata={"test_type": "retriever_test", "created_by": "test_script"},
    )

    logger.info(f"✅ Created active memory: ID={memory.id}, Title='{memory.title}'")
    logger.info(f"   Sections: {list(memory.sections.keys())}")

    # 2. Create main shortterm memory with rich data (related to query)
    logger.info("\n📝 Creating main shortterm memory (with chunks/entities)...")
    shortterm_mem = await shortterm_repo.create_memory(
        external_id=external_id,
        title="AI Agent Development Context",
        summary="Recent work on memory system implementation",
        metadata={"phase": "development", "priority": "high"},
    )
    logger.info(f"✅ Created shortterm memory: ID={shortterm_mem.id}")

    # Create 4 unrelated shortterm memories (no chunks/entities/relationships)
    logger.info("\n📝 Creating 4 unrelated shortterm memories (empty)...")
    unrelated_memories_data = [
        {
            "title": "Weekend Meal Planning",
            "summary": "Planning meals for the upcoming weekend. Considering vegetarian options and grocery shopping list.",
            "metadata": {"category": "personal", "priority": "low"},
        },
        {
            "title": "Book Club Discussion Notes",
            "summary": "Notes from book club meeting about 'The Midnight Library'. Discussion on parallel universes and life choices.",
            "metadata": {"category": "personal", "activity": "reading"},
        },
        {
            "title": "Home Renovation Ideas",
            "summary": "Collecting ideas for kitchen renovation. Considering new countertops, cabinet styles, and modern appliances.",
            "metadata": {"category": "home", "status": "planning"},
        },
        {
            "title": "Fitness Tracking Goals",
            "summary": "Monthly fitness goals and progress tracking. Target: 10k steps daily, 3 gym sessions per week.",
            "metadata": {"category": "health", "month": "October"},
        },
    ]

    for mem_data in unrelated_memories_data:
        unrelated_mem = await shortterm_repo.create_memory(
            external_id=external_id,
            title=mem_data["title"],
            summary=mem_data["summary"],
            metadata=mem_data["metadata"],
        )
        logger.info(
            f"  Created unrelated memory: ID={unrelated_mem.id}, Title='{mem_data['title']}'"
        )

    # Create shortterm chunks with embeddings
    logger.info("\n📦 Creating shortterm chunks...")
    shortterm_chunks_data = [
        {
            "content": "CentralStorage singleton service implementation. The service uses threading.Lock for thread-safety and stores data per external_id. Each agent gets isolated storage namespace to prevent data collision.",
            "section_id": "technical_decisions",
        },
        {
            "content": "Memory Retriever Agent uses two-stage pipeline: First stage searches and stores results in central storage, second stage returns pointer references by default. Synthesis is generated only when confidence is low or explicitly requested.",
            "section_id": "current_work",
        },
        {
            "content": "Implemented pointer-based memory retrieval to reduce token costs by 90%. Pointers are just IDs that reference data in central storage. The LLM works with compact references instead of full content.",
            "section_id": "technical_decisions",
        },
        {
            "content": "Database layer uses PostgreSQL for chunks with pgvector and BM25 search. Neo4j stores entities and relationships as graph. Hybrid search combines vector similarity and keyword matching.",
            "section_id": "technical_decisions",
        },
        {
            "content": "Testing strategy includes unit tests for repositories, integration tests for memory manager, and end-to-end tests for retrieval agent. Using pytest with async support.",
            "section_id": "next_steps",
        },
    ]

    for i, chunk_data in enumerate(shortterm_chunks_data):
        embedding = await embedding_service.get_embedding(chunk_data["content"])
        chunk = await shortterm_repo.create_chunk(
            shortterm_memory_id=shortterm_mem.id,
            external_id=external_id,
            content=chunk_data["content"],
            embedding=embedding,
            section_id=chunk_data["section_id"],
            metadata={"order": i},
        )
        logger.info(
            f"  Created chunk {i+1}/{len(shortterm_chunks_data)}: {chunk_data['content'][:50]}..."
        )

    # Create shortterm entities
    logger.info("\n👤 Creating shortterm entities...")
    entity_data = [
        {
            "name": "CentralStorage",
            "types": ["service", "singleton"],
            "description": "Singleton service for storing retrieval results",
            "importance": 0.9,
        },
        {
            "name": "Memory Retriever Agent",
            "types": ["agent", "ai"],
            "description": "AI agent that searches and retrieves memory",
            "importance": 0.95,
        },
        {
            "name": "PostgreSQL",
            "types": ["database", "technology"],
            "description": "Database for storing memory chunks",
            "importance": 0.7,
        },
        {
            "name": "Neo4j",
            "types": ["database", "graph", "technology"],
            "description": "Graph database for entities and relationships",
            "importance": 0.7,
        },
        {
            "name": "Pointer Pattern",
            "types": ["pattern", "optimization"],
            "description": "Design pattern using IDs instead of full content",
            "importance": 0.8,
        },
    ]

    entities = {}
    for ent_data in entity_data:
        entity = await shortterm_repo.create_entity(
            external_id=external_id,
            shortterm_memory_id=shortterm_mem.id,
            name=ent_data["name"],
            types=ent_data["types"],
            description=ent_data["description"],
            importance=ent_data["importance"],
            metadata={"source": "test_setup"},
        )
        entities[ent_data["name"]] = entity
        logger.info(f"  Created entity: {ent_data['name']} ({', '.join(ent_data['types'])})")

    # Create shortterm relationships
    logger.info("\n🔗 Creating shortterm relationships...")
    relationships_data = [
        {
            "from": "Memory Retriever Agent",
            "to": "CentralStorage",
            "types": ["uses", "stores_in"],
            "description": "Agent stores results in CentralStorage",
            "importance": 0.9,
        },
        {
            "from": "CentralStorage",
            "to": "Pointer Pattern",
            "types": ["implements"],
            "description": "Storage implements pointer pattern",
            "importance": 0.85,
        },
        {
            "from": "Memory Retriever Agent",
            "to": "PostgreSQL",
            "types": ["queries"],
            "description": "Agent searches PostgreSQL chunks",
            "importance": 0.8,
        },
        {
            "from": "Memory Retriever Agent",
            "to": "Neo4j",
            "types": ["queries"],
            "description": "Agent searches Neo4j graph",
            "importance": 0.8,
        },
    ]

    for rel_data in relationships_data:
        from_entity = entities[rel_data["from"]]
        to_entity = entities[rel_data["to"]]
        rel = await shortterm_repo.create_relationship(
            external_id=external_id,
            shortterm_memory_id=shortterm_mem.id,
            from_entity_id=from_entity.id,
            to_entity_id=to_entity.id,
            types=rel_data["types"],
            description=rel_data["description"],
            importance=rel_data["importance"],
            metadata={"source": "test_setup"},
        )
        logger.info(
            f"  Created relationship: {rel_data['from']} -> {rel_data['to']} ({', '.join(rel_data['types'])})"
        )

    # 3. Create longterm memory with rich data
    logger.info("\n📝 Creating longterm memory chunks...")
    longterm_chunks_data = [
        {
            "content": "Pointer-based reference system reduces token costs by storing full content in central storage and passing only IDs to LLM. The LLM works with compact pointer references instead of full text, achieving 90% token reduction in memory operations.",
            "importance": 0.95,
        },
        {
            "content": "CentralStorage service implements singleton pattern with thread-safe operations using threading.Lock. Data is organized by external_id for multi-agent isolation. Each pointer ID follows format: tool_call_id:type:id for unique identification.",
            "importance": 0.92,
        },
        {
            "content": "Two-stage retrieval pipeline: Stage 1 performs search and stores results in central storage. Stage 2 returns pointer IDs by default. Synthesis generated only when confidence < 0.7 or explicitly requested via synthesis flag.",
            "importance": 0.93,
        },
        {
            "content": "Memory retriever agent uses four search tools: search_shortterm_chunks, search_longterm_chunks, search_shortterm_entities, search_longterm_entities. Each tool stores raw data in central storage and returns pointer metadata.",
            "importance": 0.88,
        },
        {
            "content": "Pointer pattern enables lazy loading of memory content. Only requested pointers are resolved from storage. This prevents unnecessary data transfer and reduces LLM context window usage significantly.",
            "importance": 0.90,
        },
        {
            "content": "Implementation uses Pydantic models for type safety: ChunkPointer, EntityPointer, RelationshipPointer. Each pointer contains ID, tier, score/importance, and basic metadata for preview without full content resolution.",
            "importance": 0.85,
        },
    ]

    for i, chunk_data in enumerate(longterm_chunks_data):
        embedding = await embedding_service.get_embedding(chunk_data["content"])
        chunk = await longterm_repo.create_chunk(
            external_id=external_id,
            content=chunk_data["content"],
            embedding=embedding,
            shortterm_memory_id=shortterm_mem.id,
            importance=chunk_data["importance"],
            metadata={"order": i, "topic": "architecture" if i < 3 else "implementation"},
        )
        logger.info(
            f"  Created longterm chunk {i+1}/{len(longterm_chunks_data)}: {chunk_data['content'][:50]}..."
        )

    # Create longterm entities
    logger.info("\n👤 Creating longterm entities...")
    longterm_entity_data = [
        {
            "name": "Pointer-based Reference System",
            "types": ["pattern", "optimization"],
            "description": "Design pattern using IDs instead of full content to reduce token costs",
            "importance": 0.95,
        },
        {
            "name": "CentralStorage",
            "types": ["service", "singleton"],
            "description": "Thread-safe singleton service for storing retrieval results indexed by external_id",
            "importance": 0.92,
        },
        {
            "name": "Two-stage Retrieval Pipeline",
            "types": ["architecture", "pattern"],
            "description": "Search and store in stage 1, return pointers in stage 2",
            "importance": 0.90,
        },
        {
            "name": "Lazy Loading",
            "types": ["optimization", "pattern"],
            "description": "Load data only when needed by resolving pointers on demand",
            "importance": 0.88,
        },
        {
            "name": "Memory Retriever Agent",
            "types": ["agent", "component"],
            "description": "AI agent that implements pointer-based memory retrieval",
            "importance": 0.93,
        },
    ]

    longterm_entities = {}
    for ent_data in longterm_entity_data:
        entity = await longterm_repo.create_entity(
            external_id=external_id,
            name=ent_data["name"],
            types=ent_data["types"],
            description=ent_data["description"],
            importance=ent_data["importance"],
            metadata={"source": "test_setup"},
        )
        longterm_entities[ent_data["name"]] = entity
        logger.info(
            f"  Created longterm entity: {ent_data['name']} ({', '.join(ent_data['types'])})"
        )

    # Create longterm relationships
    logger.info("\n🔗 Creating longterm relationships...")
    longterm_relationships_data = [
        {
            "from": "Memory Retriever Agent",
            "to": "Pointer-based Reference System",
            "types": ["implements"],
            "description": "Agent implements pointer-based reference system for efficient retrieval",
            "importance": 0.95,
        },
        {
            "from": "Pointer-based Reference System",
            "to": "CentralStorage",
            "types": ["uses", "depends_on"],
            "description": "Pointer system relies on CentralStorage for data persistence",
            "importance": 0.93,
        },
        {
            "from": "Two-stage Retrieval Pipeline",
            "to": "Pointer-based Reference System",
            "types": ["enables"],
            "description": "Pipeline architecture enables efficient pointer-based retrieval",
            "importance": 0.90,
        },
        {
            "from": "Lazy Loading",
            "to": "Pointer-based Reference System",
            "types": ["powered_by"],
            "description": "Lazy loading pattern powered by pointer references",
            "importance": 0.88,
        },
        {
            "from": "CentralStorage",
            "to": "Lazy Loading",
            "types": ["supports"],
            "description": "Storage service supports lazy loading through pointer resolution",
            "importance": 0.87,
        },
    ]

    for rel_data in longterm_relationships_data:
        from_entity = longterm_entities[rel_data["from"]]
        to_entity = longterm_entities[rel_data["to"]]
        rel = await longterm_repo.create_relationship(
            external_id=external_id,
            from_entity_id=from_entity.id,
            to_entity_id=to_entity.id,
            types=rel_data["types"],
            description=rel_data["description"],
            importance=rel_data["importance"],
            metadata={"source": "test_setup"},
        )
        logger.info(
            f"  Created longterm relationship: {rel_data['from']} -> {rel_data['to']} ({', '.join(rel_data['types'])})"
        )

    # 4. Verify data exists
    logger.info("\n🔍 Verifying data setup...")
    active_memories = await memory_manager.get_active_memories(external_id)
    shortterm_chunks = await shortterm_repo.get_chunks_by_memory_id(shortterm_mem.id)
    shortterm_entities_list = await shortterm_repo.get_entities_by_memory(shortterm_mem.id)
    shortterm_rels = await shortterm_repo.get_relationships_by_memory(shortterm_mem.id)

    logger.info(f"  Active memories: {len(active_memories)}")
    logger.info(f"  Shortterm chunks: {len(shortterm_chunks)}")
    logger.info(f"  Shortterm entities: {len(shortterm_entities_list)}")
    logger.info(f"  Shortterm relationships: {len(shortterm_rels)}")
    logger.info(f"  Longterm chunks: {len(longterm_chunks_data)}")
    logger.info(f"  Longterm entities: {len(longterm_entity_data)}")
    logger.info(f"  Longterm relationships: {len(longterm_relationships_data)}")

    logger.info("\n✅ Test data setup complete!")
    logger.info("=" * 60)

    return memory


async def test_retriever_agent():
    """Test the memory retriever agent with setup data."""
    config = get_config()

    # Lower consolidation threshold for testing
    config.avg_section_update_count_for_consolidation = 2.0

    # Initialize database managers
    postgres = PostgreSQLManager(config)
    neo4j = Neo4jManager(config)

    # Initialize memory manager for setup
    memory_manager = MemoryManager(config=config)
    await memory_manager.initialize()

    # Use repositories from memory manager (they're already initialized)
    active_repo = memory_manager.active_repo
    shortterm_repo = memory_manager.shortterm_repo
    longterm_repo = memory_manager.longterm_repo
    embedding_service = memory_manager.embedding_service

    # Use unique ID for this test run
    external_id = f"test-retriever-{uuid.uuid4().hex[:8]}"
    logger.info(f"\n🆔 Test Agent ID: {external_id}")

    try:
        # Setup test data (creates active, shortterm, and longterm data)
        await setup_test_data(
            memory_manager=memory_manager,
            external_id=external_id,
            shortterm_repo=shortterm_repo,
            longterm_repo=longterm_repo,
            embedding_service=embedding_service,
        )

        # Wait a bit for any async operations
        await asyncio.sleep(1)

        logger.info("\n" + "=" * 60)
        logger.info("Testing Memory Retriever Agent")
        logger.info("=" * 60)

        logger.info("\n" + "=" * 60)
        logger.info("📝 Test: Complex Query (Force Synthesis Mode)")
        query = "What are the key technical decisions and implementation details for the pointer-based memory retrieval system?"

        result = await retrieve_memory(
            query=query,
            external_id=external_id,
            shortterm_repo=shortterm_repo,
            longterm_repo=longterm_repo,
            active_repo=active_repo,
            embedding_service=embedding_service,
            synthesis=True,  # Force full synthesis
        )

        logger.info(f"\n📊 Result Mode: {result.mode}")
        logger.info(f"🔍 Search Strategy: {result.search_strategy}")
        logger.info(f"💯 Confidence: {result.confidence:.2f}")
        logger.info(f"📦 Chunks found: {len(result.chunks)}")
        logger.info(f"👤 Entities found: {len(result.entities)}")
        logger.info(f"🔗 Relationships found: {len(result.relationships)}")

        if result.synthesis:
            logger.info(f"\n💡 Synthesis:\n{result.synthesis}")

        # Show metadata
        logger.info("\n" + "=" * 60)
        logger.info("📊 Search Metadata:")
        metadata = result.metadata or {}
        logger.info(f"  Shortterm chunks searched: {metadata.get('shortterm_chunks_searched', 0)}")
        logger.info(f"  Longterm chunks searched: {metadata.get('longterm_chunks_searched', 0)}")
        logger.info(
            f"  Shortterm entities searched: {metadata.get('shortterm_entities_searched', 0)}"
        )
        logger.info(
            f"  Longterm entities searched: {metadata.get('longterm_entities_searched', 0)}"
        )
        logger.info(f"  Total results: {metadata.get('total_results', 0)}")

        logger.info("\n" + "=" * 60)
        logger.info("✅ All tests completed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        raise

    finally:
        await memory_manager.close()
        await postgres.close()
        await neo4j.close()
        logger.info("🔒 Database connections closed")


if __name__ == "__main__":
    asyncio.run(test_retriever_agent())


