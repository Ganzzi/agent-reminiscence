"""
Simple test without AI agents - tests core database functionality.
"""

import asyncio
import logging
from agent_reminiscence.database import PostgreSQLManager, Neo4jManager
from agent_reminiscence.services import EmbeddingService
from agent_reminiscence.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def test_database_connections():
    """Test database connections without AI agents."""

    logger.info("=" * 60)
    logger.info("Testing Agent Mem Core Components")
    logger.info("=" * 60)

    config = Config()

    # Test PostgreSQL
    logger.info("\n1. Testing PostgreSQL connection...")
    pg_manager = PostgreSQLManager(config)

    try:
        await pg_manager.initialize()
        logger.info("   ✓ PostgreSQL connected successfully!")

        # Test a simple query
        async with pg_manager.connection() as conn:
            result = await conn.execute("SELECT 1 as test")
            rows = result.result()
            logger.info(f"   ✓ PostgreSQL query executed, got {len(rows)} rows")

    except Exception as e:
        logger.error(f"   ✗ PostgreSQL test failed: {e}")
        return False
    finally:
        await pg_manager.close()
        logger.info("   ✓ PostgreSQL connection closed")

    # Test Neo4j
    logger.info("\n2. Testing Neo4j connection...")
    neo4j_manager = Neo4jManager(config)

    try:
        await neo4j_manager.initialize()
        logger.info("   ✓ Neo4j connected successfully!")

        # Test a simple query
        result = await neo4j_manager.execute_read("RETURN 'Hello from Neo4j!' as message")
        if result:
            logger.info(f"   ✓ Neo4j query result: {result[0]['message']}")

    except Exception as e:
        logger.error(f"   ✗ Neo4j test failed: {e}")
        return False
    finally:
        await neo4j_manager.close()
        logger.info("   ✓ Neo4j connection closed")

    # Test Embedding Service
    logger.info("\n3. Testing Embedding Service...")
    embedding_service = EmbeddingService(config)

    try:
        # Test embedding generation
        text = "This is a test sentence for embedding."
        embedding = await embedding_service.get_embedding(text)

        logger.info(f"   ✓ Generated embedding (dimension: {len(embedding)})")
        logger.info(f"   ✓ First 5 values: {[f'{v:.4f}' for v in embedding[:5]]}")

        # Test batch embedding
        texts = ["First text", "Second text", "Third text"]
        embeddings = await embedding_service.get_embeddings_batch(texts)
        logger.info(f"   ✓ Generated {len(embeddings)} batch embeddings")

    except Exception as e:
        logger.error(f"   ✗ Embedding test failed: {e}")
        return False

    logger.info("\n" + "=" * 60)
    logger.info("✓ All core component tests passed!")
    logger.info("=" * 60)
    logger.info("\n✨ Database connections and embedding service working!\n")

    return True


if __name__ == "__main__":
    success = asyncio.run(test_database_connections())
    exit(0 if success else 1)


