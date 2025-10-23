"""Tests for configuration settings."""

import os
import pytest
from agent_reminiscence.config.settings import Config, get_config, set_config


class TestConfig:
    """Test Config class."""

    def test_config_defaults(self):
        """Test that Config loads with default values."""
        config = Config()

        # PostgreSQL defaults
        assert config.postgres_host == os.getenv("POSTGRES_HOST", "localhost")
        assert config.postgres_port == int(os.getenv("POSTGRES_PORT", "5432"))
        assert config.postgres_user == os.getenv("POSTGRES_USER", "postgres")
        assert config.postgres_db == os.getenv("POSTGRES_DB", "agent_mem")

        # Neo4j defaults
        assert config.neo4j_uri == os.getenv("NEO4J_URI", "bolt://localhost:7687")
        assert config.neo4j_user == os.getenv("NEO4J_USER", "neo4j")
        assert config.neo4j_database == os.getenv("NEO4J_DATABASE", "neo4j")

        # Ollama defaults
        assert config.ollama_base_url == os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        assert config.embedding_model == os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        assert config.vector_dimension == int(os.getenv("VECTOR_DIMENSION", "768"))

        # Agent model defaults
        assert config.memory_update_agent_model == os.getenv(
            "MEMORY_UPDATE_AGENT_MODEL", "google-gla:gemini-2.0-flash"
        )
        assert config.memorizer_agent_model == os.getenv(
            "MEMORIZER_AGENT_MODEL", "google-gla:gemini-2.0-flash"
        )
        assert config.memory_retrieve_agent_model == os.getenv(
            "MEMORY_RETRIEVE_AGENT_MODEL", "google-gla:gemini-2.0-flash"
        )

        # Agent settings defaults
        assert config.agent_temperature == 0.6
        assert config.agent_retries == 3

        # Memory configuration defaults
        assert config.consolidation_threshold == int(
            os.getenv("ACTIVE_MEMORY_UPDATE_THRESHOLD", "5")
        )
        assert config.promotion_importance_threshold == float(
            os.getenv("SHORTTERM_PROMOTION_THRESHOLD", "0.7")
        )
        assert config.entity_similarity_threshold == float(
            os.getenv("ENTITY_SIMILARITY_THRESHOLD", "0.85")
        )
        assert config.entity_overlap_threshold == float(
            os.getenv("ENTITY_OVERLAP_THRESHOLD", "0.7")
        )
        assert config.chunk_size == int(os.getenv("CHUNK_SIZE", "512"))
        assert config.chunk_overlap == int(os.getenv("CHUNK_OVERLAP", "50"))

        # Search configuration defaults
        assert config.similarity_threshold == float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
        assert config.bm25_weight == float(os.getenv("BM25_WEIGHT", "0.3"))
        assert config.vector_weight == float(os.getenv("VECTOR_WEIGHT", "0.7"))

    def test_config_custom_values(self):
        """Test that Config accepts custom values."""
        config = Config(
            postgres_host="custom_host",
            postgres_port=5433,
            postgres_user="custom_user",
            postgres_password="custom_pass",
            postgres_db="custom_db",
            neo4j_uri="bolt://custom:7687",
            neo4j_user="custom_neo4j",
            neo4j_password="custom_neo4j_pass",
            neo4j_database="custom_graph",
            ollama_base_url="http://custom:11434",
            embedding_model="custom-model",
            vector_dimension=1024,
            agent_temperature=0.8,
            agent_retries=5,
            consolidation_threshold=10,
            promotion_importance_threshold=0.8,
            entity_similarity_threshold=0.9,
            entity_overlap_threshold=0.75,
            chunk_size=1024,
            chunk_overlap=100,
            similarity_threshold=0.75,
            bm25_weight=0.4,
            vector_weight=0.6,
        )

        assert config.postgres_host == "custom_host"
        assert config.postgres_port == 5433
        assert config.postgres_user == "custom_user"
        assert config.postgres_password == "custom_pass"
        assert config.postgres_db == "custom_db"
        assert config.neo4j_uri == "bolt://custom:7687"
        assert config.neo4j_user == "custom_neo4j"
        assert config.neo4j_password == "custom_neo4j_pass"
        assert config.neo4j_database == "custom_graph"
        assert config.ollama_base_url == "http://custom:11434"
        assert config.embedding_model == "custom-model"
        assert config.vector_dimension == 1024
        assert config.agent_temperature == 0.8
        assert config.agent_retries == 5
        assert config.consolidation_threshold == 10
        assert config.promotion_importance_threshold == 0.8
        assert config.entity_similarity_threshold == 0.9
        assert config.entity_overlap_threshold == 0.75
        assert config.chunk_size == 1024
        assert config.chunk_overlap == 100
        assert config.similarity_threshold == 0.75
        assert config.bm25_weight == 0.4
        assert config.vector_weight == 0.6

    def test_config_validation(self):
        """Test that Config validates values."""
        # Valid config should work
        config = Config(
            postgres_port=5432,
            vector_dimension=768,
            agent_temperature=0.5,
            agent_retries=3,
        )
        assert config.postgres_port == 5432
        assert config.vector_dimension == 768

        # Invalid types should raise errors
        with pytest.raises((ValueError, TypeError)):
            Config(postgres_port="not_a_number")

        with pytest.raises((ValueError, TypeError)):
            Config(vector_dimension="not_a_number")


class TestConfigSingleton:
    """Test get_config() singleton pattern."""

    def test_get_config_returns_same_instance(self):
        """Test that get_config() returns the same instance."""
        # Reset global config
        set_config(None)

        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_set_config_changes_instance(self):
        """Test that set_config() changes the global instance."""
        # Create custom config
        custom_config = Config(postgres_host="custom_host_test")

        # Set it as global
        set_config(custom_config)

        # Get config should return our custom one
        retrieved_config = get_config()
        assert retrieved_config is custom_config
        assert retrieved_config.postgres_host == "custom_host_test"

        # Reset for other tests
        set_config(None)

    def test_get_config_creates_on_first_call(self):
        """Test that get_config() creates Config on first call."""
        # Reset global config
        set_config(None)

        # First call should create new Config
        config = get_config()
        assert config is not None
        assert isinstance(config, Config)


