"""Configuration settings for Agent Mem.

Supports three configuration methods:
1. Direct Python instantiation (recommended for PyPI installs)
2. Environment variables (recommended for Docker/Kubernetes)
3. .env file (convenient for local development only)

The .env file loading is optional and only attempted if python-dotenv is available.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator

# Optional .env loading - only if python-dotenv is available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, just use environment variables
    pass


class Config(BaseModel):
    """Configuration for Agent Mem package.

    This class provides type-safe configuration management and supports three
    initialization patterns:

    **Pattern 1: Direct Python (Recommended for PyPI users)**
    ```python
    from agent_reminiscence import AgentMem, Config

    config = Config(
        postgres_host="localhost",
        postgres_password="secure_password",
        neo4j_uri="bolt://localhost:7687",
        neo4j_password="neo4j_password",
        ollama_base_url="http://localhost:11434"
    )

    agent_mem = AgentMem(config=config)
    ```

    **Pattern 2: Environment Variables (Recommended for Docker/K8s)**
    ```bash
    export POSTGRES_HOST=postgres
    export POSTGRES_PASSWORD=secure_pass
    export NEO4J_URI=bolt://neo4j:7687
    export NEO4J_PASSWORD=neo4j_pass
    export OLLAMA_BASE_URL=http://ollama:11434
    python your_app.py
    ```
    ```python
    from agent_reminiscence import AgentMem

    agent_mem = AgentMem()  # Uses environment variables
    ```

    **Pattern 3: .env File (Convenient for local development)**
    ```bash
    # .env (git-ignored)
    POSTGRES_HOST=localhost
    POSTGRES_PASSWORD=devpass
    NEO4J_URI=bolt://localhost:7687
    NEO4J_PASSWORD=devpass
    OLLAMA_BASE_URL=http://localhost:11434
    ```
    ```python
    # Automatically loaded if python-dotenv is available
    from agent_reminiscence import AgentMem

    agent_mem = AgentMem()  # Uses .env file
    ```
    """

    # PostgreSQL Configuration
    postgres_host: str = Field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    postgres_port: int = Field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    postgres_user: str = Field(default_factory=lambda: os.getenv("POSTGRES_USER", "postgres"))
    postgres_password: str = Field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", ""))
    postgres_db: str = Field(default_factory=lambda: os.getenv("POSTGRES_DB", "agent_mem"))

    # Neo4j Configuration
    neo4j_uri: str = Field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = Field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = Field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))
    neo4j_database: str = Field(default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j"))

    # Ollama Configuration
    ollama_base_url: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    )
    vector_dimension: int = Field(default_factory=lambda: int(os.getenv("VECTOR_DIMENSION", "768")))

    # Agent Model Configuration
    er_extractor_agent_model: str = Field(
        default_factory=lambda: os.getenv("ER_EXTRACTOR_AGENT_MODEL", "google:gemini-2.5-flash")
    )
    memory_update_agent_model: str = Field(
        default_factory=lambda: os.getenv("MEMORY_UPDATE_AGENT_MODEL", "google:gemini-2.5-flash")
    )
    memorizer_agent_model: str = Field(
        default_factory=lambda: os.getenv("MEMORIZER_AGENT_MODEL", "google:gemini-2.5-flash")
    )
    memory_retrieve_agent_model: str = Field(
        default_factory=lambda: os.getenv("MEMORY_RETRIEVE_AGENT_MODEL", "google:gemini-2.5-flash")
    )

    # Agent Settings
    agent_temperature: float = Field(default=0.6)
    agent_retries: int = Field(default=3)

    # Memory Configuration
    avg_section_update_count_for_consolidation: float = Field(
        default_factory=lambda: float(os.getenv("AVG_SECTION_UPDATE_COUNT", "3.0")),
        description="Average update count per section before consolidation trigger",
    )
    consolidation_threshold: int = Field(
        default_factory=lambda: int(os.getenv("ACTIVE_MEMORY_UPDATE_THRESHOLD", "5")),
        description="Consolidation threshold for active memory updates",
    )
    shortterm_update_count_threshold: int = Field(
        default_factory=lambda: int(os.getenv("SHORTTERM_UPDATE_THRESHOLD", "10")),
        description="Number of shortterm memory updates before longterm promotion",
    )
    promotion_importance_threshold: float = Field(
        default_factory=lambda: float(os.getenv("SHORTTERM_PROMOTION_THRESHOLD", "0.7")),
        description="Importance score for longterm promotion",
    )
    entity_similarity_threshold: float = Field(
        default_factory=lambda: float(os.getenv("ENTITY_SIMILARITY_THRESHOLD", "0.85")),
        description="Semantic similarity threshold for entity merging",
    )
    entity_overlap_threshold: float = Field(
        default_factory=lambda: float(os.getenv("ENTITY_OVERLAP_THRESHOLD", "0.7")),
        description="Entity overlap threshold for merging",
    )
    chunk_size: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "512")),
        description="Size of memory chunks in tokens",
    )
    chunk_overlap: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50")),
        description="Overlap between chunks",
    )

    # Search Configuration
    similarity_threshold: float = Field(
        default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", "0.7")),
        description="Default similarity threshold",
    )
    bm25_weight: float = Field(
        default_factory=lambda: float(os.getenv("BM25_WEIGHT", "0.3")),
        description="Weight for BM25 in hybrid search",
    )
    vector_weight: float = Field(
        default_factory=lambda: float(os.getenv("VECTOR_WEIGHT", "0.7")),
        description="Weight for vector in hybrid search",
    )

    # LLM API Keys Configuration
    openai_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY"),
        description="OpenAI API key for GPT models",
    )
    anthropic_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"),
        description="Anthropic API key for Claude models",
    )
    google_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY"),
        description="Google API key for Gemini models",
    )
    grok_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GROK_API_KEY"),
        description="Grok API key",
    )

    # Pydantic Configuration
    model_config = ConfigDict(
        env_file=".env",  # Optional .env loading - only loaded if file exists
        env_file_encoding="utf-8",
        case_sensitive=False,  # Treat env var names case-insensitively
        extra="ignore",  # Ignore extra fields not defined in Config model
    )

    @field_validator("postgres_password", "neo4j_password")
    @classmethod
    def validate_password(cls, v: str, info) -> str:
        """Validate that passwords meet minimum security requirements.

        Only validates if a password is actually provided (non-empty).
        This allows development with default empty password while requiring
        secure passwords in production configs.
        """
        if v and len(v) < 8:
            raise ValueError(
                f"{info.field_name} must be at least 8 characters long for security. "
                f"Current length: {len(v)}"
            )
        return v


# Global config instance (for backward compatibility)
# This is optional - the recommended pattern is to pass Config directly
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance.

    **Use Case**: Fallback pattern for CLI scripts, services, and tests.
    This maintains a global singleton for convenience when a Config can't be
    passed directly through the call stack.

    **Recommended for:**
    - CLI tools
    - Service classes that accept optional config parameter
    - Test scripts that need quick config access

    **NOT Recommended for:**
    - Main application code - pass Config explicitly instead
    - Library code - accept Config as parameter

    The returned Config is loaded in this order (first match wins):
    1. Environment variables (POSTGRES_HOST, etc.)
    2. .env file (if python-dotenv is available and file exists)
    3. Default hardcoded values

    Example:
        ```python
        # For CLI scripts - OK to use get_config()
        from agent_reminiscence.config import get_config
        config = get_config()
        agent_mem = AgentMem(config)

        # For main app - pass Config explicitly (BETTER)
        from agent_reminiscence import AgentMem, Config
        config = Config(postgres_host="localhost", ...)
        agent_mem = AgentMem(config=config)
        ```

    Returns:
        Config object loaded from environment variables, .env file, or defaults
    """
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance.

    **Use Case**: Mainly for testing or CLI scripts that need to override config.

    Use this to temporarily set a different config for:
    - Unit tests (mock configurations)
    - Testing different database connections
    - CLI tools that load config dynamically

    **NOT Recommended for:**
    - Production code - pass Config to your classes instead
    - Library code - accept Config as parameter

    Example:
        ```python
        # For testing - OK to use set_config()
        test_config = Config(postgres_db="test_db", ...)
        set_config(test_config)
        config = get_config()  # Returns test_config

        # For production - pass Config directly (BETTER)
        config = Config(...)
        agent_mem = AgentMem(config=config)
        ```

    Args:
        config: Configuration object to set as global instance
    """
    global _config
    _config = config


