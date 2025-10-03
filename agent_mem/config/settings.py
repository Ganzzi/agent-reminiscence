"""Configuration settings for Agent Mem."""

import os
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator
from dotenv import load_dotenv

load_dotenv()


class Config(BaseModel):
    """Configuration for Agent Mem package."""

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
    memory_update_agent_model: str = Field(
        default_factory=lambda: os.getenv(
            "MEMORY_UPDATE_AGENT_MODEL", "google-gla:gemini-2.0-flash"
        )
    )
    memorizer_agent_model: str = Field(
        default_factory=lambda: os.getenv("MEMORIZER_AGENT_MODEL", "google-gla:gemini-2.0-flash")
    )
    memory_retrieve_agent_model: str = Field(
        default_factory=lambda: os.getenv(
            "MEMORY_RETRIEVE_AGENT_MODEL", "google-gla:gemini-2.0-flash"
        )
    )

    # Agent Settings
    agent_temperature: float = Field(default=0.6)
    agent_retries: int = Field(default=3)

    # Memory Configuration
    consolidation_threshold: int = Field(
        default_factory=lambda: int(os.getenv("ACTIVE_MEMORY_UPDATE_THRESHOLD", "5")),
        description="Number of updates before consolidation",
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

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")

    @field_validator("postgres_password", "neo4j_password")
    @classmethod
    def validate_password(cls, v: str, info) -> str:
        """Validate that passwords meet minimum security requirements."""
        if v and len(v) < 8:
            raise ValueError(
                f"{info.field_name} must be at least 8 characters long for security. "
                f"Current length: {len(v)}"
            )
        return v


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
