"""
Agent Mem - Hierarchical memory management for AI agents.

This package provides a simple interface for managing active, shortterm, and longterm
memories with vector search, graph relationships, and intelligent consolidation.
"""

from agent_reminiscence.core import AgentMem
from agent_reminiscence.config.settings import Config
from agent_reminiscence.database.models import (
    # Core models
    ActiveMemory,
    ShorttermMemory,
    LongtermMemory,
    # Legacy (v0.1.x) - Deprecated
    RetrievalResult,
    RetrievedChunk,
    RetrievedEntity,
    RetrievedRelationship,
    # New (v0.2.0) - Recommended
    RetrievalResultV2,
    ShorttermKnowledgeTriplet,
    LongtermKnowledgeTriplet,
    ShorttermRetrievedChunk,
    LongtermRetrievedChunk,
    ActiveMemorySummary,
    ActiveMemorySlim,
)

__version__ = "0.2.0"
__all__ = [
    # Core classes
    "AgentMem",
    "Config",
    # Active memory models
    "ActiveMemory",
    "ActiveMemorySummary",
    "ActiveMemorySlim",
    # Memory tier models
    "ShorttermMemory",
    "LongtermMemory",
    # Retrieval result models (v0.2.0 - recommended)
    "RetrievalResultV2",
    "ShorttermKnowledgeTriplet",
    "LongtermKnowledgeTriplet",
    "ShorttermRetrievedChunk",
    "LongtermRetrievedChunk",
    # Deprecated (v0.1.x - for backward compatibility)
    "RetrievalResult",
    "RetrievedChunk",
    "RetrievedEntity",
    "RetrievedRelationship",
]
