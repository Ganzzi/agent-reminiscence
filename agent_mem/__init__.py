"""
Agent Mem - Hierarchical memory management for AI agents.

This package provides a simple interface for managing active, shortterm, and longterm
memories with vector search, graph relationships, and intelligent consolidation.
"""

from agent_mem.core import AgentMem
from agent_mem.config.settings import Config
from agent_mem.database.models import (
    ActiveMemory,
    ShorttermMemory,
    LongtermMemory,
    Entity,
    Relationship,
    RetrievalResult,
)

__version__ = "0.1.0"
__all__ = [
    "AgentMem",
    "Config",
    "ActiveMemory",
    "ShorttermMemory",
    "LongtermMemory",
    "Entity",
    "Relationship",
    "RetrievalResult",
]
