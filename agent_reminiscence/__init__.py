"""
Agent Mem - Hierarchical memory management for AI agents.

This package provides a simple interface for managing active, shortterm, and longterm
memories with vector search, graph relationships, and intelligent consolidation.
"""

from agent_reminiscence.core import AgentMem
from agent_reminiscence.config.settings import Config
from agent_reminiscence.database.models import (
    ActiveMemory,
    ShorttermMemory,
    LongtermMemory,
    RetrievalResult,
)

__version__ = "0.1.3"
__all__ = [
    "AgentMem",
    "Config",
    "ActiveMemory",
    "ShorttermMemory",
    "LongtermMemory",
    "RetrievalResult",
]
