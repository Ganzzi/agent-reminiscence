"""
Pydantic AI Agents for intelligent memory management.

This module provides AI agents that enhance memory operations with:
- Entity and relationship extraction (ER Extractor Agent)
- Memory consolidation and conflict resolution (Memorizer Agent)
- Intelligent section updates (Memory Update Agent)
- Advanced search and synthesis (Memory Retrieve Agent)
"""

from agent_reminiscence.agents.er_extractor import (
    extract_entities,
    extract_entities_and_relationships,
    ExtractionResult,
    ExtractedEntity,
    ExtractedRelationship,
    EntityType,
    RelationshipType,
    ExtractionMode,
)
from agent_reminiscence.agents.memorizer import (
    resolve_conflicts,
    format_conflicts_as_text,
    MemorizerDeps,
    ConflictResolution,
)
from agent_reminiscence.agents.memory_updater import MemoryUpdateAgent
from agent_reminiscence.agents.memory_retriever import (
    retrieve_memory,
    ChunkPointer,
    EntityPointer,
    RelationshipPointer,
    RetrievalResult,
)
from agent_reminiscence.services.central_storage import CentralStorage, get_central_storage


# Placeholder class for backward compatibility
class MemoryRetrieveAgent:
    """Placeholder - use retrieve_memory function instead."""

    def __init__(self, config):
        self.config = config


__all__ = [
    # ER Extractor Agent
    "extract_entities",
    "extract_entities_and_relationships",
    "ExtractionResult",
    "ExtractedEntity",
    "ExtractedRelationship",
    "EntityType",
    "RelationshipType",
    "ExtractionMode",
    # Memorizer Agent
    "resolve_conflicts",
    "format_conflicts_as_text",
    "MemorizerDeps",
    "ConflictResolution",
    # Memory Retriever Agent
    "retrieve_memory",
    "ChunkPointer",
    "EntityPointer",
    "RelationshipPointer",
    "RetrievalResult",
    "CentralStorage",
    "get_central_storage",
    "MemoryRetrieveAgent",  # Backward compatibility
    # Other Agents
    "MemoryUpdateAgent",
]


