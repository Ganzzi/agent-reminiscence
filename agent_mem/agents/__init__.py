"""
Pydantic AI Agents for intelligent memory management.

This module provides AI agents that enhance memory operations with:
- Entity and relationship extraction (ER Extractor Agent)
- Memory consolidation and conflict resolution (Memorizer Agent)
- Intelligent section updates (Memory Update Agent)
- Advanced search and synthesis (Memory Retrieve Agent)
"""

from agent_mem.agents.er_extractor import (
    extract_entities_and_relationships,
    ExtractionResult,
    ExtractedEntity,
    ExtractedRelationship,
    EntityType,
    RelationshipType,
)
from agent_mem.agents.memorizer import (
    resolve_conflicts,
    format_conflicts_as_text,
    MemorizerDeps,
    ConflictResolution,
)
from agent_mem.agents.memory_updater import MemoryUpdateAgent
from agent_mem.agents.memory_retriever import MemoryRetrieveAgent

__all__ = [
    # ER Extractor Agent
    "extract_entities_and_relationships",
    "ExtractionResult",
    "ExtractedEntity",
    "ExtractedRelationship",
    "EntityType",
    "RelationshipType",
    # Memorizer Agent
    "resolve_conflicts",
    "format_conflicts_as_text",
    "MemorizerDeps",
    "ConflictResolution",
    # Other Agents
    "MemoryUpdateAgent",
    "MemoryRetrieveAgent",
]
