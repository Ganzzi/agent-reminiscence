"""
Memorizefrom agent_mem.agents.er_extractor import (
    extract_entities_and_relationships,
    ExtractionResult,
    ExtractedEntity,
    ExtractedRelationship
)t - Memory Consolidation and Conflict Resolution.

Handles consolidation of active memories to shortterm, including entity/relationship
extraction and conflict resolution.
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from agent_mem.agents.er_extractor import (
    extract_entities_and_relationships,
    ExtractionResult,
    ExtractedEntity,
    ExtractedRelationship,
)

logger = logging.getLogger(__name__)


# =========================================================================
# DEPENDENCIES
# =========================================================================


@dataclass
class MemorizerDeps:
    """Dependencies for the Memorizer Agent."""

    external_id: str
    active_memory_id: int
    shortterm_memory_id: int
    memory_manager: Any  # MemoryManager instance (avoids circular import)


# =========================================================================
# OUTPUT MODELS
# =========================================================================


class ChunkOperation(BaseModel):
    """A chunk operation to perform."""

    operation: str = Field(description="Operation type: 'update' or 'create'")
    chunk_id: Optional[int] = Field(default=None, description="Chunk ID for updates")
    content: str = Field(description="Chunk content")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EntityOperation(BaseModel):
    """An entity operation to perform."""

    operation: str = Field(description="Operation type: 'create' or 'merge' or 'conflict'")
    entity_id: Optional[int] = Field(default=None, description="Entity ID for merge/conflict")
    name: str = Field(description="Entity name")
    entity_type: str = Field(description="Entity type")
    confidence: float = Field(description="Confidence score")
    reason: str = Field(description="Reason for this operation")


class RelationshipOperation(BaseModel):
    """A relationship operation to perform."""

    from_entity: str = Field(description="Source entity name")
    to_entity: str = Field(description="Target entity name")
    relationship_type: str = Field(description="Relationship type")
    confidence: float = Field(description="Confidence score")


class ConsolidationPlan(BaseModel):
    """Plan for consolidating active memory to shortterm."""

    chunk_operations: List[ChunkOperation] = Field(
        default_factory=list, description="Operations to perform on chunks"
    )
    entity_operations: List[EntityOperation] = Field(
        default_factory=list, description="Operations to perform on entities"
    )
    relationship_operations: List[RelationshipOperation] = Field(
        default_factory=list, description="Operations to perform on relationships"
    )
    summary: str = Field(description="Summary of consolidation plan")


# =========================================================================
# SYSTEM PROMPT
# =========================================================================

SYSTEM_PROMPT = """You are a Memory Consolidation Specialist (Memorizer Agent).

**Your Role:**
Consolidate active memories into shortterm memories by:
1. Analyzing content for chunk updates
2. Identifying entity extraction needs
3. Planning relationship mapping
4. Resolving conflicts between existing and new information

**Workflow:**
1. **Analyze Input Context:**
   - Active memory content
   - Existing shortterm chunks
   - Existing shortterm entities
   - Existing shortterm relationships

2. **Plan Chunk Operations:**
   - Which chunks to UPDATE with enhanced content
   - Which NEW chunks to CREATE for additional content
   - Ensure chunks have context for standalone comprehension

3. **Plan Entity Operations:**
   - MERGE: If new entity is similar to existing (same name/type)
   - CREATE: If new entity is distinct
   - CONFLICT: If ambiguous (mark for manual review)
   - Use similarity thresholds:
     * >= 0.85 similarity AND >= 0.7 overlap → MERGE
     * < thresholds → CREATE or CONFLICT

4. **Plan Relationship Operations:**
   - Map relationships between entities
   - Ensure both entities exist before creating relationships

**Guidelines:**
- Be conservative with merges (prefer creating separate entities if unsure)
- Always provide clear reasoning for operations
- Include context in new chunks (e.g., "[Context: Section name] content...")
- Use high confidence (0.8-1.0) for explicit information
- Use medium confidence (0.6-0.8) for inferred information
- Use low confidence (0.4-0.6) for ambiguous information

**Output Format:**
Provide a structured plan with:
- chunk_operations: List of chunk operations
- entity_operations: List of entity operations
- relationship_operations: List of relationship operations
- summary: Brief description of the consolidation plan

**Important:**
- Don't execute operations, just plan them
- Memory manager will execute the plan
- Be thorough but precise
- Prioritize data quality over quantity
"""


# =========================================================================
# AGENT CREATION
# =========================================================================


def get_memorizer_agent() -> Agent[MemorizerDeps, ConsolidationPlan]:
    """
    Factory function to create the Memorizer Agent.

    Returns:
        Configured Agent instance
    """
    return Agent(
        model="google-gla:gemini-2.5-flash",  # Standard Gemini Flash
        deps_type=MemorizerDeps,
        system_prompt=SYSTEM_PROMPT,
        output_type=ConsolidationPlan,  # Correct: output_type, not result_type
        model_settings={
            "temperature": 0.6,  # Balanced for analysis and creativity
        },
        retries=2,
    )


# Create global agent instance (lazy initialization to avoid API key errors on import)
_memorizer_agent: Optional[Agent[MemorizerDeps, ConsolidationPlan]] = None


def _get_agent() -> Agent[MemorizerDeps, ConsolidationPlan]:
    """Get or create the Memorizer Agent instance."""
    global _memorizer_agent
    if _memorizer_agent is None:
        _memorizer_agent = get_memorizer_agent()
    return _memorizer_agent


# =========================================================================
# MAIN FUNCTION
# =========================================================================


async def consolidate_memory(
    external_id: str,
    active_memory_id: int,
    shortterm_memory_id: int,
    active_content: str,
    existing_chunks: List[Dict[str, Any]],
    existing_entities: List[Dict[str, Any]],
    existing_relationships: List[Dict[str, Any]],
    memory_manager: Any,
) -> ConsolidationPlan:
    """
    Create a consolidation plan for active memory → shortterm memory.

    Args:
        external_id: Agent identifier
        active_memory_id: Active memory ID
        shortterm_memory_id: Shortterm memory ID
        active_content: Content from active memory
        existing_chunks: Existing shortterm chunks
        existing_entities: Existing shortterm entities
        existing_relationships: Existing shortterm relationships
        memory_manager: MemoryManager instance

    Returns:
        ConsolidationPlan with operations to perform

    Raises:
        Exception: If planning fails after retries
    """
    logger.info(
        f"Creating consolidation plan for active memory {active_memory_id} "
        f"→ shortterm memory {shortterm_memory_id}"
    )

    try:
        # Create dependencies
        deps = MemorizerDeps(
            external_id=external_id,
            active_memory_id=active_memory_id,
            shortterm_memory_id=shortterm_memory_id,
            memory_manager=memory_manager,
        )

        # Build context for the agent
        context = _build_consolidation_context(
            active_content=active_content,
            existing_chunks=existing_chunks,
            existing_entities=existing_entities,
            existing_relationships=existing_relationships,
        )

        # Run agent to create plan
        agent = _get_agent()
        result = await agent.run(context, deps=deps)

        logger.info(
            f"Created consolidation plan: "
            f"{len(result.output.chunk_operations)} chunk ops, "
            f"{len(result.output.entity_operations)} entity ops, "
            f"{len(result.output.relationship_operations)} relationship ops"
        )

        return result.output

    except Exception as e:
        logger.error(f"Consolidation planning failed: {e}")
        raise


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================


def _build_consolidation_context(
    active_content: str,
    existing_chunks: List[Dict[str, Any]],
    existing_entities: List[Dict[str, Any]],
    existing_relationships: List[Dict[str, Any]],
) -> str:
    """
    Build context string for the Memorizer Agent.

    Args:
        active_content: Content from active memory
        existing_chunks: Existing shortterm chunks
        existing_entities: Existing shortterm entities
        existing_relationships: Existing shortterm relationships

    Returns:
        Formatted context string
    """
    context_parts = []

    # Active memory content
    context_parts.append("=== ACTIVE MEMORY CONTENT ===")
    context_parts.append(active_content)
    context_parts.append("")

    # Existing chunks
    context_parts.append("=== EXISTING SHORTTERM CHUNKS ===")
    if existing_chunks:
        for i, chunk in enumerate(existing_chunks, 1):
            context_parts.append(f"Chunk {i} (ID: {chunk.get('id', 'unknown')}):")
            context_parts.append(chunk.get("content", ""))
            context_parts.append("")
    else:
        context_parts.append("No existing chunks")
        context_parts.append("")

    # Existing entities
    context_parts.append("=== EXISTING SHORTTERM ENTITIES ===")
    if existing_entities:
        for entity in existing_entities:
            context_parts.append(
                f"- {entity.get('name')} "
                f"(Type: {entity.get('type')}, "
                f"Confidence: {entity.get('confidence', 0):.2f}, "
                f"ID: {entity.get('id')})"
            )
        context_parts.append("")
    else:
        context_parts.append("No existing entities")
        context_parts.append("")

    # Existing relationships
    context_parts.append("=== EXISTING SHORTTERM RELATIONSHIPS ===")
    if existing_relationships:
        for rel in existing_relationships:
            context_parts.append(
                f"- {rel.get('from_entity_name')} "
                f"-[{rel.get('type')}]-> "
                f"{rel.get('to_entity_name')}"
            )
        context_parts.append("")
    else:
        context_parts.append("No existing relationships")
        context_parts.append("")

    return "\n".join(context_parts)


def validate_consolidation_plan(plan: ConsolidationPlan) -> bool:
    """
    Validate a consolidation plan.

    Args:
        plan: Consolidation plan to validate

    Returns:
        True if plan is valid
    """
    # Check if we have any operations
    if not (plan.chunk_operations or plan.entity_operations or plan.relationship_operations):
        logger.warning("Consolidation plan has no operations")
        return False

    # Validate chunk operations
    for op in plan.chunk_operations:
        if op.operation not in ["update", "create"]:
            logger.error(f"Invalid chunk operation: {op.operation}")
            return False
        if op.operation == "update" and not op.chunk_id:
            logger.error("Update operation missing chunk_id")
            return False

    # Validate entity operations
    for op in plan.entity_operations:
        if op.operation not in ["create", "merge", "conflict"]:
            logger.error(f"Invalid entity operation: {op.operation}")
            return False
        if op.confidence < 0.0 or op.confidence > 1.0:
            logger.error(f"Invalid confidence score: {op.confidence}")
            return False

    # Validate relationship operations
    for op in plan.relationship_operations:
        if op.confidence < 0.0 or op.confidence > 1.0:
            logger.error(f"Invalid confidence score: {op.confidence}")
            return False

    return True
