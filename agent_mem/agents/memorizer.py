"""
Memorizer - Memory Consolidation and Conflict Resolution.

Handles consolidation of active memories to shortterm, including entity/relationship
extraction and conflict resolution using a Pydantic AI agent.
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, Tool

from agent_mem.config.settings import get_config
from agent_mem.services.llm_model_provider import model_provider
from agent_mem.database.repositories.shortterm_memory import ShorttermMemoryRepository
from agent_mem.utils.agent_logger import save_agent_run, log_usage_summary
import json
from agent_mem.database.models import (
    ConsolidationConflicts,
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
    shortterm_repo: ShorttermMemoryRepository


# =========================================================================
# RESPONSE MODELS
# =========================================================================


class ChunkUpdateAction(BaseModel):
    """Action to update a shortterm memory chunk."""

    chunk_id: int
    new_content: str
    reason: str = Field(description="Why this update is needed")


class ChunkCreateAction(BaseModel):
    """Action to create a new shortterm memory chunk."""

    content: str
    chunk_order: int
    section_id: Optional[str] = None
    reason: str = Field(description="Why this chunk should be created")


class EntityUpdateAction(BaseModel):
    """Action to update an entity."""

    entity_id: int
    name: Optional[str] = None
    types: Optional[List[str]] = None
    description: Optional[str] = None
    importance: Optional[float] = None
    reason: str = Field(description="Why this update is needed")


class RelationshipUpdateAction(BaseModel):
    """Action to update a relationship."""

    relationship_id: int
    types: Optional[List[str]] = None
    description: Optional[str] = None
    importance: Optional[float] = None
    strength: Optional[float] = None
    reason: str = Field(description="Why this update is needed")


class ConflictResolution(BaseModel):
    """Resolution decisions for conflicts."""

    chunk_updates: List[ChunkUpdateAction] = Field(default_factory=list)
    chunk_creates: List[ChunkCreateAction] = Field(default_factory=list)
    entity_updates: List[EntityUpdateAction] = Field(default_factory=list)
    relationship_updates: List[RelationshipUpdateAction] = Field(default_factory=list)
    summary: str = Field(description="Summary of all resolution decisions")


# =========================================================================
# MEMORIZER AGENT
# =========================================================================


# Register tools
async def update_chunk(
    ctx: RunContext[MemorizerDeps], chunk_id: int, new_content: str, reason: str
) -> Dict[str, Any]:
    """
    Update a shortterm memory chunk with new content.

    Args:
        chunk_id: ID of the chunk to update
        new_content: New content for the chunk
        reason: Explanation for why this update is needed

    Returns:
        Result of the update operation
    """
    try:
        logger.info(f"Updating chunk {chunk_id}. Reason: {reason}")

        updated_chunk = await ctx.deps.shortterm_repo.update_chunk(
            chunk_id=chunk_id, content=new_content
        )

        if not updated_chunk:
            return {"success": False, "error": f"Chunk {chunk_id} not found"}

        return {
            "success": True,
            "chunk_id": updated_chunk.id,
            "reason": reason,
        }
    except Exception as e:
        logger.error(f"Error updating chunk {chunk_id}: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


async def create_chunk(
    ctx: RunContext[MemorizerDeps],
    content: str,
    chunk_order: int,
    section_id: Optional[str],
    reason: str,
) -> Dict[str, Any]:
    """
    Create a new shortterm memory chunk.

    Args:
        content: Content of the new chunk
        chunk_order: Order position for the chunk
        section_id: Optional section ID reference
        reason: Explanation for why this chunk should be created

    Returns:
        Result of the create operation
    """
    try:
        logger.info(f"Creating new chunk in section {section_id}. Reason: {reason}")

        # Note: Embedding will be None - embedding service would be needed for full implementation
        new_chunk = await ctx.deps.shortterm_repo.create_chunk(
            shortterm_memory_id=ctx.deps.shortterm_memory_id,
            external_id=ctx.deps.external_id,
            content=content,
            chunk_order=chunk_order,
            section_id=section_id,
            metadata={"source": "memorizer_agent", "reason": reason},
        )

        return {
            "success": True,
            "chunk_id": new_chunk.id,
            "reason": reason,
        }
    except Exception as e:
        logger.error(f"Error creating chunk: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


async def update_entity(
    ctx: RunContext[MemorizerDeps],
    entity_id: int,
    name: Optional[str],
    types: Optional[List[str]],
    description: Optional[str],
    importance: Optional[float],
    reason: str,
) -> Dict[str, Any]:
    """
    Update an entity with new information.

    Args:
        entity_id: ID of the entity to update
        name: New name (optional)
        types: New types list (optional)
        description: New description (optional)
        importance: New importance score (optional)
        reason: Explanation for why this update is needed

    Returns:
        Result of the update operation
    """
    try:
        logger.info(f"Updating entity {entity_id}. Reason: {reason}")

        updated_entity = await ctx.deps.shortterm_repo.update_entity(
            entity_id=entity_id,
            name=name,
            types=types,
            description=description,
            importance=importance,
        )

        if not updated_entity:
            return {"success": False, "error": f"Entity {entity_id} not found"}

        return {
            "success": True,
            "entity_id": updated_entity.id,
            "reason": reason,
        }
    except Exception as e:
        logger.error(f"Error updating entity {entity_id}: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


async def update_relationship(
    ctx: RunContext[MemorizerDeps],
    relationship_id: int,
    types: Optional[List[str]],
    description: Optional[str],
    importance: Optional[float],
    strength: Optional[float],
    reason: str,
) -> Dict[str, Any]:
    """
    Update a relationship with new information.

    Args:
        relationship_id: ID of the relationship to update
        types: New types list (optional)
        description: New description (optional)
        importance: New importance score (optional)
        strength: New strength score (optional)
        reason: Explanation for why this update is needed

    Returns:
        Result of the update operation
    """
    try:
        logger.info(f"Updating relationship {relationship_id}. Reason: {reason}")

        updated_rel = await ctx.deps.shortterm_repo.update_relationship(
            relationship_id=relationship_id,
            types=types,
            description=description,
            importance=importance,
            strength=strength,
        )

        if not updated_rel:
            return {"success": False, "error": f"Relationship {relationship_id} not found"}

        return {
            "success": True,
            "relationship_id": updated_rel.id,
            "reason": reason,
        }
    except Exception as e:
        logger.error(f"Error updating relationship {relationship_id}: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def _get_memorizer_agent() -> Agent[MemorizerDeps, ConflictResolution]:
    """
    Get or create the memorizer agent (lazy initialization).

    This avoids requiring an API key at module import time.
    """
    config = get_config()

    # Get model from provider using configuration
    model = model_provider.get_model(config.memorizer_agent_model)

    # Initialize the agent
    agent = Agent(
        model=model,
        deps_type=MemorizerDeps,
        output_type=ConflictResolution,
        tools=[
            Tool(update_chunk, takes_ctx=True, docstring_format="google"),
            Tool(create_chunk, takes_ctx=True, docstring_format="google"),
            Tool(update_entity, takes_ctx=True, docstring_format="google"),
            Tool(update_relationship, takes_ctx=True, docstring_format="google"),
        ],
        system_prompt="""You are an expert memory consolidation agent responsible for resolving conflicts 
between active memory and shortterm memory.

Your goal is to:
1. Analyze conflicts in memory chunks, entities, and relationships
2. Decide how to merge conflicting information intelligently
3. Preserve important details while avoiding redundancy
4. Maintain consistency and accuracy across the memory system

When resolving conflicts:
- Favor more recent or more detailed information
- Merge complementary information rather than replacing
- Maintain entity and relationship integrity
- Use importance scores to guide decisions
- Explain your reasoning for each action

CRITICAL: You MUST use the provided tools to resolve conflicts. Do NOT just analyze - actively call the tools:
- update_chunk: Update existing shortterm memory chunk with merged content
- create_chunk: Create new shortterm memory chunk when needed
- update_entity: Update entity information with merged data
- update_relationship: Update relationship information with merged data

For each conflict, you should:
1. Call the appropriate tool(s) to make the actual changes
2. The tools will return success/failure status
3. After making all changes, provide a summary in your final response

Remember: Your job is to EXECUTE the conflict resolution, not just plan it.""",
    )

    return agent


# =========================================================================
# MAIN CONSOLIDATION FUNCTION
# =========================================================================


def format_conflicts_as_text(conflicts: ConsolidationConflicts) -> str:
    """
    Format ConsolidationConflicts as a readable text prompt for the agent.

    Args:
        conflicts: ConsolidationConflicts object

    Returns:
        Formatted text representation
    """
    lines = [
        "# Memory Consolidation Conflicts",
        "",
        f"External ID: {conflicts.external_id}",
        f"Active Memory ID: {conflicts.active_memory_id}",
        f"Shortterm Memory ID: {conflicts.shortterm_memory_id}",
        f"Total Conflicts: {conflicts.total_conflicts}",
        f"Timestamp: {conflicts.created_at.isoformat()}",
        "",
    ]

    # Section conflicts
    if conflicts.sections:
        lines.append("## Section Conflicts")
        lines.append("")
        for i, section in enumerate(conflicts.sections, 1):
            lines.append(f"### Section {i}: {section.section_id}")
            lines.append(f"- Update Count: {section.update_count}")
            lines.append(f"- Existing Chunks: {len(section.existing_chunks)}")
            lines.append(f"- New Content Length: {len(section.section_content)} chars")
            lines.append("")
            lines.append("**New Content:**")
            lines.append(f"```\n{section.section_content}\n```")
            lines.append("")
            lines.append("**Existing Chunks:**")
            for chunk in section.existing_chunks:
                lines.append(
                    f"- Chunk {chunk.id}: {chunk.content[:100]}{'...' if len(chunk.content) > 100 else ''}"
                )
            lines.append("")

    # Entity conflicts
    if conflicts.entity_conflicts:
        lines.append("## Entity Conflicts")
        lines.append("")
        for i, entity in enumerate(conflicts.entity_conflicts, 1):
            lines.append(f"### Entity {i}: {entity.name}")
            lines.append(f"- Shortterm Types: {entity.shortterm_types}")
            lines.append(f"- Active Types: {entity.active_types}")
            lines.append(f"- Merged Types: {entity.merged_types}")
            lines.append(f"- Shortterm Importance: {entity.shortterm_importance}")
            lines.append(f"- Active Importance: {entity.active_importance}")
            lines.append(f"- Merged Importance: {entity.merged_importance}")
            if entity.shortterm_description:
                lines.append(f"- Shortterm Description: {entity.shortterm_description}")
            if entity.active_description:
                lines.append(f"- Active Description: {entity.active_description}")
            if entity.merged_description:
                lines.append(f"- Merged Description: {entity.merged_description}")
            lines.append("")

    # Relationship conflicts
    if conflicts.relationship_conflicts:
        lines.append("## Relationship Conflicts")
        lines.append("")
        for i, rel in enumerate(conflicts.relationship_conflicts, 1):
            lines.append(f"### Relationship {i}: {rel.from_entity} -> {rel.to_entity}")
            lines.append(f"- Shortterm Types: {rel.shortterm_types}")
            lines.append(f"- Active Types: {rel.active_types}")
            lines.append(f"- Merged Types: {rel.merged_types}")
            lines.append(f"- Shortterm Importance: {rel.shortterm_importance}")
            lines.append(f"- Active Importance: {rel.active_importance}")
            lines.append(f"- Merged Importance: {rel.merged_importance}")
            lines.append(f"- Shortterm Strength: {rel.shortterm_strength}")
            lines.append(f"- Active Strength: {rel.active_strength}")
            lines.append(f"- Merged Strength: {rel.merged_strength}")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "Please analyze these conflicts and provide resolution actions using the available tools."
    )

    return "\n".join(lines)


async def resolve_conflicts(
    conflicts: ConsolidationConflicts, shortterm_repo: ShorttermMemoryRepository
) -> ConflictResolution:
    """
    Resolve conflicts using the memorizer agent.

    Args:
        conflicts: ConsolidationConflicts object with all conflict information
        shortterm_repo: Repository for shortterm memory operations

    Returns:
        ConflictResolution with all actions taken
    """
    logger.info(
        f"Resolving conflicts for external_id={conflicts.external_id}, "
        f"total_conflicts={conflicts.total_conflicts}"
    )

    # Create dependencies
    deps = MemorizerDeps(
        external_id=conflicts.external_id,
        active_memory_id=conflicts.active_memory_id,
        shortterm_memory_id=conflicts.shortterm_memory_id,
        shortterm_repo=shortterm_repo,
    )

    # Format conflicts as text prompt
    conflict_text = format_conflicts_as_text(conflicts)

    try:
        # Get the agent instance (lazy initialization)
        agent = _get_memorizer_agent()

        # Run the agent
        result = await agent.run(user_prompt=conflict_text, deps=deps)
        resolution = result.output

        logger.info(f"Agent resolution complete: {resolution.summary}")
        logger.info(
            f"Actions: {len(resolution.chunk_updates)} chunk updates, "
            f"{len(resolution.chunk_creates)} chunk creates, "
            f"{len(resolution.entity_updates)} entity updates, "
            f"{len(resolution.relationship_updates)} relationship updates"
        )

        # Log usage summary
        usage = result.usage()
        log_usage_summary(usage, "memorizer")

        # Save detailed logs to file
        try:
            saved_files = save_agent_run(
                result=result,
                agent_name="memorizer",
                run_id=conflicts.external_id,
                include_usage=True,
                include_messages=True,
            )
            if saved_files["messages"]:
                logger.info(f"Saved agent run logs to {saved_files['messages'].parent}")
        except Exception as file_e:
            logger.warning(f"Failed to save agent run logs: {file_e}", exc_info=True)

        return resolution

    except Exception as e:
        logger.error(f"Error resolving conflicts with agent: {e}", exc_info=True)

        # Return empty resolution on error
        return ConflictResolution(
            summary=f"Error during conflict resolution: {str(e)}",
        )
