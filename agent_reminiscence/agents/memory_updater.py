"""
Memory Update Agent - Intelligent active memory management.

This agent analyzes message history and decides whether to create new
active memories or update existing ones with new information.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, RunUsage

from agent_reminiscence.config import Config
from agent_reminiscence.database import ActiveMemoryRepository
from agent_reminiscence.database.models import ActiveMemory

logger = logging.getLogger(__name__)


# ============================================================================
# Tool Response Models
# ============================================================================


class ActiveMemoryDetail(BaseModel):
    """Detailed active memory information for agent context."""

    id: int
    title: str
    sections: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any]
    created_at: datetime
    last_updated: datetime


class MemoryUpdateDecision(BaseModel):
    """Agent's decision on how to update memory."""

    action: str = Field(description="Action to take: 'create', 'update', or 'none'")
    memory_id: Optional[int] = Field(
        default=None, description="Memory ID to update (if action='update')"
    )
    section_id: Optional[str] = Field(
        default=None, description="Section ID to update (if action='update')"
    )
    new_content: str = Field(description="New content to add/update")
    reasoning: str = Field(description="Explanation of the decision")


# ============================================================================
# Agent Dependencies
# ============================================================================


@dataclass
class MemoryUpdateDeps:
    """Dependencies for Memory Update Agent."""

    external_id: str
    active_repo: ActiveMemoryRepository


# ============================================================================
# Memory Update Agent
# ============================================================================


class MemoryUpdateAgent:
    """
    Memory Update Agent using Pydantic AI.

    Responsibilities:
    - Analyze message history
    - Identify new information
    - Decide: create new memory or update existing
    - Execute memory operations
    - Maintain active memory integrity

    Usage:
        agent = MemoryUpdateAgent(config, active_repo)
        result = await agent.process_message(
            external_id="agent-123",
            message="User reported bug in authentication module",
            context="Previous messages about auth issues..."
        )
    """

    def __init__(self, config: Config, active_repo: ActiveMemoryRepository):
        """
        Initialize Memory Update Agent.

        Args:
            config: Configuration object
            active_repo: Active memory repository
        """
        self.config = config
        self.active_repo = active_repo

        # Create Pydantic AI agent
        self.agent = Agent(
            model=config.memory_update_agent_model,
            deps_type=MemoryUpdateDeps,
            output_type=MemoryUpdateDecision,
            system_prompt=self._get_system_prompt(),
            retries=config.agent_retries,
        )

        # Register tools
        self.agent.tool(self._get_detailed_active_memories)

        logger.info(f"MemoryUpdateAgent initialized with model: {config.memory_update_agent_model}")

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        return """You are a Memory Update Agent responsible for maintaining an AI agent's active working memory.

Your role:
1. Analyze incoming messages and context
2. Identify new or updated information that should be remembered
3. Decide whether to:
   - Create a new active memory (for new topics/tasks)
   - Update an existing memory section (for related information)
   - Do nothing (if information is not worth remembering)

Guidelines:
- Active memories are organized by templates with multiple sections
- Each section can be updated independently
- Group related information into the same memory
- Create new memories for distinct topics or tasks
- Update existing sections when adding details to ongoing work
- Be selective - only remember important, actionable information

You have access to tools:
- get_detailed_active_memories: View current active memories with sections

Always provide clear reasoning for your decisions."""

    # ========================================================================
    # Agent Tools
    # ========================================================================

    async def _get_detailed_active_memories(
        self, ctx: RunContext[MemoryUpdateDeps]
    ) -> List[ActiveMemoryDetail]:
        """
        Tool: Get detailed active memories for the external_id.

        Args:
            ctx: Run context with dependencies

        Returns:
            List of detailed active memory information
        """
        try:
            memories = await ctx.deps.active_repo.get_all_by_external_id(ctx.deps.external_id)

            details = [
                ActiveMemoryDetail(
                    id=mem.id,
                    title=mem.title,
                    sections=mem.sections,
                    metadata=mem.metadata,
                    created_at=mem.created_at,
                    last_updated=mem.last_updated,
                )
                for mem in memories
            ]

            logger.debug(f"Retrieved {len(details)} active memories for {ctx.deps.external_id}")
            return details

        except Exception as e:
            logger.error(f"Failed to get active memories: {e}", exc_info=True)
            return []

    # ========================================================================
    # Public Methods
    # ========================================================================

    async def process_message(
        self,
        external_id: str,
        message: str,
        context: Optional[str] = None,
    ) -> Tuple[MemoryUpdateDecision, RunUsage]:
        """
        Process a message and decide how to update memory.

        Args:
            external_id: Agent identifier
            message: New message to process
            context: Optional additional context

        Returns:
            MemoryUpdateDecision with action and details

        Raises:
            Exception: If agent execution fails
        """
        logger.info(f"Processing message for {external_id}: {message[:50]}...")

        try:
            # Build prompt
            prompt_parts = [f"New message: {message}"]
            if context:
                prompt_parts.append(f"\nContext: {context}")

            prompt = "\n".join(prompt_parts)

            # Run agent
            deps = MemoryUpdateDeps(external_id=external_id, active_repo=self.active_repo)

            result = await self.agent.run(prompt, deps=deps)

            logger.info(
                f"Memory update decision: action={result.output.action}, "
                f"reasoning={result.output.reasoning[:100]}..."
            )

            return result.output, result.usage()

        except Exception as e:
            logger.error(f"Memory update agent failed: {e}", exc_info=True)
            # Return safe default - do nothing
            return MemoryUpdateDecision(
                action="none",
                new_content="",
                reasoning=f"Agent execution failed: {str(e)}",
            ), RunUsage(requests=0, input_tokens=0, output_tokens=0)

    async def execute_decision(
        self,
        external_id: str,
        decision: MemoryUpdateDecision,
    ) -> Optional[ActiveMemory]:
        """
        Execute the agent's memory update decision.

        Args:
            external_id: Agent identifier
            decision: Decision from process_message()

        Returns:
            Updated/created ActiveMemory or None

        Raises:
            ValueError: If decision is invalid
        """
        logger.info(f"Executing decision: {decision.action}")

        try:
            if decision.action == "none":
                logger.info("No action needed")
                return None

            elif decision.action == "update":
                if not decision.memory_id or not decision.section_id:
                    raise ValueError("Update action requires memory_id and section_id")

                # Update existing section
                updated_memory = await self.active_repo.update_section(
                    memory_id=decision.memory_id,
                    section_id=decision.section_id,
                    new_content=decision.new_content,
                )

                if updated_memory:
                    logger.info(
                        f"Updated memory {decision.memory_id}, " f"section {decision.section_id}"
                    )
                    return updated_memory
                else:
                    logger.error(f"Memory {decision.memory_id} not found")
                    return None

            elif decision.action == "create":
                # For create, we need more information
                # This is a simplified version - in production, you'd need template info
                logger.warning(
                    "Create action not fully implemented - "
                    "requires template selection and structure"
                )
                return None

            else:
                raise ValueError(f"Unknown action: {decision.action}")

        except Exception as e:
            logger.error(f"Failed to execute decision: {e}", exc_info=True)
            raise


