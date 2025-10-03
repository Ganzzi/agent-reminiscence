"""
Memory Retrieve Agent - Intelligent search and synthesis.

This agent handles memory retrieval with:
- Query understanding and intent analysis
- Cross-tier search optimization
- Result ranking and filtering
- Natural language synthesis
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from agent_mem.config import Config
from agent_mem.database.repositories import ShorttermMemoryRepository, LongtermMemoryRepository
from agent_mem.database.models import (
    ActiveMemory,
    ShorttermMemoryChunk,
    LongtermMemoryChunk,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Tool Response Models
# ============================================================================


class SearchStrategy(BaseModel):
    """Search strategy determined by agent."""

    search_active: bool = Field(default=True, description="Search active memories")
    search_shortterm: bool = Field(default=True, description="Search shortterm memory")
    search_longterm: bool = Field(default=True, description="Search longterm memory")
    vector_weight: float = Field(default=0.7, description="Weight for vector search (0-1)")
    bm25_weight: float = Field(default=0.3, description="Weight for BM25 search (0-1)")
    limit_per_tier: int = Field(default=10, description="Results per tier")
    reasoning: str = Field(description="Why this strategy was chosen")


class RetrievalSynthesis(BaseModel):
    """Synthesized response from retrieved memories."""

    summary: str = Field(description="Concise summary of findings")
    key_points: List[str] = Field(default_factory=list, description="Key points extracted")
    sources: List[str] = Field(
        default_factory=list, description="Source tiers used (active/shortterm/longterm)"
    )
    confidence: float = Field(default=0.5, description="Confidence in synthesis (0-1)")
    gaps: List[str] = Field(default_factory=list, description="Information gaps identified")


# ============================================================================
# Agent Dependencies
# ============================================================================


@dataclass
class RetrieveDeps:
    """Dependencies for Memory Retrieve Agent."""

    external_id: str
    shortterm_repo: Optional[ShorttermMemoryRepository] = None
    longterm_repo: Optional[LongtermMemoryRepository] = None
    active_memories: List[ActiveMemory] = field(default_factory=list)
    shortterm_chunks: List[ShorttermMemoryChunk] = field(default_factory=list)
    longterm_chunks: List[LongtermMemoryChunk] = field(default_factory=list)


# ============================================================================
# Memory Retrieve Agent
# ============================================================================


class MemoryRetrieveAgent:
    """
    Memory Retrieve Agent using Pydantic AI.

    Responsibilities:
    - Analyze query intent and context
    - Determine optimal search strategy
    - Synthesize results from multiple tiers
    - Provide natural language response
    - Identify information gaps

    Usage:
        agent = MemoryRetrieveAgent(config, shortterm_repo, longterm_repo)

        # Determine strategy
        strategy = await agent.determine_strategy(
            query="How does authentication work?",
            context="Implementing login system"
        )

        # Synthesize results
        synthesis = await agent.synthesize_results(
            query="How does authentication work?",
            active_memories=[...],
            shortterm_chunks=[...],
            longterm_chunks=[...]
        )
    """

    def __init__(
        self,
        config: Config,
        shortterm_repo: Optional[ShorttermMemoryRepository] = None,
        longterm_repo: Optional[LongtermMemoryRepository] = None,
    ):
        """
        Initialize Memory Retrieve Agent.

        Args:
            config: Configuration object
            shortterm_repo: Shortterm memory repository (optional)
            longterm_repo: Longterm memory repository (optional)
        """
        self.config = config
        self.shortterm_repo = shortterm_repo
        self.longterm_repo = longterm_repo

        # Create strategy agent
        self.strategy_agent = Agent(
            model=config.memory_retrieve_agent_model,
            deps_type=None,
            output_type=SearchStrategy,
            system_prompt=self._get_strategy_prompt(),
            retries=config.agent_retries,
        )

        # Create synthesis agent
        self.synthesis_agent = Agent(
            model=config.memory_retrieve_agent_model,
            deps_type=None,
            output_type=RetrievalSynthesis,
            system_prompt=self._get_synthesis_prompt(),
            retries=config.agent_retries,
        )

        logger.info(
            f"MemoryRetrieveAgent initialized with model: " f"{config.memory_retrieve_agent_model}"
        )

    def _get_strategy_prompt(self) -> str:
        """Get the system prompt for strategy determination."""
        return """You are a Memory Retrieve Agent's strategy planner.

Your role:
1. Analyze the user's query and context
2. Determine which memory tiers to search
3. Configure search parameters (vector/BM25 weights)
4. Set result limits per tier

Memory Tiers:
- Active: Current working memory (tasks in progress, recent notes)
- Shortterm: Recent searchable memory (last few sessions)
- Longterm: Consolidated knowledge (validated facts, patterns)

Search Parameters:
- vector_weight: 0-1 (semantic similarity)
- bm25_weight: 0-1 (keyword matching)
- Higher vector weight for conceptual queries
- Higher BM25 weight for specific keyword searches

Guidelines:
- For current task questions: Focus on active + shortterm
- For factual questions: Focus on longterm
- For comprehensive queries: Search all tiers
- For specific keywords: Increase BM25 weight
- For conceptual queries: Increase vector weight

Provide clear reasoning for your strategy."""

    def _get_synthesis_prompt(self) -> str:
        """Get the system prompt for result synthesis."""
        return """You are a Memory Retrieve Agent's synthesis specialist.

Your role:
1. Analyze retrieved memories from multiple tiers
2. Extract key information relevant to the query
3. Synthesize a coherent, concise response
4. Identify information gaps
5. Assess confidence in the response

Guidelines:
- Prioritize recent, specific information
- Combine information from multiple sources
- Resolve contradictions (prefer recent/longterm over older)
- Be honest about gaps and uncertainties
- Provide actionable key points
- Note which tiers provided the information

Synthesis Quality:
- High confidence (0.8-1.0): Multiple consistent sources
- Medium confidence (0.5-0.8): Single source or partial info
- Low confidence (0.0-0.5): Contradictions or sparse info

Be clear, concise, and helpful."""

    # ========================================================================
    # Public Methods
    # ========================================================================

    async def determine_strategy(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> SearchStrategy:
        """
        Determine optimal search strategy for the query.

        Args:
            query: User's search query
            context: Optional additional context

        Returns:
            SearchStrategy with tier selection and parameters

        Raises:
            Exception: If agent execution fails
        """
        logger.info(f"Determining search strategy for query: {query[:50]}...")

        try:
            # Build prompt
            prompt_parts = [f"Query: {query}"]
            if context:
                prompt_parts.append(f"\nContext: {context}")

            prompt_parts.append(
                "\n\nDetermine the optimal search strategy:"
                "\n- Which tiers to search (active/shortterm/longterm)"
                "\n- Search parameter weights (vector vs BM25)"
                "\n- Result limits per tier"
            )

            prompt = "\n".join(prompt_parts)

            # Run agent
            result = await self.strategy_agent.run(prompt)

            logger.info(
                f"Strategy determined: active={result.output.search_active}, "
                f"shortterm={result.output.search_shortterm}, "
                f"longterm={result.output.search_longterm}"
            )

            return result.output

        except Exception as e:
            logger.error(f"Strategy agent failed: {e}", exc_info=True)
            # Return safe default - search all tiers
            return SearchStrategy(
                search_active=True,
                search_shortterm=True,
                search_longterm=True,
                vector_weight=0.7,
                bm25_weight=0.3,
                limit_per_tier=10,
                reasoning=f"Agent failed: {str(e)}. Using default strategy.",
            )

    async def synthesize_results(
        self,
        query: str,
        active_memories: List[ActiveMemory],
        shortterm_chunks: List[ShorttermMemoryChunk],
        longterm_chunks: List[LongtermMemoryChunk],
    ) -> RetrievalSynthesis:
        """
        Synthesize results from multiple memory tiers.

        Args:
            query: Original search query
            active_memories: Active memories found
            shortterm_chunks: Shortterm chunks found
            longterm_chunks: Longterm chunks found

        Returns:
            RetrievalSynthesis with summary and key points

        Raises:
            Exception: If agent execution fails
        """
        logger.info(
            f"Synthesizing results: {len(active_memories)} active, "
            f"{len(shortterm_chunks)} shortterm, {len(longterm_chunks)} longterm"
        )

        try:
            # Build comprehensive prompt
            prompt_parts = [f"Original Query: {query}", "\n\n=== RETRIEVED INFORMATION ==="]

            # Add active memories
            if active_memories:
                prompt_parts.append(f"\n\nActive Memories ({len(active_memories)}):")
                for mem in active_memories[:3]:  # Limit to top 3
                    prompt_parts.append(f"\nTitle: {mem.title}")
                    for section_id, section_data in list(mem.sections.items())[:2]:
                        content = section_data.get("content", "")[:200]
                        prompt_parts.append(f"  {section_id}: {content}...")

            # Add shortterm chunks
            if shortterm_chunks:
                prompt_parts.append(f"\n\nRecent Memory Chunks ({len(shortterm_chunks)}):")
                for i, chunk in enumerate(shortterm_chunks[:5], 1):  # Top 5
                    score = getattr(chunk, "similarity_score", 0.0) or 0.0
                    prompt_parts.append(f"\n{i}. (Score: {score:.2f}) {chunk.content[:200]}...")

            # Add longterm chunks
            if longterm_chunks:
                prompt_parts.append(f"\n\nConsolidated Knowledge ({len(longterm_chunks)}):")
                for i, chunk in enumerate(longterm_chunks[:5], 1):  # Top 5
                    score = getattr(chunk, "similarity_score", 0.0) or 0.0
                    conf = getattr(chunk, "confidence_score", 0.0)
                    prompt_parts.append(
                        f"\n{i}. (Score: {score:.2f}, Confidence: {conf:.2f}) "
                        f"{chunk.content[:200]}..."
                    )

            prompt_parts.append(
                "\n\n=== YOUR TASK ==="
                "\nSynthesize the above information into:"
                "\n1. A concise summary answering the query"
                "\n2. Key points (3-5 actionable points)"
                "\n3. Sources used (active/shortterm/longterm)"
                "\n4. Confidence level in the response"
                "\n5. Any information gaps identified"
            )

            prompt = "\n".join(prompt_parts)

            # Run agent
            result = await self.synthesis_agent.run(prompt)

            logger.info(
                f"Synthesis complete: {len(result.output.key_points)} key points, "
                f"confidence={result.output.confidence:.2f}"
            )

            return result.output

        except Exception as e:
            logger.error(f"Synthesis agent failed: {e}", exc_info=True)
            # Return basic synthesis
            sources = []
            if active_memories:
                sources.append("active")
            if shortterm_chunks:
                sources.append("shortterm")
            if longterm_chunks:
                sources.append("longterm")

            return RetrievalSynthesis(
                summary=f"Found information across {len(sources)} memory tiers. "
                f"Agent synthesis failed: {str(e)}",
                key_points=[
                    f"Retrieved {len(active_memories)} active memories",
                    f"Retrieved {len(shortterm_chunks)} shortterm chunks",
                    f"Retrieved {len(longterm_chunks)} longterm chunks",
                ],
                sources=sources,
                confidence=0.3,
                gaps=["Unable to perform advanced synthesis due to agent failure"],
            )
