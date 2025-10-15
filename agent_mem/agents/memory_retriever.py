"""
Memory Retrieve Agent - Intelligent search and synthesis.

This agent handles memory retrieval with:
- Query understanding and intent analysis
- Cross-tier search optimization
- Result ranking and filtering
- Pointer-based references for efficiency
- Optional natural language synthesis

Two-stage pipeline:
1. Search and store raw data in central storage
2. Return pointer references (IDs) by default
3. Generate synthesis only when needed (low confidence, complex query, or requested)
"""

import logging
from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, Tool

from agent_mem.config.settings import get_config
from agent_mem.services.llm_model_provider import model_provider
from agent_mem.services.embedding import EmbeddingService
from agent_mem.services.central_storage import get_central_storage
from agent_mem.database.repositories import (
    ShorttermMemoryRepository,
    LongtermMemoryRepository,
    ActiveMemoryRepository,
)
from agent_mem.database.models import (
    LongtermMemoryChunk,
    RetrievedChunk,
    RetrievedEntity,
    RetrievedRelationship,
    RetrievalResult as FinalRetrievalResult,
)
from agent_mem.agents.er_extractor import extract_entities

logger = logging.getLogger(__name__)


# ============================================================================
# DEPENDENCIES
# ============================================================================


@dataclass
class RetrieverDeps:
    """Dependencies for the Memory Retrieve Agent."""

    external_id: str
    query: str
    synthesis: bool
    shortterm_repo: ShorttermMemoryRepository
    longterm_repo: LongtermMemoryRepository
    embedding_service: EmbeddingService


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class ChunkPointer(BaseModel):
    """Pointer reference to a chunk in central storage."""

    pointer_id: str = Field(description="Unique pointer ID (tool_call_id:chunk:id)")
    tier: Literal["shortterm", "longterm"] = Field(description="Memory tier")
    score: float = Field(description="Search relevance score")


class EntityPointer(BaseModel):
    """Pointer reference to an entity in central storage."""

    pointer_id: str = Field(description="Unique pointer ID (tool_call_id:entity:id)")
    name: str = Field(description="Entity name")
    types: List[str] = Field(description="Entity types")
    tier: Literal["shortterm", "longterm"] = Field(description="Memory tier")
    importance: float = Field(description="Entity importance score")


class RelationshipPointer(BaseModel):
    """Pointer reference to a relationship in central storage."""

    pointer_id: str = Field(description="Unique pointer ID (tool_call_id:relationship:id)")
    from_entity: str = Field(description="Source entity name")
    to_entity: str = Field(description="Target entity name")
    types: List[str] = Field(description="Relationship types")
    tier: Literal["shortterm", "longterm"] = Field(description="Memory tier")


class RetrievalResult(BaseModel):
    """
    Final retrieval result with pointer-based references.

    Mode determines behavior:
    - pointer: Returns IDs, caller resolves from storage (fast, low token cost)
    - synthesis: Returns synthesized summary (slower, higher token cost)
    """

    mode: Literal["pointer", "synthesis"] = Field(
        description="Result mode: pointer (IDs only) or synthesis (full summary)"
    )
    chunks: List[ChunkPointer] = Field(
        default_factory=list, description="Pointer references to retrieved chunks"
    )
    entities: List[EntityPointer] = Field(
        default_factory=list, description="Pointer references to retrieved entities"
    )
    relationships: List[RelationshipPointer] = Field(
        default_factory=list, description="Pointer references to retrieved relationships"
    )
    synthesis: Optional[str] = Field(
        default=None, description="Natural language synthesis (only in synthesis mode)"
    )
    search_strategy: str = Field(description="Brief explanation of search approach and decisions")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in result relevance (0-1)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the search (counts, timing, etc.)",
    )


# ============================================================================
# TOOLS
# ============================================================================


async def search_shortterm_chunks(
    ctx: RunContext[RetrieverDeps],
    query_text: str,
    limit: int = 10,
    shortterm_memory_id: Optional[int] = None,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
    min_similarity_score: Optional[float] = None,
    min_bm25_score: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Search shortterm memory chunks using hybrid search.

    Stores raw chunks in central storage and returns pointer references.

    Args:
        query_text: Text query for search
        limit: Maximum number of results (default: 10)
        shortterm_memory_id: Optional filter by specific shortterm memory ID
        vector_weight: Weight for vector similarity (0-1, default: 0.5)
        bm25_weight: Weight for BM25 score (0-1, default: 0.5)
        min_similarity_score: Optional minimum vector similarity threshold
        min_bm25_score: Optional minimum BM25 score threshold

    Returns:
        Dictionary with pointer IDs and metadata
    """
    try:
        tool_call_id = ctx.tool_call_id
        logger.info(
            f"[{tool_call_id}] Searching shortterm chunks: '{query_text[:50]}...' "
            f"(limit={limit}, memory_id={shortterm_memory_id})"
        )

        # Get singleton storage
        storage = get_central_storage()

        # Generate embedding for query
        query_embedding = await ctx.deps.embedding_service.get_embedding(query_text)

        # Perform hybrid search
        chunks = await ctx.deps.shortterm_repo.hybrid_search(
            external_id=ctx.deps.external_id,
            query_text=query_text,
            query_embedding=query_embedding,
            limit=limit,
            shortterm_memory_id=shortterm_memory_id,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            min_similarity_score=min_similarity_score,
            min_bm25_score=min_bm25_score,
        )

        # Store chunks and create pointers
        pointers = []
        for chunk in chunks:
            pointer_id = storage.store_chunk(ctx.deps.external_id, tool_call_id, chunk)

            combined_score = vector_weight * (chunk.similarity_score or 0.0) + bm25_weight * (
                chunk.bm25_score or 0.0
            )

            pointers.append(
                {
                    "pointer_id": pointer_id,
                    "tier": "shortterm",
                    "chunk_content": chunk.content,
                    "score": combined_score,
                }
            )

        logger.info(
            f"[{tool_call_id}] Found {len(pointers)} shortterm chunks, stored in central storage"
        )
        return {
            "success": True,
            "pointers": pointers,
            "count": len(pointers),
            "tier": "shortterm",
        }

    except Exception as e:
        logger.error(f"Error searching shortterm chunks: {e}", exc_info=True)
        return {"success": False, "error": str(e), "pointers": [], "count": 0}


async def search_longterm_chunks(
    ctx: RunContext[RetrieverDeps],
    query_text: str,
    limit: int = 10,
    min_importance: Optional[float] = None,
    shortterm_memory_id: Optional[int] = None,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
    min_similarity_score: Optional[float] = None,
    min_bm25_score: Optional[float] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Search longterm memory chunks using hybrid search.

    Stores raw chunks in central storage and returns pointer references.

    Args:
        query_text: Text query for search
        limit: Maximum number of results (default: 10)
        min_importance: Minimum importance threshold (optional)
        shortterm_memory_id: Optional filter by specific shortterm memory that chunks were merged from
        vector_weight: Weight for vector similarity (0-1, default: 0.5)
        bm25_weight: Weight for BM25 score (0-1, default: 0.5)
        min_similarity_score: Optional minimum vector similarity threshold
        min_bm25_score: Optional minimum BM25 score threshold
        start_date: Filter chunks with start_date >= this (ISO format string, optional)
        end_date: Filter chunks with start_date <= this (ISO format string, optional)

    Returns:
        Dictionary with pointer IDs and metadata
    """
    try:
        tool_call_id = ctx.tool_call_id
        logger.info(
            f"[{tool_call_id}] Searching longterm chunks: '{query_text[:50]}...' "
            f"(limit={limit}, memory_id={shortterm_memory_id}, start_date={start_date}, end_date={end_date})"
        )

        # Get singleton storage
        storage = get_central_storage()

        # Generate embedding for query
        query_embedding = await ctx.deps.embedding_service.get_embedding(query_text)

        # Parse dates if provided
        from datetime import datetime

        start_date_obj = datetime.fromisoformat(start_date) if start_date else None
        end_date_obj = datetime.fromisoformat(end_date) if end_date else None

        # Perform hybrid search
        kwargs = {
            "external_id": ctx.deps.external_id,
            "query_text": query_text,
            "query_embedding": query_embedding,
            "limit": limit,
            "vector_weight": vector_weight,
            "bm25_weight": bm25_weight,
        }

        if min_importance is not None:
            kwargs["min_importance"] = min_importance
        if shortterm_memory_id is not None:
            kwargs["shortterm_memory_id"] = shortterm_memory_id
        if min_similarity_score is not None:
            kwargs["min_similarity_score"] = min_similarity_score
        if min_bm25_score is not None:
            kwargs["min_bm25_score"] = min_bm25_score
        if start_date_obj is not None:
            kwargs["start_date"] = start_date_obj
        if end_date_obj is not None:
            kwargs["end_date"] = end_date_obj

        chunks = await ctx.deps.longterm_repo.hybrid_search(**kwargs)

        # Store chunks and create pointers
        pointers = []
        for chunk in chunks:
            pointer_id = storage.store_chunk(ctx.deps.external_id, tool_call_id, chunk)

            combined_score = vector_weight * (chunk.similarity_score or 0.0) + bm25_weight * (
                chunk.bm25_score or 0.0
            )

            pointers.append(
                {
                    "pointer_id": pointer_id,
                    "tier": "longterm",
                    "chunk_content": chunk.content,
                    "score": combined_score,
                    "importance": chunk.importance,
                    "start_date": chunk.start_date.isoformat() if chunk.start_date else None,
                }
            )

        logger.info(
            f"[{tool_call_id}] Found {len(pointers)} longterm chunks, stored in central storage"
        )
        return {
            "success": True,
            "pointers": pointers,
            "count": len(pointers),
            "tier": "longterm",
        }

    except Exception as e:
        logger.error(f"Error searching longterm chunks: {e}", exc_info=True)
        return {"success": False, "error": str(e), "pointers": [], "count": 0}


async def search_shortterm_entities(
    ctx: RunContext[RetrieverDeps],
    query_text: str,
    limit: int = 10,
    shortterm_memory_id: Optional[int] = None,
    min_importance: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Search shortterm entities and their relationships.

    Extracts entity names from query text, then searches for entities and relationships.
    Stores raw entities and relationships in central storage and returns pointer references.

    Args:
        query_text: Text query containing entities to search for
        limit: Maximum number of results per entity (default: 10)
        shortterm_memory_id: Optional filter by specific shortterm memory ID
        min_importance: Optional minimum importance threshold

    Returns:
        Dictionary with pointer IDs and metadata
    """
    try:
        tool_call_id = ctx.tool_call_id
        logger.info(f"[{tool_call_id}] Extracting entities from query: '{query_text[:50]}...'")

        # Extract entity names from query text
        entity_names = await extract_entities(query_text)
        logger.info(f"[{tool_call_id}] Extracted entities: {entity_names}")

        if not entity_names:
            logger.info(f"[{tool_call_id}] No entities extracted from query")
            return {
                "success": True,
                "entity_pointers": [],
                "relationship_pointers": [],
                "tier": "shortterm",
            }

        logger.info(f"[{tool_call_id}] Extracted entities: {entity_names}")
        logger.info(
            f"[{tool_call_id}] Searching shortterm entities: {entity_names} "
            f"(memory_id={shortterm_memory_id})"
        )

        # Get singleton storage
        storage = get_central_storage()

        result = await ctx.deps.shortterm_repo.search_entities_with_relationships(
            entity_names=entity_names,
            external_id=ctx.deps.external_id,
            limit=limit,
            shortterm_memory_id=shortterm_memory_id,
            min_importance=min_importance,
        )

        # Combine matched and related entities
        all_entities = result.matched_entities + result.related_entities

        # Store entities and create pointers
        entity_pointers = []
        for entity in all_entities:
            pointer_id = storage.store_entity(ctx.deps.external_id, tool_call_id, entity)

            entity_pointers.append(
                {
                    "pointer_id": pointer_id,
                    "name": entity.name,
                    "types": entity.types,
                    "description": entity.description,
                    "importance": entity.importance,
                }
            )

        # Store relationships and create pointers
        relationship_pointers = []
        for rel in result.relationships:
            pointer_id = storage.store_relationship(ctx.deps.external_id, tool_call_id, rel)

            relationship_pointers.append(
                {
                    "pointer_id": pointer_id,
                    "from_entity": rel.from_entity_name,
                    "to_entity": rel.to_entity_name,
                    "types": rel.types,
                    "importance": rel.importance,
                }
            )

        logger.info(
            f"[{tool_call_id}] Found {len(entity_pointers)} entities "
            f"and {len(relationship_pointers)} relationships in shortterm"
        )
        return {
            "success": True,
            "entity_pointers": entity_pointers,
            "relationship_pointers": relationship_pointers,
            "tier": "shortterm",
        }

    except Exception as e:
        logger.error(f"Error searching shortterm entities: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "entity_pointers": [],
            "relationship_pointers": [],
        }


async def search_longterm_entities(
    ctx: RunContext[RetrieverDeps],
    query_text: str,
    limit: int = 10,
    min_importance: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Search longterm entities and their relationships.

    Extracts entity names from query text, then searches for entities and relationships.
    Stores raw entities and relationships in central storage and returns pointer references.

    Args:
        query_text: Text query containing entities to search for
        limit: Maximum number of results per entity (default: 10)
        min_importance: Minimum importance threshold (optional)

    Returns:
        Dictionary with pointer IDs and metadata
    """
    try:
        tool_call_id = ctx.tool_call_id
        logger.info(f"[{tool_call_id}] Extracting entities from query: '{query_text[:50]}...'")

        # Extract entity names from query text
        entity_names = await extract_entities(query_text)
        logger.info(f"[{tool_call_id}] Extracted entities: {entity_names}")

        if not entity_names:
            logger.info(f"[{tool_call_id}] No entities extracted from query")
            return {
                "success": True,
                "entity_pointers": [],
                "relationship_pointers": [],
                "tier": "longterm",
            }

        logger.info(f"[{tool_call_id}] Extracted entities: {entity_names}")
        logger.info(f"[{tool_call_id}] Searching longterm entities: {entity_names}")

        # Get singleton storage
        storage = get_central_storage()

        result = await ctx.deps.longterm_repo.search_entities_with_relationships(
            entity_names=entity_names,
            external_id=ctx.deps.external_id,
            limit=limit,
            min_importance=min_importance,
        )

        # Combine matched and related entities
        all_entities = result.matched_entities + result.related_entities

        # Store entities and create pointers
        entity_pointers = []
        for entity in all_entities:
            pointer_id = storage.store_entity(ctx.deps.external_id, tool_call_id, entity)

            entity_pointers.append(
                {
                    "pointer_id": pointer_id,
                    "name": entity.name,
                    "types": entity.types,
                    "description": entity.description,
                    "importance": entity.importance,
                }
            )

        # Store relationships and create pointers
        relationship_pointers = []
        for rel in result.relationships:
            pointer_id = storage.store_relationship(ctx.deps.external_id, tool_call_id, rel)

            relationship_pointers.append(
                {
                    "pointer_id": pointer_id,
                    "from_entity": rel.from_entity_name,
                    "to_entity": rel.to_entity_name,
                    "types": rel.types,
                    "importance": rel.importance,
                }
            )

        logger.info(
            f"[{tool_call_id}] Found {len(entity_pointers)} entities "
            f"and {len(relationship_pointers)} relationships in longterm"
        )
        return {
            "success": True,
            "entity_pointers": entity_pointers,
            "relationship_pointers": relationship_pointers,
            "tier": "longterm",
        }

    except Exception as e:
        logger.error(f"Error searching longterm entities: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "entity_pointers": [],
            "relationship_pointers": [],
        }


# ============================================================================
# AGENT INITIALIZATION
# ============================================================================


def _get_retriever_agent() -> Agent[RetrieverDeps, RetrievalResult]:
    """
    Get or create the memory retriever agent (lazy initialization).

    This avoids requiring an API key at module import time.
    """
    config = get_config()

    # Get model from provider using configuration
    model = model_provider.get_model(config.memory_retrieve_agent_model)

    # Initialize the agent
    agent = Agent(
        model=model,
        deps_type=RetrieverDeps,
        output_type=RetrievalResult,
        tools=[
            Tool(search_shortterm_chunks, takes_ctx=True, docstring_format="google"),
            Tool(search_longterm_chunks, takes_ctx=True, docstring_format="google"),
            Tool(search_shortterm_entities, takes_ctx=True, docstring_format="google"),
            Tool(search_longterm_entities, takes_ctx=True, docstring_format="google"),
        ],
        system_prompt="""You are an expert memory retrieval agent designed for efficient, cost-effective information retrieval.

CORE MISSION:
Retrieve relevant memory data using a two-stage pipeline that optimizes for speed, relevance, and cost.

RETRIEVAL STRATEGY:

1. UNDERSTAND THE QUERY
   - Identify key concepts, entities, and relationships
   - Determine query complexity and scope
   - Assess confidence needs

2. SEARCH EXECUTION (MANDATORY TOOL USAGE)
   
   **IMPORTANT: You MUST use the search tools to retrieve information. Do not attempt to answer without using tools.**
   
   Search Strategy by Query Type:
   
   a) Text/Content Queries (keywords, descriptions, phrases):
      - ALWAYS use search_shortterm_chunks (recent context)
      - Consider search_longterm_chunks if historical depth needed
   
   b) Entity/Concept Queries (people, projects, technologies, organizations):
      - ALWAYS use search_shortterm_entities (pass query text, entities will be extracted automatically)
      - ALWAYS use search_longterm_entities (pass query text, entities will be extracted automatically)
      - These tools extract entities from your query text automatically using NLP
   
   c) Complex Queries (multiple aspects, relationships):
      - Use ALL FOUR TOOLS to gather comprehensive results:
        * search_shortterm_chunks
        * search_longterm_chunks
        * search_shortterm_entities
        * search_longterm_entities
   
   Key Notes:
   - Start with shortterm memory (recent, active information)
   - Extend to longterm memory when needed (historical context)
     * Use importance scores and temporal filtering
   - Entity search tools automatically extract entities from query text
   - Use multiple tools in parallel when query requires different types of information

3. RESULT MODE SELECTION (CRITICAL):
   
   DEFAULT MODE: "pointer"
   - Use pointer mode by DEFAULT for most queries
   - Return pointer IDs only (fast, low token cost)
   - Tool results already contain previews for context
   - Let the caller resolve full data from storage
   - This is the PREFERRED and EFFICIENT approach
   
   SYNTHESIS MODE: "synthesis" 
   - ONLY use synthesis mode when:
     a) Query is complex and requires cross-referencing multiple sources
     b) Confidence is low (< 0.7) and explanation would help
     c) Explicit synthesis is requested (check synthesis flag in context)
   - Generate natural language summary combining all findings
   - Higher token cost, slower response
   
4. SEARCH TOOLS (use ctx.tool_call_id for tracking):
   
   **Chunk Search (text-based):**
   - search_shortterm_chunks: Recent memory chunks (pointer mode)
     * Args: query_text (str), limit (int), shortterm_memory_id (int, optional), 
             vector_weight (float), bm25_weight (float), min_similarity_score (float, optional), 
             min_bm25_score (float, optional)
     * Use shortterm_memory_id to filter for a specific shortterm memory if needed
   - search_longterm_chunks: Historical chunks (pointer mode)
     * Args: query_text (str), limit (int), min_importance (float, optional), 
             shortterm_memory_id (int, optional), vector_weight (float), bm25_weight (float),
             min_similarity_score (float, optional), min_bm25_score (float, optional),
             start_date (str ISO format, optional), end_date (str ISO format, optional)
     * Use shortterm_memory_id to filter longterm chunks that were merged from specific shortterm memory
     * Use start_date/end_date for temporal filtering
   
   **Entity Search (automatically extracts entities from query text):**
   - search_shortterm_entities: Recent entities/relationships (pointer mode)
     * Args: query_text (str), limit (int), shortterm_memory_id (int, optional), 
             min_importance (float, optional)
     * Automatically extracts entity names from query_text using NLP
     * Use shortterm_memory_id to search within specific shortterm memory
   - search_longterm_entities: Historical entities/relationships (pointer mode)
     * Args: query_text (str), limit (int), min_importance (float, optional)
     * Automatically extracts entity names from query_text using NLP
   
   All tools store raw data in central storage and return:
   - Pointer IDs (tool_call_id:type:id)
   - Relevance scores
   - Metadata (tier, importance, etc.)

5. RESPONSE STRUCTURE:
   
   POINTER MODE (default):
   ```
   {
     "mode": "pointer",
     "chunks": [{"pointer_id": "...", "score": 0.9, ...}],
     "entities": [{"pointer_id": "...", "name": "...", ...}],
     "relationships": [{"pointer_id": "...", "from_entity": "...", ...}],
     "synthesis": null,  // No synthesis in pointer mode
     "search_strategy": "Searched shortterm chunks for recent task info...",
     "confidence": 0.95,
     "metadata": {"shortterm_chunks_searched": 10, ...}
   }
   ```
   
   SYNTHESIS MODE (when needed):
   ```
   {
     "mode": "synthesis",
     "chunks": [...],  // Still include pointers
     "entities": [...],
     "relationships": [...],
     "synthesis": "Based on retrieved memories...",  // Full explanation
     "search_strategy": "Complex query required cross-referencing...",
     "confidence": 0.65,
     "metadata": {...}
   }
   ```

6. CONFIDENCE SCORING:
   - High (0.8-1.0): Clear, direct matches, simple query
   - Medium (0.5-0.8): Partial matches, moderate query complexity
   - Low (0.0-0.5): Weak matches, complex/ambiguous query
   
   Low confidence may trigger synthesis mode for better explanation.

EFFICIENCY PRINCIPLES:
✓ Prefer pointer mode (fast, cheap)
✓ Use tool results for context (metadata included)
✓ Only synthesize when truly beneficial
✓ Let caller resolve full data on demand
✓ Report search metadata for transparency

REMEMBER: Your primary job is EFFICIENT RETRIEVAL, not synthesis. Return pointers by default!""",
    )

    return agent


# ============================================================================
# RESULT OUTPUT HANDLER
# ============================================================================


def resolve_and_format_results(
    retrieval_result: RetrievalResult,
    external_id: str,
) -> FinalRetrievalResult:
    """
    Resolve pointer IDs from storage and format results for return.

    Args:
        retrieval_result: Agent's output with pointer IDs
        external_id: Agent identifier to resolve storage from

    Returns:
        FinalRetrievalResult with resolved data
    """
    # Get singleton storage
    storage = get_central_storage()

    # Resolve chunks
    chunks = []
    for chunk_pointer in retrieval_result.chunks:
        raw_chunk = storage.get_chunk(external_id, chunk_pointer.pointer_id)
        if raw_chunk:
            chunk_data = {
                "id": raw_chunk.id,
                "content": raw_chunk.content,
                "tier": chunk_pointer.tier,
                "score": chunk_pointer.score,
            }

            # Add tier-specific fields
            if isinstance(raw_chunk, LongtermMemoryChunk):
                chunk_data["importance"] = raw_chunk.importance
                chunk_data["start_date"] = raw_chunk.start_date
            else:
                chunk_data["importance"] = None
                chunk_data["start_date"] = None

            chunks.append(RetrievedChunk(**chunk_data))

    # Resolve entities
    entities = []
    for entity_pointer in retrieval_result.entities:
        raw_entity = storage.get_entity(external_id, entity_pointer.pointer_id)
        if raw_entity:
            entities.append(
                RetrievedEntity(
                    id=raw_entity.id,
                    name=raw_entity.name,
                    types=raw_entity.types,
                    description=raw_entity.description,
                    tier=entity_pointer.tier,
                    importance=raw_entity.importance,
                )
            )

    # Resolve relationships
    relationships = []
    for rel_pointer in retrieval_result.relationships:
        raw_rel = storage.get_relationship(external_id, rel_pointer.pointer_id)
        if raw_rel:
            relationships.append(
                RetrievedRelationship(
                    id=raw_rel.id,
                    from_entity_name=raw_rel.from_entity_name,
                    to_entity_name=raw_rel.to_entity_name,
                    types=raw_rel.types,
                    description=raw_rel.description,
                    tier=rel_pointer.tier,
                    importance=raw_rel.importance,
                )
            )

    # Return FinalRetrievalResult
    return FinalRetrievalResult(
        mode=retrieval_result.mode,
        chunks=chunks,
        entities=entities,
        relationships=relationships,
        synthesis=retrieval_result.synthesis if retrieval_result.mode == "synthesis" else None,
        search_strategy=retrieval_result.search_strategy,
        confidence=retrieval_result.confidence,
        metadata=retrieval_result.metadata,
    )


# ============================================================================
# MAIN RETRIEVAL FUNCTION
# ============================================================================


async def retrieve_memory(
    query: str,
    external_id: str,
    shortterm_repo: ShorttermMemoryRepository,
    longterm_repo: LongtermMemoryRepository,
    active_repo: ActiveMemoryRepository,
    embedding_service: EmbeddingService,
    synthesis: bool = False,
) -> FinalRetrievalResult:
    """
    Retrieve relevant memory information based on a query.

    Uses two-stage pipeline:
    1. Search and store raw data in central storage
    2. Return pointer references by default (synthesis only when needed)

    Args:
        query: User's query text
        external_id: Agent identifier
        shortterm_repo: Shortterm memory repository
        longterm_repo: Longterm memory repository
        active_repo: Active memory repository
        embedding_service: Embedding service for vector generation
        synthesis: If True, force synthesis mode regardless of query (default: False)

    Returns:
        FinalRetrievalResult with resolved chunks, entities, relationships, and optional synthesis
    """
    logger.info(
        f"Starting memory retrieval for external_id={external_id}, "
        f"query='{query[:50]}...', synthesis={synthesis}"
    )

    # Get active memory templates for context
    active_templates = await active_repo.get_all_templates_by_external_id(external_id)
    logger.info(f"Retrieved {len(active_templates)} active memory templates")

    shortterm_memories = await shortterm_repo.get_memories_by_external_id(external_id)

    # Create dependencies (singleton storage is accessed by tools)
    deps = RetrieverDeps(
        external_id=external_id,
        query=query,
        synthesis=synthesis,
        shortterm_repo=shortterm_repo,
        longterm_repo=longterm_repo,
        embedding_service=embedding_service,
    )

    # Format active templates as context
    template_context = "\n\n".join(
        [
            f"## Active Memory: {tmpl['title'] if isinstance(tmpl, dict) else tmpl.title}\n```\n{str(tmpl['template_content']) if isinstance(tmpl, dict) else str(tmpl.template_content)}\n```"
            for tmpl in active_templates
        ]
    )

    # Format shortterm memories (without chunks) as context
    shortterm_context = "\n".join(
        [
            f"- ID: {mem.id}, Title: {mem.title}, Summary: {mem.summary or 'N/A'}, Update Count: {mem.update_count}, Metadata: {mem.metadata or {}}"
            for mem in shortterm_memories
        ]
    )

    synthesis_instruction = ""
    if synthesis:
        synthesis_instruction = "\n\n**IMPORTANT: Synthesis is REQUIRED for this query. Use mode='synthesis' and provide a comprehensive summary.**"

    user_prompt = f"""User Query: {query}

Active Memory Context (memory structures that exist):
{template_context if template_context else "No active memory templates found."}

Available Shortterm Memories:
{shortterm_context if shortterm_context else "No shortterm memories found."}
{synthesis_instruction}

Execute search and return results in the most efficient format (pointer mode by default, synthesis only if needed)."""

    try:
        # Get the agent instance (lazy initialization)
        agent = _get_retriever_agent()

        # Run the agent
        result = await agent.run(user_prompt=user_prompt, deps=deps)
        retrieval_result = result.output

        # Resolve pointers and format output
        output = resolve_and_format_results(retrieval_result, external_id)

        logger.info(
            f"Memory retrieval completed: mode={retrieval_result.mode}, "
            f"confidence={retrieval_result.confidence:.2f}, "
            f"{len(output.chunks)} chunks, {len(output.entities)} entities, "
            f"{len(output.relationships)} relationships"
        )

        logger.info(f"Memory retrieval successful")

        return output

    except Exception as e:
        logger.error(f"Error during memory retrieval: {e}", exc_info=True)
        # Return error response
        return FinalRetrievalResult(
            mode="synthesis",
            chunks=[],
            entities=[],
            relationships=[],
            synthesis=f"Error during memory retrieval: {str(e)}",
            search_strategy="Failed to execute search due to error",
            confidence=0.0,
            metadata={},
        )
    finally:
        # Clean up storage for this external_id
        storage = get_central_storage()
        storage.clear_external_id(external_id)
        logger.debug(f"Central storage cleared for external_id={external_id}")
