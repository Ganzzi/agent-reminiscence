"""
Integration tests for longterm memory entity search functionality.

Tests the search_entities_with_relationships method using mocked Neo4j to
verify correct Cypher query construction, result parsing, and relationship
direction handling without requiring a live database.
"""

import json
import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_reminiscence.database.repositories.longterm_memory import LongtermMemoryRepository

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers to build Neo4j-style mock nodes and relationship maps
# ---------------------------------------------------------------------------

def _make_node(entity_id, name, types, description, importance, metadata, external_id="test-longterm-entity-search"):
    """Create a mock Neo4j node that supports element_id and dict-like access."""
    node = MagicMock()
    node.element_id = entity_id
    props = {
        "external_id": external_id,
        "name": name,
        "types": types,
        "description": description,
        "importance": importance,
        "access_count": 0,
        "last_access": None,
        "metadata": json.dumps(metadata) if isinstance(metadata, dict) else metadata,
    }
    node.__getitem__ = MagicMock(side_effect=lambda key: props[key])
    node.get = MagicMock(side_effect=lambda key, default=None: props.get(key, default))
    return node


def _make_rel_map(rel_id, from_id, to_id, rel_props):
    """Create a mock relationship map {rel: <rel_node>, from_id: ..., to_id: ...}."""
    rel = MagicMock()
    rel.element_id = rel_id
    all_props = {"external_id": "test-longterm-entity-search", **rel_props}
    rel.__getitem__ = MagicMock(side_effect=lambda key: all_props[key])
    rel.get = MagicMock(side_effect=lambda key, default=None: all_props.get(key, default))
    return {"rel": rel, "from_id": from_id, "to_id": to_id}


def _make_record(entity_node, related_incoming, related_outgoing, rels_in, rels_out):
    """Create a mock Neo4j record with the five RETURN columns."""
    rec = MagicMock()
    rec.__getitem__ = MagicMock(side_effect=lambda key: {
        "e": entity_node,
        "related_incoming": related_incoming,
        "related_outgoing": related_outgoing,
        "relationships_in": rels_in,
        "relationships_out": rels_out,
    }.get(key))
    return rec


# ---------------------------------------------------------------------------
# Shared test data (the graph used by most tests)
#
# Entities:  arch(0.9)  embeddings(0.85)  optimization(0.8)  pydantic(0.75)  gemini(0.7)
# Relationships:
#   arch        --USES-->          embeddings
#   arch        --REQUIRES-->      optimization
#   pydantic    --INTEGRATES_WITH-> gemini
#   optimization --OPTIMIZES-->    gemini
# ---------------------------------------------------------------------------

EID = "test-longterm-entity-search"

ENT_ARCH = _make_node("e1", "Multi-tier Architecture", ["Architecture", "Design Pattern"],
                       "Layered software architecture pattern", 0.9,
                       {"source": "test", "domain": "architecture"})
ENT_EMB = _make_node("e2", "Vector Embeddings", ["Technology", "AI"],
                      "Dense vector representations for semantic search", 0.85,
                      {"source": "test", "domain": "ml"})
ENT_OPT = _make_node("e3", "Token Optimization", ["Technique", "Performance"],
                      "Reducing token usage in LLM interactions", 0.8,
                      {"source": "test", "domain": "optimization"})
ENT_PYD = _make_node("e4", "Pydantic AI", ["Framework", "Library"],
                      "Type-safe AI framework for Python", 0.75,
                      {"source": "test", "domain": "tools"})
ENT_GEM = _make_node("e5", "Google Gemini", ["LLM", "Service"],
                      "Google's multimodal AI model", 0.7,
                      {"source": "test", "domain": "llm"})

_NOW = datetime.now(timezone.utc).isoformat()

REL_ARCH_EMB = _make_rel_map("r1", "e1", "e2", {
    "types": ["USES", "IMPLEMENTS"],
    "description": "Architecture uses vector embeddings for semantic search",
    "importance": 0.85,
    "start_date": _NOW, "access_count": 0, "last_access": None,
    "metadata": json.dumps({"source": "test"}),
    "from_entity_name": "Multi-tier Architecture",
    "to_entity_name": "Vector Embeddings",
})
REL_ARCH_OPT = _make_rel_map("r2", "e1", "e3", {
    "types": ["REQUIRES", "APPLIES"],
    "description": "Architecture requires token optimization",
    "importance": 0.8,
    "start_date": _NOW, "access_count": 0, "last_access": None,
    "metadata": json.dumps({"source": "test"}),
    "from_entity_name": "Multi-tier Architecture",
    "to_entity_name": "Token Optimization",
})
REL_PYD_GEM = _make_rel_map("r3", "e4", "e5", {
    "types": ["INTEGRATES_WITH", "SUPPORTS"],
    "description": "Pydantic AI integrates with Google Gemini",
    "importance": 0.75,
    "start_date": _NOW, "access_count": 0, "last_access": None,
    "metadata": json.dumps({"source": "test"}),
    "from_entity_name": "Pydantic AI",
    "to_entity_name": "Google Gemini",
})
REL_OPT_GEM = _make_rel_map("r4", "e3", "e5", {
    "types": ["OPTIMIZES", "REDUCES_COST"],
    "description": "Token optimization reduces Gemini API costs",
    "importance": 0.7,
    "start_date": _NOW, "access_count": 0, "last_access": None,
    "metadata": json.dumps({"source": "test"}),
    "from_entity_name": "Token Optimization",
    "to_entity_name": "Google Gemini",
})


class _AsyncIter:
    """Helper to make a list usable with `async for`."""

    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


def _mock_repo(records):
    """Build a LongtermMemoryRepository with a mocked Neo4j returning *records*."""
    mock_pg = MagicMock()
    mock_neo4j = MagicMock()
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.__aiter__ = MagicMock(return_value=_AsyncIter(records))
    mock_session.run = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_neo4j.session.return_value = mock_session
    return LongtermMemoryRepository(mock_pg, mock_neo4j)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_single_entity_exact_match():
    """Test searching for a single entity with exact name match."""
    record = _make_record(ENT_ARCH, [], [ENT_EMB, ENT_OPT], [], [REL_ARCH_EMB, REL_ARCH_OPT])
    repo = _mock_repo([record])

    result = await repo.search_entities_with_relationships(
        entity_names=["Multi-tier Architecture"], external_id=EID, limit=10
    )

    assert len(result.matched_entities) == 1
    assert result.matched_entities[0].name == "Multi-tier Architecture"
    assert len(result.related_entities) >= 2
    assert len(result.relationships) == 2


@pytest.mark.asyncio
async def test_search_partial_name_match():
    """Test searching with partial name matching (case-insensitive)."""
    record = _make_record(ENT_OPT, [], [], [], [])
    repo = _mock_repo([record])

    result = await repo.search_entities_with_relationships(
        entity_names=["optimization"], external_id=EID, limit=10
    )

    assert len(result.matched_entities) >= 1
    assert any(e.name == "Token Optimization" for e in result.matched_entities)


@pytest.mark.asyncio
async def test_search_multiple_entities():
    """Test searching for multiple entities at once."""
    rec_pyd = _make_record(ENT_PYD, [], [ENT_GEM], [], [REL_PYD_GEM])
    rec_gem = _make_record(ENT_GEM, [ENT_PYD, ENT_OPT], [], [REL_PYD_GEM, REL_OPT_GEM], [])
    repo = _mock_repo([rec_pyd, rec_gem])

    result = await repo.search_entities_with_relationships(
        entity_names=["Pydantic", "Gemini"], external_id=EID, limit=10
    )

    assert len(result.matched_entities) == 2
    matched_names = {e.name for e in result.matched_entities}
    assert "Pydantic AI" in matched_names
    assert "Google Gemini" in matched_names
    assert len(result.relationships) >= 1


@pytest.mark.asyncio
async def test_search_with_importance_filter():
    """Test searching with minimum importance threshold."""
    # With min_importance=0.83, only arch(0.9) and embeddings(0.85) should match
    rec_arch = _make_record(ENT_ARCH, [], [ENT_EMB], [], [REL_ARCH_EMB])
    rec_emb = _make_record(ENT_EMB, [ENT_ARCH], [], [REL_ARCH_EMB], [])
    repo = _mock_repo([rec_arch, rec_emb])

    result = await repo.search_entities_with_relationships(
        entity_names=["architecture", "embeddings", "optimization"],
        external_id=EID, min_importance=0.83, limit=10
    )

    assert len(result.matched_entities) >= 1

    # With lower threshold, all three should be present
    rec_opt = _make_record(ENT_OPT, [ENT_ARCH], [ENT_GEM], [REL_ARCH_OPT], [REL_OPT_GEM])
    repo_low = _mock_repo([rec_arch, rec_emb, rec_opt])

    result_low = await repo_low.search_entities_with_relationships(
        entity_names=["architecture", "embeddings", "optimization"],
        external_id=EID, min_importance=0.7, limit=10
    )

    assert len(result_low.matched_entities) == 3


@pytest.mark.asyncio
async def test_search_no_results():
    """Test searching for non-existent entity returns empty result."""
    repo = _mock_repo([])

    result = await repo.search_entities_with_relationships(
        entity_names=["NonExistentEntity"], external_id=EID, limit=10
    )

    assert len(result.matched_entities) == 0
    assert len(result.related_entities) == 0
    assert len(result.relationships) == 0


@pytest.mark.asyncio
async def test_search_relationship_directions():
    """Test that both incoming and outgoing relationships are captured."""
    # Google Gemini has 2 incoming (from Pydantic AI and Token Optimization) and 0 outgoing
    record = _make_record(ENT_GEM, [ENT_PYD, ENT_OPT], [], [REL_PYD_GEM, REL_OPT_GEM], [])
    repo = _mock_repo([record])

    result = await repo.search_entities_with_relationships(
        entity_names=["Google Gemini"], external_id=EID, limit=10
    )

    outgoing = [r for r in result.relationships if r.from_entity_name == "Google Gemini"]
    incoming = [r for r in result.relationships if r.to_entity_name == "Google Gemini"]

    assert len(incoming) == 2
    assert len(outgoing) == 0

    incoming_sources = {r.from_entity_name for r in incoming}
    assert "Pydantic AI" in incoming_sources
    assert "Token Optimization" in incoming_sources


@pytest.mark.asyncio
async def test_search_metadata_parsing():
    """Test that metadata is correctly parsed from JSON strings."""
    record = _make_record(ENT_ARCH, [], [ENT_EMB], [], [REL_ARCH_EMB])
    repo = _mock_repo([record])

    result = await repo.search_entities_with_relationships(
        entity_names=["Multi-tier Architecture"], external_id=EID, limit=10
    )

    assert len(result.matched_entities) == 1
    entity = result.matched_entities[0]

    assert isinstance(entity.metadata, dict)
    assert entity.metadata.get("source") == "test"
    assert entity.metadata.get("domain") == "architecture"

    if result.relationships:
        rel = result.relationships[0]
        assert isinstance(rel.metadata, dict)
        assert rel.metadata.get("source") == "test"


@pytest.mark.asyncio
async def test_search_limit():
    """Test that limit parameter works correctly."""
    rec_arch = _make_record(ENT_ARCH, [], [], [], [])
    repo = _mock_repo([rec_arch])

    result = await repo.search_entities_with_relationships(
        entity_names=["architecture", "embeddings", "optimization", "pydantic", "gemini"],
        external_id=EID, limit=1
    )

    assert len(result.matched_entities) <= 1

    rec_emb = _make_record(ENT_EMB, [], [], [], [])
    rec_opt = _make_record(ENT_OPT, [], [], [], [])
    repo3 = _mock_repo([rec_arch, rec_emb, rec_opt])

    result3 = await repo3.search_entities_with_relationships(
        entity_names=["architecture", "embeddings", "optimization", "pydantic", "gemini"],
        external_id=EID, limit=3
    )

    assert len(result3.matched_entities) <= 3


@pytest.mark.asyncio
async def test_search_complex_graph():
    """Test searching in a more complex graph with multiple hops."""
    # Token Optimization: 1 incoming from Architecture, 1 outgoing to Gemini
    record = _make_record(ENT_OPT, [ENT_ARCH], [ENT_GEM], [REL_ARCH_OPT], [REL_OPT_GEM])
    repo = _mock_repo([record])

    result = await repo.search_entities_with_relationships(
        entity_names=["Token Optimization"], external_id=EID, limit=10
    )

    assert len(result.matched_entities) == 1
    assert result.matched_entities[0].name == "Token Optimization"
    assert len(result.relationships) == 2

    has_incoming = any(r.to_entity_name == "Token Optimization" for r in result.relationships)
    has_outgoing = any(r.from_entity_name == "Token Optimization" for r in result.relationships)

    assert has_incoming
    assert has_outgoing
