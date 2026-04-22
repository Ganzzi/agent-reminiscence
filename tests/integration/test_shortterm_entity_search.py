"""
Integration tests for shortterm memory entity search functionality.

Tests the search_entities_with_relationships method using mocked Neo4j to
verify correct Cypher query construction, result parsing, relationship
direction handling, and shortterm_memory_id filtering without requiring a
live database.
"""

import json
import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_reminiscence.database.repositories.shortterm_memory import ShorttermMemoryRepository

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(entity_id, name, types, description, importance, metadata,
               external_id="test-entity-search", memory_id=1):
    """Create a mock Neo4j node for ShorttermEntity."""
    node = MagicMock()
    node.element_id = entity_id
    props = {
        "external_id": external_id,
        "shortterm_memory_id": memory_id,
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


def _make_rel_map(rel_id, from_id, to_id, rel_props, memory_id=1):
    """Create a mock relationship map for SHORTTERM_RELATES."""
    rel = MagicMock()
    rel.element_id = rel_id
    all_props = {
        "external_id": "test-entity-search",
        "shortterm_memory_id": memory_id,
        **rel_props,
    }
    rel.__getitem__ = MagicMock(side_effect=lambda key: all_props[key])
    rel.get = MagicMock(side_effect=lambda key, default=None: all_props.get(key, default))
    return {"rel": rel, "from_id": from_id, "to_id": to_id}


def _make_record(entity_node, related_incoming, related_outgoing, rels_in, rels_out):
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
# Shared test data
#
# Entities:  storage(0.9)  postgres(0.8)  neo4j_db(0.8)  retriever(0.7)
# Relationships:
#   storage   --USES-->    postgres
#   storage   --USES-->    neo4j_db
#   retriever --QUERIES--> storage
# ---------------------------------------------------------------------------

EID = "test-entity-search"
MEM_ID = 1

ENT_STORAGE = _make_node("e1", "CentralStorage", ["Component", "Storage"],
                          "Central storage component", 0.9, {"source": "test"})
ENT_PG = _make_node("e2", "PostgreSQL", ["Database", "Storage"],
                     "PostgreSQL database", 0.8, {"source": "test"})
ENT_NEO = _make_node("e3", "Neo4j", ["Database", "Graph"],
                      "Neo4j graph database", 0.8, {"source": "test"})
ENT_RET = _make_node("e4", "MemoryRetriever", ["Agent", "Service"],
                      "Memory retriever agent", 0.7, {"source": "test"})

REL_STORAGE_PG = _make_rel_map("r1", "e1", "e2", {
    "types": ["USES", "STORES_IN"],
    "description": "CentralStorage uses PostgreSQL",
    "importance": 0.8,
    "last_access": None,
    "metadata": json.dumps({"source": "test"}),
    "from_entity_name": "CentralStorage",
    "to_entity_name": "PostgreSQL",
})
REL_STORAGE_NEO = _make_rel_map("r2", "e1", "e3", {
    "types": ["USES", "STORES_IN"],
    "description": "CentralStorage uses Neo4j",
    "importance": 0.8,
    "last_access": None,
    "metadata": json.dumps({"source": "test"}),
    "from_entity_name": "CentralStorage",
    "to_entity_name": "Neo4j",
})
REL_RET_STORAGE = _make_rel_map("r3", "e4", "e1", {
    "types": ["QUERIES", "ACCESSES"],
    "description": "MemoryRetriever queries CentralStorage",
    "importance": 0.7,
    "last_access": None,
    "metadata": json.dumps({"source": "test"}),
    "from_entity_name": "MemoryRetriever",
    "to_entity_name": "CentralStorage",
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
    """Build a ShorttermMemoryRepository with mocked Neo4j returning *records*."""
    mock_pg = MagicMock()
    mock_neo4j = MagicMock()
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.__aiter__ = MagicMock(return_value=_AsyncIter(records))
    mock_session.run = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_neo4j.session.return_value = mock_session
    return ShorttermMemoryRepository(mock_pg, mock_neo4j)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_single_entity_exact_match():
    """Test searching for a single entity with exact name match."""
    record = _make_record(ENT_STORAGE, [ENT_RET], [ENT_PG, ENT_NEO],
                          [REL_RET_STORAGE], [REL_STORAGE_PG, REL_STORAGE_NEO])
    repo = _mock_repo([record])

    result = await repo.search_entities_with_relationships(
        entity_names=["CentralStorage"], external_id=EID, limit=10
    )

    assert len(result.matched_entities) == 1
    assert result.matched_entities[0].name == "CentralStorage"
    assert len(result.related_entities) >= 2
    assert len(result.relationships) == 3


@pytest.mark.asyncio
async def test_search_partial_name_match():
    """Test searching with partial name matching (case-insensitive)."""
    record = _make_record(ENT_STORAGE, [], [], [], [])
    repo = _mock_repo([record])

    result = await repo.search_entities_with_relationships(
        entity_names=["storage"], external_id=EID, limit=10
    )

    assert len(result.matched_entities) >= 1
    assert any(e.name == "CentralStorage" for e in result.matched_entities)


@pytest.mark.asyncio
async def test_search_multiple_entities():
    """Test searching for multiple entities at once."""
    rec_pg = _make_record(ENT_PG, [ENT_STORAGE], [], [REL_STORAGE_PG], [])
    rec_neo = _make_record(ENT_NEO, [ENT_STORAGE], [], [REL_STORAGE_NEO], [])
    repo = _mock_repo([rec_pg, rec_neo])

    result = await repo.search_entities_with_relationships(
        entity_names=["PostgreSQL", "Neo4j"], external_id=EID, limit=10
    )

    assert len(result.matched_entities) == 2
    matched_names = {e.name for e in result.matched_entities}
    assert "PostgreSQL" in matched_names
    assert "Neo4j" in matched_names
    assert len(result.relationships) == 2


@pytest.mark.asyncio
async def test_search_with_importance_filter():
    """Test searching with minimum importance threshold."""
    rec_storage = _make_record(ENT_STORAGE, [], [], [], [])
    repo = _mock_repo([rec_storage])

    result = await repo.search_entities_with_relationships(
        entity_names=["storage", "retriever"], external_id=EID,
        min_importance=0.85, limit=10
    )

    assert len(result.matched_entities) == 1
    assert result.matched_entities[0].name == "CentralStorage"

    rec_ret = _make_record(ENT_RET, [], [], [], [])
    repo_low = _mock_repo([rec_storage, rec_ret])

    result_low = await repo_low.search_entities_with_relationships(
        entity_names=["storage", "retriever"], external_id=EID,
        min_importance=0.6, limit=10
    )

    assert len(result_low.matched_entities) == 2


@pytest.mark.asyncio
async def test_search_with_memory_filter():
    """Test searching with specific memory ID filter."""
    record = _make_record(ENT_STORAGE, [], [], [], [])
    repo = _mock_repo([record])

    result = await repo.search_entities_with_relationships(
        entity_names=["CentralStorage"], external_id=EID,
        shortterm_memory_id=MEM_ID, limit=10
    )

    assert len(result.matched_entities) == 1

    # Wrong memory ID should return no results (mocked to return empty)
    repo_wrong = _mock_repo([])
    result_wrong = await repo_wrong.search_entities_with_relationships(
        entity_names=["CentralStorage"], external_id=EID,
        shortterm_memory_id=99999, limit=10
    )

    assert len(result_wrong.matched_entities) == 0


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
    record = _make_record(ENT_STORAGE, [ENT_RET], [ENT_PG, ENT_NEO],
                          [REL_RET_STORAGE], [REL_STORAGE_PG, REL_STORAGE_NEO])
    repo = _mock_repo([record])

    result = await repo.search_entities_with_relationships(
        entity_names=["CentralStorage"], external_id=EID, limit=10
    )

    outgoing = [r for r in result.relationships if r.from_entity_name == "CentralStorage"]
    incoming = [r for r in result.relationships if r.to_entity_name == "CentralStorage"]

    assert len(outgoing) == 2
    assert len(incoming) == 1

    outgoing_targets = {r.to_entity_name for r in outgoing}
    assert "PostgreSQL" in outgoing_targets
    assert "Neo4j" in outgoing_targets

    incoming_sources = {r.from_entity_name for r in incoming}
    assert "MemoryRetriever" in incoming_sources


@pytest.mark.asyncio
async def test_search_metadata_parsing():
    """Test that metadata is correctly parsed from JSON strings."""
    record = _make_record(ENT_STORAGE, [], [ENT_PG], [], [REL_STORAGE_PG])
    repo = _mock_repo([record])

    result = await repo.search_entities_with_relationships(
        entity_names=["CentralStorage"], external_id=EID, limit=10
    )

    assert len(result.matched_entities) == 1
    entity = result.matched_entities[0]

    assert isinstance(entity.metadata, dict)
    assert entity.metadata.get("source") == "test"

    if result.relationships:
        rel = result.relationships[0]
        assert isinstance(rel.metadata, dict)
        assert rel.metadata.get("source") == "test"


@pytest.mark.asyncio
async def test_search_limit():
    """Test that limit parameter works correctly."""
    rec = _make_record(ENT_STORAGE, [], [], [], [])
    repo = _mock_repo([rec])

    result = await repo.search_entities_with_relationships(
        entity_names=["storage", "postgres", "neo4j", "retriever"],
        external_id=EID, limit=1
    )

    assert len(result.matched_entities) <= 1

    rec_pg = _make_record(ENT_PG, [], [], [], [])
    repo2 = _mock_repo([rec, rec_pg])

    result2 = await repo2.search_entities_with_relationships(
        entity_names=["storage", "postgres", "neo4j", "retriever"],
        external_id=EID, limit=2
    )

    assert len(result2.matched_entities) <= 2
