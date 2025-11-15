"""
Tests for v0.2.0 triplet conversion functionality.

Tests the conversion of Neo4j relationships to RDF-style triplets in both
shortterm and longterm memory repositories.
"""

import pytest
from datetime import datetime, timezone
from agent_reminiscence.database.models import (
    ShorttermRelationship,
    LongtermRelationship,
    ShorttermKnowledgeTriplet,
    LongtermKnowledgeTriplet,
)
from agent_reminiscence.database.repositories.shortterm_memory import (
    _relationship_to_triplet as st_relationship_to_triplet,
)
from agent_reminiscence.database.repositories.longterm_memory import (
    _relationship_to_triplet as lt_relationship_to_triplet,
)


class TestShorttermTripletConversion:
    """Test triplet conversion for shortterm relationships."""

    def test_single_type_conversion(self):
        """Test converting relationship with single type to triplet."""
        rel = ShorttermRelationship(
            id="rel_1",
            external_id="agent-1",
            shortterm_memory_id=1,
            from_entity_id="entity_1",
            to_entity_id="entity_2",
            from_entity_name="JWT",
            to_entity_name="Authentication",
            types=["UsedFor"],
            description="JWT is used for authentication",
            importance=0.9,
            access_count=5,
            last_access=datetime.now(timezone.utc),
            metadata={"confidence": 0.95},
        )

        triplet = st_relationship_to_triplet(rel, "JWT", "Authentication")

        assert isinstance(triplet, ShorttermKnowledgeTriplet)
        assert triplet.subject == "JWT"
        assert triplet.predicate == "UsedFor"
        assert triplet.object == "Authentication"
        assert triplet.importance == 0.9
        assert triplet.description == "JWT is used for authentication"
        assert triplet.metadata["relationship_id"] == "rel_1"
        assert triplet.metadata["confidence"] == 0.95
        assert triplet.metadata["additional_types"] == []

    def test_multiple_types_conversion(self):
        """Test converting relationship with multiple types to triplet."""
        rel = ShorttermRelationship(
            id="rel_2",
            external_id="agent-1",
            shortterm_memory_id=1,
            from_entity_id="entity_1",
            to_entity_id="entity_2",
            from_entity_name="Python",
            to_entity_name="FastAPI",
            types=["CreatedWith", "Supports", "Built"],  # Multiple types
            description="Python supports FastAPI framework",
            importance=0.85,
            access_count=10,
            last_access=datetime.now(timezone.utc),
            metadata={},
        )

        triplet = st_relationship_to_triplet(rel, "Python", "FastAPI")

        assert triplet.subject == "Python"
        assert triplet.predicate == "CreatedWith"  # First type is predicate
        assert triplet.object == "FastAPI"
        # Additional types stored in metadata
        assert triplet.metadata["additional_types"] == ["Supports", "Built"]

    def test_no_types_fallback(self):
        """Test converting relationship with no types uses fallback."""
        rel = ShorttermRelationship(
            id="rel_3",
            external_id="agent-1",
            shortterm_memory_id=1,
            from_entity_id="entity_1",
            to_entity_id="entity_2",
            from_entity_name="A",
            to_entity_name="B",
            types=[],  # No types
            description="Generic relationship",
            importance=0.5,
            access_count=1,
            last_access=None,
            metadata={},
        )

        triplet = st_relationship_to_triplet(rel, "A", "B")

        assert triplet.predicate == "RELATED_TO"  # Fallback predicate

    def test_triplet_metadata_preservation(self):
        """Test that relationship metadata is preserved in triplet."""
        rel = ShorttermRelationship(
            id="rel_4",
            external_id="agent-1",
            shortterm_memory_id=1,
            from_entity_id="entity_1",
            to_entity_id="entity_2",
            from_entity_name="Entity1",
            to_entity_name="Entity2",
            types=["ConnectedTo"],
            description="Connection description",
            importance=0.75,
            access_count=3,
            last_access=datetime(2025, 11, 15, 10, 30, 0, tzinfo=timezone.utc),
            metadata={"custom_field": "custom_value", "confidence": 0.88},
        )

        triplet = st_relationship_to_triplet(rel, "Entity1", "Entity2")

        assert triplet.access_count == 3
        assert triplet.metadata["relationship_id"] == "rel_4"
        assert triplet.metadata["confidence"] == 0.88


class TestLongtermTripletConversion:
    """Test triplet conversion for longterm relationships."""

    def test_longterm_single_type_conversion(self):
        """Test converting longterm relationship to triplet."""
        rel = LongtermRelationship(
            id="lt_rel_1",
            external_id="agent-1",
            from_entity_id="entity_1",
            to_entity_id="entity_2",
            from_entity_name="PostgreSQL",
            to_entity_name="VectorDB",
            types=["Powers"],
            description="PostgreSQL powers vector databases",
            importance=0.92,
            start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
            access_count=10,
            last_access=datetime(2025, 11, 15, tzinfo=timezone.utc),
            metadata={},
        )

        triplet = lt_relationship_to_triplet(rel, "PostgreSQL", "VectorDB")

        assert isinstance(triplet, LongtermKnowledgeTriplet)
        assert triplet.subject == "PostgreSQL"
        assert triplet.predicate == "Powers"
        assert triplet.object == "VectorDB"
        assert triplet.importance == 0.92
        # Check temporal metadata
        assert triplet.start_date == datetime(2025, 1, 1, tzinfo=timezone.utc)
        assert "last_access" in triplet.metadata

    def test_longterm_temporal_metadata(self):
        """Test that longterm metadata includes temporal info."""
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        last_access = datetime(2025, 11, 15, tzinfo=timezone.utc)

        rel = LongtermRelationship(
            id="lt_rel_2",
            external_id="agent-1",
            from_entity_id="entity_1",
            to_entity_id="entity_2",
            from_entity_name="A",
            to_entity_name="B",
            types=["Relates"],
            description="Temporal relationship",
            importance=0.7,
            start_date=start,
            access_count=5,
            last_access=last_access,
            metadata={},
        )

        triplet = lt_relationship_to_triplet(rel, "A", "B")

        assert triplet.start_date == start
        assert triplet.metadata["last_access"] == last_access.isoformat()
        assert triplet.access_count == 5

    def test_longterm_multiple_types(self):
        """Test longterm relationship with multiple types."""
        rel = LongtermRelationship(
            id="lt_rel_3",
            external_id="agent-1",
            from_entity_id="entity_1",
            to_entity_id="entity_2",
            from_entity_name="AI",
            to_entity_name="LLM",
            types=["Includes", "Enables", "Defines"],
            description="AI includes LLM",
            importance=0.95,
            start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
            access_count=20,
            last_access=datetime(2025, 11, 15, tzinfo=timezone.utc),
            metadata={"custom": "data"},
        )

        triplet = lt_relationship_to_triplet(rel, "AI", "LLM")

        assert triplet.predicate == "Includes"
        assert triplet.metadata["additional_types"] == ["Enables", "Defines"]
        assert triplet.metadata["custom"] == "data"


class TestTripletFormat:
    """Test triplet model compliance."""

    def test_triplet_importance_bounds(self):
        """Test that triplet importance is within valid bounds."""
        rel = ShorttermRelationship(
            id="rel",
            external_id="agent-1",
            shortterm_memory_id=1,
            from_entity_id="e1",
            to_entity_id="e2",
            from_entity_name="A",
            to_entity_name="B",
            types=["Relates"],
            description="Test",
            importance=0.0,  # Min bound
            access_count=0,
            last_access=None,
            metadata={},
        )

        triplet = st_relationship_to_triplet(rel, "A", "B")
        assert 0.0 <= triplet.importance <= 1.0

        # Test max bound
        rel.importance = 1.0
        triplet = st_relationship_to_triplet(rel, "A", "B")
        assert 0.0 <= triplet.importance <= 1.0

    def test_triplet_tier_designation(self):
        """Test that triplets get correct tier designation."""
        st_rel = ShorttermRelationship(
            id="st_rel",
            external_id="agent-1",
            shortterm_memory_id=1,
            from_entity_id="e1",
            to_entity_id="e2",
            from_entity_name="A",
            to_entity_name="B",
            types=["Relates"],
            description="ST",
            importance=0.5,
            access_count=0,
            last_access=None,
            metadata={},
        )

        lt_rel = LongtermRelationship(
            id="lt_rel",
            external_id="agent-1",
            from_entity_id="e1",
            to_entity_id="e2",
            from_entity_name="A",
            to_entity_name="B",
            types=["Relates"],
            description="LT",
            importance=0.5,
            start_date=datetime.now(timezone.utc),
            access_count=0,
            last_access=None,
            metadata={},
        )

        st_triplet = st_relationship_to_triplet(st_rel, "A", "B")
        lt_triplet = lt_relationship_to_triplet(lt_rel, "A", "B")

        assert isinstance(st_triplet, ShorttermKnowledgeTriplet)
        assert isinstance(lt_triplet, LongtermKnowledgeTriplet)

    def test_triplet_required_fields(self):
        """Test that all triplet required fields are populated."""
        rel = ShorttermRelationship(
            id="rel",
            external_id="agent-1",
            shortterm_memory_id=1,
            from_entity_id="e1",
            to_entity_id="e2",
            from_entity_name="Subject",
            to_entity_name="Object",
            types=["Predicate"],
            description="Test relationship",
            importance=0.8,
            access_count=1,
            last_access=None,
            metadata={},
        )

        triplet = st_relationship_to_triplet(rel, "Subject", "Object")

        # Required fields
        assert triplet.subject is not None
        assert triplet.predicate is not None
        assert triplet.object is not None
        assert triplet.importance is not None
        # Optional fields can be None
        assert triplet.description is not None
        assert triplet.metadata is not None
