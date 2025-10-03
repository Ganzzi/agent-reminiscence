"""
ER Extractor Agent - Entity and Relationship Extraction.

Extracts entities and relationships from text content for memory consolidation.
"""

import logging
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent

logger = logging.getLogger(__name__)


# =========================================================================
# ENTITY AND RELATIONSHIP TYPES
# =========================================================================


class EntityType(str, Enum):
    """Supported entity types for extraction."""

    # People and Organizations
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"

    # Technical
    TECHNOLOGY = "TECHNOLOGY"
    FRAMEWORK = "FRAMEWORK"
    LIBRARY = "LIBRARY"
    TOOL = "TOOL"
    PLATFORM = "PLATFORM"
    SERVICE = "SERVICE"
    API = "API"
    DATABASE = "DATABASE"
    OPERATING_SYSTEM = "OPERATING_SYSTEM"
    LANGUAGE = "LANGUAGE"
    VERSION = "VERSION"

    # Concepts and Information
    CONCEPT = "CONCEPT"
    TOPIC = "TOPIC"
    KEYWORD = "KEYWORD"
    PROJECT = "PROJECT"
    DOCUMENT = "DOCUMENT"
    PRODUCT = "PRODUCT"

    # Location and Time
    LOCATION = "LOCATION"
    EVENT = "EVENT"
    DATE = "DATE"

    # Technical Details
    METRIC = "METRIC"
    URL = "URL"
    EMAIL = "EMAIL"
    PHONE_NUMBER = "PHONE_NUMBER"
    IP_ADDRESS = "IP_ADDRESS"
    FILE_PATH = "FILE_PATH"
    CODE_SNIPPET = "CODE_SNIPPET"

    # Other
    OTHER = "OTHER"


class RelationshipType(str, Enum):
    """Supported relationship types for extraction."""

    # Work and Organizational
    WORKS_WITH = "WORKS_WITH"
    BELONGS_TO = "BELONGS_TO"
    CREATED_BY = "CREATED_BY"
    MANAGES = "MANAGES"
    OWNS = "OWNS"

    # Usage and Dependencies
    USED_IN = "USED_IN"
    USES = "USES"
    DEPENDS_ON = "DEPENDS_ON"
    SUPPORTS = "SUPPORTS"
    PRODUCES = "PRODUCES"
    CONSUMES = "CONSUMES"

    # Relationships
    RELATED_TO = "RELATED_TO"
    MENTIONS = "MENTIONS"
    INFLUENCED_BY = "INFLUENCED_BY"
    SIMILAR_TO = "SIMILAR_TO"
    INTERACTS_WITH = "INTERACTS_WITH"
    IMPACTS = "IMPACTS"

    # Location and Participation
    LOCATED_AT = "LOCATED_AT"
    PARTICIPATED_IN = "PARTICIPATED_IN"

    # Structure
    PART_OF = "PART_OF"
    CONTAINS = "CONTAINS"
    HAS_A = "HAS_A"
    IS_A = "IS_A"

    # Temporal
    PRECEDES = "PRECEDES"
    FOLLOWS = "FOLLOWS"

    # Other
    OTHER = "OTHER"


# =========================================================================
# OUTPUT MODELS
# =========================================================================


class ExtractedEntity(BaseModel):
    """An extracted entity from text."""

    name: str = Field(description="Entity name")
    type: EntityType = Field(description="Entity type")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence")
    description: str = Field(default="", description="Brief description")


class ExtractedRelationship(BaseModel):
    """An extracted relationship between entities."""

    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    type: RelationshipType = Field(description="Relationship type")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence")
    description: str = Field(default="", description="Brief description")


class ExtractionResult(BaseModel):
    """Result of entity and relationship extraction."""

    entities: List[ExtractedEntity] = Field(default_factory=list, description="Extracted entities")
    relationships: List[ExtractedRelationship] = Field(
        default_factory=list, description="Extracted relationships"
    )


# =========================================================================
# SYSTEM PROMPT
# =========================================================================

SYSTEM_PROMPT = """You are an Entity and Relationship Extraction Specialist.

**Your Role:**
Extract entities and relationships from text content to build a knowledge graph.

**Entity Types to Extract:**
- PERSON: People, individuals, names
- ORGANIZATION: Companies, institutions, groups
- TECHNOLOGY: Technologies, systems, platforms
- FRAMEWORK: Software frameworks (React, Django, etc.)
- LIBRARY: Software libraries (pandas, requests, etc.)
- TOOL: Development tools (Git, Docker, VS Code, etc.)
- CONCEPT: Abstract concepts, ideas, methodologies
- PROJECT: Projects, applications, products
- LOCATION: Places, addresses, regions
- EVENT: Events, milestones, releases

**Relationship Types to Extract:**
- WORKS_WITH: Person works with organization/person
- USES: Entity uses technology/tool/library
- DEPENDS_ON: Entity depends on another
- PART_OF: Entity is part of larger entity
- CREATED_BY: Entity created by person/organization
- LOCATED_AT: Entity located at place
- RELATED_TO: General relationship

**Guidelines:**
1. Extract ALL significant entities mentioned
2. Use specific entity types (avoid OTHER unless truly ambiguous)
3. Extract relationships between entities
4. Provide confidence scores (0.0-1.0):
   - 1.0: Explicitly stated, no ambiguity
   - 0.8-0.9: Clearly implied or stated
   - 0.6-0.7: Reasonable inference
   - 0.4-0.5: Weak inference or ambiguous
5. Be consistent with entity names (use canonical forms)
6. Include brief descriptions for context

**Example Input:**
"John works at Google. He uses Python and TensorFlow for ML projects."

**Example Output:**
{
  "entities": [
    {"name": "John", "type": "PERSON", "confidence": 1.0, "description": "Person working at Google"},
    {"name": "Google", "type": "ORGANIZATION", "confidence": 1.0, "description": "Technology company"},
    {"name": "Python", "type": "LANGUAGE", "confidence": 1.0, "description": "Programming language"},
    {"name": "TensorFlow", "type": "LIBRARY", "confidence": 1.0, "description": "ML library"}
  ],
  "relationships": [
    {"source": "John", "target": "Google", "type": "WORKS_WITH", "confidence": 1.0, "description": "Employment relationship"},
    {"source": "John", "target": "Python", "type": "USES", "confidence": 1.0, "description": "Uses for development"},
    {"source": "John", "target": "TensorFlow", "type": "USES", "confidence": 1.0, "description": "Uses for ML projects"}
  ]
}

**Important:**
- Be thorough but precise
- Don't invent relationships not supported by text
- Use highest confidence for explicit mentions
- Provide output in the exact structure specified
"""


# =========================================================================
# AGENT CREATION
# =========================================================================


def get_er_extractor_agent() -> Agent[None, ExtractionResult]:
    """
    Factory function to create the ER Extractor Agent.

    Returns:
        Configured Agent instance
    """
    return Agent(
        model="google-gla:gemini-2.5-flash-lite",  # Fast model for extraction
        deps_type=None,  # No dependencies needed
        system_prompt=SYSTEM_PROMPT,
        output_type=ExtractionResult,  # Correct: output_type, not result_type
        model_settings={
            "temperature": 0.3,  # Low temperature for consistency
        },
        retries=2,
    )


# Create global agent instance (lazy initialization to avoid API key errors on import)
_er_extractor_agent: Optional[Agent[None, ExtractionResult]] = None


def _get_agent() -> Agent[None, ExtractionResult]:
    """Get or create the ER Extractor Agent instance."""
    global _er_extractor_agent
    if _er_extractor_agent is None:
        _er_extractor_agent = get_er_extractor_agent()
    return _er_extractor_agent


# =========================================================================
# MAIN FUNCTION
# =========================================================================


async def extract_entities_and_relationships(content: str) -> ExtractionResult:
    """
    Extract entities and relationships from text content.

    Args:
        content: Text content to analyze

    Returns:
        ExtractionResult with entities and relationships

    Raises:
        Exception: If extraction fails after retries
    """
    logger.info(f"Extracting entities/relationships from {len(content)} characters")

    try:
        agent = _get_agent()
        result = await agent.run(content)

        logger.info(
            f"Extracted {len(result.output.entities)} entities and "
            f"{len(result.output.relationships)} relationships"
        )

        return result.output

    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        raise


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================


def validate_extraction_result(result: ExtractionResult) -> bool:
    """
    Validate extraction result quality.

    Args:
        result: Extraction result to validate

    Returns:
        True if result is valid
    """
    # Check if we have entities
    if not result.entities:
        logger.warning("No entities extracted")
        return False

    # Check confidence scores
    low_confidence_count = sum(1 for e in result.entities if e.confidence < 0.5)

    if low_confidence_count > len(result.entities) * 0.5:
        logger.warning(f"Too many low-confidence entities: {low_confidence_count}")
        return False

    # Check relationship validity
    entity_names = {e.name for e in result.entities}
    invalid_relationships = [
        r
        for r in result.relationships
        if r.source not in entity_names or r.target not in entity_names
    ]

    if invalid_relationships:
        logger.warning(f"{len(invalid_relationships)} relationships reference unknown entities")
        # Remove invalid relationships
        result.relationships = [r for r in result.relationships if r not in invalid_relationships]

    return True
