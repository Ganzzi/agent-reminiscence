"""
ER Extractor Agent - Entity and Relationship Extraction.

Extracts entities and relationships from text content for memory consolidation.
"""

import logging
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from agent_reminiscence.config.settings import get_config
from agent_reminiscence.services.llm_model_provider import model_provider

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


class ExtractionMode(str, Enum):
    """Mode for ER Extractor Agent."""

    ER = "ER"  # Entity and Relationship Extraction
    ENTITY = "ENTITY"  # Entity Extraction Only


def get_er_extractor_agent(
    mode: ExtractionMode = ExtractionMode.ER,
) -> Agent[None, ExtractionResult]:
    """
    Factory function to create the ER Extractor Agent.

    Args:
        mode: Extraction mode (ER or ENTITY)

    Returns:
        Configured Agent instance
    """
    config = get_config()
    model = model_provider.get_model(config.er_extractor_agent_model)

    if mode == ExtractionMode.ENTITY:
        system_prompt = """You are an Entity Extraction Specialist.

**Your Role:**
Extract entities from text content.

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

**Guidelines:**
1. Extract ALL significant entities mentioned
2. Use specific entity types (avoid OTHER unless truly ambiguous)
3. Provide confidence scores (0.0-1.0):
   - 1.0: Explicitly stated, no ambiguity
   - 0.8-0.9: Clearly implied or stated
   - 0.6-0.7: Reasonable inference
   - 0.4-0.5: Weak inference or ambiguous
4. Be consistent with entity names (use canonical forms)
5. Include brief descriptions for context

**Example Input:**
"John works at Google. He uses Python and TensorFlow for ML projects."

**Example Output:**
{
  "entities": [
    {"name": "John", "type": "PERSON", "confidence": 1.0, "description": "Person working at Google"},
    {"name": "Google", "type": "ORGANIZATION", "confidence": 1.0, "description": "Technology company"},
    {"name": "Python", "type": "LANGUAGE", "confidence": 1.0, "description": "Programming language"},
    {"name": "TensorFlow", "type": "LIBRARY", "confidence": 1.0, "description": "ML library"}
  ]
}

**Important:**
- Be thorough but precise
- Provide output in the exact structure specified
"""
    else:
        system_prompt = SYSTEM_PROMPT

    return Agent(
        model=model,
        deps_type=None,
        system_prompt=system_prompt,
        output_type=ExtractionResult,
        model_settings={"temperature": 0.3},
        retries=2,
    )


_er_extractor_agent: Optional[Agent[None, ExtractionResult]] = None
_entity_extractor_agent: Optional[Agent[None, ExtractionResult]] = None


def _get_agent(mode: ExtractionMode = ExtractionMode.ER) -> Agent[None, ExtractionResult]:
    """Get or create an agent instance based on the extraction mode."""
    global _er_extractor_agent, _entity_extractor_agent

    if mode == ExtractionMode.ENTITY:
        if _entity_extractor_agent is None:
            _entity_extractor_agent = get_er_extractor_agent(mode=ExtractionMode.ENTITY)
        return _entity_extractor_agent
    else:
        if _er_extractor_agent is None:
            _er_extractor_agent = get_er_extractor_agent(mode=ExtractionMode.ER)
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
    """
    logger.info(f"Extracting entities/relationships from {len(content)} characters")
    try:
        agent = _get_agent(mode=ExtractionMode.ER)
        result = await agent.run(content)
        logger.info(
            f"Extracted {len(result.output.entities)} entities and "
            f"{len(result.output.relationships)} relationships"
        )
        return result.output
    except Exception as e:
        logger.error(f"Entity/relationship extraction failed: {e}")
        raise


async def extract_entities(content: str) -> List[str]:
    """
    Extract entity names only from text content.

    Args:
        content: Text content to analyze

    Returns:
        List of unique entity names
    """
    logger.info(f"Extracting entity names from {len(content)} characters")
    try:
        agent = _get_agent(mode=ExtractionMode.ENTITY)
        result = await agent.run(content)
        entity_names = list(set(entity.name for entity in result.output.entities))
        logger.info(f"Extracted {len(entity_names)} unique entity names")
        return entity_names
    except Exception as e:
        logger.error(f"Entity name extraction failed: {e}")
        raise


