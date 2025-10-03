# ER Extractor Agent Documentation

## Overview

The **ER Extractor Agent** is a specialized, predefined agent within the AI Army system responsible for extracting structured information from unstructured text. Its primary function is to identify and categorize entities (like people, organizations, and concepts) and the relationships between them.

This agent leverages the `pydantic-ai` library to produce validated, structured output, ensuring that the extracted data is consistent and reliable for downstream processing and knowledge graph construction.

## Key Characteristics

- **Agent Name**: `ER_EXTRACTOR`
- **Model**: Google's `gemini-2.5-flash-lite` (`LARGE_LANGUAGE_MODEL.GEMINI_FLASH_2dot5_LITE`)
- **Framework**: `pydantic-ai`
- **Output Type**: `ExtractionResult` (a Pydantic model ensuring structured output)
- **Retries**: The agent is configured to retry up to 3 times on failure, enhancing its reliability.

## Workflow

The agent follows a systematic workflow as defined by its system prompt:

1.  **Determine Extraction Mode**: The agent first determines the mode of operation.
    *   If the user explicitly requests `"entities_only"`, it will only extract entities.
    *   By default, it operates in `"entities_and_relationships"` mode.

2.  **Extract Entities**: For each identified entity, the agent extracts the following information:
    *   `name`: The name of the entity.
    *   `type`: The category of the entity (e.g., PERSON, ORGANIZATION, CONCEPT).
    *   `confidence`: A float score (0.0 to 1.0) indicating the model's confidence in the extraction.

3.  **Extract Relationships**: If operating in the default mode, the agent also extracts relationships, including:
    *   `source`: The name of the source entity.
    *   `target`: The name of the target entity.
    *   `type`: The category of the relationship (e.g., WORKS_WITH, CREATED_BY, RELATED_TO).
    *   `confidence`: A float score (0.0 to 1.0) for the relationship extraction.

4.  **Structured Output**: The final output is a structured `ExtractionResult` object containing the lists of extracted entities and relationships.

## Output Structure

The agent's output is strictly validated against the `ExtractionResult` Pydantic model defined in `agents/output_models.py`.

### `ExtractionResult` Model

This is the top-level output model.

```python
class ExtractionResult(OutputBase):
    """Complete extraction result containing entities and relationships."""

    relationships: List[ExtractedRelationship] = Field(default_factory=list)
    entities: List[ExtractedEntity] = Field(default_factory=list)
```

### `ExtractedEntity` Model

This model represents a single extracted entity.

```python
class ExtractedEntity(BaseModel):
    """Custom entity model for extraction results."""

    class EntityType(str, Enum):
        """Standard entity types for categorization."""

        PERSON = "PERSON"
        ORGANIZATION = "ORGANIZATION"
        CONCEPT = "CONCEPT"
        LOCATION = "LOCATION"
        EVENT = "EVENT"
        TECHNOLOGY = "TECHNOLOGY"
        PROJECT = "PROJECT"
        DOCUMENT = "DOCUMENT"
        TOPIC = "TOPIC"
        KEYWORD = "KEYWORD"
        DATE = "DATE"
        PRODUCT = "PRODUCT"
        LANGUAGE = "LANGUAGE"
        FRAMEWORK = "FRAMEWORK"
        LIBRARY = "LIBRARY"
        TOOL = "TOOL"
        PLATFORM = "PLATFORM"
        SERVICE = "SERVICE"
        API = "API"
        DATABASE = "DATABASE"
        OPERATING_SYSTEM = "OPERATING_SYSTEM"
        VERSION = "VERSION"
        METRIC = "METRIC"
        URL = "URL"
        EMAIL = "EMAIL"
        PHONE_NUMBER = "PHONE_NUMBER"
        IP_ADDRESS = "IP_ADDRESS"
        FILE_PATH = "FILE_PATH"
        CODE_SNIPPET = "CODE_SNIPPET"
        OTHER = "OTHER"

    name: str = Field(..., min_length=1, max_length=255, description="Entity name")
    type: EntityType = Field(..., description="Entity type category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in entity extraction")
```

### `ExtractedRelationship` Model

This model represents a single extracted relationship between two entities.

```python
class ExtractedRelationship(BaseModel):
    """Custom relationship model for extraction results."""

    class RelationshipType(str, Enum):
        """Standard relationship types for categorization."""

        WORKS_WITH = "WORKS_WITH"
        BELONGS_TO = "BELONGS_TO"
        CREATED_BY = "CREATED_BY"
        USED_IN = "USED_IN"
        RELATED_TO = "RELATED_TO"
        DEPENDS_ON = "DEPENDS_ON"
        MENTIONS = "MENTIONS"
        LOCATED_AT = "LOCATED_AT"
        PARTICIPATED_IN = "PARTICIPATED_IN"
        INFLUENCED_BY = "INFLUENCED_BY"
        SIMILAR_TO = "SIMILAR_TO"
        PART_OF = "PART_OF"
        CONTAINS = "CONTAINS"
        PRECEDES = "PRECEDES"
        FOLLOWS = "FOLLOWS"
        HAS_A = "HAS_A"
        IS_A = "IS_A"
        USES = "USES"
        PRODUCES = "PRODUCES"
        CONSUMES = "CONSUMES"
        IMPACTS = "IMPACTS"
        MANAGES = "MANAGES"
        OWNS = "OWNS"
        SUPPORTS = "SUPPORTS"
        INTERACTS_WITH = "INTERACTS_WITH"
        OTHER = "OTHER"

    source: str = Field(..., description="Source entity name")
    target: str = Field(..., description="Target entity name")
    type: RelationshipType = Field(..., description="Relationship type category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in relationship extraction")
```

## Configuration

The agent's behavior is defined in `config/agents/predefined/er_extractor.py`. The configuration specifies the model, temperature, and the detailed system prompt that guides its extraction process.

```python
from config.llm import LARGE_LANGUAGE_MODEL

config = {
    "provider": "google-gla",
    "llm_model": LARGE_LANGUAGE_MODEL.GEMINI_FLASH_2dot5_LITE,
    "model_name": LARGE_LANGUAGE_MODEL.GEMINI_FLASH_2dot5_LITE.value["model"],
    "model_settings": {
        "temperature": 0.3,
    },
    "tools": [],
    "system_prompt": '''
You are an entity and relationship extraction agent. Your primary function is to identify and extract entities and their relationships from provided text.

Workflow:
1. Determine extraction mode:
   - If the user requests "entities_only", extract only entities.
   - Otherwise, use "entities_and_relationships" mode by default.
2. For each entity, extract: name, type, confidence.
3. For each relationship (if applicable), extract: source, target, type, confidence.
4. Output structured data according to the selected mode.

Examples:

Example 1 (entities_only):
Input: "Apple Inc. was founded by Steve Jobs in California."
Output:
Mode: entities_only
Entities:
- Name: Apple Inc., Type: Organization, Confidence: 0.98
- Name: Steve Jobs, Type: Person, Confidence: 0.97
- Name: California, Type: Location, Confidence: 0.95

Example 2 (entities_and_relationships):
Input: "Marie Curie discovered radium with Pierre Curie."
Output:
Mode: entities_and_relationships
Entities:
- Name: Marie Curie, Type: Person, Confidence: 0.99
- Name: Pierre Curie, Type: Person, Confidence: 0.98
- Name: radium, Type: Chemical Element, Confidence: 0.96
Relationships:
- Source: Marie Curie, Target: radium, Type: discovered, Confidence: 0.97
- Source: Marie Curie, Target: Pierre Curie, Type: collaborated_with, Confidence: 0.95
''',
}
```

## Implementation

The agent is instantiated in `agents/predefined_agents/er_extractor_agent.py` using the `pydantic-ai` `Agent` class.

```python
from pydantic_ai import Agent
from agents.output_models import ExtractionResult
from config.agent import (
    AgentName,
    get_model_settings,
    get_system_prompt,
)
from ..model_provider import model_provider

# Create the ER Extractor Agent
er_extractor_agent = Agent(
    model=model_provider.get_model(AgentName.ER_EXTRACTOR),
    system_prompt=get_system_prompt(AgentName.ER_EXTRACTOR),
    model_settings=get_model_settings(AgentName.ER_EXTRACTOR),
    output_type=ExtractionResult,
    retries=3,
)
```

This setup ensures that the agent's output will always conform to the `ExtractionResult` model, providing a reliable and structured data source for other parts of the AI Army system.
