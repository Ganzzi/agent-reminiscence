"""
Streamlit UI Configuration
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
TEMPLATES_DIR = BASE_DIR / "prebuilt-memory-tmpl" / "bmad"

# UI Configuration
APP_TITLE = "AgentMem UI"
APP_ICON = "🧠"
PAGE_LAYOUT = "wide"

# Agent types (corresponds to template directories)
AGENT_TYPES = [
    "analyst",
    "architect",
    "dev",
    "pm",
    "po",
    "qa",
    "sm",
    "ux-expert",
    "bmad-master",
    "bmad-orchestrator",
]

# Display names for agent types
AGENT_DISPLAY_NAMES = {
    "analyst": "Business Analyst",
    "architect": "System Architect",
    "dev": "Full-Stack Developer",
    "pm": "Product Manager",
    "po": "Product Owner",
    "qa": "QA/Test Architect",
    "sm": "Scrum Master",
    "ux-expert": "UX Expert",
    "bmad-master": "BMAD Master",
    "bmad-orchestrator": "BMAD Orchestrator",
}

# Database configuration (from environment or defaults)
DB_CONFIG = {
    "postgres_host": os.getenv("POSTGRES_HOST", "localhost"),
    "postgres_port": int(os.getenv("POSTGRES_PORT", 5432)),
    "postgres_db": os.getenv("POSTGRES_DB", "agent_mem"),
    "postgres_user": os.getenv("POSTGRES_USER", "postgres"),
    "postgres_password": os.getenv("POSTGRES_PASSWORD", "postgres"),
    "neo4j_uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "neo4j_user": os.getenv("NEO4J_USER", "neo4j"),
    "neo4j_password": os.getenv("NEO4J_PASSWORD", "neo4jpassword"),
    "neo4j_database": os.getenv("NEO4J_DATABASE", "agent_mem"),
}

# Memory consolidation thresholds
CONSOLIDATION_WARNING_THRESHOLD = 8
CONSOLIDATION_CRITICAL_THRESHOLD = 10

# UI Settings
MAX_CONTENT_PREVIEW_LENGTH = 200
TEMPLATES_PER_PAGE = 12
MEMORIES_PER_PAGE = 10
