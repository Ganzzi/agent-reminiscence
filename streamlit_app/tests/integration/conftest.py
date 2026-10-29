"""
Module-scoped harness for the Streamlit e2e module.

Provides the run-owned isolated-topology environment contract (no defaults,
validated before any network contact), the MemoryService/TemplateService
fixtures, and shared test helpers. See NGU-99 spec/tech design.
"""

import os
from pathlib import Path

import pytest

from streamlit_app.services.memory_service import MemoryService
from streamlit_app.services.template_service import TemplateService

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "templates"

REQUIRED_ENV = {
    "run_label": "AGENT_MEM_STREAMLIT_TEST_RUN_LABEL",
    "pg_host": "AGENT_MEM_STREAMLIT_TEST_PG_HOST",
    "pg_port": "AGENT_MEM_STREAMLIT_TEST_PG_PORT",
    "pg_db": "AGENT_MEM_STREAMLIT_TEST_PG_DB",
    "pg_user": "AGENT_MEM_STREAMLIT_TEST_PG_USER",
    "pg_password": "AGENT_MEM_STREAMLIT_TEST_PG_PASSWORD",
    "neo4j_uri": "AGENT_MEM_STREAMLIT_TEST_NEO4J_URI",
    "neo4j_user": "AGENT_MEM_STREAMLIT_TEST_NEO4J_USER",
    "neo4j_password": "AGENT_MEM_STREAMLIT_TEST_NEO4J_PASSWORD",
}


def resolve_test_db_config() -> dict:
    """Resolve and VALIDATE the run-owned target. Raises before any network use."""
    values = {}
    for key, var in REQUIRED_ENV.items():
        raw = os.getenv(var)
        if not raw:
            raise RuntimeError(f"Missing required isolated-topology variable: {var}")
        values[key] = raw
    try:
        values["pg_port"] = int(values["pg_port"])
    except ValueError as e:
        raise RuntimeError(f"AGENT_MEM_STREAMLIT_TEST_PG_PORT must be an integer: {e}")
    if not values["neo4j_uri"].startswith("bolt://"):
        raise RuntimeError("AGENT_MEM_STREAMLIT_TEST_NEO4J_URI must start with bolt://")
    for key, var in (
        ("pg_password", "AGENT_MEM_STREAMLIT_TEST_PG_PASSWORD"),
        ("neo4j_password", "AGENT_MEM_STREAMLIT_TEST_NEO4J_PASSWORD"),
    ):
        if len(values[key]) < 8:
            raise RuntimeError(
                f"{var} must be at least 8 characters (Config validator contract)"
            )
    return {
        "postgres_host": values["pg_host"],
        "postgres_port": values["pg_port"],
        "postgres_db": values["pg_db"],
        "postgres_user": values["pg_user"],
        "postgres_password": values["pg_password"],
        "neo4j_uri": values["neo4j_uri"],
        "neo4j_user": values["neo4j_user"],
        "neo4j_password": values["neo4j_password"],
    }


def external_id(base: str) -> str:
    """Run-unique, storage-auditable external id."""
    return f"ngu99-{os.environ[REQUIRED_ENV['run_label']]}-{base}"


def to_template_content(loaded: dict) -> dict:
    """Loaded Streamlit file shape -> storage contract shape (top-level sections)."""
    template = loaded["template"]
    return {
        "template": {"id": template["id"], "name": template["name"]},
        "sections": [
            {k: s[k] for k in ("id", "title", "description", "update_strategy",
                               "consolidation_trigger") if k in s}
            for s in template["sections"]
        ],
    }


@pytest.fixture
def db_config() -> dict:
    return resolve_test_db_config()


@pytest.fixture
async def memory_service(db_config):
    """Create a memory service instance against the run-owned topology."""
    service = MemoryService(db_config)
    yield service
    # Cleanup after tests
    if service._agent_mem:
        await service._agent_mem.close()


@pytest.fixture
def template_service():
    """Template service backed by the committed deterministic fixture tree."""
    return TemplateService(FIXTURES_DIR)
