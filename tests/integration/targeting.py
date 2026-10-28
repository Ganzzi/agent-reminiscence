"""Fail-closed integration targeting for the canonical integration tier.

Authority chain: AGENT_MEM_INTEGRATION_TARGETING -> manifest v1 -> structural
validation -> in-store ownership markers -> validated Config. Refuses before
any construction or network on any deviation. See NGU-100 design.

This module lives under tests/integration/ (no __init__.py) and is imported
by the integration conftest and the bootstrap regression test through the
repo's default pytest ``prepend`` import mode, which puts this directory on
sys.path. The unit tier loads it by file path via importlib instead, so it
never relies on package layout.

Refusal diagnostics are secret-free: they name the violated rule and field,
never interpolate password values, and reduce driver failures to exception
type names. Endpoint host:port values may appear.
"""

import asyncio
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from agent_reminiscence.config.settings import Config

FORBIDDEN_PG_PORTS = {5432, 5433}
FORBIDDEN_NEO4J_PORTS = {7687}
AMBIENT_ENV_VARS = (
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_ADMIN_DB",
    "NEO4J_URI",
    "NEO4J_USER",
    "NEO4J_PASSWORD",
    "OLLAMA_BASE_URL",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
)
PG_OWNERSHIP_TABLE = "agent_mem_ownership"  # gate constant, not manifest-configurable
NEO4J_OWNERSHIP_LABEL = "AgentMemRunOwnership"  # gate constant

TARGETING_ENV_VAR = "AGENT_MEM_INTEGRATION_TARGETING"
MANIFEST_KEYS = {"schema_version", "run_id", "approved_design_ref", "postgres", "neo4j"}
POSTGRES_KEYS = {"host", "port", "admin_db", "target_db", "user", "password"}
NEO4J_KEYS = {"uri", "user", "password"}
FORBIDDEN_TARGET_DBS = {"postgres", "agent_mem", "agent_mem_test"}
LOOPBACK = "127.0.0.1"

_RUN_ID_RE = re.compile(r"^ngu-100-[0-9]{8}-[0-9a-f]{8}$")
_NEO4J_URI_RE = re.compile(r"^bolt://127\.0\.0\.1:[0-9]+$")

HERMETIC_EMBEDDING_PROVIDER = "ollama"
HERMETIC_EMBEDDING_MODEL = "hermetic-no-provider"
HERMETIC_OLLAMA_BASE_URL = "http://127.0.0.1:9"  # reserved discard port, closed by construction
HERMETIC_VECTOR_DIMENSION = 768
HERMETIC_AGENT_MODEL = "google:gemini-2.5-flash"


class IntegrationTargetingRefusal(RuntimeError):
    """Secret-free refusal. Messages name the violated rule/field only."""


@dataclass(frozen=True)
class _ProbeContext:
    """Validated manifest plus connection parameters for the ownership probes."""

    manifest: dict
    admin_dsn: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str


@dataclass(frozen=True)
class TargetingState:
    """Validated run-owned targeting handed to the integration tier."""

    run_id: str
    manifest: dict
    config: Config
    admin_dsn: str
    db_dsn: str

    def dsn_for(self, db_name: str) -> str:
        """DSN for a run-owned throwaway database on the validated endpoint."""
        return _pg_dsn(self.manifest["postgres"], db_name)


def _pg_dsn(postgres: dict, db_name: str) -> str:
    return (
        f"postgresql://{postgres['user']}:{postgres['password']}"
        f"@{postgres['host']}:{postgres['port']}/{db_name}"
    )


def _pg_marker_probe(ctx: _ProbeContext) -> str | None:
    """Return the ownership run_id stored in the admin db. Sole PG network seam."""
    from psqlpy import ConnectionPool

    async def _probe() -> str | None:
        # psqlpy requires max_db_pool_size > 1.
        pool = ConnectionPool(dsn=ctx.admin_dsn, max_db_pool_size=2)
        try:
            conn = await pool.connection()
            result = await conn.execute(
                f"SELECT run_id FROM {PG_OWNERSHIP_TABLE} WHERE run_id = $1",
                [ctx.manifest["run_id"]],
            )
            rows = result.result()
            return rows[0]["run_id"] if rows else None
        finally:
            pool.close()

    return asyncio.run(_probe())


def _neo4j_marker_probe(ctx: _ProbeContext) -> str | None:
    """Return the ownership run_id stored in Neo4j. Sole Neo4j network seam."""
    from neo4j import AsyncGraphDatabase

    async def _probe() -> str | None:
        driver = AsyncGraphDatabase.driver(
            ctx.neo4j_uri, auth=(ctx.neo4j_user, ctx.neo4j_password)
        )
        try:
            session = driver.session()
            try:
                result = await session.run(
                    f"MATCH (n:{NEO4J_OWNERSHIP_LABEL} {{run_id: $run_id}}) "
                    "RETURN n.run_id AS run_id LIMIT 1",
                    {"run_id": ctx.manifest["run_id"]},
                )
                record = await result.single()
                return record["run_id"] if record is not None else None
            finally:
                await session.close()
        finally:
            await driver.close()

    return asyncio.run(_probe())


def resolve_targeting(
    root_env: Mapping[str, str] | None = None,
    repo_root: str | Path | None = None,
) -> TargetingState:
    """Run the full fail-closed authority chain and return the targeting state.

    Every deviation raises IntegrationTargetingRefusal before any Config,
    client, pool, subprocess, or network construction.
    """
    env = os.environ if root_env is None else root_env
    root = Path(__file__).resolve().parents[2] if repo_root is None else Path(repo_root)

    raw = env.get(TARGETING_ENV_VAR)
    if raw is None:
        raise IntegrationTargetingRefusal(
            f"{TARGETING_ENV_VAR} is not set: the integration tier requires "
            "run-owned targeting authority (NGU-100)"
        )
    if not raw.strip():
        raise IntegrationTargetingRefusal(
            f"{TARGETING_ENV_VAR} is empty: the integration tier requires "
            "run-owned targeting authority (NGU-100)"
        )

    if (root / ".env").is_file():
        raise IntegrationTargetingRefusal(
            "a .env file exists in the repo root: ambient dotenv authority is refused (NGU-100)"
        )

    ambient = [var for var in AMBIENT_ENV_VARS if var in env]
    if ambient:
        raise IntegrationTargetingRefusal(
            f"ambient environment authority is refused (NGU-100): {ambient}"
        )

    manifest = _load_manifest(raw.strip())
    _validate_structure(manifest)

    postgres = manifest["postgres"]
    admin_dsn = _pg_dsn(postgres, postgres["admin_db"])
    db_dsn = _pg_dsn(postgres, postgres["target_db"])
    ctx = _ProbeContext(
        manifest=manifest,
        admin_dsn=admin_dsn,
        neo4j_uri=manifest["neo4j"]["uri"],
        neo4j_user=manifest["neo4j"]["user"],
        neo4j_password=manifest["neo4j"]["password"],
    )
    _verify_ownership(ctx)

    config = _build_config(manifest)
    return TargetingState(
        run_id=manifest["run_id"],
        manifest=manifest,
        config=config,
        admin_dsn=admin_dsn,
        db_dsn=db_dsn,
    )


def _load_manifest(path_text: str) -> dict:
    path = Path(path_text)
    if not path.is_absolute():
        raise IntegrationTargetingRefusal(
            "targeting manifest path must be absolute (NGU-100)"
        )
    if not path.is_file():
        raise IntegrationTargetingRefusal(
            "targeting manifest file does not exist (NGU-100)"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        raise IntegrationTargetingRefusal(
            "targeting manifest file is not readable (NGU-100)"
        )
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise IntegrationTargetingRefusal("targeting manifest is not valid JSON (NGU-100)")
    if not isinstance(data, dict):
        raise IntegrationTargetingRefusal("targeting manifest must be a JSON object (NGU-100)")
    missing = MANIFEST_KEYS - set(data)
    if missing:
        raise IntegrationTargetingRefusal(
            f"targeting manifest is missing required keys: {sorted(missing)} (NGU-100)"
        )
    unknown = set(data) - MANIFEST_KEYS
    if unknown:
        raise IntegrationTargetingRefusal(
            f"targeting manifest has unknown top-level keys (provider/egress authority "
            f"is not granted in this slice): {sorted(unknown)} (NGU-100)"
        )
    if data["schema_version"] != 1:
        raise IntegrationTargetingRefusal(
            "targeting manifest schema_version must be 1 (NGU-100)"
        )
    return data


def _check_section(section: dict, keys: set[str], name: str) -> None:
    missing = keys - set(section)
    if missing:
        raise IntegrationTargetingRefusal(
            f"{name} is missing required keys: {sorted(missing)} (NGU-100)"
        )
    unknown = set(section) - keys
    if unknown:
        raise IntegrationTargetingRefusal(
            f"{name} has unknown keys: {sorted(unknown)} (NGU-100)"
        )


def _validate_structure(manifest: dict) -> None:
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not _RUN_ID_RE.match(run_id):
        raise IntegrationTargetingRefusal(
            "manifest.run_id must match ngu-100-<yyyymmdd>-<8hex> (NGU-100)"
        )

    design_ref = manifest.get("approved_design_ref")
    if not isinstance(design_ref, str) or not design_ref.strip():
        raise IntegrationTargetingRefusal(
            "manifest.approved_design_ref must be a non-empty string (NGU-100)"
        )

    postgres = manifest.get("postgres")
    if not isinstance(postgres, dict):
        raise IntegrationTargetingRefusal("manifest.postgres must be a JSON object (NGU-100)")
    neo4j = manifest.get("neo4j")
    if not isinstance(neo4j, dict):
        raise IntegrationTargetingRefusal("manifest.neo4j must be a JSON object (NGU-100)")
    _check_section(postgres, POSTGRES_KEYS, "manifest.postgres")
    _check_section(neo4j, NEO4J_KEYS, "manifest.neo4j")

    host = postgres["host"]
    if host != LOOPBACK:
        raise IntegrationTargetingRefusal(
            "postgres.host must be the literal 127.0.0.1 (NGU-100)"
        )

    port = postgres["port"]
    if not isinstance(port, int) or isinstance(port, bool) or not 1024 <= port <= 65535:
        raise IntegrationTargetingRefusal(
            "postgres.port must be an integer in 1024..65535 (NGU-100)"
        )
    if port in FORBIDDEN_PG_PORTS:
        raise IntegrationTargetingRefusal(
            "postgres.port must not be an ambient port (5432/5433 are forbidden) (NGU-100)"
        )

    admin_db = postgres["admin_db"]
    if not isinstance(admin_db, str) or not admin_db.strip():
        raise IntegrationTargetingRefusal(
            "postgres.admin_db must be a non-empty string (NGU-100)"
        )
    if admin_db == "postgres":
        raise IntegrationTargetingRefusal(
            'postgres.admin_db must not be the ambient default "postgres" (NGU-100)'
        )

    target_db = postgres["target_db"]
    if not isinstance(target_db, str) or not target_db.strip():
        raise IntegrationTargetingRefusal(
            "postgres.target_db must be a non-empty string (NGU-100)"
        )
    if target_db in FORBIDDEN_TARGET_DBS:
        raise IntegrationTargetingRefusal(
            "postgres.target_db must not be a default database name "
            "(postgres/agent_mem/agent_mem_test are forbidden) (NGU-100)"
        )

    pg_user = postgres["user"]
    if not isinstance(pg_user, str) or not pg_user.strip():
        raise IntegrationTargetingRefusal(
            "postgres.user must be a non-empty string (NGU-100)"
        )

    pg_password = postgres["password"]
    if not isinstance(pg_password, str) or len(pg_password) < 8:
        raise IntegrationTargetingRefusal(
            "postgres.password must be at least 8 characters (NGU-100)"
        )

    uri = neo4j["uri"]
    if not isinstance(uri, str) or not _NEO4J_URI_RE.match(uri):
        raise IntegrationTargetingRefusal(
            "neo4j.uri must be exactly bolt://127.0.0.1:<port> (NGU-100)"
        )
    neo4j_port = int(uri.rsplit(":", 1)[1])
    if neo4j_port in FORBIDDEN_NEO4J_PORTS:
        raise IntegrationTargetingRefusal(
            "neo4j port must not be the ambient port 7687 (NGU-100)"
        )

    n4j_user = neo4j["user"]
    if not isinstance(n4j_user, str) or not n4j_user.strip():
        raise IntegrationTargetingRefusal(
            "neo4j.user must be a non-empty string (NGU-100)"
        )

    n4j_password = neo4j["password"]
    if not isinstance(n4j_password, str) or len(n4j_password) < 8:
        raise IntegrationTargetingRefusal(
            "neo4j.password must be at least 8 characters (NGU-100)"
        )


def _verify_ownership(ctx: _ProbeContext) -> None:
    """Call BOTH ownership probes, then require exact run_id match on each.

    This is the first and only network the gate performs, only after the
    manifest path passes structural validation.
    """
    expected = ctx.manifest["run_id"]
    try:
        pg_run_id = _pg_marker_probe(ctx)
    except Exception as exc:
        raise IntegrationTargetingRefusal(
            f"postgres ownership marker could not be verified "
            f"({type(exc).__name__}) (NGU-100)"
        )
    try:
        neo4j_run_id = _neo4j_marker_probe(ctx)
    except Exception as exc:
        raise IntegrationTargetingRefusal(
            f"neo4j ownership marker could not be verified "
            f"({type(exc).__name__}) (NGU-100)"
        )
    if pg_run_id != expected:
        raise IntegrationTargetingRefusal(
            "postgres ownership marker does not carry this run's run_id (NGU-100)"
        )
    if neo4j_run_id != expected:
        raise IntegrationTargetingRefusal(
            "neo4j ownership marker does not carry this run's run_id (NGU-100)"
        )


def _build_config(manifest: dict) -> Config:
    """Construct a fully-explicit Config; ambient env can never override.

    Embedding/agent fields use the hermetic D6 constants: no provider
    authority is granted in this slice, and any future real embedding
    attempt fails at network time against the reserved discard port.
    """
    postgres = manifest["postgres"]
    neo4j = manifest["neo4j"]
    return Config(
        postgres_host=postgres["host"],
        postgres_port=postgres["port"],
        postgres_db=postgres["target_db"],
        postgres_user=postgres["user"],
        postgres_password=postgres["password"],
        neo4j_uri=neo4j["uri"],
        neo4j_user=neo4j["user"],
        neo4j_password=neo4j["password"],
        embedding_provider=HERMETIC_EMBEDDING_PROVIDER,
        embedding_model=HERMETIC_EMBEDDING_MODEL,
        ollama_base_url=HERMETIC_OLLAMA_BASE_URL,
        vector_dimension=HERMETIC_VECTOR_DIMENSION,
        er_extractor_agent_model=HERMETIC_AGENT_MODEL,
        memory_update_agent_model=HERMETIC_AGENT_MODEL,
        memorizer_agent_model=HERMETIC_AGENT_MODEL,
        memory_retrieve_agent_model=HERMETIC_AGENT_MODEL,
    )


def build_bootstrap_db_name(run_id: str) -> str:
    """Throwaway bootstrap database name embedding the verified run_id."""
    return f"{run_id}-bootstrap-{uuid.uuid4().hex[:8]}"


def assert_owned_bootstrap_name(db_name: str, run_id: str) -> None:
    """Refuse teardown of any database name not owned by this run."""
    if not db_name.startswith(f"{run_id}-bootstrap-"):
        raise IntegrationTargetingRefusal(
            "bootstrap teardown refused: database name is not owned by this run"
        )
