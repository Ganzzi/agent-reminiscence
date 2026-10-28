"""
Default PostgreSQL bootstrap regression tests.

Protects the NGU-98 (QA-HARDENING-005-02) repair of the checked-in SQL init
tree consumed by the PostgreSQL container entrypoint:

1. The entrypoint's execution order over `agent_reminiscence/sql/` must be
   exactly schema.sql then zz_optimize_search_indexes.sql.
2. Every executable file must replay whole, fail-fast, on a throwaway database.
3. The resulting catalog must match the default-bootstrap contract
   (22 indexes with access methods, 4 triggers, 2 functions, 4 extensions,
   `bert` tokenizer).
4. An isolated insert/search smoke must work through the repository's own
   hybrid-search query forms.

The throwaway database is created inside the owned PostgreSQL instance used
by the integration tier; the connecting user must be a superuser (CREATE
DATABASE / CREATE EXTENSION). Replay uses a dollar-quote-aware statement
splitter mirroring `agent_reminiscence/database/init_databases.py` and
aborts on the first server error, matching the entrypoint's
`ON_ERROR_STOP=1` semantics.

Since NGU-100 all DSNs derive from the session targeting gate
(`targeting_state`): the throwaway name embeds the verified run_id and the
teardown DROP is guarded by a run-prefix ownership check. Ambient-default
DSNs are no longer read.
"""

import uuid
from pathlib import Path
from typing import AsyncGenerator

import pytest
from psqlpy import ConnectionPool
from psqlpy.extra_types import PgVector

from targeting import assert_owned_bootstrap_name, build_bootstrap_db_name

SQL_DIR = Path(__file__).resolve().parents[2] / "agent_reminiscence" / "sql"

EXPECTED_ENTRYPOINT_ORDER = ["schema.sql", "zz_optimize_search_indexes.sql"]

PRIMARY_KEY_INDEXES = {
    "active_memory_pkey",
    "shortterm_memory_pkey",
    "shortterm_memory_chunk_pkey",
    "longterm_memory_chunk_pkey",
}

EXPECTED_INDEX_METHODS = {
    "idx_active_memory_external_id": "btree",
    "idx_active_memory_sections": "gin",
    "idx_active_memory_template": "gin",
    "idx_active_memory_metadata": "gin",
    "idx_shortterm_memory_external_id": "btree",
    "idx_shortterm_memory_metadata": "gin",
    "idx_shortterm_memory_dates": "btree",
    "idx_shortterm_chunk_memory_id": "btree",
    "idx_shortterm_chunk_external_id": "btree",
    "idx_shortterm_chunk_vector": "hnsw",
    "idx_shortterm_chunk_bm25": "bm25",
    "idx_shortterm_chunk_metadata": "gin",
    "idx_shortterm_chunk_section": "btree",
    "idx_shortterm_chunk_covering": "btree",
    "idx_longterm_chunk_external_id": "btree",
    "idx_longterm_chunk_temporal": "btree",
    "idx_longterm_chunk_importance": "btree",
    "idx_longterm_chunk_vector": "hnsw",
    "idx_longterm_chunk_bm25": "bm25",
    "idx_longterm_chunk_metadata": "gin",
    "idx_longterm_chunk_updated": "btree",
    "idx_longterm_chunk_covering": "btree",
}

EXPECTED_TRIGGERS = {
    "trigger_update_bm25_shortterm_chunk",
    "trigger_update_bm25_longterm_chunk",
    "trigger_update_active_memory_timestamp",
    "trigger_update_shortterm_memory_timestamp",
}

EXPECTED_FUNCTIONS = {"update_bm25_vector_chunk", "update_updated_at_column"}

EXPECTED_EXTENSIONS = {"vector", "vchord", "pg_tokenizer", "vchord_bm25"}


def _split_statements(sql_text: str) -> list[str]:
    """Dollar-quote-aware splitter mirroring init_databases.py."""
    statements: list[str] = []
    current: list[str] = []
    in_dollar = False

    for line in sql_text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue

        for _ in range(line.count("$$")):
            in_dollar = not in_dollar

        current.append(line)

        if ";" in line and not in_dollar:
            statements.append("\n".join(current))
            current = []

    if current:
        statements.append("\n".join(current))
    return statements


def _entrypoint_executable_entries() -> list[str]:
    """Entries the entrypoint executes: root-level *.sql, sorted (ASCII)."""
    return sorted(
        entry.name
        for entry in SQL_DIR.iterdir()
        if entry.is_file() and entry.suffix == ".sql"
    )


@pytest.fixture(scope="module")
async def bootstrapped_db(targeting_state) -> AsyncGenerator[dict, None]:
    """Create a throwaway database, replay the whole init tree fail-fast."""
    # The session targeting gate has already verified this run's in-store
    # ownership markers before any module fixture (or mutation) runs.
    db_name = build_bootstrap_db_name(targeting_state.run_id)

    admin_pool = ConnectionPool(dsn=targeting_state.admin_dsn, max_db_pool_size=2)
    conn = await admin_pool.connection()
    # run_id is regex-validated ([a-z0-9-]) by the gate, so the quoted
    # identifier is injection-safe (hyphens require quoting).
    await conn.execute(f'CREATE DATABASE "{db_name}"')
    admin_pool.close()

    replay: list[tuple[str, int]] = []
    pool = ConnectionPool(dsn=targeting_state.dsn_for(db_name), max_db_pool_size=2)
    try:
        conn = await pool.connection()
        for file_name in _entrypoint_executable_entries():
            statements = _split_statements(
                (SQL_DIR / file_name).read_text(encoding="utf-8")
            )
            for index, statement in enumerate(statements, 1):
                try:
                    await conn.execute(statement)
                except Exception as error:
                    pytest.fail(
                        f"Entrypoint replay of {file_name} failed at "
                        f"statement {index}: {error}"
                    )
            replay.append((file_name, len(statements)))
    finally:
        pool.close()

    yield {"db_name": db_name, "replay": replay}

    assert_owned_bootstrap_name(db_name, targeting_state.run_id)
    admin_pool = ConnectionPool(dsn=targeting_state.admin_dsn, max_db_pool_size=2)
    conn = await admin_pool.connection()
    await conn.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
    admin_pool.close()


def test_entrypoint_execution_order() -> None:
    """The init tree must execute schema first, optimizations last."""
    assert _entrypoint_executable_entries() == EXPECTED_ENTRYPOINT_ORDER


async def test_fail_fast_replay_bootstraps_database(
    bootstrapped_db: dict,
) -> None:
    """The whole init tree replays with abort-on-first-error semantics."""
    replay = dict(bootstrapped_db["replay"])
    assert list(replay) == EXPECTED_ENTRYPOINT_ORDER
    assert all(count > 0 for count in replay.values())


async def test_catalog_contract(bootstrapped_db: dict, targeting_state) -> None:
    """The replayed database must match the default-bootstrap catalog."""
    pool = ConnectionPool(
        dsn=targeting_state.dsn_for(bootstrapped_db["db_name"]), max_db_pool_size=2
    )
    conn = await pool.connection()

    result = await conn.execute(
        "SELECT i.indexname, am.amname FROM pg_indexes i "
        "JOIN pg_class c ON c.relname = i.indexname "
        "AND c.relnamespace = 'public'::regnamespace "
        "JOIN pg_am am ON am.oid = c.relam "
        "WHERE i.schemaname = 'public'"
    )
    actual = {row["indexname"]: row["amname"] for row in result.result()}
    assert set(actual) == set(EXPECTED_INDEX_METHODS) | PRIMARY_KEY_INDEXES
    assert {
        name: method
        for name, method in actual.items()
        if name in EXPECTED_INDEX_METHODS
    } == EXPECTED_INDEX_METHODS

    result = await conn.execute(
        "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'public'"
    )
    indexdefs = {row["indexname"]: row["indexdef"] for row in result.result()}
    for name in (
        "idx_shortterm_chunk_vector",
        "idx_longterm_chunk_vector",
    ):
        assert "USING hnsw (embedding vector_cosine_ops)" in indexdefs[name]
    for name in (
        "idx_shortterm_chunk_bm25",
        "idx_longterm_chunk_bm25",
    ):
        assert "USING bm25 (content_bm25 bm25_ops)" in indexdefs[name]
    for name in (
        "idx_shortterm_chunk_covering",
        "idx_longterm_chunk_covering",
    ):
        assert "INCLUDE (" in indexdefs[name]

    result = await conn.execute(
        "SELECT tgname FROM pg_trigger WHERE NOT tgisinternal"
    )
    assert {row["tgname"] for row in result.result()} == EXPECTED_TRIGGERS

    result = await conn.execute(
        "SELECT p.proname FROM pg_proc p "
        "JOIN pg_namespace n ON n.oid = p.pronamespace "
        "WHERE n.nspname = 'public'"
    )
    assert EXPECTED_FUNCTIONS <= {row["proname"] for row in result.result()}

    result = await conn.execute("SELECT extname FROM pg_extension")
    assert EXPECTED_EXTENSIONS <= {row["extname"] for row in result.result()}

    result = await conn.execute(
        "SELECT tokenize('bootstrap regression', 'bert') IS NOT NULL AS bert_ok"
    )
    assert result.result()[0]["bert_ok"] is True

    pool.close()


async def test_hybrid_search_smoke(bootstrapped_db: dict, targeting_state) -> None:
    """Insert/search smoke through the repository's own query forms."""
    external_id = f"ngu98-bootstrap-smoke-{uuid.uuid4().hex[:8]}"
    content = "The bootstrap regression smoke test validates hybrid search."
    embedding = [0.1] * 768

    pool = ConnectionPool(
        dsn=targeting_state.dsn_for(bootstrapped_db["db_name"]), max_db_pool_size=2
    )
    conn = await pool.connection()

    result = await conn.execute(
        "INSERT INTO shortterm_memory (external_id, title, summary) "
        "VALUES ($1, $2, $3) RETURNING id",
        [external_id, "NGU-98 bootstrap smoke", "smoke summary"],
    )
    memory_id = result.result()[0]["id"]

    result = await conn.execute(
        "INSERT INTO shortterm_memory_chunk "
        "(shortterm_memory_id, external_id, content, embedding) "
        "VALUES ($1, $2, $3, $4) RETURNING id",
        [memory_id, external_id, content, PgVector(embedding)],
    )
    chunk_id = result.result()[0]["id"]
    result = await conn.execute(
        "SELECT content_bm25 IS NOT NULL AS bm25_populated "
        "FROM shortterm_memory_chunk WHERE id = $1",
        [chunk_id],
    )
    assert result.result()[0]["bm25_populated"] is True, (
        "trigger did not auto-populate content_bm25"
    )

    query = """
        WITH vector_results AS (
            SELECT
                id,
                1 - (embedding <=> $1) AS vector_score
            FROM shortterm_memory_chunk
            WHERE external_id = $2 AND embedding IS NOT NULL
        ),
        bm25_results AS (
            SELECT
                id,
                content_bm25 <&> to_bm25query('idx_shortterm_chunk_bm25', tokenize($3, 'bert')) AS bm25_score
            FROM shortterm_memory_chunk
            WHERE external_id = $2 AND content_bm25 IS NOT NULL
        )
        SELECT
            c.id,
            COALESCE(v.vector_score, 0) AS vector_score,
            COALESCE(b.bm25_score, 0) AS bm25_score
        FROM shortterm_memory_chunk c
        LEFT JOIN vector_results v ON c.id = v.id
        LEFT JOIN bm25_results b ON c.id = b.id
        WHERE c.external_id = $2
          AND (v.vector_score IS NOT NULL OR b.bm25_score IS NOT NULL)
        ORDER BY (COALESCE(v.vector_score, 0) * 0.5 + COALESCE(b.bm25_score, 0) * 0.5) DESC
        LIMIT 10
    """
    result = await conn.execute(
        query,
        [PgVector(embedding), external_id, "bootstrap regression smoke test"],
    )
    rows = result.result()

    assert chunk_id in [row["id"] for row in rows]
    scores = {row["id"]: (row["vector_score"], row["bm25_score"]) for row in rows}
    vector_score, bm25_score = scores[chunk_id]
    assert vector_score > 0
    assert bm25_score is not None

    await conn.execute("DELETE FROM shortterm_memory WHERE id = $1", [memory_id])
    result = await conn.execute(
        "SELECT count(*) AS chunk_count FROM shortterm_memory_chunk "
        "WHERE external_id = $1",
        [external_id],
    )
    assert result.result()[0]["chunk_count"] == 0, (
        "chunk rows must cascade-delete with the memory row"
    )

    pool.close()
