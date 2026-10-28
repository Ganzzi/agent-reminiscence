"""Hermetic negative regressions for the NGU-100 integration targeting gate.

No network, no services, no environment prerequisites: the ownership probes
are replaced with counting sentinels and a Config sentinel proves that every
refusal path fires before any construction. The secret canary sweep asserts
manifest password values never reach refusal diagnostics.
"""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

_TARGETING_PATH = (
    Path(__file__).resolve().parents[1] / "integration" / "targeting.py"
)
_spec = importlib.util.spec_from_file_location("ngu100_targeting", _TARGETING_PATH)
assert _spec is not None and _spec.loader is not None
targeting = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(targeting)

TARGETING_ENV_VAR = targeting.TARGETING_ENV_VAR
PG_CANARY = "canary-pg-pw-0123"
N4J_CANARY = "canary-n4j-pw-4567"
FOREIGN_RUN_ID = "ngu-100-00010101-ffffffff"

VALID_MANIFEST = {
    "schema_version": 1,
    "run_id": "ngu-100-20261028-deadbeef",
    "approved_design_ref": "prea-ngu-100-integration-targeting-tech-design",
    "postgres": {
        "host": "127.0.0.1",
        "port": 23301,
        "admin_db": "ngu_admin_ngu-100-20261028-deadbeef",
        "target_db": "ngu_target_ngu-100-20261028-deadbeef",
        "user": "ngu100_runner",
        "password": PG_CANARY,
    },
    "neo4j": {
        "uri": "bolt://127.0.0.1:23302",
        "user": "ngu100_runner",
        "password": N4J_CANARY,
    },
}


def _set(path, value):
    def apply(manifest):
        node = manifest
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value

    return apply


def _drop(path):
    def apply(manifest):
        node = manifest
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]

    return apply


STRUCTURAL_CASES = {
    "schema-version-not-1": _set(("schema_version",), 2),
    "pg-host-localhost": _set(("postgres", "host"), "localhost"),
    "pg-port-5432": _set(("postgres", "port"), 5432),
    "pg-port-5433": _set(("postgres", "port"), 5433),
    "n4j-port-7687": _set(("neo4j", "uri"), "bolt://127.0.0.1:7687"),
    "admin-db-default-postgres": _set(("postgres", "admin_db"), "postgres"),
    "target-db-default-agent-mem-test": _set(("postgres", "target_db"), "agent_mem_test"),
    "pg-password-too-short": _set(("postgres", "password"), "short"),
    "unknown-top-level-key": lambda m: m.update({"extra": {"x": 1}}),
    "provider-key-present": lambda m: m.update({"provider": {"name": "ollama"}}),
}

MISSING_FIELDS = [
    ("schema_version",),
    ("run_id",),
    ("approved_design_ref",),
    ("postgres",),
    ("neo4j",),
    ("postgres", "host"),
    ("postgres", "port"),
    ("postgres", "admin_db"),
    ("postgres", "target_db"),
    ("postgres", "user"),
    ("postgres", "password"),
    ("neo4j", "uri"),
    ("neo4j", "user"),
    ("neo4j", "password"),
]


@pytest.fixture
def seam(monkeypatch):
    """Counting sentinel probes: any call means the gate reached the network."""
    calls = {"pg": 0, "n4j": 0}

    def pg_probe(ctx):
        calls["pg"] += 1
        return ctx.manifest["run_id"]

    def n4j_probe(ctx):
        calls["n4j"] += 1
        return ctx.manifest["run_id"]

    monkeypatch.setattr(targeting, "_pg_marker_probe", pg_probe)
    monkeypatch.setattr(targeting, "_neo4j_marker_probe", n4j_probe)
    return calls


@pytest.fixture
def config_boom(monkeypatch):
    """Config construction on any refusal path fails the test loudly."""

    def _no_config(*args, **kwargs):
        raise AssertionError("Config must not be constructed on a refusal path")

    monkeypatch.setattr(targeting, "Config", _no_config)


def _write_manifest(tmp_path: Path, manifest: dict) -> str:
    path = tmp_path / "targeting-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return str(path)


def _resolve_valid(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(
        TARGETING_ENV_VAR, _write_manifest(tmp_path, copy.deepcopy(VALID_MANIFEST))
    )
    return targeting.resolve_targeting(repo_root=tmp_path)


def _resolve_mutated(monkeypatch, tmp_path: Path, mutate) -> None:
    manifest = copy.deepcopy(VALID_MANIFEST)
    mutate(manifest)
    monkeypatch.setenv(TARGETING_ENV_VAR, _write_manifest(tmp_path, manifest))
    targeting.resolve_targeting(repo_root=tmp_path)


def _refusal_scenarios() -> dict:
    """Every refusal path: callable(monkeypatch, tmp_path) must refuse."""
    scenarios = {
        "env-var-missing": lambda monkeypatch, tmp_path: targeting.resolve_targeting(
            root_env={}, repo_root=tmp_path
        ),
        "env-var-empty": lambda monkeypatch, tmp_path: targeting.resolve_targeting(
            root_env={TARGETING_ENV_VAR: "   "}, repo_root=tmp_path
        ),
        "manifest-nonexistent": lambda monkeypatch, tmp_path: _resolve_at_path(
            monkeypatch, tmp_path, str(tmp_path / "missing.json")
        ),
        "manifest-invalid-json": lambda monkeypatch, tmp_path: _resolve_at_path(
            monkeypatch, tmp_path, _write_raw(tmp_path, "{not json")
        ),
        "ambient-env-refused": _scenario_ambient_env,
        "repo-dotenv-refused": _scenario_dotenv,
    }
    for case_id, mutate in STRUCTURAL_CASES.items():
        scenarios[f"structural:{case_id}"] = (
            lambda monkeypatch, tmp_path, mutate=mutate: _resolve_mutated(
                monkeypatch, tmp_path, mutate
            )
        )
    for field in MISSING_FIELDS:
        scenarios[f"missing:{'.'.join(field)}"] = (
            lambda monkeypatch, tmp_path, field=field: _resolve_mutated(
                monkeypatch, tmp_path, _drop(field)
            )
        )
    return scenarios


def _write_raw(tmp_path: Path, text: str) -> str:
    path = tmp_path / "raw-manifest.json"
    path.write_text(text, encoding="utf-8")
    return str(path)


def _resolve_at_path(monkeypatch, tmp_path: Path, manifest_path: str) -> None:
    monkeypatch.setenv(TARGETING_ENV_VAR, manifest_path)
    targeting.resolve_targeting(repo_root=tmp_path)


def _scenario_ambient_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("POSTGRES_HOST", "203.0.113.7")  # test-net sentinel, never contacted
    monkeypatch.setenv(
        TARGETING_ENV_VAR, _write_manifest(tmp_path, copy.deepcopy(VALID_MANIFEST))
    )
    targeting.resolve_targeting(repo_root=tmp_path)


def _scenario_dotenv(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("POSTGRES_HOST=localhost\n", encoding="utf-8")
    monkeypatch.setenv(
        TARGETING_ENV_VAR, _write_manifest(tmp_path, copy.deepcopy(VALID_MANIFEST))
    )
    targeting.resolve_targeting(repo_root=tmp_path)


SCENARIO_IDS = sorted(_refusal_scenarios())


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_every_refusal_path_fires_before_probes_and_config(
    scenario_id, monkeypatch, tmp_path, seam, config_boom
):
    scenario = _refusal_scenarios()[scenario_id]
    with pytest.raises(targeting.IntegrationTargetingRefusal):
        scenario(monkeypatch, tmp_path)
    assert seam == {"pg": 0, "n4j": 0}


def test_marker_run_id_mismatch_refuses(monkeypatch, tmp_path, seam, config_boom):
    def wrong_pg(ctx):
        seam["pg"] += 1
        return FOREIGN_RUN_ID

    def wrong_n4j(ctx):
        seam["n4j"] += 1
        return FOREIGN_RUN_ID

    monkeypatch.setattr(targeting, "_pg_marker_probe", wrong_pg)
    monkeypatch.setattr(targeting, "_neo4j_marker_probe", wrong_n4j)

    with pytest.raises(targeting.IntegrationTargetingRefusal):
        _resolve_valid(monkeypatch, tmp_path)
    assert seam == {"pg": 1, "n4j": 1}


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_secret_canary_never_in_diagnostics(
    scenario_id, monkeypatch, tmp_path, seam
):
    scenario = _refusal_scenarios()[scenario_id]
    with pytest.raises(targeting.IntegrationTargetingRefusal) as excinfo:
        scenario(monkeypatch, tmp_path)
    message = str(excinfo.value)
    assert PG_CANARY not in message
    assert N4J_CANARY not in message


def test_success_path_builds_fully_explicit_state(monkeypatch, tmp_path, seam):
    state = _resolve_valid(monkeypatch, tmp_path)

    assert state.run_id == VALID_MANIFEST["run_id"]
    assert state.config.postgres_host == "127.0.0.1"
    assert state.config.postgres_port == 23301
    assert state.config.postgres_db == VALID_MANIFEST["postgres"]["target_db"]
    assert state.config.postgres_user == VALID_MANIFEST["postgres"]["user"]
    assert state.config.postgres_password == PG_CANARY
    assert state.config.neo4j_uri == VALID_MANIFEST["neo4j"]["uri"]
    assert state.config.neo4j_user == VALID_MANIFEST["neo4j"]["user"]
    assert state.config.neo4j_password == N4J_CANARY
    assert state.config.embedding_provider == "ollama"
    assert state.config.embedding_model == "hermetic-no-provider"
    assert state.config.ollama_base_url == "http://127.0.0.1:9"
    assert state.config.vector_dimension == 768
    assert state.config.er_extractor_agent_model == "google:gemini-2.5-flash"
    assert state.config.memory_update_agent_model == "google:gemini-2.5-flash"
    assert state.config.memorizer_agent_model == "google:gemini-2.5-flash"
    assert state.config.memory_retrieve_agent_model == "google:gemini-2.5-flash"
    assert state.admin_dsn.startswith("postgresql://")
    assert state.admin_dsn.endswith("/" + VALID_MANIFEST["postgres"]["admin_db"])
    assert state.db_dsn.endswith("/" + VALID_MANIFEST["postgres"]["target_db"])
    assert seam == {"pg": 1, "n4j": 1}


def test_bootstrap_name_builder_ownership():
    run_id = VALID_MANIFEST["run_id"]

    name = targeting.build_bootstrap_db_name(run_id)
    assert name.startswith(f"{run_id}-bootstrap-")
    suffix = name[len(f"{run_id}-bootstrap-"):]
    assert len(suffix) == 8
    assert all(ch in "0123456789abcdef" for ch in suffix)

    targeting.assert_owned_bootstrap_name(name, run_id)  # must not raise

    with pytest.raises(targeting.IntegrationTargetingRefusal):
        targeting.assert_owned_bootstrap_name("unrelated_db_name", run_id)
    with pytest.raises(targeting.IntegrationTargetingRefusal):
        targeting.assert_owned_bootstrap_name(FOREIGN_RUN_ID + "-bootstrap-1234abcd", run_id)
