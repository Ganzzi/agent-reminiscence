"""Session-scoped fail-closed targeting gate for the integration tier.

Every test collected under tests/integration/ — including mock-backed files —
runs under the autouse session gate: without valid run-owned targeting
authority the whole tier refuses before any fixture or test executes.

``import targeting`` works because pytest's default ``prepend`` import mode
puts this directory on sys.path while collecting it (there is no
__init__.py here); the unit tier loads the module by file path instead.
"""

import pytest

from targeting import resolve_targeting


@pytest.fixture(scope="session")
def targeting_state():
    """Session gate: authority chain runs once, before any module fixture."""
    return resolve_targeting()


@pytest.fixture(scope="session", autouse=True)
def integration_targeting_gate(targeting_state):
    """Autouse so tests that request no fixtures still require the gate."""
    return targeting_state


@pytest.fixture(scope="session")
def integration_config(targeting_state):
    """Fully-explicit validated Config for integration tests."""
    return targeting_state.config
