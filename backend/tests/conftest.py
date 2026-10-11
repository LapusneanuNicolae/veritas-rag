"""Shared test setup: reset shared in-memory state before every test so tests are
independent regardless of order."""
import pytest

from app.cache import get_cache
from app.vectorstore.factory import get_vectorstore


@pytest.fixture(autouse=True)
def _reset_state():
    get_vectorstore().clear()
    get_cache().clear()
    yield
