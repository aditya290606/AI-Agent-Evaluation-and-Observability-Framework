"""
Pytest configuration and test database isolation.
Ensures tests run against an isolated test database (test_eval_tmp.db)
instead of polluting the production/demo eval_framework.db.
"""

import os
import sys
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DB_FILE = os.path.join(ROOT_DIR, "test_eval_tmp.db")

# Force DATABASE_URL to the isolated test database before any src imports
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_FILE}"


@pytest.fixture(scope="session", autouse=True)
def isolate_test_database():
    """Ensure clean test database for the test session and remove it afterwards."""
    # Clean any stale test DB
    for ext in ["", "-shm", "-wal"]:
        f = TEST_DB_FILE + ext
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

    from src.storage.db import init_db
    init_db()

    yield

    # Teardown
    from src.storage.db import engine
    engine.dispose()
    for ext in ["", "-shm", "-wal"]:
        f = TEST_DB_FILE + ext
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass
