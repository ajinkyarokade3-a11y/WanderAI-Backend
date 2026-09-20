"""Shared session fixture: migrated + seeded database for every test module.

Previously only test_backend.py carried this fixture, so any other test
file failed on an empty database (and depended on file ordering). Central
here so all suites pass standalone and in any order.
"""

import pytest
from alembic import command
from alembic.config import Config

from database.seed_data.seed import run_seed


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    command.upgrade(Config("alembic.ini"), "head")
    run_seed()
