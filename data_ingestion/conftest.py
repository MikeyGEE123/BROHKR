# tests/conftest.py

import sys
import os

# Add the project's root directory to the Python path.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Your event_loop fixture (if not already present) can go here:
import asyncio
import pytest

@pytest.fixture(scope="session")
def event_loop():
    """
    Create an instance of the default event loop for the session.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
