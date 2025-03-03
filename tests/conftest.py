# tests/conftest.py
import sys
import os

# Add the project's root directory to the Python path.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# (Remove the custom event_loop fixture so that the default from pytest-asyncio is used.)
