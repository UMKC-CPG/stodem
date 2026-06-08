"""Shared pytest fixtures for STODEM tests."""

import sys
import os
import pytest

# Add src/scripts to the path so tests can import
# simulation modules without installation.
_src = os.path.join(
    os.path.dirname(__file__), '..', 'src', 'scripts')
sys.path.insert(0, os.path.abspath(_src))


@pytest.fixture
def quicktest_dir():
    """Path to the quickTest job directory."""
    return os.path.join(
        os.path.dirname(__file__),
        '..', 'jobs', 'quickTest')


@pytest.fixture
def src_dir():
    """Path to the src/scripts directory."""
    return os.path.abspath(_src)
