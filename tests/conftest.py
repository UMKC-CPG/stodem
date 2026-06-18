"""Shared pytest configuration for STODEM tests."""

import sys
import os

# Add src/scripts to the path so the unit tests can import
# the simulation modules without installation.
_src = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'src', 'scripts'))
sys.path.insert(0, _src)
