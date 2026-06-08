"""Regression tests: compare simulation output against
stored reference data.

To generate reference outputs, run the simulation in
the quickTest directory and copy the resulting HDF5
file to tests/regression/reference_outputs/. Then
update the comparison below to check specific fields
and tolerances.

These tests are intentionally skeletal until the
simulation output is stable enough to serve as a
regression baseline.
"""

import pytest


class TestRegression:
    """Placeholder for future regression comparisons."""

    @pytest.mark.skip(
        reason="No reference outputs generated yet")
    def test_well_being_matches_reference(self):
        """Compare well-being values against stored
        reference data."""
        pass
