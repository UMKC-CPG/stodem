"""Unit tests for overlap integral edge cases and
sign conventions."""

import numpy as np
import pytest
from gaussian import Gaussian


class TestSignConvention:
    """Verify the theta sign convention: preference
    Gaussians (theta in [0, pi/2]) produce positive
    cos_theta; aversion Gaussians (theta in [pi/2, pi])
    produce negative cos_theta."""

    def test_preference_positive_cos(self):
        """Preference Gaussian has positive cos_theta."""
        g = Gaussian(
            np.array([0.0]),
            np.array([1.0]),
            np.array([0.0 + 0.3j]),
            1)
        g.update_integration_variables()
        assert g.cos_theta[0] > 0

    def test_aversion_negative_cos(self):
        """Aversion Gaussian has negative cos_theta."""
        theta = np.array([0.0 + (np.pi - 0.3) * 1j])
        g = Gaussian(
            np.array([0.0]),
            np.array([1.0]),
            theta,
            1)
        g.update_integration_variables()
        assert g.cos_theta[0] < 0

    def test_same_type_overlap_nonnegative(self):
        """Overlap between two preference Gaussians
        should be non-negative."""
        g1 = Gaussian(
            np.array([0.0]),
            np.array([1.0]),
            np.array([0.0 + 0.2j]),
            1)
        g2 = Gaussian(
            np.array([0.5]),
            np.array([1.0]),
            np.array([0.0 + 0.4j]),
            1)
        g1.update_integration_variables()
        g2.update_integration_variables()
        ol = g1.integral(g2)
        assert ol[0] >= 0

    def test_cross_type_overlap_nonpositive(self):
        """Overlap between a preference and an aversion
        Gaussian should be non-positive."""
        pref = Gaussian(
            np.array([0.0]),
            np.array([1.0]),
            np.array([0.0 + 0.2j]),
            1)
        aver = Gaussian(
            np.array([0.0]),
            np.array([1.0]),
            np.array([0.0 + (np.pi - 0.2) * 1j]),
            1)
        pref.update_integration_variables()
        aver.update_integration_variables()
        ol = pref.integral(aver)
        assert ol[0] <= 0
