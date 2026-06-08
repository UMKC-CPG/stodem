"""Unit tests for Gaussian math."""

import numpy as np
import pytest
from gaussian import Gaussian


class TestGaussianCreation:
    """Verify Gaussian initialization and derived
    variables."""

    def test_basic_creation(self):
        """A Gaussian can be created with mu, sigma,
        and theta arrays."""
        g = Gaussian(
            np.array([0.0]),
            np.array([1.0]),
            np.array([0.0 + 0j]))
        assert g.mu[0] == 0.0
        assert g.sigma[0] == 1.0

    def test_integration_variables(self):
        """Alpha and cos_theta are computed from
        sigma and theta."""
        g = Gaussian(
            np.array([0.0]),
            np.array([1.0]),
            np.array([0.0 + 0j]))
        g.update_integration_variables()
        assert np.isclose(g.alpha[0], 0.5)
        assert np.isclose(g.cos_theta[0], 1.0)


class TestOverlapIntegral:
    """Verify overlap integral properties."""

    def test_self_overlap_engaged(self):
        """Self-overlap of an engaged Gaussian should
        be close to 1."""
        g = Gaussian(
            np.array([0.0]),
            np.array([1.0]),
            np.array([0.0 + 0j]))
        g.update_integration_variables()
        ol = g.integral(g)
        assert np.isclose(ol[0], 1.0, atol=0.01)

    def test_self_overlap_apathetic(self):
        """Self-overlap of a fully apathetic Gaussian
        should be close to 0."""
        theta = np.array([0.0 + (np.pi / 2) * 1j])
        g = Gaussian(
            np.array([0.0]),
            np.array([1.0]),
            theta)
        g.update_integration_variables()
        ol = g.integral(g)
        assert np.isclose(ol[0], 0.0, atol=0.01)

    def test_distant_gaussians_weak_overlap(self):
        """Two Gaussians far apart should have near-
        zero overlap."""
        g1 = Gaussian(
            np.array([0.0]),
            np.array([1.0]),
            np.array([0.0 + 0j]))
        g2 = Gaussian(
            np.array([100.0]),
            np.array([1.0]),
            np.array([0.0 + 0j]))
        g1.update_integration_variables()
        g2.update_integration_variables()
        ol = g1.integral(g2)
        assert abs(ol[0]) < 0.01

    def test_overlap_symmetry(self):
        """I(G1, G2) should equal I(G2, G1)."""
        g1 = Gaussian(
            np.array([0.5]),
            np.array([1.0]),
            np.array([0.0 + 0.3j]))
        g2 = Gaussian(
            np.array([-0.5]),
            np.array([1.5]),
            np.array([0.0 + 0.1j]))
        g1.update_integration_variables()
        g2.update_integration_variables()
        assert np.isclose(
            g1.integral(g2)[0],
            g2.integral(g1)[0])
