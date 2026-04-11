import numpy as np

from random_state import rng


def sample_theta(orien_stddev_str, mean, size,
                 clamp_lo, clamp_hi):
    """Sample initial Im(theta) values for a set
    of Gaussians.

    Theta (orientation) is stored as a purely
    imaginary number: theta = Im(theta) * 1j.
    The imaginary part encodes a citizen's or
    politician's engagement with an issue:
      Im(theta) near 0      -> cos ~ 1 (engaged)
      Im(theta) near pi/2   -> cos ~ 0 (apathetic)
      Im(theta) near pi     -> cos ~ -1 (engaged,
                                aversion type)

    This function provides two initialization
    modes, controlled by the XML configuration
    parameter *_orien_stddev:

    1. NUMERIC string (e.g., "0.3"): draw each
       agent's Im(theta) independently from a
       normal distribution N(mean, stddev), then
       clamp to [clamp_lo, clamp_hi]. This
       produces a population with varied initial
       engagement levels.

    2. NON-NUMERIC string (e.g., "imaginary"):
       assign every agent the same Im(theta) =
       mean. The entire population starts with
       identical engagement.

    Parameters
    ----------
    orien_stddev_str : str
        The XML parameter value. If it can be
        parsed as a float, it becomes the stddev
        for the normal draw. Otherwise, the
        hardcoded mean is used for every element.
    mean : float
        Default Im(theta) value. For preference
        Gaussians this is typically 1.0 (engaged);
        for aversion Gaussians it is pi-1 (~2.14,
        engaged on the aversion side).
    size : int
        Number of Gaussian dimensions (e.g.,
        num_policy_dims or num_trait_dims).
    clamp_lo : float
        Lower bound for Im(theta). Prevents
        values from crossing into the wrong half
        of [0, pi].
    clamp_hi : float
        Upper bound for Im(theta). Same purpose.

    Returns
    -------
    numpy.ndarray (complex, shape (size,))
        Array with real part 0 and imaginary part
        set to the sampled or hardcoded values.
    """
    try:
        stddev = float(orien_stddev_str)
        im_theta = np.clip(
            rng.normal(
                loc=mean, scale=stddev,
                size=size),
            clamp_lo, clamp_hi)
        return im_theta * 1j
    except ValueError:
        return np.full(size, mean * 1j)


class Gaussian():
    """A collection of 1-D complex Gaussians that
    share the same role (e.g., all policy-preference
    dimensions for one citizen).

    Mathematical form (per dimension):

      g(x; sigma, mu, theta) =
          1/(sigma * sqrt(2*pi))
          * exp(-(x - mu)^2 / (2*sigma^2))
          * exp(i * theta)

    Each instance stores arrays of length N (one
    entry per dimension) for mu, sigma, and theta.

    Parameters
    ----------
    mu : ndarray, shape (N,)
        Position on the policy/trait axis. This is
        where the agent "stands" on an issue. The
        axis extends from -inf to +inf; absolute
        values carry no intrinsic meaning.

    sigma : ndarray, shape (N,)
        Standard deviation (spread). Encodes how
        firmly the agent is attached to a specific
        position. Narrow (small sigma) = strong
        attachment. Broad (large sigma) = flexible.
        Must be > 0; enforced by sigma_floor in the
        citizen update loop.

    theta : ndarray, shape (N,), complex
        Orientation / engagement. Stored as purely
        imaginary: theta = Im(theta) * 1j. The
        overlap integral multiplies by cos(Im(theta)),
        so:
          Im(theta) = 0    -> cos = 1  (max engaged)
          Im(theta) = pi/2 -> cos = 0  (apathetic)
          Im(theta) = pi   -> cos = -1 (max engaged,
                              aversion type)
        Preference Gaussians live in [0, pi/2);
        aversion Gaussians live in (pi/2, pi].
        This sign convention means same-type
        integrals (pref x pref) are non-negative
        and cross-type integrals (pref x aver) are
        non-positive — encoding attraction vs.
        repulsion automatically.

    Derived (cached) variables
    --------------------------
    alpha : ndarray
        1 / (2 * sigma^2). Used in the overlap
        integral formula. Cached to avoid
        recomputation.

    cos_theta : ndarray
        cos(Im(theta)). The "engagement factor"
        that scales the Gaussian's contribution
        to any overlap integral. Positive for
        preferences, negative for aversions.

    self_norm : ndarray
        (pi * sigma^2)^0.25 * |cos_theta|.
        The square root of the unnormalized 1-D
        Gaussian's self-overlap I(G,G). Used to
        normalize the overlap integral to [-1, +1].
        Cached so that integral() only pays one
        multiply and one divide per call.
    """

    def __init__(self, pos, stddev, orien,
                 initialize):
        """Create a Gaussian collection.

        Parameters
        ----------
        pos : array-like, shape (N,)
            Initial mu values.
        stddev : array-like, shape (N,)
            Initial sigma values.
        orien : array-like, shape (N,), complex
            Initial theta values (purely imaginary).
        initialize : int
            If 1, immediately compute and cache
            the derived integration variables
            (alpha, cos_theta, self_norm). Pass 0
            to defer this step (used when the
            Gaussian will be populated via
            accumulate() and average() before any
            integrals are needed).
        """
        self.mu = np.array(pos)
        self.sigma = np.array(stddev)
        self.theta = np.array(orien)

        if (initialize == 1):
            self.update_integration_variables()


    def update_integration_variables(self):
        """Recompute cached derived variables from
        the current mu, sigma, and theta.

        Must be called after any direct modification
        to sigma or theta (e.g., after
        apply_influence_shifts() in citizen.py, or
        after adapt_to_patch() in politician.py).
        Failing to call this before the next overlap
        integral will cause stale cached values to
        be used, producing incorrect results.

        Computed quantities:
          alpha     = 1 / (2 * sigma^2)
          cos_theta = cos(Im(theta))
          self_norm = (pi * sigma^2)^0.25
                      * |cos_theta|

        The self_norm is the square root of the
        self-overlap of a unit-area 1-D Gaussian
        (times the theta engagement factor). For a
        normalized 1-D Gaussian g(x) = [1/(s*sqrt(
        2*pi))] * exp(-(x-mu)^2/(2*s^2)):

          I(G,G) = integral of g^2
                 = 1 / (2*s*sqrt(pi))
                 = 1 / (4*pi*s^2)^(1/2)

        so sqrt(I(G,G)) = (4*pi*s^2)^(-1/4)
                        = 1 / (pi*s^2)^0.25 / sqrt(2)

        However, because both the raw integral and
        self_norm use the UNNORMALIZED Gaussian
        exp(-(x-mu)^2/(2*s^2)) (area = s*sqrt(2*pi)
        rather than 1), the normalization prefactors
        cancel in the ratio I_raw/(norm1*norm2).
        The self-overlap of the unnormalized 1-D
        Gaussian is s*sqrt(pi), so:

          sqrt(I_unnorm(G,G)) = (s*sqrt(pi))^0.5
                              = (pi*s^2)^0.25
        """
        self.alpha = 0.5 / self.sigma**2
        self.cos_theta = np.cos(self.theta.imag)
        self.self_norm = (
            (np.pi * self.sigma**2)**0.25
            * np.abs(self.cos_theta))


    def integral(self, g):
        """Compute the normalized overlap integral
        between this Gaussian and another.

        The raw overlap integral between two 1-D
        (unnormalized) complex Gaussians G1, G2 is:

          I_raw = (pi / zeta)^0.5
                  * exp(-xi * d^2)
                  * cos(theta_1) * cos(theta_2)

        where:
          zeta = alpha_1 + alpha_2
          xi   = alpha_1 * alpha_2 / zeta
          d    = mu_1 - mu_2

        Derivation: for two Gaussians
        exp(-a1*(x-m1)^2) and exp(-a2*(x-m2)^2),
        completing the square in the product
        integral yields sqrt(pi/zeta) for the
        Gaussian integral factor and exp(-xi*d^2)
        for the separation penalty, where
        xi = a1*a2/zeta is the reduced exponent
        (analogous to reduced mass in classical
        mechanics).

        To make the result independent of sigma
        and bounded to [-1, +1], we normalize by
        the geometric mean of the two self-overlaps:

          I_norm = I_raw
                   / (self_norm_1 * self_norm_2)

        Physical interpretation:
          +1 = identical Gaussians (perfect
               alignment)
          -1 = identical except opposite engagement
               signs (pref vs aver; max
               disagreement)
           0 = no overlap (distant positions or at
               least one is fully apathetic)

        Parameters
        ----------
        g : Gaussian
            The other Gaussian to integrate against.
            Must have the same number of dimensions.

        Returns
        -------
        ndarray, shape (N,)
            Per-dimension normalized overlap values
            in [-1, +1]. Returns 0 for any dimension
            where either Gaussian has cos_theta = 0
            (fully apathetic / zero self_norm).
        """
        zeta = self.alpha + g.alpha
        xi = self.alpha * g.alpha / zeta
        dist = self.mu - g.mu
        exp_factor = np.exp(-xi * dist**2)
        raw = (
            (np.pi / zeta)**0.5 * exp_factor
            * self.cos_theta * g.cos_theta)
        norm = self.self_norm * g.self_norm
        return np.where(
            norm > 0.0, raw / norm, 0.0)


    def accumulate(self, gaussian):
        """Element-wise add another Gaussian's
        parameters into this one. Used when
        computing zone averages: each citizen's
        Gaussian is accumulated into a running
        total, which is then divided by the citizen
        count via average().

        Note: this does NOT update the cached
        integration variables. Call
        update_integration_variables() after
        averaging is complete.
        """
        self.mu += gaussian.mu
        self.sigma += gaussian.sigma
        self.theta += gaussian.theta


    def average(self, total_count):
        """Divide accumulated parameters by the
        total number of contributing Gaussians to
        obtain the zone average. Called after all
        citizens have been accumulated via
        accumulate().

        After calling this, call
        update_integration_variables() to refresh
        the cached derived quantities before using
        this Gaussian in any overlap integral.

        Parameters
        ----------
        total_count : int
            Number of Gaussians that were
            accumulated.
        """
        self.mu /= total_count
        self.sigma /= total_count
        self.theta /= total_count
