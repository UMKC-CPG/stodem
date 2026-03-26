import numpy as np

from gaussian import Gaussian
from random_state import rng


class Government():
    """Represents the government and its enacted
    policies.

    The government maintains a single set of
    policy Gaussians (one per policy dimension)
    called "enacted policy" (notation: Pge). These
    Gaussians represent the current laws/policies
    in effect. During the govern phase, elected
    politicians exert forces on Pge to shift it
    toward their preferred positions and away from
    their aversions. Citizens then compare their
    ideal and stated policy positions against Pge
    to determine well-being and satisfaction.

    The government has no personality traits —
    only policy positions. Traits are a property
    of individual agents (citizens, politicians),
    not institutions.

    Attributes
    ----------
    num_policy_dims : int
        Number of independent policy dimensions.
    enacted_policy : Gaussian
        The current government policy. One Gaussian
        per policy dimension. Modified each govern
        step by elected politicians' forces
        (DESIGN.md §7.5.3).
    spread_rate : float
        Controls the natural broadening of enacted
        policy each govern cycle. Without active
        politician forces, policy sigma drifts
        upward over time, representing institutional
        entropy — the tendency for policy precision
        to erode without active political will.
    """

    def __init__(self, settings):
        self.num_policy_dims = int(
                settings.infile_dict[1][
                    "world"]["num_policy_dims"])

        # Enacted policy Gaussians: one per policy
        #   dimension. These represent the current
        #   government policy and are modified each
        #   govern step by elected politicians'
        #   forces (DESIGN.md §7.5.3).
        #
        #   mu: drawn from a zero-mean normal
        #     distribution — initial policy positions
        #     are random.
        #   sigma: drawn from a normal distribution
        #     and wrapped in abs() to guarantee a
        #     positive spread.
        #   theta: set to [1]*N, i.e., Im(theta)=1
        #     for all dimensions. This places the
        #     government on the preference side of
        #     the theta sign convention (cos(1)~0.54,
        #     positive). The government's enacted
        #     policies are always "preferences" (they
        #     represent what IS enacted, not what the
        #     government avoids). The theta is fixed
        #     at initialization and does not change
        #     during the simulation — only mu and
        #     sigma are updated by politician forces
        #     in the govern phase.
        self.enacted_policy = Gaussian(
                rng.normal(loc=0.0,
                    scale=float(
                        settings.infile_dict[1][
                            "government"][
                            "policy_pos_stddev"]),
                    size=self.num_policy_dims),
                np.abs(rng.normal(loc=0.0,
                    scale=float(
                        settings.infile_dict[1][
                            "government"][
                            "policy_stddev_stddev"]),
                    size=self.num_policy_dims)),
                [1] * self.num_policy_dims, 1)

        # spread_rate controls the natural
        #   broadening of enacted policy each
        #   govern cycle (DESIGN.md §7.5.4).
        #   The formula is:
        #     sigma += spread_rate / sigma
        #   This means narrow (precise) policies
        #   spread faster than broad (vague) ones:
        #   a narrow policy has small sigma, so
        #   spread_rate/sigma is large, causing
        #   rapid broadening. A broad policy has
        #   large sigma, so the broadening is
        #   slow. This reflects the idea that
        #   specific policy details are harder to
        #   maintain than vague directives.
        #   Applied once per govern cycle (not per
        #   step) to prevent runaway broadening.
        self.spread_rate = float(
                settings.infile_dict[1][
                    "government"]["spread_rate"])

        # sigma_floor is the minimum allowed value
        #   for Pge.sigma. It prevents sigma from
        #   reaching zero or going negative under
        #   politician forces (DESIGN.md §7.5.3).
        #   This is the same floor used for citizen
        #   Gaussians — read from the citizens
        #   section of the input file.
        self.sigma_floor = float(
                settings.infile_dict[1][
                    "citizens"]["sigma_floor"])
