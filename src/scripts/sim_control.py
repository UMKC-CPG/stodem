import numpy as np
from dataclasses import dataclass


@dataclass
class SimProperty():
    group : str
    name : str
    datatype : str
    data : np.ndarray


class SimControl():

    def __init__(self, settings):

        # Initialize instance variables.
        self.curr_step = 0

        # Extract simulation control parameters
        #   from the xml input file.
        self.num_cycles = int(
                settings.infile_dict[1][
                    "sim_control"]["num_cycles"])
        self.num_campaign_steps = int(
                settings.infile_dict[1][
                    "sim_control"]["num_campaign_steps"])
        self.num_govern_steps = int(
                settings.infile_dict[1][
                    "sim_control"]["num_govern_steps"])

        # Get the resolution and negligability limit
        #   of the data that may be output.
        self.data_resolution = int(
                settings.infile_dict[1][
                    "sim_control"]["data_resolution"])
        self.data_neglig = float(
                settings.infile_dict[1][
                    "sim_control"]["data_neglig"])

        # Compute the total number of simulation steps.
        self.total_num_steps = (
                (self.num_campaign_steps
                 + self.num_govern_steps)
                * self.num_cycles)


    def compute_data_range(self, settings, world):
        """Compute the bounding range of all
        Gaussians across every agent and the
        government, for future visualization use.

        Each Gaussian occupies a finite region of
        the real axis. The "extent" of a Gaussian
        is defined as the distance from mu at which
        its amplitude drops to data_neglig times
        the peak. From the Gaussian formula:

          exp(-(x-mu)^2 / (2*sigma^2)) = neglig
          => |x - mu| = sigma * sqrt(-2*ln(neglig))

        This method sweeps through every citizen
        Gaussian (stated policy pref/aver, ideal
        policy pref, stated trait pref/aver), every
        politician Gaussian (innate and external
        policy pref/aver, innate and external trait),
        and the government's enacted policy to find
        the global min/max along each policy and
        trait dimension. A 20% buffer is added on
        each side to accommodate drift during the
        simulation.

        The results are stored in world.policy_limits
        and world.trait_limits as [min_array,
        max_array] pairs, one element per dimension.
        These are used by the output module for
        visualization scaling.

        Parameters
        ----------
        settings : ScriptSettings
            The simulation settings (provides
            data_neglig via self).
        world : World
            The populated simulation world
            containing all agents and government.
        """
        # Compute the extent multiplier from the
        #   negligibility threshold. At distance
        #   extent_mult * sigma from mu, the
        #   Gaussian amplitude = data_neglig * peak.
        extent_mult = np.sqrt(
            -2.0 * np.log(self.data_neglig))

        # Initialize min/max arrays for policy and
        #   trait dimensions. Starting at +/-inf
        #   ensures the first Gaussian encountered
        #   will set the initial bounds.
        policy_min = np.full(
            world.num_policy_dims, np.inf)
        policy_max = np.full(
            world.num_policy_dims, -np.inf)
        trait_min = np.full(
            world.num_trait_dims, np.inf)
        trait_max = np.full(
            world.num_trait_dims, -np.inf)

        def update_limits(gaussian, dim_min,
                          dim_max):
            """Update min/max from one Gaussian's
            extent: [mu - extent, mu + extent]."""
            extent = (
                np.abs(gaussian.sigma) * extent_mult)
            dim_min[:] = np.minimum(
                dim_min, gaussian.mu - extent)
            dim_max[:] = np.maximum(
                dim_max, gaussian.mu + extent)

        # Process all citizens (5 Gaussians each).
        for citizen in world.citizens:
            update_limits(
                citizen.stated_policy_pref,
                policy_min, policy_max)
            update_limits(
                citizen.stated_policy_aver,
                policy_min, policy_max)
            update_limits(
                citizen.ideal_policy_pref,
                policy_min, policy_max)
            update_limits(
                citizen.stated_trait_pref,
                trait_min, trait_max)
            update_limits(
                citizen.stated_trait_aver,
                trait_min, trait_max)

        # Process all politicians (6 Gaussians
        #   each: innate+external for policy
        #   pref/aver, innate+external for trait).
        for politician in world.politicians:
            update_limits(
                politician.innate_policy_pref,
                policy_min, policy_max)
            update_limits(
                politician.innate_policy_aver,
                policy_min, policy_max)
            update_limits(
                politician.ext_policy_pref,
                policy_min, policy_max)
            update_limits(
                politician.ext_policy_aver,
                policy_min, policy_max)
            update_limits(
                politician.innate_trait,
                trait_min, trait_max)
            update_limits(
                politician.ext_trait,
                trait_min, trait_max)

        # Process the single government enacted
        #   policy (policy dims only; government
        #   has no traits).
        update_limits(
            world.government.enacted_policy,
            policy_min, policy_max)

        # Extend the range by 20% on each side
        #   to accommodate simulation drift.
        buffer_factor = 1.2
        policy_range = policy_max - policy_min
        trait_range = trait_max - trait_min

        policy_min -= (
            0.5 * (buffer_factor - 1.0)
            * policy_range)
        policy_max += (
            0.5 * (buffer_factor - 1.0)
            * policy_range)
        trait_min -= (
            0.5 * (buffer_factor - 1.0)
            * trait_range)
        trait_max += (
            0.5 * (buffer_factor - 1.0)
            * trait_range)

        # Store the computed limits in the World
        #   instance for use by output modules.
        world.policy_limits = [
            policy_min, policy_max]
        world.trait_limits = [
            trait_min, trait_max]
