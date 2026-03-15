import numpy as np
from dataclasses import dataclass


@dataclass
class SimProperty():
    group : str
    name : str
    datatype : str
    data : np.ndarray


class SimControl():

    # Declare and initialize the class variables.
    curr_step = 0  # The current overall timestep number.
    num_cycles = 0  # The number of campaign cycles.
    num_campaign_steps = 0  # Number of time steps in a campaign.
    num_govern_steps = 0  # Number of time steps to govern.
    # Number of time steps in a primary campaign.
    num_primary_campaign_steps = 0
    total_num_steps = 0  # Total number of simulation steps.
    data_resolution = 1 # data points per real number
    data_neglig = 0.01 # negligability limit for determining min/max data range


    def __init__(self, settings):
        # Extract simulation control parameters from the xml input file.
        SimControl.num_cycles = int(
                settings.infile_dict[1]["sim_control"]["num_cycles"])
        SimControl.num_campaign_steps = int(
                settings.infile_dict[1]["sim_control"]["num_campaign_steps"])
        SimControl.num_govern_steps = int(
                settings.infile_dict[1]["sim_control"]["num_govern_steps"])
        SimControl.num_primary_campaign_steps = int(
                settings.infile_dict[1]["sim_control"]
                ["num_primary_campaign_steps"])

        # Get the resolution and negligabilty limit of the data that may be
        #   output.
        SimControl.data_resolution = \
                int(settings.infile_dict[1]["sim_control"]["data_resolution"])
        SimControl.data_neglig = \
                float(settings.infile_dict[1]["sim_control"]["data_neglig"])

        # Compute the total number of simulation steps.
        SimControl.total_num_steps = (SimControl.num_campaign_steps + \
                SimControl.num_govern_steps + \
                SimControl.num_primary_campaign_steps) * \
                SimControl.num_cycles


    # Consider all citizens, politicians, and the government. For each, take
    #   the position of each policy and trait (if applicable) and compute the
    #   maximum and minimum extent of the Gaussian (to the data negligability
    #   limit). Use the maximum and minimum of each Gaussian to find the min
    #   and max of every dimension. Then extend the min/max a little bit so
    #   that (hopefully) if the ranges change during the simulation they will
    #   not go past the limits.
    def compute_data_range(self, settings, world):

        # Compute the extent multiplier from the negligibility threshold.
        # At x = mu ± extent_mult * sigma, the Gaussian amplitude =
        #   data_neglig * peak
        extent_mult = np.sqrt(-2.0 * np.log(SimControl.data_neglig))

        # Initialize min/max arrays for policy and trait dimensions.
        policy_min = np.full(world.num_policy_dims, np.inf)
        policy_max = np.full(world.num_policy_dims, -np.inf)
        trait_min = np.full(world.num_trait_dims, np.inf)
        trait_max = np.full(world.num_trait_dims, -np.inf)

        # Helper to update min/max from a Gaussian's extent.
        def update_limits(gaussian, dim_min, dim_max):
            extent = np.abs(gaussian.sigma) * extent_mult
            dim_min[:] = np.minimum(dim_min, gaussian.mu - extent)
            dim_max[:] = np.maximum(dim_max, gaussian.mu + extent)

        # Process all citizens.
        for citizen in world.citizens:
            update_limits(citizen.stated_policy_pref, policy_min, policy_max)
            update_limits(citizen.stated_policy_aver, policy_min, policy_max)
            update_limits(citizen.ideal_policy_pref, policy_min, policy_max)
            update_limits(citizen.stated_trait_pref, trait_min, trait_max)
            update_limits(citizen.stated_trait_aver, trait_min, trait_max)

        # Process all politicians.
        for politician in world.politicians:
            update_limits(politician.innate_policy_pref, policy_min, policy_max)
            update_limits(politician.innate_policy_aver, policy_min, policy_max)
            update_limits(politician.ext_policy_pref, policy_min, policy_max)
            update_limits(politician.ext_policy_aver, policy_min, policy_max)
            update_limits(politician.innate_trait, trait_min, trait_max)
            update_limits(politician.ext_trait, trait_min, trait_max)

        # Process government.
        update_limits(world.government.enacted_policy, policy_min, policy_max)

        # Extend the range by some factor to allow for simulation drift.
        buffer_factor = 1.2  # 20% buffer
        policy_range = policy_max - policy_min
        trait_range = trait_max - trait_min

        policy_min -= 0.5 * (buffer_factor - 1.0) * policy_range
        policy_max += 0.5 * (buffer_factor - 1.0) * policy_range
        trait_min -= 0.5 * (buffer_factor - 1.0) * trait_range
        trait_max += 0.5 * (buffer_factor - 1.0) * trait_range

        # Store the computed limits in the World instance.
        world.policy_limits = [policy_min, policy_max]
        world.trait_limits = [trait_min, trait_max]
