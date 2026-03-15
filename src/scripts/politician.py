import numpy as np

from gaussian import Gaussian
from random_state import rng


class Politician():
    # Politicians have an innate set of personality "traits" positioned on a
    #   one-dimensional spectrum. The distance between a citizen's trait
    #   position and a politician's trait position will influence (1) the
    #   ability of the politician to persuade a citizen with respect to their
    #   policy positions; (2) the probability that a citizen will vote for a
    #   politician; (3) the probability that a citizen will allow policy
    #   position misrepresentations to go "unpunished".

    # Politicians have an innate position for each policy dimension. The
    #   innate position should not be understood as a "values" statement in
    #   any definite sense. For some politicians, it may be representative of
    #   their "values" but for others it may be better thought of as their
    #   "desire". Presently, the innate position is static for each
    #   politician.

    # Politicians have an apparent "externally visible" position for each
    #   policy dimension. The external position is what a politician presents
    #   to citizens. The apparent position will differ from the innate
    #   position due to factors: (1) The innate position and the average
    #   position of the citizens who's votes the politician wants to obatian
    #   tend to differ. So, the politician may present an apparent position
    #   to the citizens in an effort to persuade them. (2) The politician is
    #   willing to misrepresent their innate position by an amount in
    #   proportion to their propensity to lie/pander and believe that they
    #   will not turn off citizens by being detected.

    # It is assumed that citizens may or may not be able to detect (or may
    #   not care about) the difference between an apparent policy position
    #   and the innate position that a politician has.

    def __init__(self, settings, zone_type, zone, patch):

        # Convert some settings variables to instance variables.
        self.num_policy_dims = int(
                settings.infile_dict[1]["world"]["num_policy_dims"])
        self.num_trait_dims = int(
                settings.infile_dict[1]["world"]["num_trait_dims"])

        # Define the initial instance variables of this politician obtained
        #   from the input file.
        self.reset_to_input(settings)

        # Select strategies for politician activities.
        self.move_strategy = self.select_strategy(settings, "move")
        self.adapt_strategy = self.select_strategy(settings, "adapt")
        self.campaign_strategy = self.select_strategy(settings, "campaign")

        # Assign instance variables from passed initialization parameters.
        self.zone_type = zone_type
        self.zone = zone
        self.patch = patch

        # Initialize any other instance variables to their default value.
        self.elected = False
        self.votes = 0


    def reset_to_input(self, settings):
        # Use a uniform initial policy orientation
        self.innate_policy_pref = Gaussian(rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["policy_pref_pos_stddev"]),
                size=self.num_policy_dims), rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["policy_pref_stddev_stddev"]),
                size=self.num_policy_dims), rng.uniform(high=2.0*np.pi,
                size=self.num_policy_dims), 1)
        self.innate_policy_aver = Gaussian(rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["policy_aver_pos_stddev"]),
                size=self.num_policy_dims), rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["policy_aver_stddev_stddev"]),
                size=self.num_policy_dims), rng.uniform(high=2.0*np.pi,
                size=self.num_policy_dims), 1)
        self.ext_policy_pref = Gaussian(rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["policy_pref_pos_stddev"]),
                size=self.num_policy_dims), rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["policy_pref_stddev_stddev"]),
                size=self.num_policy_dims), rng.uniform(high=2.0*np.pi,
                size=self.num_policy_dims), 1)
        self.ext_policy_aver = Gaussian(rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["policy_aver_pos_stddev"]),
                size=self.num_policy_dims), rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["policy_aver_stddev_stddev"]),
                size=self.num_policy_dims), rng.uniform(high=2.0*np.pi,
                size=self.num_policy_dims), 1)

        # Use a uniform initial trait orientation
        self.innate_trait = Gaussian(rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["trait_innate_pos_stddev"]),
                size=self.num_trait_dims), rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["trait_innate_stddev_stddev"]),
                size=self.num_trait_dims), rng.uniform(high=2.0*np.pi,
                size=self.num_trait_dims), 1)
        self.ext_trait = Gaussian(rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["trait_ext_pos_stddev"]),
                size=self.num_trait_dims), rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["trait_ext_stddev_stddev"]),
                size=self.num_trait_dims), rng.uniform(high=2.0*np.pi,
                size=self.num_trait_dims), 1)

        self.policy_influence = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["policy_influence_stddev"]))
        self.trait_influence = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["trait_influence_stddev"]))
        self.policy_lie = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["policy_lie_stddev"]))
        self.trait_lie = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["trait_lie_stddev"]))
        self.pander = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["pander_stddev"]))


    def reset_votes(self):
        self.votes = 0

    # Add self to the list of politicians that each citizen in the zone could
    #   vote for.
    def present_to_citizens(self, world):
        # Consider every patch in the zone that this politician competes in.
        for patch in self.zone.patches:
            #print("patch.citizen_list", patch.citizen_list)

            # Consider every citizen in this patch.
            for citizen in patch.citizen_list:
                world.citizens[citizen].add_politician(self)


    def select_strategy(self, settings, strat_type):
        # Select a strategy using the xml given probabilities. The strategy
        #   is selected according to the cumulative probabilities given in
        #   the xml file. For example, if the cumulative probabilities for
        #   three strategies are 0.5,0.75,1.0 then 50% of the time the first
        #   strategy is used, and 25% of the time each of the other
        #   strategies is used. To make the selection, we obtain a random
        #   number and look for the first index where the random number is
        #   less than one of the cumulative probabilty values.
        strategy_index = -1
        random_float = rng.uniform()
        strategy_distribution = [float(prob) for prob in settings.infile_dict[1]
                ["politicians"]
                [f"cumul_{strat_type}_strategy_probs"].split(',')]
        for index in range(len(strategy_distribution)):
            if (random_float < strategy_distribution[index]):
                strategy_index = index
                break

        return strategy_index


    def move(self):
        # Move according to the strategy that this politician is following.
        if (self.move_strategy == 0):
            # Select a random patch within the same zone.
            self.patch = self.zone.random_patch()


    def adapt_to_patch(self, world):
        # Set apparent policy positions and personality positions according
        #   to the strategy that this politician is following.
        if (self.adapt_strategy == 0):
            # No: present innate positions as external positions
            #   without modification. The politician says what they believe.
            self.ext_policy_pref.mu[:] = self.innate_policy_pref.mu
            self.ext_policy_pref.sigma[:] = self.innate_policy_pref.sigma
            self.ext_policy_pref.theta[:] = self.innate_policy_pref.theta
            self.ext_policy_pref.update_integration_variables()

            self.ext_policy_aver.mu[:] = self.innate_policy_aver.mu
            self.ext_policy_aver.sigma[:] = self.innate_policy_aver.sigma
            self.ext_policy_aver.theta[:] = self.innate_policy_aver.theta
            self.ext_policy_aver.update_integration_variables()

            self.ext_trait.mu[:] = self.innate_trait.mu
            self.ext_trait.sigma[:] = self.innate_trait.sigma
            self.ext_trait.theta[:] = self.innate_trait.theta
            self.ext_trait.update_integration_variables()


    def persuade(self, world):
        # Iterate over the list of citizens in the patch that the politician
        #   is currently on.

        #

        for citizen in self.patch.citizen_list:
            pass
            # Compare each policy position of the citizen with the current
            #   apparent policy position of the politician.
            #
            # In the explanation given below, mirror symmetry has the same
            #   result.
            # If the governing policy is left of the citizen and the
            #   politician policy is left

        world.properties[0].data = (world.properties[0].data
                + rng.uniform(high=0.01))
