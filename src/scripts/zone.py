import numpy as np

from gaussian import Gaussian
from random_state import rng


class Zone():

    def __init__(self, settings, zone_type, patch):
        self.zone_type = zone_type
        self.patches = [patch]

        # Get the statistical parameters for the number of politicians for this zone.
        self.min_politicians = int(settings.infile_dict[1]["world"]
                [f"zone_type_{zone_type}"]["min_politicians"])
        self.max_politicians = int(settings.infile_dict[1]["world"]
                [f"zone_type_{zone_type}"]["max_politicians"])
        self.num_politicians_mean = float(settings.infile_dict[1]["world"]
                [f"zone_type_{zone_type}"]["num_politicians_mean"])
        self.num_politicians_stddev = float(settings.infile_dict[1]["world"]
                [f"zone_type_{zone_type}"]["num_politicians_stddev"])

        # Determine the initial number of politicians for this zone.
        self.num_politicians = int(rng.normal(loc=self.num_politicians_mean,
                scale=self.num_politicians_stddev))
        if (self.num_politicians < self.min_politicians):
            self.num_politicians = self.min_politicians
        elif (self.num_politicians > self.max_politicians):
            self.num_politicians = self.max_politicians

        # Each zone maintains a list of the politicians who vie for election in it.
        self.politician_list = []

        # Each zone maintains a list of the citizen average values for each policy and
        #   trait preference and aversion.
        self.avg_Pcp = []
        self.avg_Pca = []
        self.avg_Tcp = []
        self.avg_Tca = []

        # The zone notes the total number of citizens in it when computing
        #   averages.
        self.curr_num_citizens = 0


    def compute_zone_averages(self, world):
        # Initialize empty Gaussians for accumulating.
        self.avg_Pcp = Gaussian(np.zeros(world.num_policy_dims),
                                np.zeros(world.num_policy_dims),
                                np.zeros(world.num_policy_dims)+0j, 0)
        self.avg_Pca = Gaussian(np.zeros(world.num_policy_dims),
                                np.zeros(world.num_policy_dims),
                                np.zeros(world.num_policy_dims)+0j, 0)
        self.avg_Tcp = Gaussian(np.zeros(world.num_trait_dims),
                                np.zeros(world.num_trait_dims),
                                np.zeros(world.num_trait_dims)+0j, 0)
        self.avg_Tca = Gaussian(np.zeros(world.num_trait_dims),
                                np.zeros(world.num_trait_dims),
                                np.zeros(world.num_trait_dims)+0j, 0)

        # Initialize the count of the number of citizens in this zone.
        self.curr_num_citizens = 0

        # Visit each patch in this zone and accumulate the citizen policy+trait
        #   prefs+aversions from each citizen in this zone.
        for patch in self.patches:
            self.curr_num_citizens += len(patch.citizen_list)
            for citizen in patch.citizen_list:
                self.avg_Pcp.accumulate(world.citizens[citizen].stated_policy_pref)
                self.avg_Pca.accumulate(world.citizens[citizen].stated_policy_aver)
                self.avg_Tcp.accumulate(world.citizens[citizen].stated_trait_pref)
                self.avg_Tca.accumulate(world.citizens[citizen].stated_trait_aver)

        # Divide the values by the number of citizens that contributed to each average.
        self.avg_Pcp.average(self.curr_num_citizens)
        self.avg_Pca.average(self.curr_num_citizens)
        self.avg_Tcp.average(self.curr_num_citizens)
        self.avg_Tca.average(self.curr_num_citizens)

        # Update the integration variables using the newly computed averages.
        self.avg_Pcp.update_integration_variables()
        self.avg_Pca.update_integration_variables()
        self.avg_Tcp.update_integration_variables()
        self.avg_Tca.update_integration_variables()



    def add_politician(self, politician):
        self.politician_list.append(politician)


    def clear_politician_list(self):
        self.politician_list.clear()


    def add_patch(self, patch):
        self.patches.append(patch)


    def random_patch(self):
        return rng.choice(self.patches)


    def set_elected_politician(self, top_vote_getter):
        # Reset all politicians in this zone to the not elected state.
        for politician in self.politician_list:
            politician.elected = False

        # Set the elected politician to the elected state and set a
        #   specific variable pointing to it.
        top_vote_getter.elected = True
        self.elected_politician = top_vote_getter
