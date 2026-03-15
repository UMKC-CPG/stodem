from gaussian import Gaussian
from random_state import rng


class Government():

    # Define class variables.
    num_policy_dims = 0

    def __init__(self, settings):
        self.num_policy_dims = int(
                settings.infile_dict[1]["world"]["num_policy_dims"])

        # Real policy orientation.
        self.enacted_policy = Gaussian(rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["government"]
                ["policy_pos_stddev"]),
                size=self.num_policy_dims), rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["government"]
                ["policy_stddev_stddev"]),
                size=self.num_policy_dims), [1] * self.num_policy_dims, 1)
