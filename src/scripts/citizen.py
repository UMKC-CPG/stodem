import numpy as np

from gaussian import Gaussian
from random_state import rng


class Citizen():
    # Citizens have an innate "personality trait" preference and aversion
    #   that can change to align with a politician.

    # Citizens have a stated policy position for each policy. This is the
    #   policy position that the citizen claims to align with and it will
    #   affect their preference for a particular politician. Similarly, each
    #   citizen has an aversion associated with each policy.

    # Citizens have a most-beneficial policy position that the citizen does
    #   not directly know. I.e., the well-being of the citizen will depend on
    #   the alignment between governing policy and this most-beneficial
    #   policy, but the stated policy may be quite different than the
    #   most-beneficial policy. For example, if a policy represented "tax
    #   rates", it is hard to know what the best tax rate for an individual
    #   should be. If it was set to zero, the individual would pay nothing,
    #   but also likely have no services. If it was set to 100% they would
    #   have no money, but would have many servies. The "correct" number is
    #   not easy for any individual to know and that individual's stated
    #   preference may easily be different from whatever number actually
    #   benefits them the most.

    # Citizens have a probability of participation. It is affected (in no
    #   order) by:
    #   (1) The personality alignment between a citizen and a politician.
    #   (2) The cumulative policy position alignment between a citizen and a
    #       politician.
    #   (3) Whether or not the citizen voted previously.
    #   (4) The number of citizens in the same zone that vote in agreement
    #       with the citizen.
    #   (5) The well-being of the citizen.
    #   (6) Whether the person that they voted for last time won or not.

    # Citizens have a well-being factor that weights the degree to which
    #   they will use personality or policy alignment when deciding how to
    #   cast their vote.


    def __init__(self, settings, patch, zones):

        # Get temporary local names for settings variables.
        self.num_policy_dims = int(
                settings.infile_dict[1]["world"]["num_policy_dims"])
        self.num_trait_dims = int(
                settings.infile_dict[1]["world"]["num_trait_dims"])

        # Define the initial instance variables of this citizen.

        # Define instance variables given in the input file.
        self.participation_prob = rng.normal(loc=float(
                settings.infile_dict[1]["citizens"]["participation_prob_pos"]),
                scale=float(settings.infile_dict[1]["citizens"]
                ["participation_prob_stddev"]))

        # For each policy, create a preference. The preference is represented
        #   using a Gaussian function that is centered near 0 (following a
        #   Gaussian distribution with the given standard deviation) and that
        #   has a spread as a random number using a Gaussian distribution
        #   centered at 0 with a
        self.stated_policy_pref = Gaussian(rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["citizens"]
                ["policy_pref_pos_stddev"]),
                size=self.num_policy_dims), rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["citizens"]
                ["policy_pref_stddev_stddev"]),
                size=self.num_policy_dims),
                (rng.integers(low=0, high=1, endpoint=True,
                size=self.num_policy_dims)*2 - 1) * 1j, 1)

        self.stated_policy_aver = Gaussian(rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["citizens"]
                ["policy_aver_pos_stddev"]),
                size=self.num_policy_dims), rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["citizens"]
                ["policy_aver_stddev_stddev"]),
                size=self.num_policy_dims),
                (rng.integers(low=0, high=1, endpoint=True,
                size=self.num_policy_dims)*2 - 1) * 1j, 1)

        self.ideal_policy_pref = Gaussian([x + rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["citizens"]
                ["ideal_policy_pref_pos_stddev"]))
                for x in self.stated_policy_pref.mu],
                rng.normal(loc=0.0, scale=[float(
                settings.infile_dict[1]["citizens"]
                ["ideal_policy_pref_stddev_stddev"])
                for x in range(self.num_policy_dims)]),
                [1 + 0j for x in range(self.num_policy_dims)], 1)

        self.stated_trait_pref = Gaussian(rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["citizens"]
                ["trait_pref_pos_stddev"]),
                size=self.num_trait_dims), rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["citizens"]
                ["trait_pref_stddev_stddev"]),
                size=self.num_trait_dims),
                (rng.integers(low=0, high=1, endpoint=True,
                size=self.num_trait_dims)*2 - 1) * 1j, 1)

        self.stated_trait_aver = Gaussian(rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["citizens"]
                ["trait_aver_pos_stddev"]),
                size=self.num_trait_dims), rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["citizens"]
                ["trait_aver_stddev_stddev"]),
                size=self.num_trait_dims),
                (rng.integers(low=0, high=1, endpoint=True,
                size=self.num_trait_dims)*2 - 1) * 1j, 1)

        self.policy_consistency = self.policy_alignment()

        self.policy_trait_ratio = np.clip(rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["citizens"]
                ["policy_trait_ratio_stddev"])), -0.5, 0.5)

        # Initialize instance variables that do not come from the input file.
        self.current_patch = patch
        self.politician_list = []  # Politicians that this citizen can vote for.
        self.set_zone_list(zones)  # Zones that this citizen is in.


    def set_zone_list(self, zones):
        # Initialize that this citizen belongs to no zones.
        self.zone_list = []

        # Consider each of the zone types (accessed by looking at the length of
        #   the zone_index array in the current patch of this citizen). The
        #   zone_index array will have one entry for each type of zone.
        for zone_type in range(len(self.current_patch.zone_index)):
            # Of all zones of the current zone_type in the World, append the
            #   zone_index of the current zone_type associated with the
            #   current_patch that this citizen is on.
            self.zone_list.append(zones[zone_type]
                                  [self.current_patch.zone_index[zone_type]])


    # Add a politician.
    def add_politician(self, politician):
        self.politician_list.append(politician)


    # Clear the politicians that this citizen could vote for.
    def clear_politicians(self):
        self.politician_list.clear()


    def compute_all_overlaps(self, world):

        # Initialize the integral solution lists.
        self.initialize_lists()

        # Compute the integrals between the current self citizen and all
        #   relevant politicians
        self.policy_politician_integrals()
        self.trait_politician_integrals()

        # Assume that the citizen zone averages have been computed. Then compute
        #   the integrals between the current self citizen and all the relevant
        #   citizen zone averages.
        self.policy_citizen_integrals()
        self.trait_citizen_integrals()

        # Compute the integrals between the current self citizen and all
        #   government enacted policies.
        self.policy_government_integrals(world)


    def initialize_lists(self):

        # Initialize lists that will hold overlap integral solutions between
        #   citizen and politicians (policy and trait, preference and
        #   aversion).
        # Policy: citizen preference vs politician preference
        self.Pcp_Ppp_ol = []
        self.Pcp_Ppa_ol = [] # Policy: citizen preference vs politician aversion
        self.Pca_Ppp_ol = [] # Policy: citizen aversion vs politician preference
        self.Pca_Ppa_ol = [] # Policy: citizen aversion vs politician aversion
        self.Tcp_Tpx_ol = [] # Trait: citizen preference vs politician external
        self.Tca_Tpx_ol = [] # Trait: citizen aversion vs politician external

        # Initialize lists that will hold overlap integral solutions between
        #   citizen and the zone average values of other citizens for both
        #   policies and traits
        # Policy: citizen preference vs zone avg citizen preference
        self.Pcp_Pcp_ol = []
        # Policy: citizen preference vs zone avg citizen aversion
        self.Pcp_Pca_ol = []
        # Policy: citizen aversion vs zone avg citizen preference
        self.Pca_Pcp_ol = []
        # Policy: citizen aversion vs zone avg citizen aversion
        self.Pca_Pca_ol = []
        # Trait: citizen preference vs zone avg citizen preference
        self.Tcp_Tcp_ol = []
        # Trait: citizen preference vs zone avg citizen aversion
        self.Tcp_Tca_ol = []
        # Trait: citizen aversion vs zone avg citizen preference
        self.Tca_Tcp_ol = []
        # Trait: citizen aversion vs zone avg citizen aversion
        self.Tca_Tca_ol = []

        # Initialize lists that will hold overlap integral solutions between
        #   citizen and the government policies.
        self.Pcp_Pge_ol = [] # Policy: citizen preference vs government enacted
        self.Pca_Pge_ol = [] # Policy: citizen aversion vs government enacted
        self.Pci_Pge_ol = [] # Policy: citizen ideal vs government enacted


    def policy_politician_integrals(self):

        # Compute the overlaps between the citizen and each relevant politician.
        for politician in self.politician_list:

            # Obtain the overlap between each citizen policy preference and
            #   aversion and each politician policy preference and aversion.
            self.Pcp_Ppp_ol.append(self.stated_policy_pref.integral(
                    politician.ext_policy_pref))
            self.Pca_Ppa_ol.append(self.stated_policy_aver.integral(
                    politician.ext_policy_aver))
            self.Pcp_Ppa_ol.append(self.stated_policy_pref.integral(
                    politician.ext_policy_aver))
            self.Pca_Ppp_ol.append(self.stated_policy_aver.integral(
                    politician.ext_policy_pref))


    def trait_politician_integrals(self):

        # Compute the overlaps between the citizen and each relevant politician.
        for politician in self.politician_list:

            # Obtain the overlap between each citizen trait preference and
            #   aversion and each politician externally exposed trait.
            self.Tcp_Tpx_ol.append(
                    self.stated_trait_pref.integral(politician.ext_trait))
            self.Tca_Tpx_ol.append(
                    self.stated_trait_aver.integral(politician.ext_trait))


    def policy_citizen_integrals(self):

        for zone in self.zone_list:
            # Obtain the overlap between each citizen policy preference and
            #   aversion and the zone average values across all citizen of
            #   the zone.
            self.Pcp_Pcp_ol.append(
                    self.stated_policy_pref.integral(zone.avg_Pcp))
            self.Pca_Pca_ol.append(
                    self.stated_policy_aver.integral(zone.avg_Pca))
            self.Pcp_Pca_ol.append(
                    self.stated_policy_pref.integral(zone.avg_Pca))
            self.Pca_Pcp_ol.append(
                    self.stated_policy_aver.integral(zone.avg_Pcp))


    def trait_citizen_integrals(self):

        for zone in self.zone_list:
            # Obtain the overlap between each citizen trait preference and
            #   aversion and the zone average values across all citizen of
            #   the zone.
            self.Tcp_Tcp_ol.append(
                    self.stated_trait_pref.integral(zone.avg_Tcp))
            self.Tca_Tca_ol.append(
                    self.stated_trait_aver.integral(zone.avg_Tca))
            self.Tcp_Tca_ol.append(
                    self.stated_trait_pref.integral(zone.avg_Tca))
            self.Tca_Tcp_ol.append(
                    self.stated_trait_aver.integral(zone.avg_Tcp))


    def policy_government_integrals(self, world):

        # Compute the overlaps between the citizen and the enacted policies
        #   of the government.
        self.Pcp_Pge_ol.append(self.stated_policy_pref.integral(
                world.government.enacted_policy))
        self.Pca_Pge_ol.append(self.stated_policy_aver.integral(
                world.government.enacted_policy))
        self.Pci_Pge_ol.append(self.ideal_policy_pref.integral(
                world.government.enacted_policy))


    def prepare_for_influence(self, num_policy_dims, num_trait_dims):
        # Initialize variables to accumulate orientation, position, and
        #   standard deviation shifts that are caused by influences from
        #   politicians, other citizens and the citizen's own sense of
        #   well-being.
        self.policy_orien_shift = [0 for x in range(num_policy_dims)]
        self.policy_pos_shift = [0 for x in range(num_policy_dims)]
        self.policy_stddev_shift = [0 for x in range(num_policy_dims)]
        self.trait_orien_shift = [0 for x in range(num_trait_dims)]
        self.trait_pos_shift = [0 for x in range(num_trait_dims)]
        self.trait_stddev_shift = [0 for x in range(num_trait_dims)]


    def build_response_to_politician_influence(self):

        # This citizen must incorporate all influence from all politicians
        #   that they could vote for.

        # The influence takes the form of shifts to the orientation,
        #   position, and standard deviation of each policy and trait Gaussian.

        # The orientation of each policy will shift to make the citizen more
        #   or less engaged.

        # The sum of trait integrals can influence

        # Consider each politician that this citizen could vote for from
        #   each zone.
        for politician in self.politician_list:
            self.policy_orien_shift += self.Pcp_Ppp_ol
            self.policy_orien_shift += self.Pca_Ppa_ol
            self.policy_orien_shift += self.Pcp_Ppa_ol
            self.policy_orien_shift += self.Pca_Ppp_ol
            self.trait_orien_shift += self.Tcp_Tpx_ol
            self.trait_orien_shift += self.Tca_Tpx_ol


    def build_response_to_citizen_collective(self):
        self.policy_orien_shift += self.Pcp_Pcp_ol
        self.policy_orien_shift += self.Pca_Pca_ol
        self.policy_orien_shift += self.Pcp_Pca_ol
        self.policy_orien_shift += self.Pca_Pcp_ol
        self.trait_orien_shift += self.Tcp_Tcp_ol
        self.trait_orien_shift += self.Tca_Tca_ol
        self.trait_orien_shift += self.Tcp_Tca_ol
        self.trait_orien_shift += self.Tca_Tcp_ol


    def build_response_to_well_being(self):
        self.well_being = sum(self.Pci_Pge_ol[0])


    def score_candidates(self, world):

        w_policy = 0.5 + self.policy_trait_ratio
        w_trait = 0.5 - self.policy_trait_ratio

        pol_index = 0
        self.politician_score = []
        for politician in self.politician_list:
            policy_sum = (sum(self.Pcp_Ppp_ol[pol_index])
                    + sum(self.Pca_Ppa_ol[pol_index])
                    + sum(self.Pcp_Ppa_ol[pol_index])
                    + sum(self.Pca_Ppp_ol[pol_index]))
            trait_sum = (sum(self.Tcp_Tpx_ol[pol_index])
                    + sum(self.Tca_Tpx_ol[pol_index]))
            self.politician_score.append(
                    w_policy * policy_sum + w_trait * trait_sum)
            pol_index += 1


    def vote_for_candidates(self, world):
        # The assumption is that a citizen who decides to vote, will vote for
        #   every one of their top candidates. If a citizen decides to not
        #   vote, then they vote for none of their candidates. (Clearly, this
        #   could be modified so that citizens make a decision to "vote-at-all"
        #   followed by separate decisions about making a vote for each zone.
        #   This approach is a bit more complicated and so it is not done yet.
        # Determine if the citizen will vote. If not, return. If so, continue.
        if (rng.random() > self.participation_prob):
            return

        # Each zone level (district, state, country, …) holds an independent
        # election. The citizen evaluates all candidates in each zone
        # separately and casts one vote per zone for the best-scoring
        # candidate at that level.
        #
        # enumerate() gives us both the zone_type (an integer identifying which
        # level of the geographic hierarchy we are at, e.g. district=0,
        # state=1, country=2) and the zone_index (the integer index of the
        # specific zone of that type that this citizen's patch belongs to).
        # Both values are needed to unambiguously identify a zone: zone_index
        # alone is not sufficient because two zones at different hierarchy
        # levels can share the same integer index.
        for zone_type, zone_index in enumerate(self.current_patch.zone_index):

            # Use None as a sentinel meaning "no candidate seen yet for this
            # zone". We cannot default to index 0 because politician 0 in the
            # global list may belong to a completely different zone and should
            # not be pre-selected as the initial best.
            top_pol_index = None

            # enumerate() here keeps pol_index in sync with self.politician_list
            # so that self.politician_score[pol_index] always refers to the
            # score of the same politician. Without enumerate, a separate
            # counter that only advances for non-skipped politicians would
            # become misaligned with the score array.
            for pol_index, politician in enumerate(self.politician_list):

                # Skip politicians that do not belong to the zone currently
                # being considered. A politician's zone is identified by two
                # attributes: zone_type (the hierarchy level the politician
                # runs in) and zone.zone_index (the integer index of the
                # specific zone at that level). Both must match; checking only
                # one would incorrectly include politicians from a different
                # hierarchy level that happen to share the same integer index.
                if (politician.zone_type != zone_type or
                        politician.zone.zone_index != zone_index):
                    continue

                # Accept the first candidate seen in this zone, then keep
                # replacing with any higher-scoring candidate found later.
                if (top_pol_index is None or
                        self.politician_score[pol_index] >
                        self.politician_score[top_pol_index]):
                    top_pol_index = pol_index

            # Cast a vote for the best-scoring candidate in this zone.
            # (top_pol_index should always be set here because every zone the
            # citizen belongs to must have at least one politician, but the
            # None guard prevents a crash in case of a misconfigured world.)
            if top_pol_index is not None:
                self.politician_list[top_pol_index].votes += 1



    # Compute the relationship between this citizen's stated policy positions
    #   and the actual policies as implemented by the government.
    def policy_attitude(self, government):
        self.attitude = 0
        for (stated, govern) in zip(self.stated_policy_pref.mu,
                government.policy_pos):
            self.attitude += abs(stated - govern)


    # Compute the relationship between this citizen's policy positions and
    #   the ideal (unknown to the citizen) policies that will benefit this
    #   citizen the most.
    def policy_alignment(self):
        self.alignment = 0 # Represents perfect alignment.
        for (stated, ideal) in zip(self.stated_policy_pref.mu,
                self.ideal_policy_pref.mu):
            self.alignment += abs(stated - ideal)

        return self.alignment
