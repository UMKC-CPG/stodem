import numpy as np

from gaussian import Gaussian, sample_theta
from random_state import rng


class Citizen():
    """A citizen agent in the democracy simulation.

    Citizens are the primary agents whose collective
    behavior produces emergent democratic outcomes.
    Each citizen maintains a set of Gaussians that
    encode their political positions and engagement:

    POLICY GAUSSIANS (one per policy dimension):
      stated_policy_pref (Pcp): The citizen's
        conscious policy preference. Used when
        comparing with politicians, other citizens,
        and the government. Positive-valued
        (cos(theta) > 0).
      stated_policy_aver (Pca): The citizen's
        conscious policy aversion. Represents what
        the citizen actively opposes. Negative-valued
        (cos(theta) < 0).
      ideal_policy_pref (Pci): The citizen's true
        best interest, which they do not directly
        know. Compared with government enacted
        policy to determine well-being. Never
        modified by influence — represents an
        objective ground truth.

    TRAIT GAUSSIANS (one per trait dimension):
      stated_trait_pref (Tcp): Personality
        preference. Positive-valued.
      stated_trait_aver (Tca): Personality aversion.
        Negative-valued.

    Key behavioral properties:
      - Citizen Gaussians are modified each campaign
        step by three influence sources: politician
        persuasion, well-being feedback, and citizen
        collective (community norms).
      - The "trait gates policy" principle governs
        how much policy positions shift: trait
        alignment with a source determines the
        magnitude and type of policy shift.
      - Engagement (theta) determines both the
        probability of voting and the citizen's
        resistance to position shifts
        (susceptibility).
      - Well-being (overlap of Pci with Pge) drives
        engagement: both very positive and very
        negative well-being increase engagement.
    """

    # --- Detailed conceptual notes for students ---
    #
    # Citizens have an innate "personality trait"
    #   preference and aversion that can change to
    #   align with a politician.
    #
    # Citizens have a stated policy position for
    #   each policy. This is the policy position
    #   that the citizen claims to align with and
    #   it will affect their preference for a
    #   particular politician. Similarly, each
    #   citizen has an aversion associated with
    #   each policy.
    #
    # Citizens have a most-beneficial policy
    #   position that the citizen does not directly
    #   know. I.e., the well-being of the citizen
    #   will depend on the alignment between
    #   governing policy and this most-beneficial
    #   policy, but the stated policy may be quite
    #   different than the most-beneficial policy.
    #   For example, if a policy represented "tax
    #   rates", it is hard to know what the best
    #   tax rate for an individual should be. If it
    #   was set to zero, the individual would pay
    #   nothing, but also likely have no services.
    #   If it was set to 100% they would have no
    #   money, but would have many services. The
    #   "correct" number is not easy for any
    #   individual to know and that individual's
    #   stated preference may easily be different
    #   from whatever number actually benefits them
    #   the most.
    #
    # Citizens have a probability of participation
    #   that is computed from their average
    #   engagement: P(vote) = mean(|cos(theta)|)
    #   across all stated Gaussians. This is
    #   recomputed each time the citizen votes.
    #
    # Citizens have a well-being factor that weights
    #   the degree to which they will use personality
    #   or policy alignment when deciding how to
    #   cast their vote.


    def __init__(self, settings, patch, zones):
        """Initialize a citizen with random Gaussian
        positions drawn from the XML configuration.

        Each citizen receives:
          - 3 policy Gaussians per policy dimension
            (stated pref, stated aver, ideal pref)
          - 2 trait Gaussians per trait dimension
            (stated pref, stated aver)
          - Scalar parameters: policy_trait_ratio,
            collective_influence_rate, sigma_floor,
            engagement_decay_rate, defensive_ratio
          - A patch assignment and zone membership

        Parameters
        ----------
        settings : ScriptSettings
            Provides XML configuration for Gaussian
            initialization parameters.
        patch : Patch
            The patch this citizen is placed on.
        zones : list of list of Zone
            The full zone hierarchy. The citizen
            determines which zones it belongs to
            based on its patch's zone_index.
        """
        # Get temporary local names for settings
        #   variables.
        self.num_policy_dims = int(
                settings.infile_dict[1][
                    "world"]["num_policy_dims"])
        self.num_trait_dims = int(
                settings.infile_dict[1][
                    "world"]["num_trait_dims"])

        # Define the initial instance variables of this citizen.

        # Theta (orientation) sign convention:
        #   preference Gaussians use Im(theta) in
        #   [0, pi/2), giving cos(theta) > 0
        #   (positive-valued). Aversion Gaussians use
        #   Im(theta) in (pi/2, pi], giving
        #   cos(theta) < 0 (negative-valued). This
        #   ensures same-type integrals (pref x pref,
        #   aver x aver) are non-negative and
        #   cross-type integrals (pref x aver) are
        #   non-positive — encoding attraction vs.
        #   repulsion without special-casing.
        #   The default means are 1 (preferences) and
        #   pi-1 (aversions), giving cos(1) ~ 0.54
        #   and cos(pi-1) ~ -0.54 respectively.
        #   If the *_orien_stddev XML parameter is
        #   numeric, Im(theta) is drawn from a normal
        #   distribution with the default as the mean
        #   and the parameter as the stddev, clamped
        #   to the appropriate half of [0, pi].
        #   Otherwise (e.g., "imaginary"), the default
        #   mean is used for all agents.
        half_pi = np.pi / 2.0
        cit = settings.infile_dict[1]["citizens"]

        self.stated_policy_pref = Gaussian(
                rng.normal(loc=0.0,
                    scale=float(
                        cit["policy_pref_pos_stddev"]),
                    size=self.num_policy_dims),
                np.abs(rng.normal(loc=0.0,
                    scale=float(
                        cit["policy_pref_stddev_stddev"]),
                    size=self.num_policy_dims)),
                sample_theta(
                    cit["policy_pref_orien_stddev"],
                    1.0, self.num_policy_dims,
                    0.0, half_pi),
                1)

        self.stated_policy_aver = Gaussian(
                rng.normal(loc=0.0,
                    scale=float(
                        cit["policy_aver_pos_stddev"]),
                    size=self.num_policy_dims),
                np.abs(rng.normal(loc=0.0,
                    scale=float(
                        cit["policy_aver_stddev_stddev"]),
                    size=self.num_policy_dims)),
                sample_theta(
                    cit["policy_aver_orien_stddev"],
                    np.pi - 1.0, self.num_policy_dims,
                    half_pi, np.pi),
                1)

        self.ideal_policy_pref = Gaussian(
                [x + rng.normal(loc=0.0,
                    scale=float(
                        cit["ideal_policy_pref_pos_stddev"]))
                    for x in self.stated_policy_pref.mu],
                np.abs(rng.normal(loc=0.0,
                    scale=[float(
                        cit["ideal_policy_pref_stddev_stddev"])
                        for x in range(
                            self.num_policy_dims)])),
                sample_theta(
                    cit["ideal_policy_pref_orien_stddev"],
                    0.0, self.num_policy_dims,
                    0.0, half_pi),
                1)

        self.stated_trait_pref = Gaussian(
                rng.normal(loc=0.0,
                    scale=float(
                        cit["trait_pref_pos_stddev"]),
                    size=self.num_trait_dims),
                np.abs(rng.normal(loc=0.0,
                    scale=float(
                        cit["trait_pref_stddev_stddev"]),
                    size=self.num_trait_dims)),
                sample_theta(
                    cit["trait_pref_orien_stddev"],
                    1.0, self.num_trait_dims,
                    0.0, half_pi),
                1)

        self.stated_trait_aver = Gaussian(
                rng.normal(loc=0.0,
                    scale=float(
                        cit["trait_aver_pos_stddev"]),
                    size=self.num_trait_dims),
                np.abs(rng.normal(loc=0.0,
                    scale=float(
                        cit["trait_aver_stddev_stddev"]),
                    size=self.num_trait_dims)),
                sample_theta(
                    cit["trait_aver_orien_stddev"],
                    np.pi - 1.0, self.num_trait_dims,
                    half_pi, np.pi),
                1)

        self.policy_consistency = self.policy_alignment()

        self.policy_trait_ratio = np.clip(rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["citizens"]
                ["policy_trait_ratio_stddev"])), -0.5, 0.5)

        self.collective_influence_rate = float(
                settings.infile_dict[1]["citizens"]
                ["collective_influence_rate"])

        # sigma_floor is the minimum allowed value
        #   for any Gaussian's sigma (standard
        #   deviation / spread). Without it, sigma
        #   could reach zero, making the Gaussian
        #   infinitely narrow and causing a
        #   division-by-zero in alpha = 1/(2*sigma^2).
        #   It also embodies a physical idea: no
        #   citizen becomes so certain about any
        #   position that their Gaussian collapses to
        #   a delta function. The floor is also the
        #   target sigma for defensive narrowing: a
        #   citizen under backlash (negative trait
        #   alignment with a politician) rigidifies
        #   their preference sigma toward sigma_floor,
        #   not toward zero. See the defensive branch
        #   in build_response_to_politician_influence().
        self.sigma_floor = float(
                settings.infile_dict[1]["citizens"]
                ["sigma_floor"])

        # engagement_decay_rate controls how quickly
        #   citizens drift back toward apathy each step
        #   in the absence of active stimulation. The
        #   formula is proportional: theta *= (1 + rate),
        #   where theta is the "engagement angle" (see
        #   apply_influence_shifts). Proportional decay
        #   means a perfectly engaged citizen (angle=0)
        #   stays engaged, while a slightly disengaged
        #   citizen accelerates toward full apathy. This
        #   creates a fundamentally unstable equilibrium
        #   at apathy: campaigns must continually re-engage
        #   citizens, not just engage them once. Stored as
        #   an instance variable so it can be made dynamic
        #   in the future (e.g., driven by well-being).
        self.engagement_decay_rate = float(
                settings.infile_dict[1]["citizens"]
                ["engagement_decay_rate"])

        # defensive_ratio scales the targeted backlash response
        #   when a citizen dislikes a politician (negative trait
        #   alignment; DESIGN.md §8.1). In the defensive branch of
        #   build_response_to_politician_influence(), the citizen's
        #   policy aversion mu shifts toward the politician's
        #   PREFERENCE positions — the citizen develops aversion to
        #   exactly what the disliked politician stands FOR, not
        #   toward the politician's own aversion targets. For
        #   example: a citizen who dislikes a pro-tax politician
        #   does not merely stay put; they develop a stronger
        #   aversion specifically in the direction of that
        #   politician's pro-tax stance.
        #   A value of 1.0 applies backlash at the same rate as
        #   same-sign attraction shifts. Values > 1 amplify the
        #   reaction; values < 1 dampen it. Stored as an instance
        #   variable for potential future dynamic modulation (e.g.,
        #   by well-being or accumulated resentment).
        self.defensive_ratio = float(
                settings.infile_dict[1]["citizens"]
                ["defensive_ratio"])

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
        """Compute all overlap integrals between
        this citizen and every relevant entity:
        politicians, zone-average citizens, and
        government.

        This is the "sensing" step of each campaign
        iteration. The overlap integrals quantify
        how aligned or opposed this citizen is to
        each politician, each community norm, and
        the government's enacted policy. These
        integrals are then consumed by the three
        build_response_to_*() methods to compute
        engagement and position shifts.

        The integrals are stored in lists indexed
        by politician or zone. For example,
        self.Pcp_Ppp_ol[i] is a numpy array of
        per-dimension overlaps between this
        citizen's policy preference and politician
        i's external policy preference.

        Prerequisite: zone averages must have been
        computed for the current step (done in
        campaign() before this method is called).
        """

        # Clear and re-initialize all integral
        #   storage lists.
        self.initialize_lists()

        # Citizen ↔ politician integrals (one entry
        #   per politician in self.politician_list).
        self.policy_politician_integrals()
        self.trait_politician_integrals()

        # Citizen ↔ zone-average-citizen integrals
        #   (one entry per zone in self.zone_list).
        self.policy_citizen_integrals()
        self.trait_citizen_integrals()

        # Citizen ↔ government integrals (one entry
        #   total, since there is one government).
        self.policy_government_integrals(world)


    def initialize_lists(self):
        """Initialize (or reset) all overlap
        integral storage lists to empty.

        Naming convention for overlap lists:
          {citizen_type}_{other_type}_ol

        Where the abbreviations are:
          Pcp = Policy citizen preference
          Pca = Policy citizen aversion
          Pci = Policy citizen ideal
          Ppp = Policy politician preference
          Ppa = Policy politician aversion
          Tcp = Trait citizen preference
          Tca = Trait citizen aversion
          Tpx = Trait politician external
          Pge = Policy government enacted

        Each list element is a numpy array of shape
        (num_dims,) — one overlap value per policy
        or trait dimension. The list index
        corresponds to the politician index (for
        citizen↔politician lists) or zone index
        (for citizen↔zone-average lists).
        """

        # --- Citizen ↔ politician overlaps ---
        # One entry per politician. Four policy
        #   combinations + two trait combinations
        #   = six lists.
        # Policy: citizen preference vs politician
        #   preference.
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
            self.Tcp_Tpx_ol.append(self.stated_trait_pref.integral(
                politician.ext_trait))
            self.Tca_Tpx_ol.append(self.stated_trait_aver.integral(
                politician.ext_trait))


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
        # Per DESIGN.md §8.6.1: twelve separate shift arrays, three
        #   per Gaussian type (Pcp, Pca, Tcp, Tca). Separate arrays
        #   per type are REQUIRED because preference and aversion
        #   Gaussians respond differently under negative trait
        #   alignment (§8.1, the "defensive branch"):
        #
        #   Positive trait overlap (attraction):
        #     Both Pcp and Pca shift toward the source positions.
        #
        #   Negative trait overlap (defensive rigidity):
        #     Pcp sigma narrows toward sigma_floor.
        #     Pca mu shifts toward source's PREFERENCE (backlash).
        #     Pcp mu and Pca sigma do NOT change in this branch.
        #
        #   If Pcp and Pca shared pos/stddev arrays, the two
        #   branches could not accumulate independent shifts into
        #   each type: a Pca-only backlash mu shift would
        #   contaminate Pcp, and a Pcp-only sigma narrowing would
        #   also shrink Pca.
        #
        #   All arrays are numpy so += accumulates element-wise.
        #   (A plain Python list would extend, not add.)
        #   Naming convention: the prefix names the Gaussian type
        #   whose parameter is being shifted.
        n = num_policy_dims
        m = num_trait_dims
        self.Pcp_orien_shift  = np.zeros(n)  # policy pref theta
        self.Pcp_pos_shift    = np.zeros(n)  # policy pref mu
        self.Pcp_stddev_shift = np.zeros(n)  # policy pref sigma
        self.Pca_orien_shift  = np.zeros(n)  # policy aver theta
        self.Pca_pos_shift    = np.zeros(n)  # policy aver mu
        self.Pca_stddev_shift = np.zeros(n)  # policy aver sigma
        self.Tcp_orien_shift  = np.zeros(m)  # trait pref theta
        self.Tcp_pos_shift    = np.zeros(m)  # trait pref mu
        self.Tcp_stddev_shift = np.zeros(m)  # trait pref sigma
        self.Tca_orien_shift  = np.zeros(m)  # trait aver theta
        self.Tca_pos_shift    = np.zeros(m)  # trait aver mu
        self.Tca_stddev_shift = np.zeros(m)  # trait aver sigma


    def build_response_to_politician_influence(self):
        # Accumulate engagement (theta), position (mu), and spread
        #   (sigma) shifts from each politician sharing this citizen's
        #   patch. Called once per campaign step after all overlap
        #   integrals have been computed.
        #
        # ---- "Trait gates policy" (DESIGN.md §8.1, §8.6.3) -------
        #
        # For each politician, compute a signed scalar:
        #
        #   trait_sum = sum_m( I(Tcp,Tpx)[m] + I(Tca,Tpx)[m] )
        #
        #   trait_sum > 0  →  citizen feels affinity (attraction)
        #   trait_sum < 0  →  citizen feels aversion (defensive)
        #   trait_sum = 0  →  no policy position/spread shift
        #                     (engagement shifts may still occur)
        #
        # The absolute value |trait_sum| = mag sets the magnitude of
        # the policy shift; the sign selects one of two branches:
        #
        #   Attraction branch (trait_sum >= 0):
        #     Pcp mu/sigma drift toward Ppp (politician's policy pref).
        #     Pca mu/sigma drift toward Ppa (politician's policy aver).
        #
        #   Defensive branch (trait_sum < 0):
        #     Pcp sigma narrows toward sigma_floor (rigidity: the
        #       citizen digs in and becomes certain of their stance).
        #     Pca mu shifts toward Ppp — the politician's PREFERENCE
        #       (targeted backlash: the citizen develops aversion
        #       specifically to what the disliked politician stands
        #       FOR, not to their aversion targets). Scaled by
        #       defensive_ratio.
        #     Pcp mu does NOT shift; Pca sigma does NOT shift.
        #
        # ---- Susceptibility (DESIGN.md §8.6.3) -------------------
        #
        #   S(sigma, theta) = sigma * (1 - |cos(theta)|)
        #
        # A citizen's resistance to having a position or spread
        # shifted depends on two properties of the Gaussian:
        #
        #   sigma (spread): narrow = strong attachment = hard to move.
        #     S increases with sigma (broader Gaussians move more).
        #
        #   theta (engagement): engaged (|cos|≈1) = the citizen has
        #     thought about this issue and resists change.
        #     Apathetic (|cos|≈0) = positions are held loosely.
        #     S increases as engagement falls.
        #
        #   Key property: at full engagement S=0, so campaigns first
        #   change ENGAGEMENT levels; position shifts only accumulate
        #   once citizens begin to disengage.
        #
        #   Note on |cos θ|: The DESIGN writes S = σ*(1-cos θ).
        #   That is exact for preference Gaussians (θ∈[0,π/2], cos≥0).
        #   For aversion Gaussians (θ∈[π/2,π], cos≤0) the literal
        #   formula gives S > σ at full engagement — counterintuitive.
        #   Using |cos θ| generalizes correctly to both types:
        #     preference at θ=0 (engaged): S = σ*(1-1) = 0 ✓
        #     preference at θ=π/2 (apathetic): S = σ*(1-0) = σ ✓
        #     aversion  at θ=π (engaged): S = σ*(1-|-1|) = 0 ✓
        #     aversion  at θ=π/2 (apathetic): S = σ*(1-0) = σ ✓
        #
        # ---- Engagement shifts (DESIGN.md §8.6.2) ----------------
        #
        # Each |overlap integral| drives the corresponding Gaussian's
        # theta toward real (more engaged). Pcp is driven by integrals
        # that involve Pcp; Pca by integrals involving Pca. Both
        # same-type (agreement) and cross-type (disagreement) integrals
        # contribute: you become engaged by opposition as well as
        # support. Scaled by the politician's persuasion factors
        # (f_pol for policy overlaps, f_trait for trait overlaps).
        # f_pol and f_trait are drawn from zero-mean Gaussians and can
        # be positive (amplifies) or negative (dampens).
        for pol_idx, politician in enumerate(self.politician_list):
            f_pol   = politician.policy_persuasion
            f_trait = politician.trait_persuasion

            # --- Engagement shifts ---
            # Pcp is driven by policy integrals involving Pcp.
            # Pca is driven by policy integrals involving Pca.
            self.Pcp_orien_shift += f_pol * (
                np.abs(self.Pcp_Ppp_ol[pol_idx])
                + np.abs(self.Pcp_Ppa_ol[pol_idx]))
            self.Pca_orien_shift += f_pol * (
                np.abs(self.Pca_Ppa_ol[pol_idx])
                + np.abs(self.Pca_Ppp_ol[pol_idx]))
            self.Tcp_orien_shift += (
                f_trait * np.abs(self.Tcp_Tpx_ol[pol_idx]))
            self.Tca_orien_shift += (
                f_trait * np.abs(self.Tca_Tpx_ol[pol_idx]))

            # --- Trait sum: the gate for policy shifts ---
            # Signed sum of all trait overlaps citizen↔politician.
            # Positive = affinity; negative = aversion.
            trait_sum = (
                np.sum(self.Tcp_Tpx_ol[pol_idx])
                + np.sum(self.Tca_Tpx_ol[pol_idx]))
            mag = abs(trait_sum)

            # --- Susceptibility per policy Gaussian ---
            # cos_theta is cached in the Gaussian; it is negative
            # for aversion types, so abs() is applied to get |cos θ|.
            S_Pcp = (
                self.stated_policy_pref.sigma
                * (1.0 - np.abs(
                    self.stated_policy_pref.cos_theta)))
            S_Pca = (
                self.stated_policy_aver.sigma
                * (1.0 - np.abs(
                    self.stated_policy_aver.cos_theta)))

            if trait_sum >= 0:
                # Attraction branch: Pcp → Ppp, Pca → Ppa.
                # Direction is sign(source_mu - citizen_mu):
                # a one-unit impulse per step, gated by
                # susceptibility and trait magnitude.
                Ppp_mu    = politician.ext_policy_pref.mu
                Ppp_sigma = politician.ext_policy_pref.sigma
                Ppa_mu    = politician.ext_policy_aver.mu
                Ppa_sigma = politician.ext_policy_aver.sigma
                self.Pcp_pos_shift += (mag * f_pol * S_Pcp
                    * np.sign(Ppp_mu - self.stated_policy_pref.mu))
                self.Pcp_stddev_shift += (mag * f_pol * S_Pcp
                    * np.sign(Ppp_sigma - self.stated_policy_pref.sigma))
                self.Pca_pos_shift += (mag * f_pol * S_Pca
                    * np.sign(Ppa_mu - self.stated_policy_aver.mu))
                self.Pca_stddev_shift += (mag * f_pol * S_Pca
                    * np.sign(Ppa_sigma - self.stated_policy_aver.sigma))
            else:
                # Defensive branch: rigidity + targeted backlash.
                # Pcp sigma narrows toward sigma_floor.
                # Pca mu shifts toward politician's PREFERENCE
                #   (not aversion), scaled by defensive_ratio.
                Ppp_mu = politician.ext_policy_pref.mu
                self.Pcp_stddev_shift += (mag * f_pol * S_Pcp
                    * np.sign(self.sigma_floor - self.stated_policy_pref.sigma))
                self.Pca_pos_shift += (mag * f_pol * self.defensive_ratio
                    * S_Pca * np.sign(Ppp_mu - self.stated_policy_aver.mu))


    def build_response_to_citizen_collective(self):
        # Accumulate engagement, policy, and trait shifts from the
        #   zone-average Gaussians of other citizens (DESIGN.md §8.4,
        #   §8.6.2–§8.6.4). Called once per campaign step after all
        #   overlap integrals have been computed.
        #
        # Three key differences from politician influence:
        #
        # 1. UNCONDITIONAL — no sign-gating, no defensive branch.
        #    Policy and trait Gaussians always drift toward zone
        #    averages regardless of trait alignment. The community
        #    exerts a steady background pull: people absorb the norms
        #    around them even when their personality opposes those
        #    norms; they just absorb them more slowly (trait_rate is
        #    smaller when same-type trait alignment is weaker).
        #
        # 2. TRAIT RATE (not trait sum).
        #    The shift rate uses only SAME-TYPE trait overlaps
        #    (pref×pref, aver×aver); cross-terms are excluded.
        #    Same-type overlaps are non-negative by the theta sign
        #    convention (both factors carry the same sign), so
        #    trait_rate >= 0 always — there is no sign to gate on:
        #
        #      trait_rate = sum_m( I(Tcp, avg_Tcp)[m]
        #                        + I(Tca, avg_Tca)[m] )
        #
        # 3. TRAIT SHIFTS INCLUDED (§8.6.4).
        #    The same trait_rate drives trait acclimatization: Tcp
        #    and Tca drift toward the zone-average trait Gaussians.
        #    This is the ONLY mechanism by which citizen traits
        #    change; politicians never directly alter citizen traits.
        #
        # Susceptibility: S = sigma * (1 - |cos_theta|), same as for
        #   politician influence. Computed once before the zone loop
        #   because citizen Gaussians are read-only during
        #   accumulation — only the shift arrays change.
        cir = self.collective_influence_rate
        S_Pcp = (
            self.stated_policy_pref.sigma
            * (1.0 - np.abs(
                self.stated_policy_pref.cos_theta)))
        S_Pca = (
            self.stated_policy_aver.sigma
            * (1.0 - np.abs(
                self.stated_policy_aver.cos_theta)))
        S_Tcp = (
            self.stated_trait_pref.sigma
            * (1.0 - np.abs(
                self.stated_trait_pref.cos_theta)))
        S_Tca = (
            self.stated_trait_aver.sigma
            * (1.0 - np.abs(
                self.stated_trait_aver.cos_theta)))

        for zone_idx, zone in enumerate(self.zone_list):

            # --- Engagement shifts (§8.6.2) ---
            # |overlap| drives theta toward real. Cross-terms
            #   (pref×aver) are included: both agreement and
            #   disagreement with the community increase engagement
            #   on community-relevant issues.
            self.Pcp_orien_shift += cir * (
                np.abs(self.Pcp_Pcp_ol[zone_idx])
                + np.abs(self.Pcp_Pca_ol[zone_idx]))
            self.Pca_orien_shift += cir * (
                np.abs(self.Pca_Pca_ol[zone_idx])
                + np.abs(self.Pca_Pcp_ol[zone_idx]))
            self.Tcp_orien_shift += cir * (
                np.abs(self.Tcp_Tcp_ol[zone_idx])
                + np.abs(self.Tcp_Tca_ol[zone_idx]))
            self.Tca_orien_shift += cir * (
                np.abs(self.Tca_Tca_ol[zone_idx])
                + np.abs(self.Tca_Tcp_ol[zone_idx]))

            # --- Trait rate: the community shift gate ---
            # Same-type overlaps only. Non-negative by sign
            #   convention. Larger when citizen traits resemble
            #   the community norm; smaller when they differ.
            trait_rate = (
                np.sum(self.Tcp_Tcp_ol[zone_idx])
                + np.sum(self.Tca_Tca_ol[zone_idx]))

            # --- Policy position and spread shifts (§8.6.3) ---
            # Unconditional: Pcp drifts toward avg_Pcp,
            #   Pca toward avg_Pca. No defensive branch.
            # Direction = sign(zone_avg_mu - citizen_mu):
            #   one-unit impulse per step, gated by susceptibility
            #   and trait_rate.
            self.Pcp_pos_shift += (cir * trait_rate * S_Pcp
                * np.sign(zone.avg_Pcp.mu - self.stated_policy_pref.mu))
            self.Pcp_stddev_shift += (cir * trait_rate * S_Pcp
                * np.sign(zone.avg_Pcp.sigma - self.stated_policy_pref.sigma))
            self.Pca_pos_shift += (cir * trait_rate * S_Pca
                * np.sign(zone.avg_Pca.mu - self.stated_policy_aver.mu))
            self.Pca_stddev_shift += (cir * trait_rate * S_Pca
                * np.sign(zone.avg_Pca.sigma - self.stated_policy_aver.sigma))

            # --- Trait position and spread shifts (§8.6.4) ---
            # Same trait_rate drives trait acclimatization. Tcp drifts
            #   toward avg_Tcp; Tca toward avg_Tca. This is the sole
            #   mechanism for trait change.
            self.Tcp_pos_shift += (cir * trait_rate * S_Tcp
                * np.sign(zone.avg_Tcp.mu - self.stated_trait_pref.mu))
            self.Tcp_stddev_shift += (cir * trait_rate * S_Tcp
                * np.sign(zone.avg_Tcp.sigma - self.stated_trait_pref.sigma))
            self.Tca_pos_shift += (cir * trait_rate * S_Tca
                * np.sign(zone.avg_Tca.mu - self.stated_trait_aver.mu))
            self.Tca_stddev_shift += (cir * trait_rate * S_Tca
                * np.sign(zone.avg_Tca.sigma - self.stated_trait_aver.sigma))


    def apply_influence_shifts(self):
        # This method is the second half of a deliberate
        #   two-phase accumulate-then-apply design (see
        #   DESIGN.md §8.6). During the accumulation phase,
        #   build_response_to_politician_influence(),
        #   build_response_to_well_being(), and
        #   build_response_to_citizen_collective() each add
        #   their contributions to twelve shared shift
        #   arrays — three per Gaussian type (Pcp, Pca,
        #   Tcp, Tca), one each for orientation (theta),
        #   position (mu), and spread (sigma). Because
        #   those methods write to arrays rather than
        #   mutating Gaussian parameters directly, the
        #   order in which the three sources are processed
        #   is irrelevant — they all observe the same
        #   unchanged Gaussian state from the start of the
        #   step. This method is the apply phase: it
        #   flushes all accumulated shifts into the actual
        #   Gaussian parameters in one pass, then enforces
        #   constraints and decays engagement.
        #
        # Three independent Gaussian quantities are updated:
        #
        #   mu (position): shifts the center of the
        #     Gaussian along the policy/trait axis — where
        #     the citizen stands on an issue. Accumulated by:
        #     - build_response_to_politician_influence():
        #       attraction branch moves Pcp toward Ppp and
        #       Pca toward Ppa; defensive branch moves Pca
        #       toward Ppp (targeted backlash).
        #     - build_response_to_citizen_collective():
        #       unconditionally moves Pcp, Pca, Tcp, Tca
        #       toward their respective zone averages.
        #     mu is unbounded on the real line (no clamping).
        #
        #   sigma (spread): shifts the Gaussian's width —
        #     how firmly the citizen is attached to that
        #     specific position. A narrow sigma encodes
        #     strong, rigid attachment; a wide sigma means
        #     many nearby positions are acceptable. Clamped
        #     to sigma_floor after application to prevent
        #     degenerate near-zero widths that would cause
        #     division-by-zero in alpha = 1/(2*sigma^2).
        #
        #   theta (engagement): shifts Im(theta) toward
        #     the "engaged" pole of the Gaussian, then
        #     decays it proportionally back toward apathy.
        #     The theta mechanics are described in detail
        #     below.

        # -------------------------------------------------------
        # Theta (engagement) mechanics
        # -------------------------------------------------------
        #
        # The citizen's engagement with each issue is
        #   encoded in the imaginary part of theta,
        #   Im(theta). The overlap integral formula uses
        #   cos(Im(theta)), so:
        #     cos = 1 (or -1)  -> fully engaged
        #     cos = 0          -> fully apathetic; the
        #                         Gaussian contributes
        #                         nothing to any integral
        #
        # Preference and aversion Gaussians are assigned
        #   opposite halves of [0, pi] so that their
        #   overlap integrals carry the right sign
        #   automatically — no special-casing required:
        #
        #   Preference: Im(theta) in [0, pi/2)
        #     cos(Im(theta)) in (0, 1]  (positive)
        #     "more engaged" means Im(theta) -> 0
        #     "more apathetic" means Im(theta) -> pi/2
        #
        #   Aversion: Im(theta) in (pi/2, pi]
        #     cos(Im(theta)) in [-1, 0) (negative)
        #     "more engaged" means Im(theta) -> pi
        #     "more apathetic" means Im(theta) -> pi/2
        #
        # Consequence: the engagement shift must push
        #   Im(theta) in opposite directions for the two
        #   types. Subtracting a positive shift from a
        #   preference moves it toward 0 (more engaged);
        #   adding a positive shift to an aversion moves
        #   it toward pi (more engaged). Both converge
        #   on apathy at pi/2 from their respective sides.
        #
        # Each Gaussian type has its own independent shift
        #   arrays (Pcp, Pca, Tcp, Tca), so each receives
        #   only the contributions intended for it. The
        #   defensive branch accumulates into Pcp and Pca
        #   independently; no cross-contamination occurs.
        #
        # Engagement decay (DESIGN.md §8.6.6):
        #   After the influence-driven shift, each theta
        #   decays proportionally back toward apathy via
        #   alpha *= (1 + engagement_decay_rate), where
        #   "alpha" is the "engagement angle" — always in
        #   [0, pi/2] regardless of Gaussian type:
        #     preference: alpha = Im(theta) directly
        #     aversion:   alpha = pi - Im(theta) (mirrored,
        #                 so alpha=0 means Im(theta)=pi,
        #                 which is maximum aversion
        #                 engagement)
        #   Multiplying alpha by (1 + rate) > 1 makes it
        #   grow toward pi/2 (apathy) each step. Key
        #   properties:
        #     - alpha=0 (full engagement): 0*(1+rate)=0,
        #       so perfectly engaged citizens do not decay.
        #     - Larger alpha decays faster: disengagement
        #       is self-reinforcing and accelerates.
        #   After decay, aversion Im(theta) is recovered
        #   as pi - alpha_decayed.

        half_pi = np.pi / 2.0
        edr = self.engagement_decay_rate

        # --- Policy preference (Pcp) ---
        # Apply Pcp-specific shifts (from the attraction branch
        #   and from citizen-collective unconditional drift).
        self.stated_policy_pref.mu += self.Pcp_pos_shift
        self.stated_policy_pref.sigma = np.maximum(
                self.stated_policy_pref.sigma
                + self.Pcp_stddev_shift,
                self.sigma_floor)
        # Subtract the engagement shift: drives Im(theta)
        #   toward 0 (maximum preference engagement).
        #   Clip to [0, pi/2] enforces the preference
        #   convention after the shift.
        pref_im = np.clip(
                self.stated_policy_pref.theta.imag
                - self.Pcp_orien_shift,
                0.0, half_pi)
        # Apply proportional decay: alpha grows toward
        #   pi/2. Re-clip to guard against floating-point
        #   drift past pi/2.
        pref_im = np.clip(
                pref_im * (1.0 + edr), 0.0, half_pi)
        self.stated_policy_pref.theta = pref_im * 1j

        # --- Policy aversion (Pca) ---
        # Apply Pca-specific shifts (from the attraction OR
        #   defensive branch, and from collective drift).
        self.stated_policy_aver.mu += self.Pca_pos_shift
        self.stated_policy_aver.sigma = np.maximum(
                self.stated_policy_aver.sigma
                + self.Pca_stddev_shift,
                self.sigma_floor)
        # Add the engagement shift: drives Im(theta)
        #   toward pi (maximum aversion engagement).
        #   Clip to [pi/2, pi] enforces the aversion
        #   convention after the shift.
        aver_im = np.clip(
                self.stated_policy_aver.theta.imag
                + self.Pca_orien_shift,
                half_pi, np.pi)
        # Compute the engagement angle alpha = pi - Im(theta)
        #   (so alpha=0 is full engagement, alpha=pi/2 is
        #   full apathy), decay it, then mirror back to
        #   Im(theta) = pi - alpha_decayed.
        alpha = np.clip(
                (np.pi - aver_im) * (1.0 + edr),
                0.0, half_pi)
        self.stated_policy_aver.theta = (np.pi - alpha) * 1j

        # --- Trait preference (Tcp) ---
        # Identical pattern to policy preference, applied to
        #   trait dimensions. Tcp shifts come only from
        #   citizen-collective influence (§8.6.4); politicians
        #   do not directly alter citizen traits.
        self.stated_trait_pref.mu += self.Tcp_pos_shift
        self.stated_trait_pref.sigma = np.maximum(
                self.stated_trait_pref.sigma
                + self.Tcp_stddev_shift,
                self.sigma_floor)
        pref_im = np.clip(
                self.stated_trait_pref.theta.imag
                - self.Tcp_orien_shift,
                0.0, half_pi)
        pref_im = np.clip(
                pref_im * (1.0 + edr), 0.0, half_pi)
        self.stated_trait_pref.theta = pref_im * 1j

        # --- Trait aversion (Tca) ---
        # Identical pattern to policy aversion, applied to
        #   trait dimensions.
        self.stated_trait_aver.mu += self.Tca_pos_shift
        self.stated_trait_aver.sigma = np.maximum(
                self.stated_trait_aver.sigma
                + self.Tca_stddev_shift,
                self.sigma_floor)
        aver_im = np.clip(
                self.stated_trait_aver.theta.imag
                + self.Tca_orien_shift,
                half_pi, np.pi)
        alpha = np.clip(
                (np.pi - aver_im) * (1.0 + edr),
                0.0, half_pi)
        self.stated_trait_aver.theta = (np.pi - alpha) * 1j

        # §8.6.7: Recompute derived variables (alpha,
        #   cos_theta, self_norm) cached inside each
        #   Gaussian. These are used by integral() in the
        #   next campaign step. Because we changed sigma
        #   and theta above, the cached values are now
        #   stale and must be refreshed before any overlap
        #   integral is computed. The ideal_policy_pref
        #   Gaussian is intentionally excluded: it
        #   represents the citizen's true (hidden) best
        #   interest and is never subject to influence or
        #   decay (DESIGN.md §8.6).
        self.stated_policy_pref.update_integration_variables()
        self.stated_policy_aver.update_integration_variables()
        self.stated_trait_pref.update_integration_variables()
        self.stated_trait_aver.update_integration_variables()


    def recompute_well_being(self, world):
        """Recompute well-being from the current
        Pge without accumulating engagement shifts.

        Used during the govern phase to update the
        well-being scalar for output after each
        govern step changes Pge. Unlike
        build_response_to_well_being(), this method
        does NOT call _well_being_to_engagement()
        because the govern phase has no
        accumulate-then-apply cycle.

        Pge.update_integration_variables() must be
        called before this method so that the
        integral uses the updated Pge parameters.
        """
        ol = self.ideal_policy_pref.integral(
            world.government.enacted_policy)
        self.well_being = sum(ol)


    def build_response_to_well_being(self):
        """Compute well-being and accumulate the
        resulting engagement shifts.

        Well-being is computed from the overlap
        between the citizen's IDEAL policy positions
        (Pci — the objectively best policy for this
        citizen, which they do not consciously know)
        and the government's enacted policy (Pge).
        A positive value means the government's
        policy is objectively benefiting this
        citizen; a negative value means it is
        harming them.

        Note the key distinction: well-being is
        based on IDEAL positions (Pci), not STATED
        positions (Pcp). A citizen may feel
        dissatisfied (stated pref far from Pge) yet
        have high well-being (ideal pref close to
        Pge), or vice versa. This captures the
        reality that people don't always know what
        policies actually benefit them.

        The well-being scalar is then mapped to
        engagement shifts via
        _well_being_to_engagement(). This method is
        encapsulated separately so that the richer
        well-being model (DESIGN.md §8.5: resource,
        perceived satisfaction, resentment) can
        replace the mapping without restructuring
        this call site.

        This method writes only to the engagement
        shift arrays (orien_shift), not to the
        citizen's Gaussians directly. The actual
        application happens in apply_influence_shifts().
        """
        self.well_being = sum(self.Pci_Pge_ol[0])
        self._well_being_to_engagement()


    def _well_being_to_engagement(self):
        """Map the well-being scalar to engagement
        shifts for all four Gaussian types.

        The core idea: |well_being| drives all
        Gaussians toward their engaged poles.
        Citizens doing very well (high positive
        well-being) are engaged because the system
        is working for them and they want to
        preserve it. Citizens doing very poorly
        (large negative) are engaged because they
        are motivated to change things. Only
        near-zero well-being (policy is neutral
        to the citizen) provides no engagement
        stimulus.

        This follows the same "absolute value
        drives engagement" principle used in
        politician influence (where |overlap|
        shifts theta toward real) and citizen
        collective influence. Engagement is about
        caring, not about approval.

        The shift magnitude |well_being| is applied
        uniformly across all policy and trait
        dimensions. Well-being is a whole-citizen
        state (derived from all policy dimensions
        summed together), so it affects engagement
        on ALL issues equally, not dimension by
        dimension.

        The actual theta modification happens later
        in apply_influence_shifts(), where these
        accumulated orien_shift values are
        subtracted from preference Im(theta) (driving
        it toward 0) or added to aversion Im(theta)
        (driving it toward pi).
        """
        mag = abs(self.well_being)
        n = self.num_policy_dims
        m = self.num_trait_dims

        self.Pcp_orien_shift += np.full(n, mag)
        self.Pca_orien_shift += np.full(n, mag)
        self.Tcp_orien_shift += np.full(m, mag)
        self.Tca_orien_shift += np.full(m, mag)


    def score_candidates(self, world):
        """Score each politician this citizen could
        vote for, producing a single scalar per
        politician that determines vote choice.

        The score is a weighted combination of
        policy alignment and trait alignment:

          score = w_policy * policy_sum
                  + w_trait * trait_sum

        where:
          w_policy = 0.5 + policy_trait_ratio
          w_trait  = 0.5 - policy_trait_ratio

        Since policy_trait_ratio is clamped to
        [-0.5, +0.5] at initialization, both
        weights are non-negative and sum to 1.
        A citizen with ratio = 0 weights policy
        and trait equally; ratio > 0 favors policy;
        ratio < 0 favors trait (personality).

        policy_sum includes ALL four citizen-
        politician policy overlaps per politician:
          I(Pcp, Ppp) + I(Pca, Ppa)  (same-type:
            agreement, positive)
          + I(Pcp, Ppa) + I(Pca, Ppp)  (cross-type:
            disagreement, negative)
        A politician aligned with the citizen's
        preferences and aversions gets a high
        positive score; one who opposes them gets
        a negative score.

        trait_sum includes both trait overlaps:
          I(Tcp, Tpx) + I(Tca, Tpx)
        Same logic: trait affinity adds positively,
        trait aversion adds negatively.

        The politician_score list is parallel to
        politician_list: politician_score[i] is the
        score for politician_list[i].
        """
        w_policy = 0.5 + self.policy_trait_ratio
        w_trait = 0.5 - self.policy_trait_ratio

        pol_index = 0
        self.politician_score = []
        for politician in self.politician_list:
            policy_sum = (
                sum(self.Pcp_Ppp_ol[pol_index])
                + sum(self.Pca_Ppa_ol[pol_index])
                + sum(self.Pcp_Ppa_ol[pol_index])
                + sum(self.Pca_Ppp_ol[pol_index]))
            trait_sum = (
                sum(self.Tcp_Tpx_ol[pol_index])
                + sum(self.Tca_Tpx_ol[pol_index]))
            self.politician_score.append(
                w_policy * policy_sum
                + w_trait * trait_sum)
            pol_index += 1


    def compute_vote_probability(self):
        # Compute the probability that this citizen will vote based on their
        #   average engagement across all stated Gaussians. Each Gaussian's
        #   cos(theta) measures engagement: 1 = fully engaged, 0 = fully
        #   apathetic. The mean across all stated policy and trait Gaussians
        #   gives a natural vote probability.
        #
        # Note: In the future, a discriminability term could be included.
        #   The idea is that if a citizen's top candidate score is barely
        #   above the second-best, the citizen has weak preference among
        #   candidates and may be less motivated to vote. The magnitude of
        #   the score gap between the top two candidates could multiply the
        #   engagement-based probability.
        all_cos_theta = np.concatenate([
                self.stated_policy_pref.cos_theta,
                self.stated_policy_aver.cos_theta,
                self.stated_trait_pref.cos_theta,
                self.stated_trait_aver.cos_theta])
        self.participation_prob = np.mean(np.abs(all_cos_theta))


    def vote_for_candidates(self, world):
        # The assumption is that a citizen who decides to vote, will vote for
        #   every one of their top candidates. If a citizen decides to not
        #   vote, then they vote for none of their candidates. (Clearly, this
        #   could be modified so that citizens make a decision to "vote-at-all"
        #   followed by separate decisions about making a vote for each zone.
        #   This approach is a bit more complicated and so it is not done yet.

        # Compute the vote probability from the citizen's current engagement.
        self.compute_vote_probability()

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



    # Compute the relationship between this citizen's policy positions and
    #   the ideal (unknown to the citizen) policies that will benefit this
    #   citizen the most.
    def policy_alignment(self):
        self.alignment = 0 # Represents perfect alignment.
        for (stated, ideal) in zip(self.stated_policy_pref.mu,
                self.ideal_policy_pref.mu):
            self.alignment += abs(stated - ideal)

        return self.alignment
