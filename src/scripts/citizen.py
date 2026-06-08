import numpy as np

from gaussian import Gaussian, sample_theta
from random_state import rng


class DirectionalShift():
    """Accumulates no-overshoot shifts for one Gaussian
    parameter (a per-dimension mu vector or sigma vector)
    coming from ONE influence source (e.g. all politicians,
    or all neighbor zones). Implements DESIGN.md §8.6.3.

    Two ideas combine here:

    1. Per-contribution cap. Each contributor (one
       politician, one zone) pulls the parameter toward a
       target at a speed set by trait alignment and
       susceptibility — NOT by the distance to the target.
       Distance only caps the step so a single contributor
       can never carry the citizen past its own target:

           gap          = target - current
           contribution = sign(gap) * min(speed, |gap|)

       The speed may be large (the approach can be rapid);
       the cap merely forbids crossing the target.

    2. Span clamp. Several contributors on the same side of
       the citizen can still sum past all of them. To stop
       that, we remember the lowest and highest target any
       contributor pulled toward, and at application time we
       clamp the moved value into that span — widened to
       include the starting value so a citizen already
       outside the span is never dragged inward.

    The net result: the citizen never ends up more extreme,
    in either direction, than the most extreme target it was
    actually pulled toward.

    Attributes
    ----------
    shift : np.ndarray
        Running sum of capped contributions, one per
        dimension.
    target_lo, target_hi : np.ndarray
        Lowest / highest target seen so far, per dimension.
        Initialized to +inf / -inf so the first contribution
        sets them; if no contribution arrives, the span
        defaults to the current value at apply() time and the
        clamp is inert.
    """

    def __init__(self, num_dims):
        self.shift = np.zeros(num_dims)
        self.target_lo = np.full(num_dims, np.inf)
        self.target_hi = np.full(num_dims, -np.inf)

    def add(self, current, target, speed):
        """Accumulate one contributor's capped pull of
        `current` toward `target` at the given `speed`.

        Parameters
        ----------
        current : np.ndarray
            The parameter's present value (read-only here),
            per dimension.
        target : np.ndarray or float
            Where this contributor pulls the parameter.
        speed : np.ndarray or float
            Non-negative distance-free pull magnitude
            (trait alignment x persuasion x susceptibility).
        """
        gap = target - current
        self.shift = self.shift + (
            np.sign(gap) * np.minimum(speed, np.abs(gap)))
        self.target_lo = np.minimum(self.target_lo, target)
        self.target_hi = np.maximum(self.target_hi, target)

    def apply(self, current):
        """Return `current` moved by the accumulated shift,
        clamped to the target span (widened by `current` so
        an already-outside citizen is not dragged inward).
        """
        moved = current + self.shift
        low = np.minimum(self.target_lo, current)
        high = np.maximum(self.target_hi, current)
        return np.clip(moved, low, high)


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
        positions drawn from the TOML configuration.

        Each citizen receives:
          - 3 policy Gaussians per policy dimension
            (stated pref, stated aver, ideal pref)
          - 2 trait Gaussians per trait dimension
            (stated pref, stated aver)
          - Scalar parameters: policy_trait_ratio,
            collective_influence_rate, sigma_floor,
            engagement_decay_rate, defensive_ratio,
            threat_weight, govt_engagement_rate, sat_ref
          - A patch assignment and zone membership

        Parameters
        ----------
        settings : ScriptSettings
            Provides TOML configuration for Gaussian
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
        world_config = settings.infile_dict["world"]
        self.num_policy_dims = world_config["num_policy_dims"]
        self.num_trait_dims = world_config["num_trait_dims"]

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
        #   If the *_orien_stddev TOML parameter is
        #   numeric, Im(theta) is drawn from a normal
        #   distribution with the default as the mean
        #   and the parameter as the stddev, clamped
        #   to the appropriate half of [0, pi].
        #   Otherwise (e.g., "imaginary"), the default
        #   mean is used for all agents.
        half_pi = np.pi / 2.0
        cit = settings.infile_dict["citizens"]

        self.stated_policy_pref = Gaussian(
                rng.normal(loc=0.0,
                    scale=cit["policy_pref_pos_stddev"],
                    size=self.num_policy_dims),
                np.abs(rng.normal(loc=0.0,
                    scale=cit["policy_pref_stddev_stddev"],
                    size=self.num_policy_dims)),
                sample_theta(
                    cit["policy_pref_orien_stddev"],
                    1.0, self.num_policy_dims,
                    0.0, half_pi),
                1)

        self.stated_policy_aver = Gaussian(
                rng.normal(loc=0.0,
                    scale=cit["policy_aver_pos_stddev"],
                    size=self.num_policy_dims),
                np.abs(rng.normal(loc=0.0,
                    scale=cit["policy_aver_stddev_stddev"],
                    size=self.num_policy_dims)),
                sample_theta(
                    cit["policy_aver_orien_stddev"],
                    np.pi - 1.0, self.num_policy_dims,
                    half_pi, np.pi),
                1)

        self.ideal_policy_pref = Gaussian(
                [x + rng.normal(loc=0.0,
                    scale=cit["ideal_policy_pref_pos_stddev"])
                    for x in self.stated_policy_pref.mu],
                np.abs(rng.normal(loc=0.0,
                    scale=[cit["ideal_policy_pref_stddev_stddev"]
                        for x in range(
                            self.num_policy_dims)])),
                sample_theta(
                    cit["ideal_policy_pref_orien_stddev"],
                    0.0, self.num_policy_dims,
                    0.0, half_pi),
                1)

        self.stated_trait_pref = Gaussian(
                rng.normal(loc=0.0,
                    scale=cit["trait_pref_pos_stddev"],
                    size=self.num_trait_dims),
                np.abs(rng.normal(loc=0.0,
                    scale=cit["trait_pref_stddev_stddev"],
                    size=self.num_trait_dims)),
                sample_theta(
                    cit["trait_pref_orien_stddev"],
                    1.0, self.num_trait_dims,
                    0.0, half_pi),
                1)

        self.stated_trait_aver = Gaussian(
                rng.normal(loc=0.0,
                    scale=cit["trait_aver_pos_stddev"],
                    size=self.num_trait_dims),
                np.abs(rng.normal(loc=0.0,
                    scale=cit["trait_aver_stddev_stddev"],
                    size=self.num_trait_dims)),
                sample_theta(
                    cit["trait_aver_orien_stddev"],
                    np.pi - 1.0, self.num_trait_dims,
                    half_pi, np.pi),
                1)

        self.policy_consistency = self.policy_alignment()

        self.policy_trait_ratio = np.clip(rng.normal(loc=0.0,
                scale=cit["policy_trait_ratio_stddev"]),
                -0.5, 0.5)

        self.collective_influence_rate = (
                cit["collective_influence_rate"])

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
        self.sigma_floor = cit["sigma_floor"]

        # engagement_decay_rate controls the steady fade
        #   back toward apathy each step (DESIGN §8.6.6).
        #   The fade is proportional to the Gaussian's own
        #   spread: fade = engagement_decay_rate * sigma,
        #   measured in radians of theta moved toward pi/2
        #   per step. Because sigma is floored at
        #   sigma_floor, every citizen fades a little every
        #   step, so no one freezes at full engagement. A
        #   sharp (narrow-sigma) view fades slowly and holds
        #   its engagement; a broad, unsettled view fades
        #   fast and lapses to apathy. The fade depends on
        #   spread, NOT on the current engagement level, so
        #   it makes no claim that the most engaged fade
        #   fastest. (This replaced the older proportional
        #   theta *= (1 + rate) rule, which could not act at
        #   full engagement and trapped citizens there.)
        #   Stored as an instance variable so it can be made
        #   dynamic in the future (e.g., driven by crisis).
        self.engagement_decay_rate = (
                cit["engagement_decay_rate"])

        # threat_weight (w) is the multiplier applied to
        #   every engagement contribution that involves an
        #   aversion Gaussian — a direct threat or a shared
        #   opposition (DESIGN §8.6.2). Pure preference-meets-
        #   preference agreement counts at weight 1; every
        #   aversion-touching term counts at w. A shared
        #   enemy or a direct threat mobilizes harder than
        #   shared enthusiasm, so w is typically about 2.
        self.threat_weight = cit["threat_weight"]

        # govt_engagement_rate scales the government-driven
        #   engagement response (DESIGN §8.6.2): anger when a
        #   stated aversion is enacted (engagement up) and
        #   resignation when a stated preference goes unmet
        #   (engagement down). Applied every step in both the
        #   campaign and govern phases. Tunable; see §8.6.8.
        self.govt_engagement_rate = (
                cit["govt_engagement_rate"])

        # sat_ref is the reference "fully satisfied" overlap
        #   level for resignation (DESIGN §8.6.2): resignation
        #   grows as the conscious satisfaction overlap
        #   I(Pcp, Pge) falls below sat_ref. Set near the
        #   matched-policy self-overlap. Tunable; see §8.6.8.
        self.sat_ref = cit["sat_ref"]

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
        self.defensive_ratio = cit["defensive_ratio"]

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
        #
        #   Engagement (theta) shifts stay as plain summed arrays:
        #   they are not clamped, and both politician and community
        #   contributions accumulate into one array per Gaussian.
        n = num_policy_dims
        m = num_trait_dims
        self.Pcp_orien_shift = np.zeros(n)  # policy pref theta
        self.Pca_orien_shift = np.zeros(n)  # policy aver theta
        self.Tcp_orien_shift = np.zeros(m)  # trait pref theta
        self.Tca_orien_shift = np.zeros(m)  # trait aver theta

        #   Position (mu) and spread (sigma) shifts use the
        #   no-overshoot DirectionalShift accumulator (§8.6.3),
        #   and they split by influence SOURCE so each source can
        #   be clamped to its own target span before the two are
        #   summed (§8.6.5). Politicians never alter citizen
        #   traits (§8.6.4), so Tcp/Tca have a community source
        #   only.
        self.Pcp_mu_pol  = DirectionalShift(n)  # politician source
        self.Pcp_sig_pol = DirectionalShift(n)
        self.Pca_mu_pol  = DirectionalShift(n)
        self.Pca_sig_pol = DirectionalShift(n)

        self.Pcp_mu_com  = DirectionalShift(n)  # community source
        self.Pcp_sig_com = DirectionalShift(n)
        self.Pca_mu_com  = DirectionalShift(n)
        self.Pca_sig_com = DirectionalShift(n)
        self.Tcp_mu_com  = DirectionalShift(m)
        self.Tcp_sig_com = DirectionalShift(m)
        self.Tca_mu_com  = DirectionalShift(m)
        self.Tca_sig_com = DirectionalShift(m)


    def _definedness(self, gaussian):
        # Definedness factor d for the engagement push
        #   (DESIGN §8.6.2): d = sigma_floor / sigma, capped
        #   at 1. It is near 1 for a sharp (narrow) view —
        #   easy to rouse — and near 0 for a vague (broad)
        #   one — hard to rouse. The cap guards the rare
        #   first-step case where an initial sigma was drawn
        #   below the floor (the floor is otherwise enforced
        #   on every apply step). Returned per-dimension.
        return np.minimum(
                1.0, self.sigma_floor / gaussian.sigma)


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
        # f_pol and f_trait are non-negative magnitudes (half-
        # normal draws), so a more persuasive politician always
        # raises engagement faster — persuasion never drives a
        # citizen toward apathy. Disengagement is the job of the
        # separate engagement-decay mechanism, not persuasion.
        for pol_idx, politician in enumerate(self.politician_list):
            f_pol   = politician.policy_persuasion
            f_trait = politician.trait_persuasion

            # --- Engagement shifts (DESIGN §8.6.2) ---
            # Pcp is driven by policy integrals involving Pcp;
            #   Pca by integrals involving Pca. Each |overlap|
            #   is scaled two ways: by the definedness
            #   d = sigma_floor/sigma of the Gaussian being
            #   shifted (sharp views rouse easily, vague ones
            #   barely move) and by the threat weight w on
            #   every term that touches an aversion. Only the
            #   pure preference-meets-preference terms
            #   (Pcp-Ppp and Tcp-Tpx) escape w.
            w = self.threat_weight
            d_Pcp = self._definedness(self.stated_policy_pref)
            d_Pca = self._definedness(self.stated_policy_aver)
            d_Tcp = self._definedness(self.stated_trait_pref)
            d_Tca = self._definedness(self.stated_trait_aver)
            self.Pcp_orien_shift += f_pol * d_Pcp * (
                np.abs(self.Pcp_Ppp_ol[pol_idx])
                + w * np.abs(self.Pcp_Ppa_ol[pol_idx]))
            self.Pca_orien_shift += f_pol * d_Pca * w * (
                np.abs(self.Pca_Ppa_ol[pol_idx])
                + np.abs(self.Pca_Ppp_ol[pol_idx]))
            self.Tcp_orien_shift += (
                f_trait * d_Tcp * np.abs(self.Tcp_Tpx_ol[pol_idx]))
            self.Tca_orien_shift += (f_trait * d_Tca * w
                * np.abs(self.Tca_Tpx_ol[pol_idx]))

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

            # speed = trait magnitude x persuasion x
            #   susceptibility (>= 0, distance-free). The
            #   DirectionalShift accumulator caps each pull at
            #   the gap so no single politician overshoots, and
            #   records the target for the span clamp (§8.6.3).
            speed_Pcp = mag * f_pol * S_Pcp
            speed_Pca = mag * f_pol * S_Pca
            if trait_sum >= 0:
                # Attraction branch: Pcp → Ppp, Pca → Ppa.
                Ppp_mu    = politician.ext_policy_pref.mu
                Ppp_sigma = politician.ext_policy_pref.sigma
                Ppa_mu    = politician.ext_policy_aver.mu
                Ppa_sigma = politician.ext_policy_aver.sigma
                self.Pcp_mu_pol.add(
                    self.stated_policy_pref.mu, Ppp_mu, speed_Pcp)
                self.Pcp_sig_pol.add(
                    self.stated_policy_pref.sigma, Ppp_sigma,
                    speed_Pcp)
                self.Pca_mu_pol.add(
                    self.stated_policy_aver.mu, Ppa_mu, speed_Pca)
                self.Pca_sig_pol.add(
                    self.stated_policy_aver.sigma, Ppa_sigma,
                    speed_Pca)
            else:
                # Defensive branch: rigidity + targeted backlash.
                # Pcp sigma narrows toward sigma_floor; Pca mu
                #   shifts toward the politician's PREFERENCE (not
                #   aversion), scaled by defensive_ratio. Pcp mu
                #   and Pca sigma do not move in this branch.
                Ppp_mu = politician.ext_policy_pref.mu
                self.Pcp_sig_pol.add(
                    self.stated_policy_pref.sigma,
                    self.sigma_floor, speed_Pcp)
                self.Pca_mu_pol.add(
                    self.stated_policy_aver.mu, Ppp_mu,
                    self.defensive_ratio * speed_Pca)


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

        # Threat weight and definedness for the engagement
        #   push, the same rule as politician influence
        #   (DESIGN §8.6.2). d = sigma_floor/sigma scales the
        #   push by how sharply each view is held; w weights
        #   every aversion-touching term.
        w = self.threat_weight
        d_Pcp = self._definedness(self.stated_policy_pref)
        d_Pca = self._definedness(self.stated_policy_aver)
        d_Tcp = self._definedness(self.stated_trait_pref)
        d_Tca = self._definedness(self.stated_trait_aver)

        for zone_idx, zone in enumerate(self.zone_list):

            # --- Engagement shifts (DESIGN §8.6.2) ---
            # |overlap| drives theta toward the engaged pole,
            #   scaled by definedness d and threat weight w.
            #   Cross-terms (pref×aver) carry w: both agreement
            #   and shared/opposed stances raise engagement on
            #   community-relevant issues, threats more so.
            self.Pcp_orien_shift += cir * d_Pcp * (
                np.abs(self.Pcp_Pcp_ol[zone_idx])
                + w * np.abs(self.Pcp_Pca_ol[zone_idx]))
            self.Pca_orien_shift += cir * d_Pca * w * (
                np.abs(self.Pca_Pca_ol[zone_idx])
                + np.abs(self.Pca_Pcp_ol[zone_idx]))
            self.Tcp_orien_shift += cir * d_Tcp * (
                np.abs(self.Tcp_Tcp_ol[zone_idx])
                + w * np.abs(self.Tcp_Tca_ol[zone_idx]))
            self.Tca_orien_shift += cir * d_Tca * w * (
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
            # Unconditional: Pcp drifts toward avg_Pcp, Pca toward
            #   avg_Pca. No defensive branch. speed = community rate
            #   x trait_rate x susceptibility (>= 0, distance-free).
            #   The same no-overshoot cap and span clamp apply, so
            #   community drift never passes the zone average.
            speed_Pcp = cir * trait_rate * S_Pcp
            speed_Pca = cir * trait_rate * S_Pca
            self.Pcp_mu_com.add(
                self.stated_policy_pref.mu, zone.avg_Pcp.mu,
                speed_Pcp)
            self.Pcp_sig_com.add(
                self.stated_policy_pref.sigma, zone.avg_Pcp.sigma,
                speed_Pcp)
            self.Pca_mu_com.add(
                self.stated_policy_aver.mu, zone.avg_Pca.mu,
                speed_Pca)
            self.Pca_sig_com.add(
                self.stated_policy_aver.sigma, zone.avg_Pca.sigma,
                speed_Pca)

            # --- Trait position and spread shifts (§8.6.4) ---
            # Same trait_rate drives trait acclimatization. Tcp
            #   drifts toward avg_Tcp; Tca toward avg_Tca. This is
            #   the sole mechanism for trait change.
            speed_Tcp = cir * trait_rate * S_Tcp
            speed_Tca = cir * trait_rate * S_Tca
            self.Tcp_mu_com.add(
                self.stated_trait_pref.mu, zone.avg_Tcp.mu,
                speed_Tcp)
            self.Tcp_sig_com.add(
                self.stated_trait_pref.sigma, zone.avg_Tcp.sigma,
                speed_Tcp)
            self.Tca_mu_com.add(
                self.stated_trait_aver.mu, zone.avg_Tca.mu,
                speed_Tca)
            self.Tca_sig_com.add(
                self.stated_trait_aver.sigma, zone.avg_Tca.sigma,
                speed_Tca)


    def apply_influence_shifts(self):
        # This method is the second half of a deliberate
        #   two-phase accumulate-then-apply design (see
        #   DESIGN.md §8.6). During the accumulation phase,
        #   build_response_to_politician_influence(),
        #   build_response_to_government(), and
        #   build_response_to_citizen_collective() each add
        #   their contributions to twelve shared shift
        #   arrays — three per Gaussian type (Pcp, Pca,
        #   Tcp, Tca), one each for orientation (theta),
        #   position (mu), and spread (sigma). Because
        #   those methods write to arrays rather than
        #   mutating Gaussian parameters directly, the
        #   order in which the sources are processed is
        #   irrelevant — they all observe the same unchanged
        #   Gaussian state from the start of the step. This
        #   method is the apply phase: it flushes the
        #   position (mu) and spread (sigma) shifts into the
        #   actual Gaussian parameters, then hands the
        #   engagement (theta) update to apply_engagement_
        #   shifts() (shared with the govern phase).
        #
        # Two of the three quantities are updated here:
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
        #   theta (engagement) is updated separately in
        #     apply_engagement_shifts(), called at the end —
        #     see that method for the sign convention and the
        #     spread-proportional fade.

        # --- Policy preference (Pcp) ---
        # Apply the politician sub-total first, then the community
        #   sub-total, each clamped to its own target span
        #   (§8.6.5). Clamping per source — using the value BEFORE
        #   that source is added as the widening bound — guarantees
        #   neither source carries the citizen past its own most
        #   extreme target. sigma_floor is applied last as a safety
        #   floor (the span clamp already keeps sigma within the
        #   contributed targets).
        mu = self.Pcp_mu_pol.apply(self.stated_policy_pref.mu)
        mu = self.Pcp_mu_com.apply(mu)
        self.stated_policy_pref.mu = mu
        sigma = self.Pcp_sig_pol.apply(self.stated_policy_pref.sigma)
        sigma = self.Pcp_sig_com.apply(sigma)
        self.stated_policy_pref.sigma = np.maximum(
                sigma, self.sigma_floor)

        # --- Policy aversion (Pca) ---
        # Same clamp-then-sum pattern as Pcp. The politician
        #   sub-total here may come from the attraction OR the
        #   defensive branch; the community sub-total is always
        #   attraction toward the zone average.
        mu = self.Pca_mu_pol.apply(self.stated_policy_aver.mu)
        mu = self.Pca_mu_com.apply(mu)
        self.stated_policy_aver.mu = mu
        sigma = self.Pca_sig_pol.apply(self.stated_policy_aver.sigma)
        sigma = self.Pca_sig_com.apply(sigma)
        self.stated_policy_aver.sigma = np.maximum(
                sigma, self.sigma_floor)

        # --- Trait preference (Tcp) ---
        # Trait shifts have a community source only (politicians
        #   never alter citizen traits, §8.6.4), so there is one
        #   sub-total to clamp and apply.
        self.stated_trait_pref.mu = self.Tcp_mu_com.apply(
                self.stated_trait_pref.mu)
        self.stated_trait_pref.sigma = np.maximum(
                self.Tcp_sig_com.apply(
                    self.stated_trait_pref.sigma),
                self.sigma_floor)

        # --- Trait aversion (Tca) ---
        # Identical pattern to trait preference.
        self.stated_trait_aver.mu = self.Tca_mu_com.apply(
                self.stated_trait_aver.mu)
        self.stated_trait_aver.sigma = np.maximum(
                self.Tca_sig_com.apply(
                    self.stated_trait_aver.sigma),
                self.sigma_floor)

        # Engagement (theta) is updated last, in the shared
        #   helper used by BOTH the campaign and govern phases
        #   (§8.6.5): the net theta_shift drives toward the
        #   engaged pole and the spread-proportional fade pulls
        #   back toward apathy. The helper also refreshes the
        #   cached integration variables, so they reflect the
        #   sigma changes applied just above as well.
        self.apply_engagement_shifts()


    def apply_engagement_shifts(self):
        # Apply the accumulated engagement (theta) shifts and the
        #   steady fade toward apathy, then refresh the cached
        #   integration variables. This runs EVERY step in BOTH
        #   the campaign and the govern phases (DESIGN §8.6,
        #   §8.6.5–§8.6.6), so engagement is one continuous
        #   process; the phases differ only in which sources fed
        #   the theta_shift arrays during accumulation.
        #
        # Theta sign convention (see §8.6.5):
        #   Preference Gaussians: Im(theta) in [0, pi/2], engaged
        #     pole at 0. Subtracting the (net) theta_shift drives
        #     toward 0 (more engaged); the fade adds back toward
        #     pi/2 (apathy).
        #   Aversion Gaussians: Im(theta) in [pi/2, pi], engaged
        #     pole at pi. Adding the theta_shift drives toward pi;
        #     the fade subtracts back toward pi/2.
        #
        # theta_shift is the NET drive: most sources add to it
        #   (toward engaged), while government resignation
        #   subtracts from it (toward apathy), so the net value
        #   may be positive or negative.
        #
        # The fade (DESIGN §8.6.6) is proportional to each
        #   Gaussian's own spread: fade = engagement_decay_rate *
        #   sigma. Because sigma is floored at sigma_floor, every
        #   citizen fades a little every step — no one freezes at
        #   full engagement. A sharp (narrow-sigma) view fades
        #   slowly and holds engagement; a broad view fades fast
        #   and lapses to apathy. The fade depends on spread, not
        #   on the current engagement level, so it does not make
        #   the most engaged fade fastest.
        half_pi = np.pi / 2.0
        edr = self.engagement_decay_rate

        # --- Preference types (Pcp, Tcp) ---
        for gaussian, orien_shift in (
                (self.stated_policy_pref, self.Pcp_orien_shift),
                (self.stated_trait_pref, self.Tcp_orien_shift)):
            fade = edr * gaussian.sigma
            pref_im = np.clip(
                    gaussian.theta.imag - orien_shift,
                    0.0, half_pi)
            pref_im = np.clip(pref_im + fade, 0.0, half_pi)
            gaussian.theta = pref_im * 1j

        # --- Aversion types (Pca, Tca) ---
        for gaussian, orien_shift in (
                (self.stated_policy_aver, self.Pca_orien_shift),
                (self.stated_trait_aver, self.Tca_orien_shift)):
            fade = edr * gaussian.sigma
            aver_im = np.clip(
                    gaussian.theta.imag + orien_shift,
                    half_pi, np.pi)
            aver_im = np.clip(aver_im - fade, half_pi, np.pi)
            gaussian.theta = aver_im * 1j

        # §8.6.7: Refresh derived variables (alpha, cos_theta,
        #   self_norm) cached inside each shifted Gaussian. They
        #   are stale after the sigma/theta changes and must be
        #   current before the next overlap integral. The
        #   ideal_policy_pref Gaussian is intentionally excluded:
        #   it is the citizen's true (hidden) interest and is
        #   never subject to influence or fade (DESIGN §8.6).
        self.stated_policy_pref.update_integration_variables()
        self.stated_policy_aver.update_integration_variables()
        self.stated_trait_pref.update_integration_variables()
        self.stated_trait_aver.update_integration_variables()


    def recompute_government_overlaps(self, world):
        """Reset and recompute the three citizen-vs-
        government overlap lists against the current
        enacted policy.

        Used in the govern phase, where Pge changes every
        step but the full compute_all_overlaps() sensing
        pass (which also rebuilds the politician and
        community overlaps) is not run. This refreshes only
        the government overlaps the engagement response and
        the well-being measure depend on: the conscious
        preference and aversion overlaps (Pcp/Pca vs Pge)
        and the ideal overlap (Pci vs Pge).

        Pge.update_integration_variables() must be called
        before this method so the integrals use the updated
        Pge parameters.
        """
        self.Pcp_Pge_ol = []
        self.Pca_Pge_ol = []
        self.Pci_Pge_ol = []
        self.policy_government_integrals(world)


    def build_response_to_government(self):
        """Set the well-being outcome measure and accumulate
        the government-driven engagement response.

        The government affects engagement through the
        citizen's CONSCIOUS (stated) policy positions
        (DESIGN §8.6.2), in two opposing channels:

          Anger (engagement UP): when a stated aversion is
            realized by the enacted policy. The overlap
            I(Pca, Pge) is most negative when the hated
            thing is being done, so -I(Pca, Pge) is the
            positive anger signal. It touches an aversion,
            so it carries the threat weight w.

          Resignation (engagement DOWN): when a stated
            preference goes unmet. Resignation grows as the
            satisfaction overlap I(Pcp, Pge) falls below the
            reference level sat_ref. It is subtracted from
            the engagement drive, pushing theta toward
            apathy, and is scaled by definedness so it bites
            hardest on SHARP citizens — the well-informed
            voter who knows what they want, sees it ignored,
            and stops participating while keeping a sharp
            opinion.

        Both channels are scaled by govt_engagement_rate and
        by the definedness d = sigma_floor/sigma of the
        Gaussian being shifted. Government acts on policy
        only, so the trait Gaussians are untouched here.

        The well-being scalar is the OUTCOME measure
        (DESIGN §8.5): the overlap between the citizen's
        IDEAL policy (Pci — the objectively best policy,
        which they do not consciously know) and Pge. It is
        recorded for output but, unlike the conscious
        overlaps above, no longer feeds engagement — a
        citizen cannot perceive their own hidden ideal.

        This method writes only to the engagement shift
        arrays (orien_shift) and the well_being scalar, not
        to the Gaussians directly; the theta update happens
        in apply_engagement_shifts(). It runs every step in
        BOTH phases, against the current Pge, with no stored
        state — a change of government washes the old
        response out.
        """
        w = self.threat_weight
        ger = self.govt_engagement_rate

        # Outcome measure (recorded for output, not an
        #   engagement input).
        self.well_being = sum(self.Pci_Pge_ol[0])

        # Definedness of the conscious policy Gaussians.
        d_Pcp = self._definedness(self.stated_policy_pref)
        d_Pca = self._definedness(self.stated_policy_aver)

        # Anger: aversion realized -> engagement up. Carries
        #   the threat weight (it touches an aversion).
        anger = np.maximum(0.0, -self.Pca_Pge_ol[0])
        self.Pca_orien_shift += ger * w * d_Pca * anger

        # Resignation: preference unmet -> engagement down.
        #   Subtracted so it drives theta toward apathy.
        resignation = np.maximum(
                0.0, self.sat_ref - self.Pcp_Pge_ol[0])
        self.Pcp_orien_shift -= ger * d_Pcp * resignation


    def reset_orientation_shifts(self, num_policy_dims,
            num_trait_dims):
        """Zero the four engagement (theta) shift arrays.

        The campaign phase resets these inside
        prepare_for_influence() (which also allocates the
        position/spread accumulators). The govern phase has
        no position/spread shifts, so it uses this lighter
        reset before each step's government engagement
        response (DESIGN §8.6).
        """
        n = num_policy_dims
        m = num_trait_dims
        self.Pcp_orien_shift = np.zeros(n)
        self.Pca_orien_shift = np.zeros(n)
        self.Tcp_orien_shift = np.zeros(m)
        self.Tca_orien_shift = np.zeros(m)


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
