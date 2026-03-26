import numpy as np

from gaussian import Gaussian, sample_theta
from random_state import rng


class Politician():
    """A political agent who campaigns for votes
    and, if elected, exerts forces on government
    enacted policy.

    Each politician has two layers of positions:

    INNATE positions: The politician's true beliefs
      or desires. These are static — they do not
      change during the simulation. They include:
      - innate_policy_pref: where the politician
        truly stands on each policy dimension.
      - innate_policy_aver: policies the politician
        truly opposes.
      - innate_trait: the politician's personality
        profile.

    EXTERNAL (apparent) positions: What the
      politician presents to citizens during
      campaigns. These may differ from innate
      positions based on the politician's adapt
      strategy:
      - ext_policy_pref: apparent policy preferences.
      - ext_policy_aver: apparent policy aversions.
      - ext_trait: apparent personality traits.

    The gap between innate and external positions
    represents the degree to which a politician is
    willing to misrepresent themselves to win votes.
    Citizens compare their Gaussians against the
    politician's EXTERNAL positions when computing
    overlaps and scoring candidates.

    During the govern phase, elected politicians use
    their INNATE positions (not external) to exert
    forces on government enacted policy — they
    govern according to their true beliefs, not
    their campaign promises.

    Key scalar attributes:
    - policy_persuasion: scales engagement and
      position shifts driven by policy overlaps.
    - trait_persuasion: scales engagement shifts
      driven by trait overlaps.
    - policy_lie: controls how far external policy
      positions deviate from innate ones (adapt
      strategies 1-2).
    - trait_lie: same for trait positions.
    - margin_of_victory: set during the vote phase;
      used to compute political_power.
    - political_power: determines the magnitude of
      forces this politician exerts on enacted
      policy during the govern phase.
    - pref_weight, aver_weight: per-dimension
      weights allocating political_power across
      policy dimensions, derived from innate sigma.
    """

    # --- Conceptual overview for students ---
    #
    # Politicians have an innate set of personality
    #   "traits" positioned on a one-dimensional
    #   spectrum. The distance between a citizen's
    #   trait position and a politician's trait
    #   position will influence (1) the ability of
    #   the politician to persuade a citizen with
    #   respect to their policy positions; (2) the
    #   probability that a citizen will vote for a
    #   politician; (3) the probability that a
    #   citizen will allow policy position
    #   misrepresentations to go "unpunished".
    #
    # Politicians have an innate position for each
    #   policy dimension. The innate position should
    #   not be understood as a "values" statement in
    #   any definite sense. For some politicians, it
    #   may be representative of their "values" but
    #   for others it may be better thought of as
    #   their "desire". Presently, the innate
    #   position is static for each politician.
    #
    # Politicians have an apparent "externally
    #   visible" position for each policy dimension.
    #   The external position is what a politician
    #   presents to citizens. The apparent position
    #   will differ from the innate position due to
    #   factors: (1) The innate position and the
    #   average position of the citizens whose votes
    #   the politician wants to obtain tend to
    #   differ. So, the politician may present an
    #   apparent position to the citizens in an
    #   effort to persuade them. (2) The politician
    #   is willing to misrepresent their innate
    #   position by an amount in proportion to their
    #   propensity to lie/pander and believe that
    #   they will not turn off citizens by being
    #   detected.
    #
    # It is assumed that citizens may or may not be
    #   able to detect (or may not care about) the
    #   difference between an apparent policy
    #   position and the innate position that a
    #   politician has.

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
        self.move_strategy = self.select_strategy(
                settings, "move")
        self.adapt_strategy = self.select_strategy(
                settings, "adapt")

        # Assign instance variables from passed initialization parameters.
        self.zone_type = zone_type
        self.zone = zone
        self.patch = patch

        # Initialize any other instance variables to their default value.
        self.elected = False
        self.votes = 0


    def reset_to_input(self, settings):
        """(Re)initialize all Gaussians and scalar
        attributes from the XML configuration.

        Called once during __init__() and again at
        the start of each cycle for any politician
        who lost the previous election (see
        World.repopulate_politicians()). Losing
        politicians are effectively replaced by
        new challengers with fresh random positions.

        Each Gaussian is initialized with three
        random draws:
          mu: from N(0, pos_stddev) — random
            position on the policy/trait axis.
          sigma: from |N(0, stddev_stddev)| —
            random spread, abs() ensures positive.
          theta: from sample_theta() — random
            engagement, using the theta sign
            convention described below.

        Theta sign convention (mirrors citizens):
          Preference Gaussians use Im(theta) in
          [0, pi/2), giving cos(theta) > 0
          (positive-valued). Aversion Gaussians
          use Im(theta) in (pi/2, pi], giving
          cos(theta) < 0 (negative-valued). Traits
          (innate and external) are positive-valued;
          politicians have no trait aversions.

          If the *_orien_stddev XML parameter is
          numeric, Im(theta) is drawn from a normal
          distribution centered on the default mean
          (1.0 for preferences, pi-1 for aversions),
          producing a population with varied initial
          engagement. If non-numeric (e.g.,
          "uniform"), the default mean is used for
          all agents, giving uniform initial
          engagement.
        """
        half_pi = np.pi / 2.0
        pol = settings.infile_dict[1]["politicians"]

        self.innate_policy_pref = Gaussian(
                rng.normal(loc=0.0,
                    scale=float(
                        pol["policy_pref_pos_stddev"]),
                    size=self.num_policy_dims),
                np.abs(rng.normal(loc=0.0,
                    scale=float(
                        pol["policy_pref_stddev_stddev"]),
                    size=self.num_policy_dims)),
                sample_theta(
                    pol["policy_pref_orien_stddev"],
                    1.0, self.num_policy_dims,
                    0.0, half_pi),
                1)
        self.innate_policy_aver = Gaussian(
                rng.normal(loc=0.0,
                    scale=float(
                        pol["policy_aver_pos_stddev"]),
                    size=self.num_policy_dims),
                np.abs(rng.normal(loc=0.0,
                    scale=float(
                        pol["policy_aver_stddev_stddev"]),
                    size=self.num_policy_dims)),
                sample_theta(
                    pol["policy_aver_orien_stddev"],
                    np.pi - 1.0,
                    self.num_policy_dims,
                    half_pi, np.pi),
                1)
        self.ext_policy_pref = Gaussian(
                rng.normal(loc=0.0,
                    scale=float(
                        pol["policy_pref_pos_stddev"]),
                    size=self.num_policy_dims),
                np.abs(rng.normal(loc=0.0,
                    scale=float(
                        pol["policy_pref_stddev_stddev"]),
                    size=self.num_policy_dims)),
                sample_theta(
                    pol["policy_pref_orien_stddev"],
                    1.0, self.num_policy_dims,
                    0.0, half_pi),
                1)
        self.ext_policy_aver = Gaussian(
                rng.normal(loc=0.0,
                    scale=float(
                        pol["policy_aver_pos_stddev"]),
                    size=self.num_policy_dims),
                np.abs(rng.normal(loc=0.0,
                    scale=float(
                        pol["policy_aver_stddev_stddev"]),
                    size=self.num_policy_dims)),
                sample_theta(
                    pol["policy_aver_orien_stddev"],
                    np.pi - 1.0,
                    self.num_policy_dims,
                    half_pi, np.pi),
                1)

        # Trait Gaussians: politicians have no
        #   *_orien_stddev params for traits, so
        #   theta uses the hardcoded default (1j).
        self.innate_trait = Gaussian(
                rng.normal(loc=0.0,
                    scale=float(
                        pol["trait_innate_pos_stddev"]),
                    size=self.num_trait_dims),
                np.abs(rng.normal(loc=0.0,
                    scale=float(
                        pol["trait_innate_stddev_stddev"]),
                    size=self.num_trait_dims)),
                np.full(self.num_trait_dims, 1j),
                1)
        self.ext_trait = Gaussian(
                rng.normal(loc=0.0,
                    scale=float(
                        pol["trait_ext_pos_stddev"]),
                    size=self.num_trait_dims),
                np.abs(rng.normal(loc=0.0,
                    scale=float(
                        pol["trait_ext_stddev_stddev"]),
                    size=self.num_trait_dims)),
                np.full(self.num_trait_dims, 1j),
                1)

        # policy_persuasion and trait_persuasion scale how
        #   effectively this politician shifts citizen Gaussians
        #   during the campaign phase (DESIGN.md §8.6.2–§8.6.3).
        #   f_pol = policy_persuasion scales engagement and
        #   position/spread shifts driven by policy overlaps;
        #   f_trait = trait_persuasion scales engagement shifts
        #   driven by trait overlaps. Both are drawn from
        #   zero-mean Gaussians so that politicians vary in
        #   how persuasive they are, with the sign encoding
        #   direction of effect (positive = amplifies shifts,
        #   negative = dampens).
        self.policy_persuasion = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["policy_persuasion_stddev"]))
        self.trait_persuasion = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["trait_persuasion_stddev"]))

        self.policy_lie = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["policy_lie_stddev"]))
        self.trait_lie = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["trait_lie_stddev"]))


    def reset_votes(self):
        """Reset this politician's vote tally and
        margin of victory to zero at the start of
        a new election cycle. Called by
        World.repopulate_politicians() before each
        campaign.

        margin_of_victory is set by the vote()
        function after the election and is used by
        compute_political_power() during the govern
        phase. It represents the normalized
        difference between the winner's votes and
        the runner-up's votes, divided by total
        votes cast in the zone. A landslide winner
        has margin close to 1; a narrow winner has
        margin close to 0.
        """
        self.votes = 0
        self.margin_of_victory = 0.0


    def compute_political_power(self):
        """Compute this politician's political power
        for the govern phase (DESIGN.md §7.5.1).

        Political power determines the magnitude of
        the forces this politician exerts on the
        government's enacted policy (Pge) during
        each govern step. The formula is:

          political_power =
              zone_population
              * margin_of_victory
              * (agreement / (|disagreement| + eps))

        The three factors capture different aspects
        of political mandate:

        1. zone_population: a politician representing
           a larger zone has more power. A district
           representative with 1000 citizens has
           less sway than a state representative
           with 100,000.

        2. margin_of_victory: a politician who won
           by a landslide has a stronger mandate
           than one who barely won. This is the
           normalized vote margin from the election.

        3. agreement/disagreement ratio: measures
           how well this politician's positions
           align with what their constituents
           actually want vs. what they oppose.
           Agreement is the sum of same-type
           overlaps (citizen pref × politician pref,
           citizen aver × politician aver, citizen
           trait × politician trait). Disagreement
           is the sum of cross-type overlaps
           (citizen aver × politician pref, citizen
           pref × politician aver). A politician
           whose positions match constituent desires
           and avoid their aversions gets a high
           ratio and thus more governing power.
           The eps term prevents division by zero
           when disagreement is negligible.

        Prerequisites: zone averages must be current
        (computed during the last campaign step) and
        margin_of_victory must be set by the vote
        phase.
        """
        zone = self.zone

        # Agreement: same-type overlaps — how
        #   well the politician's apparent
        #   positions align with citizen
        #   preferences. Non-negative by the
        #   theta sign convention (cos_pref > 0,
        #   cos_aver < 0, same-type product ≥ 0).
        agreement = (
            np.sum(zone.avg_Pcp.integral(
                self.ext_policy_pref))
            + np.sum(zone.avg_Pca.integral(
                self.ext_policy_aver))
            + np.sum(zone.avg_Tcp.integral(
                self.ext_trait)))

        # Disagreement: cross-type overlaps —
        #   how much the politician clashes with
        #   citizen aversions/preferences. These
        #   integrals are non-positive by the
        #   theta sign convention (cos_pref > 0
        #   × cos_aver < 0 = negative product).
        disagreement = (
            np.sum(zone.avg_Pca.integral(
                self.ext_policy_pref))
            + np.sum(zone.avg_Pcp.integral(
                self.ext_policy_aver)))

        # Mandate: net alignment = agreement
        #   + disagreement. Since disagreement ≤
        #   0, this equals agreement - |disagree|.
        #   Clamped to 0 so that a politician
        #   with negative net alignment has zero
        #   governing power (DESIGN.md §7.5.1).
        mandate = max(0.0, agreement + disagreement)

        self.political_power = (
            zone.curr_num_citizens
            * self.margin_of_victory
            * mandate)


    def compute_dimension_weights(self):
        """Compute per-dimension weights that
        distribute this politician's political power
        across policy dimensions (DESIGN.md §7.5.2).

        The principle: a politician who holds a
        strong opinion on a dimension (small innate
        sigma = narrow, firm Gaussian) will direct
        more of their governing force to that
        dimension. A politician with a weak opinion
        (large innate sigma = broad, flexible
        Gaussian) will exert less force there.

        Mathematically:
          raw_weight_d = 1 / sigma_d
          weight_d = raw_weight_d / sum(raw_weight)

        This produces a weight vector that sums to
        1 across all policy dimensions. Two separate
        weight vectors are computed: pref_weight
        (from innate preference sigma) controls
        preference-attraction forces, and aver_weight
        (from innate aversion sigma) controls
        aversion-repulsion forces.

        Example with 3 policy dims:
          sigma = [0.5, 1.0, 2.0]
          raw   = [2.0, 1.0, 0.5]  (inverse)
          weight = [0.571, 0.286, 0.143]  (normalized)
          -> 57% of pref force goes to dimension 0
        """
        raw_pref = (
            1.0 / self.innate_policy_pref.sigma)
        self.pref_weight = (
            raw_pref / np.sum(raw_pref))

        raw_aver = (
            1.0 / self.innate_policy_aver.sigma)
        self.aver_weight = (
            raw_aver / np.sum(raw_aver))

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
        """Randomly select a strategy index from a
        cumulative probability distribution defined
        in the XML configuration.

        The XML parameter cumul_{strat_type}_strategy_probs
        contains a comma-separated list of cumulative
        probabilities. For example, "0.5,0.75,1.0"
        means:
          - 50% chance of strategy 0  (U < 0.50)
          - 25% chance of strategy 1  (0.50 <= U < 0.75)
          - 25% chance of strategy 2  (0.75 <= U < 1.00)

        A single uniform random draw U in [0,1) is
        compared against each cumulative threshold
        in order; the first threshold that exceeds
        U selects the strategy index.

        Parameters
        ----------
        settings : ScriptSettings
            Provides the XML configuration.
        strat_type : str
            Either "move" or "adapt", selecting
            which set of cumulative probabilities
            to use from the XML.

        Returns
        -------
        int
            The selected strategy index (0-based).
        """
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
        """Move this politician to a new patch
        within their zone, according to their
        assigned move strategy.

        Politicians can only influence citizens who
        share their patch (overlap integrals are
        computed between each citizen and the
        politicians on their patch). The move
        strategy determines which citizens the
        politician will interact with each campaign
        step.

        Strategy 0 — Random:
          Move to a random patch in the zone each
          step. Simple baseline behavior; the
          politician covers the zone evenly over
          time but without tactical intent.

        Strategy 1 — Population center:
          Move to the most populous patch in the
          zone and stay there. Maximizes the number
          of citizens influenced per step.
          Represents a "play to the base" approach.

        Strategy 2 — Grassroots outreach:
          Cycle through the least-populous patches
          (bottom half by population) in round-robin
          order. Represents a strategy of reaching
          underserved communities. The politician
          visits fewer citizens per step but may
          encounter citizens with different views
          than the mainstream.
        """
        if (self.move_strategy == 0):
            self.patch = self.zone.random_patch()

        elif (self.move_strategy == 1):
            best = self.zone.patches[0]
            for p in self.zone.patches[1:]:
                if len(p.citizen_list) > len(
                        best.citizen_list):
                    best = p
            self.patch = best

        elif (self.move_strategy == 2):
            # Sort patches by population ascending,
            #   take the bottom half, and advance
            #   one step through them each call.
            sorted_patches = sorted(
                self.zone.patches,
                key=lambda p: len(p.citizen_list))
            mid = max(1, len(sorted_patches) // 2)
            targets = sorted_patches[:mid]
            try:
                idx = targets.index(self.patch)
                self.patch = targets[
                    (idx + 1) % len(targets)]
            except ValueError:
                # Current patch not in bottom half;
                #   start at the first target.
                self.patch = targets[0]


    def _copy_innate_to_external(self):
        """Copy innate positions to external
        positions without modification.

        This is the baseline for all adapt
        strategies: every strategy starts from
        the politician's true (innate) positions.
        Strategy 0 (honest) stops here; strategies
        1+ call this first, then modify the
        external arrays.

        The [:] slice assignment copies values
        into the existing numpy array rather than
        replacing the array object. This preserves
        the Gaussian's identity (other code may
        hold references to the ext_* Gaussians).
        """
        self.ext_policy_pref.mu[:] = (
            self.innate_policy_pref.mu)
        self.ext_policy_pref.sigma[:] = (
            self.innate_policy_pref.sigma)
        self.ext_policy_pref.theta[:] = (
            self.innate_policy_pref.theta)

        self.ext_policy_aver.mu[:] = (
            self.innate_policy_aver.mu)
        self.ext_policy_aver.sigma[:] = (
            self.innate_policy_aver.sigma)
        self.ext_policy_aver.theta[:] = (
            self.innate_policy_aver.theta)

        self.ext_trait.mu[:] = (
            self.innate_trait.mu)
        self.ext_trait.sigma[:] = (
            self.innate_trait.sigma)
        self.ext_trait.theta[:] = (
            self.innate_trait.theta)


    def adapt_to_patch(self, world):
        """Set apparent (external) policy and trait
        positions according to this politician's
        adapt strategy.

        This method is called each campaign step
        after the politician moves to a new patch.
        It determines what positions the politician
        "shows" to citizens during interactions.

        All strategies start from innate (true)
        positions. Strategies 1+ then blend the
        external positions toward a target using
        the policy_lie and trait_lie parameters:

          ext = innate + lie * (target - innate)

        This is a linear interpolation:
          lie = 0 -> ext = innate (honest)
          lie = 1 -> ext = target (full adoption)
          0 < lie < 1 -> partial shift toward
                         the target

        The lie parameters are drawn from zero-mean
        Gaussians at initialization, so they can be
        negative (which would shift AWAY from the
        target — a contrarian response). Typical
        values are small and positive.

        Strategy 0 — Honest:
          Present innate positions unchanged. The
          politician says what they truly believe.

        Strategy 1 — Pander:
          Shift external policy positions toward
          zone-average CITIZEN PREFERENCES and
          AVERSIONS, and external trait toward
          zone-average CITIZEN TRAIT PREFERENCES.
          The politician tells citizens what they
          want to hear, modulated by how willing
          the politician is to lie (policy_lie,
          trait_lie).

        Strategy 2 — Avoidance:
          Shift external positions AWAY from what
          citizens dislike. Policy preference mu
          moves away from citizen aversions; policy
          aversion mu moves toward citizen
          preferences (pretending to dislike what
          citizens like less); trait moves away from
          citizen trait aversions. This is subtler
          than pandering: rather than adopting
          popular positions, the politician avoids
          unpopular ones.

        After any modification, cached integration
        variables on the external Gaussians are
        refreshed so that the next overlap integral
        computation uses the updated positions.
        """
        # Always start from innate positions.
        self._copy_innate_to_external()

        if (self.adapt_strategy == 0):
            # Honest: present innate positions
            #   unchanged. Nothing more to do.
            pass

        elif (self.adapt_strategy == 1):
            # Pander toward citizen preferences:
            #   shift external policy positions
            #   toward zone-average citizen
            #   preferences and aversions, and
            #   shift external trait toward
            #   zone-average citizen trait
            #   preferences.
            avg = self.zone
            pl = self.policy_lie
            tl = self.trait_lie
            self.ext_policy_pref.mu += (
                pl * (avg.avg_Pcp.mu
                      - self.innate_policy_pref.mu))
            self.ext_policy_pref.sigma += (
                pl * (avg.avg_Pcp.sigma
                      - self.innate_policy_pref.sigma))
            self.ext_policy_aver.mu += (
                pl * (avg.avg_Pca.mu
                      - self.innate_policy_aver.mu))
            self.ext_policy_aver.sigma += (
                pl * (avg.avg_Pca.sigma
                      - self.innate_policy_aver.sigma))
            self.ext_trait.mu += (
                tl * (avg.avg_Tcp.mu
                      - self.innate_trait.mu))
            self.ext_trait.sigma += (
                tl * (avg.avg_Tcp.sigma
                      - self.innate_trait.sigma))

        elif (self.adapt_strategy == 2):
            # Avoid citizen aversions: shift
            #   external policy positions away
            #   from what citizens dislike. The
            #   target is the zone-average citizen
            #   aversion (for policy pref) and
            #   the zone-average citizen preference
            #   (for policy aver), so that the
            #   politician moves away from what
            #   citizens are averse to on both
            #   sides. Trait shifts toward
            #   zone-average trait aversion (so the
            #   politician avoids personality
            #   traits that citizens dislike).
            avg = self.zone
            pl = self.policy_lie
            tl = self.trait_lie
            # Move pref mu AWAY from citizen
            #   aversion: subtract the blend.
            self.ext_policy_pref.mu -= (
                pl * (avg.avg_Pca.mu
                      - self.innate_policy_pref.mu))
            # Move aver mu TOWARD citizen pref:
            #   the politician pretends to dislike
            #   what citizens like less.
            self.ext_policy_aver.mu -= (
                pl * (avg.avg_Pcp.mu
                      - self.innate_policy_aver.mu))
            # Move trait AWAY from citizen trait
            #   aversion.
            self.ext_trait.mu -= (
                tl * (avg.avg_Tca.mu
                      - self.innate_trait.mu))

        # Refresh cached integration variables
        #   after any modification.
        self.ext_policy_pref.update_integration_variables()
        self.ext_policy_aver.update_integration_variables()
        self.ext_trait.update_integration_variables()


