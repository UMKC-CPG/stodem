# STODEM Design Document

## 1. Purpose and Scope

STODEM (Stochastic Democracy Simulation) is a multi-agent based
simulation that models democratic processes. The simulation
investigates whether stochastic voting can help a democracy navigate
a high-dimensional policy space and find alignment between the
internalized policy preferences of a population and the actual
(unknown) policies that lead to positive outcomes.

The simulation is not tied to any real-world political system. All
policy dimensions and personality traits are abstract and carry no
intrinsic meaning. "Extreme" and "centrist" are relative labels
determined only by the distribution of agents.

This document describes the architecture, mathematical foundations,
agent interactions, and simulation flow. It is intended to be read
alongside `TODO.md`, which tracks concrete implementation tasks.

---

## 2. Repository Layout

```
stodem/
  .stodem/              RC files (stodemrc.py, defaults)
  bin/                  Installed copies of Python modules
  build/release/        CMake build artifacts
  jobs/
    quickTest/          Small test case (stodem.in.xml, outputs)
    test1/              Larger test case
  src/
    scripts/            Primary source code (see Module Map)
  CMakeLists.txt        Build system (Fortran future expansion)
  CLAUDE.md             AI assistant guidance
  DESIGN.md             This document
  TODO.md               Bug tracker and implementation task list
```

---

## 3. Module Map

All simulation logic lives in `src/scripts/`. Each module has a
single clear responsibility, organized here by functional group.

**Entry Point and Control**

- `stodem.py` — Entry point. Contains `main()`, the
  simulation loop, `campaign()`, `vote()`, `govern()`.
- `settings.py` (`ScriptSettings`) — XML input parsing,
  command-line argument handling, RC file loading.
- `sim_control.py` (`SimControl`, `SimProperty`) — Simulation
  phase counts, total step computation, data range
  computation, property dataclass.

**World and Spatial Structure**

- `world.py` (`World`) — Top-level container. Creates
  patches, zones, citizens, politicians, government.
  Aggregates patch-level well-being for output.
- `zone.py` (`Zone`) — Geographic region at one hierarchy
  level. Maintains politician lists and citizen zone
  averages.
- `patch.py` (`Patch`) — Basic grid unit. Knows its (x,y)
  location, zone membership, and resident citizen indices.

**Agents**

- `citizen.py` (`Citizen`) — Agent with Gaussian policy/trait
  preferences, aversions, and ideal positions. Computes
  overlap integrals and voting logic.
- `politician.py` (`Politician`) — Agent with innate and
  external policy/trait positions. Strategies for movement,
  adaptation, and campaigning.
- `government.py` (`Government`) — Holds enacted policy
  Gaussians that affect citizen well-being.

**Mathematics and Infrastructure**

- `gaussian.py` (`Gaussian`) — Complex Gaussian
  representation with overlap integral computation.
- `random_state.py` — Global `rng` instance (numpy
  `default_rng`, seed=8675309).
- `output.py` (`Hdf5`, `Xdmf`) — HDF5 data output and XDMF
  metadata generation for Paraview visualization.

### Dependency Graph

```
stodem.py
  +-- settings.py
  +-- sim_control.py
  +-- world.py
  |     +-- patch.py
  |     +-- zone.py ----------+-- gaussian.py
  |     +-- citizen.py -------+-- gaussian.py
  |     +-- politician.py ----+-- gaussian.py
  |     +-- government.py ----+-- gaussian.py
  |     +-- sim_control.py (SimProperty)
  |     +-- random_state.py
  +-- output.py
```

All modules that use randomness import the shared `rng` from
`random_state.py` to ensure reproducibility from a single seed.

---

## 4. Mathematical Foundations

### 4.1 Complex Gaussians

The fundamental data structure is a one-dimensional complex
Gaussian defined on each policy or trait axis:

```
g(x; sigma, mu, theta) =
    1/(sigma * sqrt(2*pi)) * exp(-(x-mu)^2 / (2*sigma^2))
    * exp(i*theta)
```

Each Gaussian has three parameters:

- **mu (position)**: Location on the real number line. Where
  the agent stands on a policy or trait axis. Extreme vs.
  centrist is purely relative to other agents.

- **sigma (spread/standard deviation)**: Strength of attachment
  to a specific position. A narrow Gaussian implies strong
  attachment; a broad Gaussian implies many nearby variations
  are acceptable.

- **theta (orientation/engagement)**: Degree of engagement with
  an issue. theta=0 means fully real (fully engaged); theta=pi/2
  means fully imaginary (fully apathetic). The projection onto
  the real axis, cos(theta), determines how much weight the
  issue carries in overlap integrals.

Derived quantities used in integration:

```
alpha     = 1 / (2 * sigma^2)
cos_theta = cos(Im(theta))
```

### 4.2 Overlap Integral

The key relationship between any two Gaussians G1 and G2 is
their overlap integral, which measures alignment:

```
I(G1, G2) = (pi / zeta)^1.5 * exp(-xi * d^2)
            * cos(theta_1) * cos(theta_2)
```

where:

```
zeta = alpha_1 + alpha_2
xi   = 1 / (2 * zeta)       = 0.5 / zeta
d    = mu_1 - mu_2
```

Properties of the integral:

- Maximum value (+1) when both Gaussians have identical
  parameters.
- Minimum value (-1) when parameters match except theta values
  are 0 and pi respectively.
- When theta is restricted to [0, pi/2], the integral range
  is [0, +1] for same-sign Gaussians.
- Both agreement and disagreement produce large |I|; only weak
  overlap produces small |I|.

⚠️ **DESIGN QUESTION (Normalization)**: The current
implementation does not normalize the integral by dividing
by the geometric mean of self-overlaps. Without
normalization, the raw integral value may not be bounded
to [-1, +1] for arbitrary Gaussian parameters. Should
the integral be normalized? If so, should normalization
occur inside `Gaussian.integral()` or at the call sites?
(See also TODO #16.)

### 4.3 Gaussian Notation

Each agent type maintains specific Gaussians. The naming
convention uses a three-letter code:

| Code    | Meaning |
|---------|---------|
| `Pcp;n` | Policy citizen stated preference, dimension n |
| `Pca;n` | Policy citizen stated aversion, dimension n |
| `Pci;n` | Policy citizen ideal preference, dimension n |
| `Ppp;n` | Policy politician apparent preference, dimension n |
| `Ppa;n` | Policy politician apparent aversion, dimension n |
| `Pge;n` | Policy government enacted, dimension n |
| `Tcp;m` | Trait citizen stated preference, dimension m |
| `Tca;m` | Trait citizen stated aversion, dimension m |
| `Tpx;m` | Trait politician external personality, dimension m |

---

## 5. World Structure

### 5.1 Spatial Organization

The world is a two-dimensional grid of **patches**. Patches are
the atomic spatial unit. Each patch knows its (x,y) grid
coordinates and which zones it belongs to at every hierarchy level.

### 5.2 Zone Hierarchy

Zones are nested geographic regions organized in a hierarchy of
**zone types**. For example:

```
Zone Type 0: districts (smallest)
Zone Type 1: states (composed of districts)
Zone Type 2: countries (composed of states)
```

Each zone type is defined by:

- `x_sub_units`, `y_sub_units`: How many sub-units (patches or
  lower-level zones) compose one zone of this type in each
  dimension.
- `static`: Whether zone boundaries can change during
  simulation (flexible zones are defined but not yet
  implemented).
- Politician count parameters: `min_politicians`,
  `max_politicians`, `num_politicians_mean`,
  `num_politicians_stddev`.

The total world size in patches is the product of all zone type
sub-unit counts:

```
x_num_patches = product(zone_type[i].x_sub_units for all i)
y_num_patches = product(zone_type[i].y_sub_units for all i)
```

Each patch computes its own zone index for every zone type based
on its (x,y) position and the cumulative zone sizes. This allows
zones to be constructed by scanning patches and grouping those
with matching zone indices.

### 5.3 Zone Averages

Each zone computes and maintains average Gaussians across all
its citizens for:

- `avg_Pcp`: Average stated policy preference
- `avg_Pca`: Average stated policy aversion
- `avg_Tcp`: Average stated trait preference
- `avg_Tca`: Average stated trait aversion

These averages are recomputed each campaign step and are used
in citizen-citizen collective influence calculations.

---

## 6. Agent Design

### 6.1 Citizens

Each citizen occupies a single patch and belongs to one zone of
each zone type (determined by that patch's location).

**State variables (Gaussians)**:

- `stated_policy_pref`: Conscious policy position used for
  voting, comparing with politicians, and citizen-citizen
  interaction. Positive-valued. One Gaussian per policy
  dimension.
- `stated_policy_aver`: Conscious policy aversion. Negative-
  valued. Used alongside preference in scoring candidates.
- `ideal_policy_pref`: The citizen's true best-interest policy
  position, unknown to the citizen. Initialized near the stated
  preference with perturbation. Used to compute well-being
  against enacted policy. Always positive, always fully real
  (theta=0).
- `stated_trait_pref`: Personality trait affinity. Interacts
  with politicians' and other citizens' traits.
- `stated_trait_aver`: Personality trait aversion.

**Scalar state**:

- `policy_trait_ratio`: Weights policy vs. trait in candidate
  scoring. Drawn from N(0, stddev), clamped to [-0.5, +0.5].
  Produces weights: w_policy = 0.5 + ratio, w_trait = 0.5 -
  ratio.
- `well_being`: Currently computed as the overlap between
  ideal policy preference and enacted government policy:
  `sum(overlap(Pci, Pge))`. This simple implementation is
  a placeholder. A richer model incorporating resource
  accumulation, perceived satisfaction, community fit,
  and other factors is under design (§8.5). An earlier
  idea suggested a sensitivity function (e.g., A e^(-a x))
  where low well-being makes citizens more susceptible to
  political lies; this could be integrated into the final
  model.

  ⚠️ **DESIGN QUESTION (Well-Being Scope)**: Should the
  simple overlap-based well_being be kept as a first
  implementation with downstream effects on engagement,
  deferring the richer model from §8.5? Or should the
  richer model be designed fully before implementing any
  well-being downstream effects? (See also TODO #9.)
- `participation_prob`: Probability of voting, computed
  dynamically as mean(|cos(theta)|) across all stated
  Gaussians.
- `politician_list`: Politicians this citizen can vote for
  (populated each campaign).
- `politician_score`: Accumulated scores for each politician.

**Influence accumulation**: During each campaign step, six
shift arrays accumulate changes to the citizen's Gaussians
from all influence sources before application:

- `policy_orien_shift`, `policy_pos_shift`,
  `policy_stddev_shift`
- `trait_orien_shift`, `trait_pos_shift`,
  `trait_stddev_shift`

### 6.2 Politicians

Each politician is assigned to a specific zone at a specific
zone type level. Politicians compete for votes only within
their assigned zone.

**State variables (Gaussians)**:

- `innate_policy_pref` / `innate_policy_aver`: The
  politician's true policy positions. Currently static.
- `ext_policy_pref` / `ext_policy_aver`: The externally
  presented (apparent) policy positions. May differ from
  innate positions depending on strategy.
- `innate_trait` / `ext_trait`: Innate and externally
  presented personality traits.

**Scalar state**:

- `policy_influence` / `trait_influence`: Scale factors for
  how effectively the politician shifts citizen engagement.
- `policy_lie` / `trait_lie`: Propensity to misrepresent
  positions (not yet used).
- `pander`: Propensity to pander to citizen preferences (not
  yet used).
- `elected`: Whether this politician won the last election.
- `votes`: Vote count (reset each cycle).

**Strategies**: Each politician has three independently
selected strategies, chosen probabilistically from cumulative
distribution parameters in the input XML:

- `move_strategy`: How the politician moves between patches.
  Only strategy 0 (random patch in zone) is implemented.
- `adapt_strategy`: How the politician sets apparent positions.
  Only strategy 0 (present innate positions unchanged) is
  implemented.
- `campaign_strategy`: Selected but not yet used anywhere
  (TODO #11).

Only `move_strategy` 0 and `adapt_strategy` 0 are
implemented. Higher strategies are silently no-ops
(TODO #10, #10b).

### 6.3 Government

A single `Government` instance holds:

- `enacted_policy`: Gaussian for each policy dimension
  representing current government policy. Initialized
  randomly and currently static (govern phase is
  unimplemented — see §7.5 for design, TODO #5 for
  implementation).

---

## 7. Simulation Flow

### 7.1 Initialization

```
main()
  ScriptSettings()          Read RC file, parse command line
  settings.read_input_file()  Parse stodem.in.xml
  SimControl(settings)      Extract phase counts
  World(settings, sim_control)
    Create patches (2D grid)
    Create zones (scan patches, group by zone index)
  world.populate(settings)
    Create politicians (random patches within zones)
    Create Government
    Create citizens (assigned to patches)
  Xdmf / Hdf5              Initialize output files
```

### 7.2 Main Loop

The simulation runs for `num_cycles` iterations. Each cycle
consists of three phases (with two more planned):

```
for cycle in num_cycles:
    campaign(...)     # Multi-step campaigning phase
    vote(...)         # Single-step election
    govern(...)       # Multi-step governing phase (stub)
    # primary(...)    # Not implemented
    # primary_vote(...)  # Not implemented
```

### 7.3 Campaign Phase (Multi-Step)

The campaign phase is the most developed part of the
simulation. Each campaign consists of `num_campaign_steps`
time steps.

**One-time setup (start of campaign)**:

1. `world.repopulate_politicians()` — Elected politicians
   keep their positions; losers are replaced with fresh
   random politicians. All vote counts reset.
2. Each citizen clears their politician list.
3. Each politician presents itself to all citizens in its zone
   (adds itself to their `politician_list`).

**Per-step activities**:

1. **Compute zone averages** — Each zone aggregates its
   citizens' policy and trait Gaussians into zone-level
   averages.

2. **Politicians move** — Each politician moves to a new
   patch within its zone according to its move strategy.

3. **Politicians adapt** — Each politician sets its apparent
   (external) policy and trait positions according to its
   adapt strategy.

4. **Citizens compute overlaps** — Each citizen computes
   the full set of overlap integrals (§9):
   - Citizen-politician integrals (policy and trait)
   - Citizen-citizen zone average integrals (policy
     and trait)
   - Citizen-government integrals (policy only)

5. **Citizens prepare for influence** — Shift
   accumulation arrays are initialized to zero (§8.6).

6. **Politician influence** — Citizens accumulate
   engagement and policy/spread shifts from politician
   overlap integrals (§8.3, §8.6.2–§8.6.3).

7. **Well-being response** — Citizens compute
   well-being from ideal policy vs. enacted policy
   overlap (§8.5).

8. **Citizen collective influence** — Citizens
   accumulate engagement and policy/spread shifts from
   zone average overlap integrals (§8.4, §8.6.2–
   §8.6.4).

9. **Score candidates** — Each citizen computes a
   weighted score for each politician using policy and
   trait overlaps, weighted by `policy_trait_ratio`
   (§10.1).

10. **Apply influence shifts** — Accumulated shifts are
    applied to each citizen's Gaussian parameters in a
    single pass, followed by engagement decay and
    derived variable updates (§8.6.5–§8.6.7). **Not
    yet implemented** (TODO #8).

11. **Aggregate and output** — Patch-level well-being
    is computed and written to HDF5.

### 7.4 Vote Phase

The vote phase runs once per cycle after the campaign:

1. **Vote probability**: Each citizen computes P(vote) =
   mean(|cos(theta)|) across all stated Gaussians. A random
   draw against this probability determines participation.

2. **Candidate selection**: For each zone the citizen belongs
   to, the citizen selects the politician with the highest
   score (computed during the last campaign step).

3. **Election**: Each zone determines the top vote-getter
   and marks them as elected via
   `zone.set_elected_politician()`.

### 7.5 Govern Phase

The govern phase runs for `num_govern_steps` time steps
after each election. Currently a stub that returns
immediately (TODO #5). The design below describes the
intended mechanics.

#### 7.5.1 Policy Force Model

Each govern step, enacted policy (`Pge`) shifts under the net
force exerted by all elected politicians. Each politician has
two policy positions — innate (what they actually believe) and
external (what they promised voters). These pull in potentially
different directions, and the balance between them determines
what actually happens during governance.

**Effective governing target per politician:**

The politician's existing `pander` parameter controls how much
they govern according to their promises vs. their ideology:

```
target_mu = pander * ext_policy_pref.mu
          + (1 - pander) * innate_policy_pref.mu

target_sigma = pander * ext_policy_pref.sigma
             + (1 - pander) * innate_policy_pref.sigma
```

When innate and external positions are close (an honest
politician), the distinction is moot. When they differ (a
liar), `pander` controls whether the lie carries through into
governance or was purely electoral.

**Force on enacted policy:**

```
for each policy dimension n:
    force_mu = 0
    force_sigma = 0

    for each elected politician:
        target = effective_target(politician)
        w = political_capital(politician, n)
        force_mu += w * influence * (target.mu - Pge.mu)
        force_sigma += w * influence * (target.sigma
                                        - Pge.sigma)

    Pge.mu += governance_rate * force_mu
    Pge.sigma += governance_rate * force_sigma
```

The `policy_influence` parameter (already on each politician)
scales how effectively they move policy. The
`governance_rate` provides institutional inertia — policy
changes gradually regardless of how strong the forces are.
The `governance_rate` is stored as a variable so that it can
be made dynamic in the future (e.g., modulated by incumbency,
supermajority conditions, or crisis).

**Aversion forces**: Politicians also have policy aversions
(innate and external). These exert a repulsive force — pushing
`Pge` *away* from the politician's aversion positions, using
the same influence, political capital, and rate scaling. A
politician can shape policy both by what they want and by what
they actively oppose.

#### 7.5.2 Political Capital

Politicians cannot affect all policies equally. Each
politician has a finite amount of political capital that must
be distributed across policy dimensions. The distribution is
determined by the politician's own Gaussian parameters:

- **Narrow sigma dimensions get more capital**: A politician
  with a narrow (sharp) innate preference on a policy
  dimension has strong attachment to a specific position on
  that issue. They will focus their capital there.
- **Strong aversion dimensions get more capital**: If the
  enacted policy overlaps strongly with a politician's
  aversion on some dimension, that dimension is urgent — the
  politician spends capital to push policy away from what
  they oppose.

A candidate formulation for the capital weight on dimension n:

```
raw_n = 1/sigma_innate_n + |overlap(Ppa_n, Pge_n)|
capital_n = raw_n / sum(raw_n for all dimensions)
```

The first term gives more weight to narrow (strongly held)
positions. The second term gives more weight to dimensions
where enacted policy currently aligns with the politician's
aversion — urgent problems demand attention. The
normalization ensures capital sums to 1 across all dimensions,
so a politician who focuses heavily on one policy necessarily
has less influence on others.

This creates emergent strategic behavior without modeling
strategy explicitly: a politician with broadly held positions
spreads their capital thin and moves each dimension only
slightly, while a politician with one sharp preference and
one urgent aversion concentrates force on those dimensions.

#### 7.5.3 Zone Population Weighting

Not all elected politicians carry equal institutional weight.
A politician elected from a large zone (many citizens)
represents more constituents and should have proportionally
more ability to effect change than one from a small zone.
The population of the zone affects the relative weight of
each politician's force contribution.

#### 7.5.4 Downstream Effects

Once `Pge` shifts, existing mechanisms produce feedback:

1. `overlap(Pci, Pge)` changes — citizen well-being changes.
2. `overlap(Pcp, Pge)` changes — perceived satisfaction
   changes.
3. Well-being feeds into engagement (via
   `build_response_to_well_being()`).
4. Engagement affects vote probability.
5. Vote probability affects who wins next cycle.

A governing phase that drifts policy away from citizens'
ideal positions will erode well-being, change voting
behavior, change who gets elected, and change the forces on
policy. The feedback loop closes naturally.

#### 7.5.5 Emergent Behaviors

The force model produces several interesting dynamics without
special-casing them:

- **Gridlock**: When elected politicians pull in opposing
  directions, forces partially cancel and policy moves slowly
  or not at all.
- **Rapid change**: When politicians align, forces reinforce
  and policy moves quickly (still bounded by
  `governance_rate`).
- **Broken promises**: A low-pander politician governs based
  on ideology, not promises. Citizens experience the
  consequences through well-being and eventually vote
  differently.
- **Focused politicians**: Narrow-sigma politicians
  concentrate capital on a few dimensions and move those
  effectively. Broad-sigma politicians spread thin and change
  little.
- **Reactive governance**: High aversion overlap with enacted
  policy redirects capital toward urgent dimensions, creating
  a natural prioritization mechanism.

---

## 8. Interaction Physics

### 8.1 Fundamental Principle: Trait Gates Policy

A consistent rule governs how citizens' policy positions
shift in response to any influence source (politician or
citizen collective):

- The **magnitude** of trait alignment (|trait_sum|)
  determines how much the policy shift is.
- The **sign** of trait alignment determines what kind of
  shift occurs.

```
trait_sum = sum of trait overlap integrals with the source
magnitude = |trait_sum|   --> amount of shift
sign      = sign(trait_sum)  --> type of shift
```

**Positive trait overlap** (citizen likes the source):

- Policy mu shifts toward the source's mu, proportional
  to magnitude.
- Policy sigma shifts toward the source's sigma,
  proportional to magnitude.

**Negative trait overlap** (citizen dislikes the source):

- No preference mu movement.
- Preference sigma narrows (citizen becomes more rigid),
  proportional to magnitude.
- Aversion mu shifts toward the source's policy positions
  (targeted backlash), proportional to magnitude times
  a `defensive_ratio` parameter.

Physical interpretation: People who feel personality affinity
with an influence source are susceptible to adopting that
source's policy positions. People who feel personality aversion
become defensive, dig in on their preferences, and develop
targeted aversions to the disliked source's specific
policies.

For the detailed shift formulas, see §8.6.3. For
engagement effects, see §8.6.2.

### 8.2 Engagement Dynamics

**Engagement from overlap integrals**: Both agreement and
disagreement increase engagement. The |absolute value| of
each overlap integral shifts the corresponding citizen
Gaussian's theta toward real (toward theta=0, more engaged).
A politician who advocates for a policy you have a strong
aversion to will make you engage to fight against them, just
as a politician who aligns with your preferences will make
you engage in support.

**Engagement decay**: Every simulation step, every citizen's
theta for every Gaussian drifts toward pi/2 (fully imaginary /
fully apathetic) by a constant `engagement_decay_rate`. Without
active campaigning or citizen-citizen interaction, citizens
gradually disengage from all issues. This creates a fundamental
tension: campaigns must actively maintain engagement, not just
create it once. The decay rate is stored as a variable
for future dynamic behavior.

For the precise decay formula, see §8.6.6. For the
new parameter, see §8.6.8.

### 8.3 Politician-Driven Influence

**Engagement shifts**: When a politician and citizen share a
patch, the comprehensive citizen-politician integration set is
computed. Each |overlap integral| shifts the corresponding
citizen Gaussian's theta toward real:

```
|I(Pcp,Ppp)| --> shifts Pcp theta toward real
|I(Pca,Ppa)| --> shifts Pca theta toward real
|I(Pcp,Ppa)| --> shifts Pcp theta toward real
|I(Pca,Ppp)| --> shifts Pca theta toward real
|I(Tcp,Tpx)| --> shifts Tcp theta toward real
|I(Tca,Tpx)| --> shifts Tca theta toward real
```

The politician's `policy_influence` and `trait_influence`
parameters scale these shifts.

**Policy position and spread shifts**: Follow the "trait
gates policy" principle (Section 8.1). The trait sum between
citizen and politician determines whether the citizen's policy
Gaussians attract toward or rigidify against the politician's
positions.

**No direct trait shifts**: Politicians do not directly
change citizen trait preferences. Citizen traits are
influenced only through citizen-citizen interactions. If a
politician converts citizens' policy views, those citizens
then influence other citizens' traits through the collective
mechanism. Politician influence on citizen traits is thus
indirect and emergent.

For detailed shift formulas, see §8.6.2 (engagement)
and §8.6.3 (position/spread). Implementation status:
TODO #19, #21.

### 8.4 Citizen-Driven Influence

**Policy acclimatization**: Citizens unconditionally
acclimatize toward the policy views of their community (zone
average). The rate follows the same "trait gates policy"
principle: trait overlap with the zone average determines how
much policy positions shift toward the zone average.

**Trait acclimatization**: Citizen traits acclimatize toward
the zone average trait values. This is the sole mechanism by
which traits change. The rate is governed by trait overlap
integrals themselves (traits gate their own movement, unlike
the cross-domain gating of policy by traits).

**Engagement**: Citizen-citizen overlaps also affect engagement
following the same absolute-value rule as politician-driven
engagement (§8.6.2).

For detailed trait acclimatization formulas, see
§8.6.4. Implementation status: TODO #20, #21.

### 8.5 Well-Being, Resource, and Resentment

These concepts are under active design discussion. Current
thinking:

**Perceived satisfaction** (relatively settled): Direct overlap
between stated policy preferences and enacted policy:
overlap(Pcp, Pge). Represents how satisfied the citizen *feels*,
regardless of whether the policy actually benefits them.

**Resource** (candidate concept): An abstract economic stock per
citizen that accumulates based on alignment between ideal policy
and enacted policy:

```
resource(t+1) = clamp(
    resource(t) + alpha * overlap(Pci, Pge, t),
    floor, ceiling)
```

Key properties: inertia/lag (enacted policy changes the rate,
not the level directly), asymmetric dynamics (depletion faster
than accumulation), diminishing returns (concave mapping to
well-being).

**Well-being** (candidate model): Composite scalar with
candidate inputs: resource, perceived satisfaction, community
fit (trait overlap with zone averages), policy consistency
(stated vs. ideal preference alignment), and policy stability
(variance of Pge over a rolling window). Exact functional form
is TBD.

**Resentment** (candidate concept): Driven by gap between a
citizen's resource and the zone average resource. Asymmetric:
being below average generates resentment; being above does not
produce an inverse effect. Separate from well-being and
engagement. May affect `policy_trait_ratio` (resentful citizens
vote more on personality than policy), aversion intensity, and
trait shifts.

**Three independent axes**: Engagement (theta), happiness/
well-being, and resentment are orthogonal. An unhappy
citizen can be engaged or disengaged. A resentful citizen
can have moderate well-being.

⚠️ **DESIGN QUESTION (Well-Being Model)**: The concepts
in this section — resource, resentment, and the composite
well-being function — are candidate models. Before
implementation, the following decisions are needed:

1. **Resource**: Should the resource model be
   implemented? If so, what are the accumulation rates
   (α_gain, α_loss), floor, and ceiling values? Should
   resource be per-citizen or per-patch?

2. **Well-being composite**: What is the functional form
   of `f(resource, perceived_satisfaction,
   community_fit, policy_consistency,
   policy_stability)`? How should the inputs be
   weighted?

3. **Resentment**: Should resentment be implemented as a
   separate scalar? If so, which behavioral channels
   should it affect (policy_trait_ratio, aversion
   intensity, trait shifts)?

4. **Implementation order**: Should the simple
   overlap-based well-being (`overlap(Pci, Pge)`) be
   kept as a first step with downstream effects on
   engagement, deferring the richer model? Or should
   the full model be designed before implementing any
   downstream effects? (See also §6.1 and TODO #9.)

### 8.6 Influence Application

Each campaign step proceeds in two phases: accumulation and
application. During accumulation, all influence sources
(politicians, citizen collective, well-being response) compute
their contributions and add them to per-citizen shift arrays.
During application, the accumulated shifts modify the citizen's
Gaussian parameters in a single pass. This two-phase pattern
prevents order-of-evaluation artifacts: the sequence in which
influence sources are processed does not affect the outcome.

Ideal policy preferences (Pci) are never subject to influence
shifts or engagement decay. They represent the citizen's true
(unknown) interest and remain static throughout the simulation.

#### 8.6.1 Shift Array Structure

Separate arrays are needed for preference and aversion
Gaussians because they respond differently under negative
trait alignment (Section 8.1). Each citizen maintains twelve
shift arrays per campaign step:

```
Pcp:  theta_shift[n],  mu_shift[n],  sigma_shift[n]
Pca:  theta_shift[n],  mu_shift[n],  sigma_shift[n]
Tcp:  theta_shift[m],  mu_shift[m],  sigma_shift[m]
Tca:  theta_shift[m],  mu_shift[m],  sigma_shift[m]
```

All arrays are initialized to zero at the start of each step.
The current code uses six arrays (three per domain);
these must be expanded to twelve (TODO #23). The
current accumulation code also has bugs that must be
fixed first (TODO #19, #20).

#### 8.6.2 Engagement (Theta) Accumulation

Each |overlap integral| contributes to the theta shift of the
citizen Gaussian it involves. All contributions are
non-negative — both agreement and disagreement drive
engagement.

From each politician p (f_pol = policy_influence,
f_trait = trait_influence):

```
Pcp.theta_shift[n] += f_pol * (|I(Pcp,Ppp)[n]|
                              + |I(Pcp,Ppa)[n]|)
Pca.theta_shift[n] += f_pol * (|I(Pca,Ppa)[n]|
                              + |I(Pca,Ppp)[n]|)
Tcp.theta_shift[m] += f_trait * |I(Tcp,Tpx)[m]|
Tca.theta_shift[m] += f_trait * |I(Tca,Tpx)[m]|
```

From zone averages (no external scaling factor):

```
Pcp.theta_shift[n] += |I(Pcp,avg_Pcp)[n]|
                     + |I(Pcp,avg_Pca)[n]|
Pca.theta_shift[n] += |I(Pca,avg_Pca)[n]|
                     + |I(Pca,avg_Pcp)[n]|
Tcp.theta_shift[m] += |I(Tcp,avg_Tcp)[m]|
                     + |I(Tcp,avg_Tca)[m]|
Tca.theta_shift[m] += |I(Tca,avg_Tca)[m]|
                     + |I(Tca,avg_Tcp)[m]|
```

#### 8.6.3 Policy Position and Spread Accumulation

Follows the trait-gates-policy principle (Section 8.1).
However, the shift magnitude should NOT be proportional to
the distance between source and citizen positions. A
spring-like formula (shift ~ source_mu - citizen_mu) would
make distant positions attract more strongly, which is not
the intended physics. Instead, trait alignment drives the
magnitude, and the citizen's own Gaussian parameters
determine susceptibility to movement.

**Susceptibility**: A citizen's resistance to having a
position or spread shifted depends on two properties of
the Gaussian being shifted:

- **Sigma (spread)**: A narrow Gaussian implies strong
  attachment to a specific position. Narrow positions are
  harder to move. Broader positions are easier to move.
  Susceptibility increases with sigma.

- **Theta (engagement)**: A disengaged citizen (theta near
  pi/2) holds positions loosely and is more susceptible to
  influence. An engaged citizen (theta near 0) has thought
  about the issue and resists movement. Susceptibility
  increases with theta.

A candidate susceptibility function:

```
S(sigma, theta) = sigma * (1 - c * cos(theta))
```

At full engagement (theta=0): S = sigma * (1-c) — reduced
but nonzero (even an engaged citizen can be influenced).
At full apathy (theta=pi/2): S = sigma — maximum
susceptibility. The parameter c (0 < c < 1) controls how
much engagement protects against position shifts.

**Shift direction**: The direction of movement is toward
the source, independent of distance:

```
direction = sign(source_mu[n] - citizen_mu[n])
```

**From each politician p:**

Compute the trait sum across all trait dimensions:

```
trait_sum = sum_m( I(Tcp,Tpx)[m] + I(Tca,Tpx)[m] )
mag       = |trait_sum|
f         = policy_influence
```

If trait_sum >= 0 (citizen likes politician):

```
for each policy dim n:
    Pcp.mu_shift[n]    += mag * f
        * S(Pcp.sigma[n], Pcp.theta[n])
        * sign(Ppp.mu[n] - Pcp.mu[n])
    Pcp.sigma_shift[n] += mag * f
        * S(Pcp.sigma[n], Pcp.theta[n])
        * sign(Ppp.sigma[n] - Pcp.sigma[n])
    Pca.mu_shift[n]    += mag * f
        * S(Pca.sigma[n], Pca.theta[n])
        * sign(Ppa.mu[n] - Pca.mu[n])
    Pca.sigma_shift[n] += mag * f
        * S(Pca.sigma[n], Pca.theta[n])
        * sign(Ppa.sigma[n] - Pca.sigma[n])
```

If trait_sum < 0 (citizen dislikes politician):

```
for each policy dim n:
    Pcp.sigma_shift[n] += mag * f
        * S(Pcp.sigma[n], Pcp.theta[n])
        * sign(sigma_floor - Pcp.sigma[n])
    Pca.mu_shift[n]    += mag * f * defensive_ratio
        * S(Pca.sigma[n], Pca.theta[n])
        * sign(Ppp.mu[n] - Pca.mu[n])
```

Note: in the defensive case the citizen's aversion mu shifts
toward the politician's *preference* positions (Ppp), not the
politician's aversion. The citizen develops an aversion to
what the disliked politician stands *for*.

**From zone averages (citizen collective):**

Compute the trait sum using all four citizen-citizen trait
integrals (the citizen collective has separate pref and aver
Gaussians, unlike a politician's single trait Gaussian):

```
trait_sum = sum_m( I(Tcp,avg_Tcp)[m]
                 + I(Tca,avg_Tca)[m]
                 + I(Tcp,avg_Tca)[m]
                 + I(Tca,avg_Tcp)[m] )
mag       = |trait_sum|
```

If trait_sum >= 0: policy pref and aver mu and sigma shift
toward the zone average policy values. Same formulas as the
politician case with f = 1, zone averages replacing
politician Gaussians, and susceptibility S applied to the
citizen Gaussian being shifted.

If trait_sum < 0: same defensive response — pref sigma
narrows toward sigma_floor, aver mu shifts toward zone
average policy pref positions (scaled by
mag * defensive_ratio), all modulated by susceptibility.

**Candidate alternative — force/momentum dynamics:**

The susceptibility model above is memoryless: each step's
shift depends only on the current state. A richer
alternative introduces Newtonian dynamics where influence
produces a *force* rather than a direct shift:

```
mass[n]         = f(1/sigma[n], cos(theta[n]))
force[n]        = mag * influence * direction[n]
acceleration[n] = force[n] / mass[n]
velocity[n]    += acceleration[n]
mu[n]          += velocity[n]
velocity[n]    *= (1 - damping)
```

Mass depends inversely on sigma and proportionally on
cos(theta): narrow + engaged = heavy = hard to accelerate.
This naturally captures both susceptibility factors.
Momentum means past influences have lingering effects — a
citizen pushed in one direction for several steps continues
drifting even after the influence stops. Damping prevents
runaway velocity.

Trade-offs: The dynamics model adds a velocity state
variable per Gaussian per dimension, plus mass and damping
parameters. It produces smoother, more physically motivated
trajectories but is more complex to tune. The overlap
integrals already encode some distance sensitivity (far-apart
Gaussians have weaker overlap), so the force model layers on
top of that existing physics.

⚠️ **DESIGN QUESTION (Shift Model)**: Either the
susceptibility model or the force/momentum model could
be implemented first. The susceptibility model is
simpler (no additional state variables) but memoryless.
The dynamics model adds velocity per Gaussian per
dimension and produces smoother trajectories with
inertia. Which model should be implemented first?
(See also §8.6.9 for related questions, TODO #8 and
#21 for implementation.)

#### 8.6.4 Trait Position and Spread Accumulation

Trait acclimatization occurs only through the citizen
collective (Section 8.4 — politicians do not directly shift
citizen traits). It is unconditional: trait Gaussians always
shift toward the zone average, regardless of the sign of
trait overlap. The magnitude of trait overlap controls only
the rate.

The same susceptibility considerations from Section 8.6.3
apply: narrow, engaged trait Gaussians resist movement more
than broad, disengaged ones.

```
for each trait dim m:
    Tcp.mu_shift[m]    += |trait_sum|
        * S(Tcp.sigma[m], Tcp.theta[m])
        * sign(avg_Tcp.mu[m] - Tcp.mu[m])
    Tcp.sigma_shift[m] += |trait_sum|
        * S(Tcp.sigma[m], Tcp.theta[m])
        * sign(avg_Tcp.sigma[m] - Tcp.sigma[m])
    Tca.mu_shift[m]    += |trait_sum|
        * S(Tca.sigma[m], Tca.theta[m])
        * sign(avg_Tca.mu[m] - Tca.mu[m])
    Tca.sigma_shift[m] += |trait_sum|
        * S(Tca.sigma[m], Tca.theta[m])
        * sign(avg_Tca.sigma[m] - Tca.sigma[m])
```

Here trait_sum is the same citizen-collective trait sum
computed in Section 8.6.3. Even a citizen with negative
trait overlap slowly drifts toward community trait norms;
they simply drift more slowly than a citizen who fits in.

If the force/momentum model (Section 8.6.3) is adopted,
trait shifts would use the same dynamics framework with
per-trait-dimension velocities and mass.

#### 8.6.5 Application Step

After all sources have accumulated their contributions, the
twelve shift arrays are applied to the citizen's Gaussian
parameters in a single pass.

**Position:**

```
mu_new = mu + mu_shift
```

Mu is unbounded on the real line.

**Spread:**

```
sigma_new = max(sigma + sigma_shift, sigma_floor)
```

sigma_floor prevents degenerate Gaussians (division by zero
in alpha = 1/(2*sigma^2)) and ensures every Gaussian retains
finite width.

**Engagement:**

```
theta_new = theta - theta_shift
```

The theta_shift is non-negative, so subtraction always drives
theta toward zero (more engaged). Engagement decay is then
applied before clamping (Section 8.6.6).

#### 8.6.6 Engagement Decay

After influence-driven engagement shifts, a constant decay
pulls every citizen's theta for every stated Gaussian toward
pi/2 (full apathy):

```
theta = clamp(theta + engagement_decay_rate, 0, pi/2)
```

engagement_decay_rate is a positive simulation parameter.
Without active influence, citizens disengage from all issues
over time. The rate is stored as a variable for future
dynamic behavior (e.g., modulated by well-being or crisis).

#### 8.6.7 Post-Application Update

After all parameters are modified and clamped, recompute the
derived quantities needed for the next step's overlap
integrals:

```
alpha     = 1 / (2 * sigma^2)
cos_theta = cos(theta)
```

#### 8.6.8 New Parameters

This section introduces simulation parameters not yet in the
XML configuration:

| Parameter | Purpose | Initial |
|---|---|---|
| `engagement_decay_rate` | Per-step theta drift toward pi/2 | TBD |
| `defensive_ratio` | Scales targeted backlash mu shift | 1.0 |
| `sigma_floor` | Minimum sigma for all Gaussians | TBD |
| `engagement_protection` | c in S(); how much engagement resists shifts | TBD |

If the force/momentum model is adopted, additional
parameters would be needed:

| Parameter | Purpose | Initial |
|---|---|---|
| `damping` | Velocity decay per step (0 = no friction, 1 = no momentum) | TBD |

The mass function f(1/sigma, cos(theta)) would also need
a specific functional form and possibly scaling parameters.

#### 8.6.9 Open Design Questions

The following questions must be resolved before the
shift mechanics can be fully implemented (TODO #8,
#21). The susceptibility vs. dynamics question (Q1)
is the most fundamental — it determines the overall
shift architecture. The remaining questions are
refinements that can be resolved during or after
initial implementation.

⚠️ **DESIGN QUESTION (Q1 — Shift model choice)**:
Susceptibility model (§8.6.3) vs. force/momentum
model (§8.6.3)? The susceptibility model is
memoryless and simpler; the dynamics model adds
velocity state and inertia. See §8.6.3 for the full
description of both.

⚠️ **DESIGN QUESTION (Q2 — Susceptibility function
form)**: If the susceptibility model is chosen, which
formula for S(sigma, theta)?
- `S = sigma * (1 - c * cos(theta))` — nonzero at
  full engagement (controlled by c)
- `S = sigma * sin(theta)` — zero at full engagement
  (fully engaged citizens are immovable)
- Product of separate sigma/theta terms with
  different functional forms

The choice depends on whether full engagement should
make a citizen nearly immovable or merely resistant.

⚠️ **DESIGN QUESTION (Q3 — Defensive narrowing
rate)**: The defensive sigma shift uses
`sign(sigma_floor - sigma)`, giving constant-rate
narrowing. With susceptibility, broader Gaussians
narrow faster (they are more susceptible). Does this
produce the right behavior, or should the narrowing
rate be independent of susceptibility?

⚠️ **DESIGN QUESTION (Q4 — Citizen-collective
scaling)**: Politician influence is scaled by
`policy_influence` / `trait_influence`. Zone average
influence currently has no scaling factor (f = 1).
Should a `collective_influence_rate` parameter be
added for tuning community vs. politician influence
strength?

⚠️ **DESIGN QUESTION (Q5 — Trait self-gating
rate)**: Trait acclimatization uses |trait_sum| (net
magnitude). If positive and negative overlaps
partially cancel, the net could be small even when
individual overlaps are large. Alternative:
`sum(|individual overlaps|)`, where any interaction
accelerates drift. The |net sum| form is used for
policy-case consistency, but the behavioral
difference is significant.

⚠️ **DESIGN QUESTION (Q6 — Sigma shift direction)**:
Current formulas use `sign(source_sigma -
citizen_sigma)` for constant-magnitude steps. An
alternative makes shift magnitude depend on the
spread difference (hybrid between sign-only and full
proportional). Same directional question as for mu
shifts, applied to sigma.

---

## 9. Overlap Integral Catalog

### 9.1 Citizen-Politician Integrals

Computed for each citizen against each politician in their
`politician_list`:

| Integral | Components | Sign by Construction |
|---|---|---|
| `Pcp_Ppp_ol` | Citizen pref vs. politician pref | Positive |
| `Pca_Ppa_ol` | Citizen aver vs. politician aver | Positive (negatives cancel) |
| `Pcp_Ppa_ol` | Citizen pref vs. politician aver | Negative |
| `Pca_Ppp_ol` | Citizen aver vs. politician pref | Negative |
| `Tcp_Tpx_ol` | Citizen trait pref vs. politician trait | Varies |
| `Tca_Tpx_ol` | Citizen trait aver vs. politician trait | Varies |

### 9.2 Citizen-Citizen (Zone Average) Integrals

Computed for each citizen against each zone's average
Gaussians:

| Integral | Components |
|---|---|
| `Pcp_Pcp_ol` | Citizen policy pref vs. zone avg policy pref |
| `Pca_Pca_ol` | Citizen policy aver vs. zone avg policy aver |
| `Pcp_Pca_ol` | Citizen policy pref vs. zone avg policy aver |
| `Pca_Pcp_ol` | Citizen policy aver vs. zone avg policy pref |
| `Tcp_Tcp_ol` | Citizen trait pref vs. zone avg trait pref |
| `Tca_Tca_ol` | Citizen trait aver vs. zone avg trait aver |
| `Tcp_Tca_ol` | Citizen trait pref vs. zone avg trait aver |
| `Tca_Tcp_ol` | Citizen trait aver vs. zone avg trait pref |

### 9.3 Citizen-Government Integrals

Computed once per citizen (single government):

| Integral | Components | Purpose |
|---|---|---|
| `Pcp_Pge_ol` | Stated pref vs. enacted | Perceived satisfaction |
| `Pca_Pge_ol` | Stated aver vs. enacted | Policy frustration |
| `Pci_Pge_ol` | Ideal pref vs. enacted | True well-being |

---

## 10. Candidate Scoring and Voting

### 10.1 Scoring

Each citizen scores each politician by computing a weighted
sum of policy and trait overlap integrals:

```
policy_sum = sum(Pcp_Ppp + Pca_Ppa + Pcp_Ppa + Pca_Ppp)
trait_sum  = sum(Tcp_Tpx + Tca_Tpx)

w_policy = 0.5 + policy_trait_ratio
w_trait  = 0.5 - policy_trait_ratio

score = w_policy * policy_sum + w_trait * trait_sum
```

The `policy_trait_ratio` is clamped to [-0.5, +0.5], ensuring
both weights are non-negative and sum to 1.

### 10.2 Vote Probability

```
P(vote) = mean(|cos(theta)|)
```

over all stated Gaussians (policy pref, policy aver, trait
pref, trait aver). A citizen whose Gaussians are mostly real
is highly engaged and very likely to vote. A citizen whose
Gaussians are mostly imaginary is disengaged and unlikely to
vote.

Future extension: a discriminability term based on the score
gap between the top two candidates could multiply the
engagement-based probability.

### 10.3 Election

Each zone holds an independent election. The politician with
the most votes in a zone is marked as elected. The zone
stores a reference to its elected politician. All other
politicians in that zone are marked as not elected.

---

## 11. Configuration (stodem.in.xml)

The XML input file defines all simulation parameters. Key
sections:

### sim_control

| Parameter | Purpose |
|---|---|
| `num_cycles` | Number of campaign-vote-govern cycles |
| `num_campaign_steps` | Time steps per campaign phase |
| `num_govern_steps` | Time steps per govern phase |
| `num_primary_campaign_steps` | Time steps per primary (unimplemented) |
| `data_resolution` | Data points per real number for output |
| `data_neglig` | Negligibility threshold for data range |

### world

| Parameter | Purpose |
|---|---|
| `patch_size` | Spatial extent of a patch (currently unused in movement) |
| `num_policy_dims` | Number of abstract policy dimensions |
| `num_trait_dims` | Number of abstract trait dimensions |
| `num_zone_types` | Number of hierarchy levels |
| `zone_type_N` | Per-type: name, sub-units, static flag, politician count distribution |

### citizens

Standard deviations for initializing citizen Gaussians
(position and spread) for each of: policy pref, policy aver,
ideal policy, trait pref, trait aver. Also
`policy_trait_ratio_stddev`.

### politicians

Standard deviations for initializing politician Gaussians.
Also: influence, lie, and pander standard deviations, and
cumulative strategy probability distributions for move,
adapt, and campaign strategies.

### government

Standard deviations for initializing enacted policy position
and spread.

---

## 12. Output System

### 12.1 HDF5

Binary data file with gzip compression. Organized as:

```
<outfile>.hdf5
  /CitizenGeoData/
    WellBeing0    (x_patches x y_patches)
    WellBeing1
    ...
```

Each dataset corresponds to one time step. The `Hdf5` class
creates groups at initialization and adds datasets
incrementally as the simulation runs.

### 12.2 XDMF

XML metadata file that describes the HDF5 data layout for
Paraview visualization. Written entirely at initialization
time based on `total_num_steps`. This creates a potential
mismatch if the simulation terminates early (see TODO #18).

Structure: a temporal collection grid containing one uniform
grid per time step, each with 2DCoRectMesh topology and
ORIGIN_DXDY geometry pointing to the HDF5 datasets.

---

## 13. Randomness and Reproducibility

All random number generation uses a single shared `rng`
instance (numpy `default_rng` with seed 8675309), imported
from `random_state.py` by every module that needs randomness.
This ensures full reproducibility given the same seed and
execution order.

---

## 14. Build System

CMake is used for installation (copying scripts to
`$STODEM_DIR/bin`) and for future Fortran expansion. The
primary simulation is pure Python and does not require
compilation. Dependencies: Python 3, numpy, h5py, lxml.
Optional: Fortran compiler (gfortran or ifort) for future
expansion.

---

## 15. Known Architectural Issues

These issues are tracked in `TODO.md`. Brief summary
with cross-references:

- **Influence shifts never applied** — Campaign
  interactions currently have no lasting effect on
  citizen state (TODO #8). Accumulation code also has
  bugs (TODO #19, #20).
- **Class-level mutable state** — `World`, `SimControl`,
  `Hdf5` use class-level lists that would cause
  cross-instance contamination (TODO #17).
- **XDMF/HDF5 mismatch** — XDMF written eagerly for
  all steps; early termination leaves dangling
  references (TODO #18).
- **Settings via deep dictionary indexing** — Parameters
  extracted as
  `settings.infile_dict[1]["section"]["param"]`
  throughout the codebase rather than named attributes.
  Not yet tracked as a TODO item.

---

## 16. Design Principles

1. **Abstract dimensions**: All policy and personality trait
   dimensions are abstract. The simulation makes no assumption
   about what they represent. Meaning emerges from agent
   interactions.

2. **Engagement as a first-class quantity**: Engagement
   (theta) is not a separate boolean or scalar. It is built
   into the Gaussian representation itself via the complex
   orientation, making it inseparable from the agent's position
   and spread on every issue.

3. **Trait gates policy**: Personality determines
   susceptibility to policy influence. This creates rich
   emergent dynamics where politicians can shift policy
   views only through citizens who find them personally
   agreeable, while alienated citizens become more rigid.

4. **Symmetric engagement from asymmetric alignment**: Both
   agreement and disagreement increase engagement. Only
   indifference (weak overlap) leaves engagement unchanged.
   This prevents the common modeling pitfall where only
   positive interactions create participation.

5. **Separation of ideal and stated preferences**: Citizens
   can hold stated preferences that do not serve their actual
   interests. The gap between stated and ideal preferences
   creates the core tension of the simulation: democracy
   must navigate this gap to find good outcomes.

6. **Accumulate-then-apply influence**: Influence from all
   sources (politicians, citizen collective, well-being) is
   accumulated into shift arrays before being applied. This
   prevents order-of-evaluation artifacts where the first
   influence source processed has a different effect than the
   last.
