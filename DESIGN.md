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

  **Sign convention**: Preference Gaussians (Pcp, Tcp) keep
  Im(theta) in [0, pi/2), giving cos(theta) > 0 (positive-
  valued). Aversion Gaussians (Pca, Tca) keep Im(theta) in
  (pi/2, pi], giving cos(theta) < 0 (negative-valued). This
  encodes semantic attraction vs. repulsion without special-
  casing: same-type integrals (pref x pref, aver x aver) are
  non-negative while cross-type integrals (pref x aver) are
  non-positive. Citizens and politicians are initialized at
  theta = 1j for preferences (cos(1) ≈ 0.54) and theta =
  (pi-1)j for aversions (cos(pi-1) = -cos(1) ≈ -0.54),
  matching engagement magnitude with opposite sign. The ideal
  policy Gaussian (Pci) is initialized at theta = 1+0j
  (Im = 0, cos = 1, fully engaged, positive-valued).
  Politician traits (Tpx, innate and external) are positive-
  valued; politicians have no trait aversions.

  **Initial theta distribution**: Theta is currently
  hardcoded at initialization (1j for preferences,
  (pi-1)j for aversions). The `*_orien_stddev` XML
  parameters (e.g., `policy_pref_orien_stddev`) are
  reserved for a future extension in which the initial
  Im(theta) is drawn from a configurable distribution,
  allowing a population of citizens and politicians with
  varied initial engagement levels rather than a uniform
  starting engagement. See TODO #29.

Derived quantities used in integration:

```
alpha     = 1 / (2 * sigma^2)
cos_theta = cos(Im(theta))
```

### 4.2 Overlap Integral

The key relationship between any two Gaussians G1 and G2 is
their normalized overlap integral, which measures alignment
on a bounded [-1, +1] scale:

```
I_norm(G1, G2) = I_raw(G1, G2) / sqrt(I(G1,G1) * I(G2,G2))
```

where the raw integral is:

```
I_raw(G1, G2) = (pi / zeta)^1.5 * exp(-xi * d^2)
                * cos(theta_1) * cos(theta_2)
```

and:

```
zeta = alpha_1 + alpha_2
xi   = 1 / (2 * zeta)       = 0.5 / zeta
d    = mu_1 - mu_2
```

The normalization denominator factorizes across the two
Gaussians, so each self-norm is cached per Gaussian object:

```
self_norm = sqrt(I(G,G)) = (pi * sigma^2)^0.75 * |cos(theta)|
```

This is recomputed in `update_integration_variables()` whenever
sigma or theta change, and costs only one multiply and one
divide per `integral()` call. Returns 0 when either agent is
fully apathetic (cos_theta = 0).

Properties of the integral:

- Maximum value (+1) when both Gaussians have identical
  parameters.
- Minimum value (-1) when parameters match except theta values
  are 0 and pi respectively.
- Same-type integrals (pref x pref, aver x aver) are ≥ 0;
  cross-type integrals (pref x aver) are ≤ 0, following from
  the sign convention (see §4.1 theta sign convention).
- Both agreement and disagreement produce large |I|; only weak
  overlap produces small |I|.

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
  with politicians' and other citizens' traits. Positive-
  valued (same theta sign convention as stated_policy_pref).
- `stated_trait_aver`: Personality trait aversion. Negative-
  valued (same theta sign convention as stated_policy_aver).

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

  **Decision**: Implement the simple overlap-based
  `well_being` first with downstream engagement effects.
  Design the code so the richer model from §8.5 can slot
  in to replace it without restructuring the call sites.
  (See TODO #9; §8.5 for the richer model design.)
- `participation_prob`: Probability of voting, computed
  dynamically as mean(|cos(theta)|) across all stated
  Gaussians.
- `politician_list`: Politicians this citizen can vote for
  (populated each campaign).
- `politician_score`: Accumulated scores for each politician.

**Influence accumulation**: During each campaign step,
twelve shift arrays accumulate changes to the citizen's
Gaussians from all influence sources before application —
one set of three (theta, mu, sigma) per Gaussian type
(Pcp, Pca, Tcp, Tca). See §8.6.1. The current code uses
only six arrays (TODO #23); these must be expanded.

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

- `policy_persuasion` / `trait_persuasion`: Scale factors
  for how effectively the politician shifts citizen Gaussians
  during the campaign phase. `policy_persuasion` (f_pol)
  scales policy-overlap-driven engagement and position/
  spread shifts; `trait_persuasion` (f_trait) scales
  trait-overlap-driven engagement shifts. Initialized in
  `Politician.reset_to_input()` from XML stddev params;
  applied by TODO #21.
- `policy_lie` / `trait_lie`: Propensity to misrepresent
  positions (not yet used).
- `political_power`: Computed each govern phase from zone
  population, margin of victory, and agreement/disagreement
  ratio (§7.5.1). Not initialized from XML.
- `margin_of_victory`: Computed after each election as the
  normalized vote margin over the next-closest rival.
  Not initialized from XML.
- `elected`: Whether this politician won the last election.
- `votes`: Vote count (reset each cycle).

**Strategies**: Each politician has three independently
selected strategies, chosen probabilistically from cumulative
distribution parameters in the input XML:

- `move_strategy`: How the politician moves between
  patches each campaign step.
  - 0: Random patch within zone.
  - 1: Teleport to the highest-population patch in
    the zone and stay there.
  - 2: Cycle through the bottom half of patches
    (sorted by population, ascending) in round-robin
    order, visiting low-to-middle population areas.
- `adapt_strategy`: How the politician sets apparent
  positions. Strategy 0: honest (innate unchanged).
  Strategy 1: pander toward zone-average citizen
  preferences using `policy_lie`/`trait_lie`.
  Strategy 2: shift away from zone-average citizen
  aversions.

### 6.3 Government

A single `Government` instance holds:

- `enacted_policy`: Gaussian for each policy dimension
  representing current government policy. Initialized
  randomly. During the govern phase, `Pge.mu` and
  `Pge.sigma` shift each step under politician forces
  (§7.5.3), and `Pge.sigma` naturally broadens each
  cycle via the policy spread mechanism (§7.5.4).

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
consists of three phases:

```
for cycle in num_cycles:
    campaign(...)     # Multi-step campaigning phase
    vote(...)         # Single-step election
    govern(...)       # Multi-step governing phase
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

The govern phase runs for `num_govern_steps` time
steps after each election. The design below
describes the implemented mechanics.

The govern phase focuses on how elected politicians
move enacted government policy (`Pge`). Each
politician exerts force on `Pge` in proportion to
their `political_power` — a scalar weight derived
from institutional and electoral factors. The
direction of force comes from the relative position
of `Pge` and the politician's true innate policy
positions. Distance of separation has no effect on
force magnitude.

#### 7.5.1 Political Power

Each elected politician's `political_power` is a
scalar computed from three components:

**Component 1 — Zone population**: Politicians
representing larger zones (more citizens) carry more
institutional weight. A state-level politician has
more power than a district-level one; a country-level
politician has more than a state-level one. The zone
population is `zone.curr_num_citizens`.

**Component 2 — Margin of victory**: A politician who
won by a large margin has a stronger mandate than one
who barely won. The margin is the difference between
the winner's vote count and the next-closest rival's
vote count, normalized by the total votes cast in
the zone.

**Component 3 — Mandate (net alignment)**:
The net alignment between a politician's apparent
positions and what citizens want, using zone citizen
averages and the politician's external positions:

```
agreement    = I(avg_Pcp, Ppp) + I(avg_Pca, Ppa)
             + I(avg_Tcp, Tpx)
disagreement = I(avg_Pca, Ppp) + I(avg_Pcp, Ppa)
mandate      = agreement + disagreement
```

The theta sign convention (§3) makes same-type
integrals non-negative and cross-type integrals
non-positive. Therefore `agreement ≥ 0` and
`disagreement ≤ 0`, so `mandate = agreement -
|disagreement|`. Mandate increases as the
politician's apparent positions align with citizen
preferences and decreases as they clash with citizen
aversions. Mandate is bounded because each
normalized integral is in `[-1, +1]`, so
`mandate ∈ [-(N_policy + N_trait),
             +(N_policy + N_trait)]`.

A politician whose apparent positions strongly
align with citizen preferences and actively avoid
their aversions has a large positive mandate. One
whose positions clash with citizen aversions has a
reduced or negative mandate. Because this uses
external positions, a politician who lied
effectively during the campaign may have an
inflated mandate relative to their true innate
positions.

**Combining the components**:

```
political_power = zone_population
                * margin_of_victory
                * max(0, mandate)
```

The `max(0, ...)` clamp keeps political power
non-negative. A politician with negative mandate
(cross-type clash exceeds same-type agreement) has
zero governing power — their electoral win does not
translate to policy influence. The product form
means each component scales the others
multiplicatively.

**Population normalization**: The raw
`political_power` values are proportional to zone
population (which ranges from tens to thousands of
citizens depending on zone level). If these raw
values are applied directly as force magnitudes on
`Pge`, the forces overwhelm the policy-space scale —
a single govern step can shift `Pge.mu` by hundreds
of units when the typical policy position is O(1).

To decouple force magnitude from world size, the
govern phase normalizes each politician's power by
the total population represented by all elected
politicians:

```
total_pop = sum(zone_population_p) for all elected p
political_power_p /= total_pop
```

After normalization, the sum of all raw powers
(before margin and ratio scaling) equals 1.0. This
makes the force magnitudes dimensionless and
independent of world size — doubling the number of
citizens does not double the force on `Pge`. The
relative distribution of power among politicians is
preserved: a state-level politician still has more
power than a district-level one, but the absolute
scale is bounded.

Note that zone populations can overlap in the
hierarchical zone structure (a citizen in a district
is also in a state). The `total_pop` sum counts
each elected politician's zone population, so a
citizen may be counted more than once if politicians
from multiple zone levels are elected. This is
intentional — it means higher-zone politicians share
their larger population weight with other elected
politicians in the normalization pool, preventing
a single country-level politician from dominating
all district-level ones by raw population alone.

**Future extensions**: The political power
computation may later incorporate:
- An incumbency factor.
- Change-desire integrals: `I(avg_Pca, Pge)` added
  to the agreement numerator and `I(avg_Pcp, Pge)`
  added to the disagreement denominator, capturing
  the degree to which citizens want policy to move
  away from the status quo.

#### 7.5.2 Per-Dimension Allocation

Political power is a scalar, but politicians do not
exert equal force on every policy dimension. Each
politician's power is distributed across dimensions
according to how strongly they hold positions on each
dimension, as measured by the inverse of their innate
Gaussian sigma.

**For preference-driven forces** (attraction toward
innate preference):

```
raw_pref_n = 1 / sigma_innate_pref_n
pref_weight_n = raw_pref_n / sum(raw_pref_n)
```

**For aversion-driven forces** (repulsion away from
innate aversion):

```
raw_aver_n = 1 / sigma_innate_aver_n
aver_weight_n = raw_aver_n / sum(raw_aver_n)
```

A narrow sigma (strong, focused opinion) allocates
more of the politician's power to that dimension.
A broad sigma (weak, diffuse opinion) allocates less.
The normalization ensures each set of weights sums
to 1 across all policy dimensions.

#### 7.5.3 Policy Movement

Each govern step, enacted policy (`Pge`) shifts under
the forces exerted by all elected politicians. Each
politician contributes two types of force:

**Preference attraction**: Pulls `Pge` toward the
politician's innate policy preference position.

```
for each policy dimension n:
    pref_force_mu_n = 0
    pref_force_sigma_n = 0

    for each elected politician p:
        w = political_power_p * pref_weight_p_n
        dir_mu = sign(innate_pref_p.mu_n
                      - Pge.mu_n)
        dir_sigma = sign(innate_pref_p.sigma_n
                         - Pge.sigma_n)
        pref_force_mu_n += w * dir_mu
        pref_force_sigma_n += w * dir_sigma

    Pge.mu_n += pref_force_mu_n
    Pge.sigma_n += pref_force_sigma_n
```

**Aversion repulsion**: Pushes `Pge` away from the
politician's innate policy aversion position.

```
for each policy dimension n:
    aver_force_mu_n = 0
    aver_force_sigma_n = 0

    for each elected politician p:
        w = political_power_p * aver_weight_p_n
        dir_mu = sign(Pge.mu_n
                      - innate_aver_p.mu_n)
        dir_sigma = sign(Pge.sigma_n
                         - innate_aver_p.sigma_n)
        aver_force_mu_n += w * dir_mu
        aver_force_sigma_n += w * dir_sigma

    Pge.mu_n += aver_force_mu_n
    Pge.sigma_n += aver_force_sigma_n
```

The `sign()` function returns +1, 0, or -1. Only the
direction matters — the distance between `Pge` and
the politician's position does not scale the force.
Politicians push both `Pge.mu` (position) and
`Pge.sigma` (specificity) toward their innate values,
so they shape both where policy sits and how sharply
defined it is.

**Pge sigma floor**: After forces are applied each
govern step, `Pge.sigma` is clamped to a minimum
value (`sigma_floor`) to prevent sigma from reaching
zero or going negative. This is the same floor used
for citizen Gaussians (§8.6.8). Without this floor,
opposing sigma forces can drive `Pge.sigma` below
zero, which is physically meaningless for a Gaussian
width and causes cascading numerical failures
(negative alpha, sign-flipping in subsequent steps).
The floor is applied element-wise after each step's
force accumulation and again after the natural spread
(§7.5.4):

```
Pge.sigma_n = max(Pge.sigma_n, sigma_floor)
```

**Future extension — governance rate**: A global
`governance_rate` parameter may be introduced to
provide institutional inertia, bounding how fast
`Pge` can move per step regardless of force
magnitude. For now, the simulation runs without this
parameter to observe the natural dynamics of
political power weighting alone.

**Future extension — position blending**: A per-
politician parameter may be introduced to blend
innate and external positions when computing the
governing target. This would allow politicians to
govern partly according to their campaign promises
rather than purely from ideology. The blending
concept is deferred until the basic mechanics are
validated.

#### 7.5.4 Natural Policy Spread

Enacted policy Gaussians naturally broaden over time,
representing the tendency of policy specificity to
erode without active political effort to maintain it.
The broadening is faster for sharp (narrow sigma)
policies and slower for broad ones:

```
Pge.sigma_n += spread_rate / Pge.sigma_n
```

A narrow `Pge` (small sigma) spreads quickly —
precise policy details are hard to maintain. A broad
`Pge` (large sigma) spreads slowly — vague policy is
already diffuse. This is analogous to the natural
tendency of citizen engagement to decay toward apathy
(§8.2), but applied to the specificity of enacted
policy rather than to agent orientation.

The natural spread is applied once per govern cycle
(not per govern step), using a small `spread_rate`
parameter. The `spread_rate` is stored as a variable
so it can be made dynamic in the future. After the
spread, `Pge.sigma` is clamped to `sigma_floor`
(same floor as in §7.5.3).

The spread creates a natural tension: politicians
must continuously exert force to keep enacted policy
sharp and aligned with their positions. Without
sustained political effort, policy drifts toward
vagueness.

#### 7.5.5 Downstream Effects

Once `Pge` shifts, existing mechanisms produce
feedback:

1. `overlap(Pci, Pge)` changes — citizen well-being
   changes.
2. `overlap(Pcp, Pge)` changes — perceived
   satisfaction changes.
3. Well-being feeds into engagement (via
   `build_response_to_well_being()`).
4. Engagement affects vote probability.
5. Vote probability affects who wins next cycle.

A governing phase that drifts policy away from
citizens' ideal positions will erode well-being,
change voting behavior, change who gets elected, and
change the forces on policy. The feedback loop closes
naturally.

#### 7.5.6 Emergent Behaviors

The political power model produces several dynamics
without special-casing them:

- **Gridlock**: When elected politicians pull in
  opposing directions, forces partially cancel and
  policy moves slowly or not at all.
- **Rapid change**: When politicians align, forces
  reinforce and policy moves quickly.
- **Mandate effects**: A politician who won by a
  large margin with high citizen agreement exerts
  substantially more force than one who barely won
  with mixed support.
- **Hierarchical power**: Higher-zone politicians
  (state, country) naturally dominate lower-zone
  ones (district) through the population component.
- **Policy erosion**: Without sustained political
  effort, enacted policy naturally broadens and
  loses specificity via the natural spread
  mechanism (§7.5.4).
- **Focused politicians**: Narrow-sigma politicians
  concentrate their power on a few dimensions and
  move those effectively. Broad-sigma politicians
  spread thin and change little.

---

## 8. Interaction Physics

### 8.1 Fundamental Principle: Trait Gates Policy

**Politician influence** on citizen policy positions is
sign-gated by trait alignment:

- The **magnitude** of trait alignment (|trait_sum|)
  determines how much the policy shift is.
- The **sign** of trait alignment determines what kind of
  shift occurs.

```
trait_sum = sum of trait overlap integrals with politician
magnitude = |trait_sum|   --> amount of shift
sign      = sign(trait_sum)  --> type of shift
```

**Positive trait overlap** (citizen likes the politician):

- Policy mu shifts toward the politician's mu, proportional
  to magnitude.
- Policy sigma shifts toward the politician's sigma,
  proportional to magnitude.

**Negative trait overlap** (citizen dislikes the politician):

- No preference mu movement.
- Preference sigma narrows (citizen becomes more rigid),
  proportional to magnitude.
- Aversion mu shifts toward the politician's policy positions
  (targeted backlash), proportional to magnitude times
  a `defensive_ratio` parameter.

**Citizen-collective influence** is unconditional: policy
and trait Gaussians always drift toward zone averages
regardless of trait alignment. There is no defensive
backlash against the community. The drift rate uses only
same-type overlaps (pref x pref, aver x aver); cross terms
are excluded. See §8.6.3 for the full formulas.

Physical interpretation: Personality affinity with a
politician makes a citizen open to adopting that politician's
positions; personality aversion triggers defensive rigidity
and targeted backlash. Community norms exert a background
pull with no analogous backlash — people absorb community
norms gradually, including norms they are opposed to, just
more slowly.

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
fully apathetic) proportionally to theta itself:
`theta *= (1 + engagement_decay_rate)`. The more disengaged a
citizen already is, the faster they drift toward full apathy —
disengagement is self-reinforcing. A perfectly engaged citizen
(theta=0) experiences no decay. Without active campaigning or
citizen-citizen interaction, citizens gradually disengage from
all issues. The rate is stored as a variable for future dynamic
behavior.

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

The politician's `policy_persuasion` (f_pol) and
`trait_persuasion` (f_trait) parameters scale these
shifts.

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

4. **Implementation order**: ✅ **Decided**: Implement
   the simple `overlap(Pci, Pge)` well-being first with
   downstream engagement effects. Encapsulate the
   well-being→engagement mapping so the richer model
   can slot in without restructuring call sites.
   (See §6.1 and TODO #9.)

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

From each politician p (f_pol = policy_persuasion,
f_trait = trait_persuasion):

```
Pcp.theta_shift[n] += f_pol * (|I(Pcp,Ppp)[n]|
                              + |I(Pcp,Ppa)[n]|)
Pca.theta_shift[n] += f_pol * (|I(Pca,Ppa)[n]|
                              + |I(Pca,Ppp)[n]|)
Tcp.theta_shift[m] += f_trait * |I(Tcp,Tpx)[m]|
Tca.theta_shift[m] += f_trait * |I(Tca,Tpx)[m]|
```

From zone averages (scaled by `collective_influence_rate`
from the XML; no per-politician factor like f_pol/f_trait):

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

The susceptibility function (c=1 settled, see §8.6.8):

```
S(sigma, theta) = sigma * (1 - cos(theta))
```

At full engagement (theta=0): S = 0 — fully engaged
citizens are completely immovable; no mu or sigma shifts
occur regardless of influence magnitude.
At full apathy (theta=pi/2): S = sigma — maximum
susceptibility.
Near theta=0: S ≈ sigma * theta²/2 — susceptibility
grows slowly (quadratically) from zero, so positions
only shift meaningfully once disengagement is substantial.
This means campaigns primarily alter engagement levels;
position shifts follow only as citizens disengage.

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
f         = policy_persuasion
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

Community policy influence is unconditional: policy
Gaussians always drift toward zone average policy values
regardless of trait alignment. There is no defensive
response. The rate uses only same-type trait overlaps;
cross terms are excluded:

```
trait_rate = sum_m( I(Tcp,avg_Tcp)[m]
                  + I(Tca,avg_Tca)[m] )
```

trait_rate is always ≥ 0. Policy pref and aver mu and
sigma always shift toward the zone average policy values.
Same formulas as the politician positive-trait case with
f = 1, zone averages replacing politician Gaussians, and
susceptibility S applied to the citizen Gaussian being
shifted.

**Decision**: The susceptibility model is the chosen
approach. The force/momentum alternative is back-burnered
(see note below).

*Back-burnered — force/momentum dynamics*: An alternative
approach introduces Newtonian dynamics where influence
produces a *force* rather than a direct shift, with
velocity state per Gaussian per dimension and a damping
parameter. This produces inertia and lingering effects from
past influence, but adds complexity and is unappealing at
this stage. Revisit after the susceptibility model is
implemented and evaluated. (See §8.6.8 for the parameters
that would be needed if this is ever revisited.)

#### 8.6.4 Trait Position and Spread Accumulation

Trait acclimatization occurs only through the citizen
collective (Section 8.4 — politicians do not directly shift
citizen traits). It is unconditional: trait Gaussians always
shift toward the zone average, regardless of trait alignment.
The rate uses only same-type trait overlaps (pref x pref,
aver x aver); cross terms are excluded. This is the same
trait_rate computed in §8.6.3 for the citizen-collective
policy case.

The same susceptibility considerations from Section 8.6.3
apply: narrow, engaged trait Gaussians resist movement more
than broad, disengaged ones.

```
trait_rate = sum_m( I(Tcp,avg_Tcp)[m]
                  + I(Tca,avg_Tca)[m] )

for each trait dim m:
    Tcp.mu_shift[m]    += trait_rate
        * S(Tcp.sigma[m], Tcp.theta[m])
        * sign(avg_Tcp.mu[m] - Tcp.mu[m])
    Tcp.sigma_shift[m] += trait_rate
        * S(Tcp.sigma[m], Tcp.theta[m])
        * sign(avg_Tcp.sigma[m] - Tcp.sigma[m])
    Tca.mu_shift[m]    += trait_rate
        * S(Tca.sigma[m], Tca.theta[m])
        * sign(avg_Tca.mu[m] - Tca.mu[m])
    Tca.sigma_shift[m] += trait_rate
        * S(Tca.sigma[m], Tca.theta[m])
        * sign(avg_Tca.sigma[m] - Tca.sigma[m])
```

trait_rate is always ≥ 0. A citizen whose traits are
opposed to community norms still drifts toward those norms;
trait_rate simply becomes small when same-type alignment
is weak.

*Back-burnered*: If the force/momentum model is ever
revisited, trait shifts would use the same dynamics
framework with per-trait-dimension velocities and mass.

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

After influence-driven engagement shifts, a proportional
decay pulls every citizen's theta for every stated Gaussian
toward pi/2 (full apathy):

```
theta = clamp(theta * (1 + engagement_decay_rate), 0, pi/2)
```

`engagement_decay_rate` is a small positive dimensionless
constant (the fractional increase in theta per step).

Key properties of this formulation:
- **No decay at full engagement**: when theta = 0, the
  decay term is zero — a perfectly engaged citizen stays
  engaged unless actively disturbed.
- **Self-reinforcing**: the higher theta (the more
  disengaged), the larger the absolute decay step. Once
  disengagement begins, it accelerates.
- **Maximum decay near apathy**: the decay is largest
  when theta is near pi/2, producing the strongest pull
  toward full apathy.

The rate is stored as a variable for future dynamic
behavior (e.g., modulated by well-being or crisis).

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
| `engagement_decay_rate` | Fractional per-step theta increase (proportional decay: theta *= (1 + rate)) | TBD |
| `defensive_ratio` | Scales targeted backlash mu shift | 1.0 |
| `sigma_floor` | Minimum sigma for all Gaussians; also the target of defensive narrowing | 0.05 (~20× narrower than a typical initial sigma of O(1)) |
| `engagement_protection` | c in S() = sigma*(1 - c*cos(theta)); c=1 makes fully engaged citizens completely immovable | 1.0 |

*Back-burnered — force/momentum parameters*: If the
dynamics model is ever revisited, additional parameters
would be needed: `damping` (velocity decay per step) and
a mass function f(1/sigma, cos(theta)) with its own
scaling parameters.

#### 8.6.9 Open Design Questions

The following questions must be resolved before the
shift mechanics can be fully implemented (TODO #8,
#21). The shift model choice (Q1) is settled; the
remaining questions are refinements that can be
resolved during or after initial implementation.

✅ **Q1 — Shift model choice**: The susceptibility
model is chosen. Force/momentum dynamics are
back-burnered (see §8.6.3).

✅ **Q2 — Susceptibility function form**:
`S(sigma, theta) = sigma * (1 - cos(theta))` with c=1.
Fully engaged citizens (theta=0) are completely
immovable. See §8.6.3 for full description.

✅ **Q3 — Defensive narrowing rate**: Susceptibility-
dependent narrowing is correct. Broad Gaussians
(undecided citizens) narrow faster than already-narrow
ones, which aligns with the empirical observation that
people form firm opinions more quickly than expected.
Narrowing rate remains coupled to susceptibility.

✅ **Q4 — Citizen-collective scaling**: A
`collective_influence_rate` parameter is added to
independently scale community influence relative to
politician influence. Default value is 1.0 (no
change to current behavior). Configured in the
`<citizens>` XML block.

✅ **Q5 — Trait self-gating rate**: Rate is determined
by same-type overlaps only: `trait_rate =
I(Tcp,avg_Tcp) + I(Tca,avg_Tca)`. Cross terms are
excluded and do not affect acclimatization speed.
trait_rate is always ≥ 0, eliminating cancellation.
Citizen-collective policy influence is also
unconditional (always attract, same-type rate),
parallel to trait acclimatization. The full
trait_sum (all four terms, signed) is used for
sign-gating in politician influence only.

✅ **Q6 — Sigma shift direction**: Keep
`sign(source_sigma - citizen_sigma)` for
constant-magnitude steps. Spring-like behavior
(shift proportional to distance) is explicitly
rejected. The anti-spring character already present
in `S = sigma_citizen * (1 - cos(theta))` is
sufficient: broader citizens take larger steps and
naturally decelerate as they narrow toward the
source.

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
Also: persuasion and lie standard deviations, and
cumulative strategy probability distributions for move,
adapt, and campaign strategies.

### government

Standard deviations for initializing enacted policy
position and spread. The `sigma_floor` parameter is
shared with the citizen section — it is the same
minimum sigma used for all Gaussians in the
simulation. In the govern phase, it prevents
`Pge.sigma` from reaching zero or going negative
under politician forces (§7.5.3) and after natural
spread (§7.5.4).

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

No open architectural issues. All previously tracked
issues (TODO #8, #17, #18, #19, #20 and others) are
resolved — see `TODO.md` Resolved Items.

The one remaining known limitation is that the govern
phase forces still have no hard cap on the per-step
movement of `Pge` (a `governance_rate` parameter is
described as a future extension in §7.5.3). The
current combination of population normalization and
bounded mandate (§7.5.1) keeps forces in the O(10^-2)
range for typical world configurations, but very
large worlds or extreme margin/mandate values could
still produce large per-step shifts. This is noted
as a future parameter rather than an active bug.

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
