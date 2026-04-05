# STODEM Pseudocode

> **Document hierarchy:** VISION → ARCHITECTURE →
> DESIGN → **PSEUDOCODE** → Code. This file specifies
> key algorithms in language-agnostic form. For the
> mathematical derivations and design rationale behind
> these algorithms, see `DESIGN.md`. For the source
> implementation, see `src/scripts/`.

---

## 1. Main Loop

```
function main():
    settings  = parse_config("stodem.in.xml")
    sim       = SimControl(settings)
    world     = create_world(settings, sim)
    populate(world, settings)    # citizens, politicians, government
    hdf5      = open_hdf5(settings, world)
    glyph     = open_glyph_hdf5(settings, world)

    for cycle in 0 .. num_cycles - 1:
        campaign(sim, settings, world, hdf5, glyph, cycle)
        vote(sim, world)
        govern(sim, world, hdf5, glyph, cycle)

    write_xdmf(settings, sim.current_step, world)
    write_glyph_xdmf(settings, glyph, sim.current_step)
    write_paraview_script(settings, world)
    close(hdf5, glyph)
```

---

## 2. Campaign Phase

One campaign runs for `num_campaign_steps` time steps.
Each step follows a fixed sequence; the order of
steps 6–8 is irrelevant because all three read from
the same unchanged Gaussian state (accumulate-then-
apply, DESIGN §8.6).

```
function campaign(sim, settings, world, hdf5, glyph,
                  cycle):
    # --- One-time setup ---
    repopulate_politicians(world, settings)
        # Elected politicians keep positions;
        # losers are replaced with fresh randoms.
    for each citizen in world:
        citizen.clear_politician_list()
    for each politician in world:
        politician.present_to_citizens(world)
            # Citizens in the politician's zone
            # add it to their candidate list.

    # --- Per-step loop ---
    for step in 0 .. num_campaign_steps - 1:

        # 1. Aggregate citizen stats per zone.
        for each zone_type in world:
            for each zone in zone_type:
                zone.compute_zone_averages(world)

        # 2. Politicians move within their zone.
        for each politician in world:
            politician.move()

        # 3. Politicians set external positions.
        for each politician in world:
            politician.adapt_to_patch(world)

        # 4. Citizens compute all overlap integrals.
        for each citizen in world:
            citizen.compute_all_overlaps(world)

        # 5. Initialize twelve shift arrays to zero.
        for each citizen in world:
            citizen.prepare_for_influence(
                num_policy_dims, num_trait_dims)

        # 6. Accumulate politician influence.
        for each citizen in world:
            citizen.accumulate_politician_influence()

        # 7. Accumulate well-being engagement.
        for each citizen in world:
            citizen.accumulate_well_being_response()

        # 8. Accumulate citizen collective influence.
        for each citizen in world:
            citizen.accumulate_collective_influence()

        # 9. Apply all accumulated shifts (single pass).
        for each citizen in world:
            citizen.apply_influence_shifts()

        # 10. Score candidates for voting.
        for each citizen in world:
            citizen.score_candidates(world)

        # 11. Aggregate patch-level stats and write.
        compute_patch_well_being(world)
        compute_patch_gaussian_stats(world)
        compute_patch_politician_stats(world)
        write_step(hdf5, glyph, world, sim.current_step)
        sim.current_step += 1
```

---

## 3. Influence Physics

### 3.1 Susceptibility

A citizen's resistance to having a Gaussian parameter
shifted. Used for all position and spread shifts.

```
function S(sigma, cos_theta):
    return sigma * (1 - |cos_theta|)
```

Properties:
- Full engagement (|cos_theta| = 1): S = 0; immovable.
- Full apathy (|cos_theta| = 0): S = sigma; maximum.
- Campaigns primarily change engagement; position
  shifts follow only as citizens disengage.

### 3.2 Politician Influence (per citizen)

For each politician sharing the citizen's patch,
compute engagement shifts and trait-gated policy
shifts.

```
function accumulate_politician_influence(citizen):
    for each politician in citizen.politician_list:
        f_pol   = politician.policy_persuasion
        f_trait = politician.trait_persuasion

        # --- Engagement shifts (DESIGN §8.6.2) ---
        # |overlap| drives theta toward engaged pole.
        Pcp.theta_shift += f_pol * (|I(Pcp,Ppp)|
                                  + |I(Pcp,Ppa)|)
        Pca.theta_shift += f_pol * (|I(Pca,Ppa)|
                                  + |I(Pca,Ppp)|)
        Tcp.theta_shift += f_trait * |I(Tcp,Tpx)|
        Tca.theta_shift += f_trait * |I(Tca,Tpx)|

        # --- Trait sum: gate for policy shifts ---
        trait_sum = sum_m(I(Tcp,Tpx)[m]
                       + I(Tca,Tpx)[m])
        mag = |trait_sum|

        S_Pcp = S(Pcp.sigma, Pcp.cos_theta)
        S_Pca = S(Pca.sigma, Pca.cos_theta)

        if trait_sum >= 0:
            # Attraction: drift toward politician.
            for each policy dim n:
                Pcp.mu_shift[n] += mag * f_pol
                    * S_Pcp[n]
                    * sign(Ppp.mu[n] - Pcp.mu[n])
                Pcp.sigma_shift[n] += mag * f_pol
                    * S_Pcp[n]
                    * sign(Ppp.sigma[n]
                           - Pcp.sigma[n])
                Pca.mu_shift[n] += mag * f_pol
                    * S_Pca[n]
                    * sign(Ppa.mu[n] - Pca.mu[n])
                Pca.sigma_shift[n] += mag * f_pol
                    * S_Pca[n]
                    * sign(Ppa.sigma[n]
                           - Pca.sigma[n])
        else:
            # Defensive: rigidity + backlash.
            for each policy dim n:
                # Pcp sigma narrows toward floor.
                Pcp.sigma_shift[n] += mag * f_pol
                    * S_Pcp[n]
                    * sign(sigma_floor
                           - Pcp.sigma[n])
                # Pca mu shifts toward politician's
                # PREFERENCE (targeted backlash).
                Pca.mu_shift[n] += mag * f_pol
                    * defensive_ratio * S_Pca[n]
                    * sign(Ppp.mu[n] - Pca.mu[n])
```

### 3.3 Citizen Collective Influence (per citizen)

Unconditional drift toward zone averages. No
defensive branch. Rate uses same-type trait overlaps
only (always >= 0).

```
function accumulate_collective_influence(citizen):
    cir = collective_influence_rate

    for each zone in citizen.zone_list:

        # --- Engagement shifts ---
        Pcp.theta_shift += cir * (|I(Pcp,avg_Pcp)|
                                + |I(Pcp,avg_Pca)|)
        Pca.theta_shift += cir * (|I(Pca,avg_Pca)|
                                + |I(Pca,avg_Pcp)|)
        Tcp.theta_shift += cir * (|I(Tcp,avg_Tcp)|
                                + |I(Tcp,avg_Tca)|)
        Tca.theta_shift += cir * (|I(Tca,avg_Tca)|
                                + |I(Tca,avg_Tcp)|)

        # --- Trait rate (same-type only, >= 0) ---
        trait_rate = sum_m(I(Tcp,avg_Tcp)[m]
                         + I(Tca,avg_Tca)[m])

        # --- Policy shifts (unconditional) ---
        S_Pcp = S(Pcp.sigma, Pcp.cos_theta)
        S_Pca = S(Pca.sigma, Pca.cos_theta)
        for each policy dim n:
            Pcp.mu_shift[n] += cir * trait_rate
                * S_Pcp[n]
                * sign(avg_Pcp.mu[n] - Pcp.mu[n])
            Pcp.sigma_shift[n] += cir * trait_rate
                * S_Pcp[n]
                * sign(avg_Pcp.sigma[n]
                       - Pcp.sigma[n])
            Pca.mu_shift[n] += cir * trait_rate
                * S_Pca[n]
                * sign(avg_Pca.mu[n] - Pca.mu[n])
            Pca.sigma_shift[n] += cir * trait_rate
                * S_Pca[n]
                * sign(avg_Pca.sigma[n]
                       - Pca.sigma[n])

        # --- Trait shifts (sole mechanism) ---
        S_Tcp = S(Tcp.sigma, Tcp.cos_theta)
        S_Tca = S(Tca.sigma, Tca.cos_theta)
        for each trait dim m:
            Tcp.mu_shift[m] += cir * trait_rate
                * S_Tcp[m]
                * sign(avg_Tcp.mu[m] - Tcp.mu[m])
            Tcp.sigma_shift[m] += cir * trait_rate
                * S_Tcp[m]
                * sign(avg_Tcp.sigma[m]
                       - Tcp.sigma[m])
            Tca.mu_shift[m] += cir * trait_rate
                * S_Tca[m]
                * sign(avg_Tca.mu[m] - Tca.mu[m])
            Tca.sigma_shift[m] += cir * trait_rate
                * S_Tca[m]
                * sign(avg_Tca.sigma[m]
                       - Tca.sigma[m])
```

### 3.4 Well-Being Response (per citizen)

Well-being is the overlap between ideal policy (Pci)
and enacted policy (Pge). |well_being| drives
engagement uniformly across all dimensions.

```
function accumulate_well_being_response(citizen):
    well_being = sum_n(I(Pci, Pge)[n])
    mag = |well_being|

    # Uniform engagement shift across all dims.
    Pcp.theta_shift += mag  (broadcast to n dims)
    Pca.theta_shift += mag  (broadcast to n dims)
    Tcp.theta_shift += mag  (broadcast to m dims)
    Tca.theta_shift += mag  (broadcast to m dims)
```

### 3.5 Apply Influence Shifts (per citizen)

Single-pass application of all twelve accumulated
shift arrays, followed by engagement decay and
derived-variable refresh.

```
function apply_influence_shifts(citizen):
    edr = engagement_decay_rate

    # For each Gaussian type (Pcp, Pca, Tcp, Tca):
    #   1. Apply mu shift (unbounded).
    #   2. Apply sigma shift (clamped to sigma_floor).
    #   3. Apply theta shift + engagement decay.

    # --- Preference types (Pcp, Tcp) ---
    # Theta convention: Im(theta) in [0, pi/2].
    # Subtracting the shift drives toward 0 (engaged).
    for G in [Pcp, Tcp]:
        G.mu    += G.mu_shift
        G.sigma  = max(G.sigma + G.sigma_shift,
                       sigma_floor)
        im       = clamp(Im(G.theta) - G.theta_shift,
                         0, pi/2)
        im       = clamp(im * (1 + edr), 0, pi/2)
        G.theta  = im * i

    # --- Aversion types (Pca, Tca) ---
    # Theta convention: Im(theta) in [pi/2, pi].
    # Adding the shift drives toward pi (engaged).
    # Decay operates on alpha = pi - Im(theta).
    for G in [Pca, Tca]:
        G.mu    += G.mu_shift
        G.sigma  = max(G.sigma + G.sigma_shift,
                       sigma_floor)
        im       = clamp(Im(G.theta) + G.theta_shift,
                         pi/2, pi)
        alpha    = clamp((pi - im) * (1 + edr),
                         0, pi/2)
        G.theta  = (pi - alpha) * i

    # Refresh cached integration variables.
    for G in [Pcp, Pca, Tcp, Tca]:
        G.alpha      = 1 / (2 * G.sigma^2)
        G.cos_theta  = cos(Im(G.theta))
```

---

## 4. Vote Phase

```
function vote(sim, world):
    # Stage 1: Citizens cast votes.
    for each citizen in world:
        # Compute vote probability from engagement.
        all_cos = concatenate(
            Pcp.cos_theta, Pca.cos_theta,
            Tcp.cos_theta, Tca.cos_theta)
        P_vote = mean(|all_cos|)

        if random() > P_vote:
            continue    # citizen abstains

        # Vote for best candidate per zone level.
        for each zone_type, zone_index in
                citizen.patch.zone_indices:
            best = None
            for each politician in
                    citizen.politician_list:
                if politician.zone_type == zone_type
                   and politician.zone_index
                       == zone_index:
                    if best is None
                       or score[politician]
                          > score[best]:
                        best = politician
            if best is not None:
                best.votes += 1

    # Stage 2: Determine winners per zone.
    for each zone_type in world:
        for each zone in zone_type:
            winner = politician with most votes
            runner_up_votes = second-highest count
            total_votes = sum of all votes in zone

            if total_votes > 0:
                winner.margin_of_victory =
                    (winner.votes - runner_up_votes)
                    / total_votes
            else:
                winner.margin_of_victory = 0

            zone.elected_politician = winner
```

---

## 5. Govern Phase

Elected politicians exert forces on the government's
enacted policy (Pge). Forces are direction-only
(sign function), so magnitude comes solely from
political power, not distance.

```
function govern(sim, world, hdf5, glyph, cycle):
    Pge = government.enacted_policy
    n   = num_policy_dims

    # --- Compute political power (once per cycle) ---
    elected = []
    for each zone_type in world:
        for each zone in zone_type:
            pol = zone.elected_politician
            pol.compute_political_power()
            pol.compute_dimension_weights()
            elected.append(pol)

    # Normalize by total population.
    total_pop = sum(pol.zone.num_citizens
                    for pol in elected)
    for each pol in elected:
        pol.political_power /= total_pop

    # --- Per-step force accumulation ---
    for step in 0 .. num_govern_steps - 1:

        pref_force_mu    = zeros(n)
        pref_force_sigma = zeros(n)
        aver_force_mu    = zeros(n)
        aver_force_sigma = zeros(n)

        for each pol in elected:
            pw = pol.political_power

            # Preference attraction: pull toward
            # politician's innate preference.
            w_pref = pw * pol.pref_weight
            pref_force_mu += w_pref
                * sign(pol.innate_pref.mu - Pge.mu)
            pref_force_sigma += w_pref
                * sign(pol.innate_pref.sigma
                       - Pge.sigma)

            # Aversion repulsion: push away from
            # politician's innate aversion.
            w_aver = pw * pol.aver_weight
            aver_force_mu += w_aver
                * sign(Pge.mu - pol.innate_aver.mu)
            aver_force_sigma += w_aver
                * sign(Pge.sigma
                       - pol.innate_aver.sigma)

        # Apply forces.
        Pge.mu    += pref_force_mu + aver_force_mu
        Pge.sigma += pref_force_sigma
                     + aver_force_sigma
        Pge.sigma  = max(Pge.sigma, sigma_floor)

        # Refresh and recompute well-being.
        Pge.update_integration_variables()
        for each citizen in world:
            citizen.well_being =
                sum(I(Pci, Pge))
        write_step(hdf5, glyph, world,
                   sim.current_step)
        sim.current_step += 1

    # --- Natural policy spread (once per cycle) ---
    Pge.sigma += spread_rate / Pge.sigma
    Pge.sigma  = max(Pge.sigma, sigma_floor)
    Pge.update_integration_variables()
```

### 5.1 Political Power

```
function compute_political_power(politician):
    zone_pop = politician.zone.num_citizens
    margin   = politician.margin_of_victory

    # Mandate: net alignment with constituents.
    agreement = I(avg_Pcp, Ppp) + I(avg_Pca, Ppa)
              + I(avg_Tcp, Tpx)
    disagreement = I(avg_Pca, Ppp) + I(avg_Pcp, Ppa)

    # Same-type integrals >= 0, cross-type <= 0
    # by the theta sign convention.
    mandate = max(0, agreement + disagreement)

    political_power = zone_pop * margin * mandate
```

### 5.2 Dimension Weights

```
function compute_dimension_weights(politician):
    # Narrow sigma = strong opinion = more weight.
    for each policy dim n:
        raw_pref[n] = 1 / innate_pref.sigma[n]
        raw_aver[n] = 1 / innate_aver.sigma[n]

    pref_weight = raw_pref / sum(raw_pref)
    aver_weight = raw_aver / sum(raw_aver)
```

---

## 6. Candidate Scoring

```
function score_candidates(citizen, world):
    w_policy = 0.5 + policy_trait_ratio
    w_trait  = 0.5 - policy_trait_ratio

    for each politician in citizen.politician_list:
        policy_sum = sum_n(
            I(Pcp,Ppp)[n] + I(Pca,Ppa)[n]     # same-type
          + I(Pcp,Ppa)[n] + I(Pca,Ppp)[n])    # cross-type

        trait_sum = sum_m(
            I(Tcp,Tpx)[m] + I(Tca,Tpx)[m])

        score = w_policy * policy_sum
              + w_trait  * trait_sum
```

---

## 7. Overlap Integral

The fundamental measurement of alignment between two
complex Gaussians. See DESIGN §4 for the derivation.

```
function integral(G1, G2):
    # G1, G2 each have: mu[d], sigma[d], theta[d]
    #   (arrays over d dimensions)
    for each dim d:
        alpha1 = 1 / (2 * G1.sigma[d]^2)
        alpha2 = 1 / (2 * G2.sigma[d]^2)
        zeta   = alpha1 + alpha2
        xi     = alpha1 * alpha2 / zeta
        dist   = G1.mu[d] - G2.mu[d]

        result[d] = sqrt(pi / zeta)
                  * exp(-xi * dist^2)
                  * cos(Im(G1.theta[d]))
                  * cos(Im(G2.theta[d]))
                  * G1.self_norm[d]
                  * G2.self_norm[d]
    return result
```

where `self_norm = 1 / sqrt(pi / (2 * alpha))` is the
normalization factor that makes each Gaussian have unit
self-overlap when theta = 0.

---

## 8. Debug Visualization — Policy/Trait Space

pyqtgraph visualization of individual agent
Gaussians in policy/trait space. 2-D projected
curves with colour saturation encoding engagement.
Interactive display only. See DESIGN §12.6 for
the full specification.

### 8.1 Initialization

```
function PolicySpaceViz(world, settings):
    n_policy = world.num_policy_dims
    n_trait  = world.num_trait_dims
    n_cols   = max(n_policy, n_trait)

    # Create Qt application and window.
    app    = pyqtgraph.mkQApp()
    window = QWidget with QGridLayout

    # Build 2-row grid of PlotWidgets.
    #   Row 0 = policy dims, row 1 = trait dims.
    plots = 2-D array [2, n_cols], all None

    for col in 0 .. n_policy - 1:
        pw = PlotWidget(title="Policy Dim " + col)
        pw.x_range = [world.policy_limits[0][col],
                       world.policy_limits[1][col]]
        pw.y_range = [-1.1, 1.1]
        pw.add_zero_line(y=0, grey)
        plots[0, col] = pw
        add pw to grid at (row=0, col)

    for col in 0 .. n_trait - 1:
        pw = PlotWidget(title="Trait Dim " + col)
        pw.x_range = [world.trait_limits[0][col],
                       world.trait_limits[1][col]]
        pw.y_range = [-1.1, 1.1]
        pw.add_zero_line(y=0, grey)
        plots[1, col] = pw
        add pw to grid at (row=1, col)

    # Pre-create curve items (§8.2).
    create_all_curves()

    window.show()
    app.processEvents()

    store world, app, window, plots,
          n_policy, n_trait, viz_delay
```

### 8.2 Curve Pre-Creation

Pre-create one PlotDataItem per agent Gaussian
per dimension. These objects persist for the life
of the visualization — only their data and pen
colour change each frame.

```
function create_all_curves():
    citizen_curves = []
    for each citizen in world.citizens:
        items = {}
        for dim in 0 .. n_policy - 1:
            pw = plots[0, dim]
            items[("pp", dim)] = pw.plot(BLUE)
            items[("pa", dim)] = pw.plot(BLUE)
            items[("ip", dim)] = pw.plot(GREEN)
        for dim in 0 .. n_trait - 1:
            pw = plots[1, dim]
            items[("tp", dim)] = pw.plot(BLUE)
            items[("ta", dim)] = pw.plot(BLUE)
        citizen_curves.append(items)

    politician_curves = []
    for each politician in world.politicians:
        items = {}
        for dim in 0 .. n_policy - 1:
            pw = plots[0, dim]
            items[("pp", dim)] = pw.plot(RED)
            items[("pa", dim)] = pw.plot(RED)
        for dim in 0 .. n_trait - 1:
            pw = plots[1, dim]
            items[("tr", dim)] = pw.plot(RED)
        politician_curves.append(items)

    government_curves = {}
    for dim in 0 .. n_policy - 1:
        pw = plots[0, dim]
        government_curves[dim] = pw.plot(BLACK)
```

### 8.3 Gaussian Curve Computation

Compute the 2-D projected curve for a single
Gaussian on one dimension. The bell curve amplitude
is multiplied by cos(theta) to project onto the
real (engaged) plane.

Unit-peak normalization: the `1/(sigma*sqrt(2*pi))`
prefactor is omitted because peak height is
determined solely by sigma, which is already
visible as curve width.

```
function projected_curve(mu, sigma, cos_theta,
                         n_points = 80):
    x = linspace(mu - 4 * sigma,
                 mu + 4 * sigma, n_points)

    amplitude = exp(-(x - mu)^2
                    / (2 * sigma^2))

    y = amplitude * cos_theta

    return x, y
```

Sign convention (DESIGN §4.1) produces correct
visual geometry:
- Preference (theta in [0, pi/2)):
    cos > 0, curve above axis (+y).
- Aversion (theta in (pi/2, pi]):
    cos < 0, curve below axis (-y).
- Apathy (theta near pi/2):
    cos near 0, curve flattens.

### 8.4 Engagement Colour

Compute colour with saturation reflecting the
engagement level |cos(theta)|. Base colour is
lerped toward white as engagement drops. A
minimum floor prevents fully apathetic agents
from becoming invisible.

```
function engagement_color(base_rgb, cos_theta):
    raw = abs(cos_theta)
    engagement = MIN_ENGAGEMENT
                 + (1 - MIN_ENGAGEMENT) * raw
    fade = 1 - engagement
    r = base_rgb.r + (255 - base_rgb.r) * fade
    g = base_rgb.g + (255 - base_rgb.g) * fade
    b = base_rgb.b + (255 - base_rgb.b) * fade
    return (int(r), int(g), int(b))
```

### 8.5 Per-Curve Update

Update one pre-created curve item with new
projected data and engagement-coloured pen.

```
function update_curve(curve_item, mu, sigma,
                      theta_imag, base_rgb, w):
    cos_theta = cos(theta_imag)
    x, y = projected_curve(mu, sigma, cos_theta)
    color = engagement_color(base_rgb, cos_theta)
    curve_item.setData(x, y, pen=(color, w))
```

### 8.6 Update (per simulation step)

Called once per simulation step. Refreshes every
pre-created curve item with the current agent
state and flushes the Qt event loop.

```
function update(step_label):
    window.title = "STODEM — " + step_label

    # --- Citizens (blue) and ideal (green) ---
    for each (cit_idx, citizen) in citizens:
        items = citizen_curves[cit_idx]
        for dim in 0 .. n_policy - 1:
            update_curve(items["pp", dim],
                pref.mu[dim], pref.sigma[dim],
                pref.theta[dim].imag, BLUE, 1)
            update_curve(items["pa", dim],
                aver.mu[dim], aver.sigma[dim],
                aver.theta[dim].imag, BLUE, 1)
            update_curve(items["ip", dim],
                ideal.mu[dim], ideal.sigma[dim],
                ideal.theta[dim].imag, GREEN, 1)
        for dim in 0 .. n_trait - 1:
            update_curve(items["tp", dim], ...)
            update_curve(items["ta", dim], ...)

    # --- Politicians (red) ---
    for each (pol_idx, pol) in politicians:
        items = politician_curves[pol_idx]
        for dim in 0 .. n_policy - 1:
            update_curve(items["pp", dim], ...)
            update_curve(items["pa", dim], ...)
        for dim in 0 .. n_trait - 1:
            update_curve(items["tr", dim], ...)

    # --- Government (black, policy only) ---
    for dim in 0 .. n_policy - 1:
        update_curve(government_curves[dim],
            Pge.mu[dim], Pge.sigma[dim],
            Pge.theta[dim].imag, BLACK, 2)

    # Flush display, throttle via event loop.
    app.processEvents()
    if viz_delay > 0:
        spin on processEvents until deadline
```

### 8.7 Finalize

No-op — retained for interface consistency with
the stodem.py call site.

```
function finalize():
    pass
```

### 8.8 Main Loop Integration

Minimal additions to the main loop (§1) and phase
functions (§2, §5). The visualization is
conditionally created and threaded through as an
optional argument.

```
# In main():
if settings.debug_viz:
    viz = PolicySpaceViz(world, settings)
else:
    viz = None

for cycle in 0 .. num_cycles - 1:
    campaign(sim, settings, world, hdf5,
             glyph, cycle, viz)
    vote(sim, world)
    govern(sim, world, hdf5, glyph, cycle, viz)

if viz is not None:
    viz.finalize()

# In campaign(), after apply_influence_shifts():
if viz is not None:
    viz.update("Campaign  Cycle " + cycle
               + "  Step " + step)

# In govern(), after Pge force application:
if viz is not None:
    viz.update("Govern  Cycle " + cycle
               + "  Step " + step)
```
