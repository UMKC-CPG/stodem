# STODEM Implementation Task List

## Summary

The simulation framework is well-structured and the
core data model (Gaussians, zones, patches, citizens,
politicians, government) is largely in place. The
campaign loop skeleton exists with full influence shift
mechanics (accumulation, trait-gates-policy, engagement
decay, application) and data output (HDF5/XDMF) works.
The governing phase is designed (DESIGN.md §7.5) but
unimplemented, and several agent strategies remain
stubs.

This document tracks concrete implementation tasks.
Architectural design and mathematical specifications
are in `DESIGN.md`. Items are numbered for stable
cross-referencing; gaps in numbering indicate resolved
items (see Resolved Items below).

---

## Resolved Items

The following issues have been fixed and are recorded
here for reference:

| # | Issue | File | Resolution |
|---|-------|------|------------|
| 1 | `compute_patch_well_being` called before defined | `stodem.py` | Moved definition above campaign loop |
| 2 | `hdf5.close` property ref, not method call | `output.py` | Changed to `hdf5.close()`; added `close()` method to `Hdf5` |
| 3 | `Citizen.policy_attitude()` refs `.pos` not `.mu` | `citizen.py` | Changed to `.mu` |
| 4 | Zone object vs. integer comparison in `vote_for_candidates()` | `citizen.py` | Added `zone_index` to `Zone`; filter on both `zone_type` and `zone_index` |
| 6 | `compute_overlap()` incomplete | `gaussian.py` | Superseded by fully implemented `Gaussian.integral()` |
| 14 | `score_candidates()` ignores `policy_trait_ratio` | `citizen.py` | Applied weighting: `w_policy = 0.5 + ratio`, `w_trait = 0.5 - ratio` |
| 15 | Static participation probability | `citizen.py` | Now computed dynamically as `P(vote) = mean(\|cos(theta)\|)` across all stated Gaussians; removed static XML parameters |
| Q4 | No scaling for citizen-collective influence | `citizen.py`, `stodem.in.xml` | `collective_influence_rate` parameter added to `<citizens>` XML (default 1.0); applied as multiplier in `build_response_to_citizen_collective()` |
| 24 | Politician Gaussian theta not sign-convention-compliant | `politician.py` | Preference Gaussians (innate/ext policy pref, innate/ext trait) initialized to `±1j`; aversion Gaussians (innate/ext policy aver) initialized to `(pi-1)j`; matches citizen convention |
| 25 | `collective_influence_rate` missing from XML files | `stodem.in.xml` | Added `<collective_influence_rate>1.0</collective_influence_rate>` to `<citizens>` block in `src/scripts/stodem.in.xml` and `jobs/test1/stodem.in.xml`; `quickTest` already had it |
| 26 | Preference Gaussian theta initialized as ±1j | `citizen.py`, `politician.py` | All preference Gaussian theta changed to `np.full(N, 1j)`; resolves sign-convention violation and prepares for decay clamping (#22) |
| 27 | Sigma initialization allows negative values | `citizen.py`, `politician.py` | All sigma initializations wrapped with `np.abs()`; sigma is now guaranteed positive |
| 28 | `policy_attitude()` dead code with broken `government.policy_pos` reference | `citizen.py` | Removed. Perceived satisfaction uses `Pcp_Pge_ol` overlaps (already computed) |
| 30 | Stale/incomplete comment block in `vote()` | `stodem.py` | Removed orphaned comment block |
| 19 | `build_response_to_politician_influence()` accumulation bugs | `citizen.py` | Rewrote with `enumerate`, per-politician indexing, and `np.abs()`; shift arrays converted to numpy zeros in `prepare_for_influence()` |
| 20 | `build_response_to_citizen_collective()` same accumulation bugs | `citizen.py` | Fixed simultaneously with #19 (numpy conversion required both to be fixed together): rewrote with `enumerate(zone_list)`, per-zone indexing, and `np.abs()` |
| 8 | Influence shifts accumulated but never applied | `citizen.py`, `stodem.py`, `stodem.in.xml` | Implemented `apply_influence_shifts()`: applies orien/pos/stddev shifts per §8.6.5 (pref subtracts toward 0, aver adds toward pi), enforces `sigma_floor`, applies proportional engagement decay per §8.6.6, recomputes derived vars per §8.6.7. Added `sigma_floor=0.05` and `engagement_decay_rate=0.01` to XML/`__init__`. Called after all `build_response_to_*()` calls in campaign loop. |
| 23 | Expand shift arrays from 6 to 12 | `citizen.py` | Done alongside #21: `prepare_for_influence()` now initializes twelve named arrays (Pcp/Pca/Tcp/Tca × orien/pos/stddev); `apply_influence_shifts()` applies each to its own Gaussian; stale "combined arrays" note in docstring removed. |
| 21 | Trait-gates-policy shift mechanics | `citizen.py`, `stodem.in.xml` | Implemented full §8.6.3–§8.6.4 physics. `build_response_to_politician_influence()`: applies f_pol/f_trait to engagement shifts; computes signed trait_sum per politician; attraction branch shifts Pcp→Ppp and Pca→Ppa; defensive branch narrows Pcp.sigma and shifts Pca.mu toward Ppp (targeted backlash), scaled by `defensive_ratio`. `build_response_to_citizen_collective()`: computes non-negative trait_rate from same-type overlaps; unconditionally shifts policy and trait mu/sigma toward zone averages (no sign-gate, no defensive response). Susceptibility `S = sigma*(1-\|cos θ\|)` applied to all pos/stddev shifts. Added `defensive_ratio=1.0` to all XML `<citizens>` blocks. Corrected stale stodem.py comment that falsely described a community defensive branch. |
| 22 | Engagement decay not applied | `citizen.py`, `stodem.in.xml` | Implemented as part of #8. Inside `apply_influence_shifts()`, each Gaussian type applies `alpha *= (1 + engagement_decay_rate)` (preference: `alpha = Im(theta)`; aversion: `alpha = pi - Im(theta)`) then clamps to `[0, pi/2]` before writing back to theta. A perfectly engaged citizen (alpha=0) experiences zero decay; disengagement is self-reinforcing. `engagement_decay_rate=0.01` added to all XML `<citizens>` blocks and `__init__`. Govern-phase decay will be added alongside TODO #5. |
| 7 | `Politician.persuade()` stub, never called | `politician.py` | Removed. All citizen Gaussian modifications are computed from the citizen side via `build_response_to_politician_influence()`. No politician-side entry point needed. |
| 10 | Politician move strategies 1+ unimplemented | `politician.py` | Strategy 1: teleport to highest-population patch and stay. Strategy 2: cycle through bottom-half patches by population (round-robin). |

---

## Active Bugs

No active bugs. See Resolved Items above.

---

## Core Simulation Physics

These tasks implement the interaction mechanics needed
for meaningful simulation results. See DESIGN.md §8
for the full interaction physics specification.

### 5. Implement `govern()` — `stodem.py`, `government.py`, `politician.py`

`govern()` returns immediately with no implementation.
Per DESIGN.md §7.5, it should run for
`num_govern_steps` time steps each cycle. Sub-tasks:

#### 5a. Compute `margin_of_victory` — `stodem.py`

After each election in `vote()`, compute each winning
politician's normalized margin over the next-closest
rival and store it as `politician.margin_of_victory`.
Currently the vote phase only identifies the top
vote-getter; it does not record the margin. This value
is needed by the political power calculation (§7.5.1).

#### 5b. Compute `political_power` — `politician.py` or `government.py`

For each elected politician, compute the scalar
`political_power` from three components (§7.5.1):

1. Zone population (`zone.curr_num_citizens`).
2. Margin of victory (from 5a).
3. Agreement/disagreement ratio using zone citizen
   averages and the politician's external positions:
   ```
   agreement = I(avg_Pcp, Ppp) + I(avg_Pca, Ppa)
             + I(avg_Tcp, Tpx)
   disagreement = I(avg_Pca, Ppp) + I(avg_Pcp, Ppa)
   political_power = zone_pop * margin
                   * (agreement
                      / (|disagreement| + eps))
   ```

Zone averages must be recomputed (or carried forward
from the last campaign step) before this calculation.

#### 5c. Compute per-dimension weights — `politician.py` or `government.py`

For each elected politician, compute preference and
aversion dimension weights from innate sigma (§7.5.2):
```
pref_weight_n = (1/sigma_innate_pref_n)
              / sum(1/sigma_innate_pref)
aver_weight_n = (1/sigma_innate_aver_n)
              / sum(1/sigma_innate_aver)
```

#### 5d. Apply forces to `Pge` — `government.py`, `stodem.py`

Each govern step, accumulate direction-only forces
from all elected politicians on each policy dimension
for both `Pge.mu` and `Pge.sigma` (§7.5.3):
```
w = political_power * dim_weight
force += w * sign(innate.param - Pge.param)
```
Preference forces pull `Pge` toward innate pref;
aversion forces push `Pge` away from innate aver.
Apply forces after all politicians are accumulated.

#### 5e. Apply natural policy spread — `government.py`, `stodem.py`

Once per govern cycle (not per step), broaden
`Pge.sigma` (§7.5.4):
```
Pge.sigma_n += spread_rate / Pge.sigma_n
```

New parameter needed: `spread_rate` (small positive
float, added to `<government>` in XML and to
`Government.__init__`).

### 9. Implement `build_response_to_well_being()` downstream effects — `citizen.py`

The method currently only computes
`self.well_being = sum(self.Pci_Pge_ol[0])`. Per
DESIGN.md §7.5.5 and §8.5, well-being should modulate
citizen engagement (orientation shifts) and potentially
policy position shifts.

**Decision**: Use the simple overlap-based `well_being`
for downstream engagement modulation. Implement so the
richer model from DESIGN.md §8.5 can slot in without
restructuring call sites. Specifically: encapsulate the
well-being→engagement mapping in a method so it can be
replaced independently of the overlap computation.

---

## Feature Completion

### ~~10. Implement politician move strategies 1+ — `politician.py`~~ ✓ DONE

All three move strategies are now implemented:
- Strategy 0: Random patch within zone (unchanged).
- Strategy 1: Teleport to highest-population patch
  in zone and stay there.
- Strategy 2: Cycle through low-to-middle population
  patches (bottom half by population, round-robin).

### 10b. Implement politician adapt strategies 1+ — `politician.py`

Only strategy 0 (present innate positions unchanged)
is coded. Strategies 1+ — including misrepresentation
via `policy_lie` and `trait_lie` — are unimplemented
despite the instance variables being set in
`reset_to_input()`.

**Action**: Define and implement strategies 1+. See
DESIGN.md §6.2 for the existing parameters.

### 11. Use `campaign_strategy` — `politician.py`

`self.campaign_strategy` is set in `__init__` via
`select_strategy()` but is never referenced anywhere.

**Action**: Define what campaign strategies control
and implement their effects, or remove the parameter
if not needed.

### 12. (Removed — primary election deferred)

### 13. Call `compute_data_range()` — `sim_control.py`, `stodem.py`

`compute_data_range()` computes policy and trait axis
limits for output but is never called. It is fully
implemented.

⚠️ **TODO QUESTION**: Should this be called at
simulation start for visualization setup, or is it
deferred until needed?

---

## Robustness / Architecture

### ~~16. Clarify overlap integral normalization — `gaussian.py`~~ ✓ DONE

Normalized inside `Gaussian.integral()`. Each Gaussian caches
`self_norm = (π·σ²)^0.75 · |cos θ|` in
`update_integration_variables()`; `integral()` divides raw
by `self_norm₁ · self_norm₂`. See DESIGN.md §4.2.

### 17. Move mutable class-level lists to `__init__()` — `world.py`

`World.zone_types`, `World.zones`, `World.citizens`,
`World.politicians`, and `World.properties` are
class-level mutable lists. If `World` is ever
instantiated more than once (tests, parameter sweeps),
these would accumulate across instances. The same
pattern exists in `SimControl` and `Hdf5`.

**Action**: Move list declarations into `__init__()`.

### 18. Address XDMF/HDF5 step count mismatch — `output.py`, `stodem.py`

`Xdmf.print_xdmf_xml()` writes entries for all
`total_num_steps` up front, but HDF5 datasets are
created incrementally. If the simulation terminates
early, the XDMF references nonexistent HDF5 datasets.

**Action**: Either write XDMF incrementally alongside
HDF5, or write XDMF only after simulation completes.

### 31. Remove `policy_influence`, `trait_influence`, `pander` — `politician.py`, `stodem.py`, `stodem.in.xml`

These parameters are superseded by the new govern-phase
design (DESIGN.md §7.5). `policy_influence` and
`trait_influence` are replaced by the computed
`political_power` scalar. `pander` is removed; the
blending concept may return as a future extension
(§7.5.3) but will use a different, more abstract
parameter name.

**Code changes needed**:

- `politician.py`: Remove `self.policy_influence`,
  `self.trait_influence`, and `self.pander`
  initialization from `reset_to_input()`. Remove the
  stale comment (lines 142–144) referencing
  `policy_influence` and `trait_influence`.
- `stodem.py`: Remove stale comment block (around
  line 376) that references `policy_influence`.
- `stodem.in.xml` (all three copies): Remove
  `<policy_influence_stddev>`,
  `<trait_influence_stddev>`, and `<pander_stddev>`
  from `<politicians>`.

### 29. Implement configurable initial theta distribution — `citizen.py`, `politician.py`, `stodem.in.xml`

Theta (orientation/engagement) is currently hardcoded
at initialization: preferences at `1j`, aversions at
`(pi-1)j`. All agents in a run start with identical
engagement magnitude. A more realistic initialization
would draw Im(theta) from a configurable distribution,
producing a population with varied initial engagement
levels.

The `*_orien_stddev` XML parameters
(`policy_pref_orien_stddev`, `policy_aver_orien_stddev`,
`trait_pref_orien_stddev`, `trait_aver_orien_stddev` in
`<citizens>`, and analogues in `<politicians>`) are
already present in all XML files and reserved for this
purpose. They are not yet read by the code. See
DESIGN.md §4.1 for the design rationale.

**Action**: When implementing, read the `*_orien_stddev`
parameters and use them to sample Im(theta) from a
truncated normal distribution with the hardcoded value
as the mean and the parameter as the stddev, clamped to
[0, π/2) for preferences and (π/2, π] for aversions.
