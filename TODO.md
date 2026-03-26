# STODEM Implementation Task List

## Summary

The simulation is fully implemented through the
campaign → vote → govern loop. All core mechanics are
in place: Gaussian influence physics, trait-gates-
policy, engagement decay, political power, govern-phase
force application, well-being feedback, and diagnostic
output. The govern-phase force scale has been
stabilized via population normalization and a bounded
mandate formula (TODO #33, #34); Pge sigma is
protected by a floor (TODO #32).

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
| 31 | `policy_influence`, `trait_influence`, `pander` obsolete | `politician.py`, `stodem.py`, `stodem.in.xml` | Removed all three parameters from `reset_to_input()` and all three XML files. Updated stale `stodem.py` design comments to reference `policy_persuasion` instead. |
| 13 | `compute_data_range()` never called | `sim_control.py`, `stodem.py` | Called in `main()` after `world.populate()`, grouped with XDMF/HDF5 setup. Results stored in `world.policy_limits` and `world.trait_limits` for future visualization use. |
| 17 | Mutable class-level lists/variables | `world.py`, `sim_control.py`, `output.py`, `government.py` | Moved all mutable class-level declarations into `__init__()` as instance variables. Changed `World.xxx`, `SimControl.xxx`, `Hdf5.xxx`, `Xdmf.xxx` references to `self.xxx` throughout. |
| 18 | XDMF/HDF5 step count mismatch | `output.py`, `stodem.py` | XDMF is now written after the simulation completes using `sim_control.curr_step` (actual steps completed) instead of `total_num_steps` (planned). Handles early termination and govern-phase gaps. |
| 29 | Configurable initial theta distribution | `gaussian.py`, `citizen.py`, `politician.py` | Added `sample_theta()` helper in `gaussian.py`. Citizen and politician Gaussian initialization now reads `*_orien_stddev` XML params; numeric values sample Im(theta) from a clamped normal distribution, non-numeric values (e.g. "imaginary") keep the hardcoded default. |
| 10b | Politician adapt strategies 1+ unimplemented | `politician.py` | Strategy 1: pander toward zone-average citizen preferences using `policy_lie`/`trait_lie` blend. Strategy 2: shift away from zone-average citizen aversions. Both start from innate positions via `_copy_innate_to_external()`. |
| 11 | `campaign_strategy` unused | `politician.py`, `stodem.in.xml` | Removed. Move and adapt strategies already cover campaign behavior. Removed `self.campaign_strategy` and `cumul_campaign_strategy_probs` from all XML files and DESIGN.md. |
| 5 | Implement `govern()` (5a–5e) | `stodem.py`, `politician.py`, `government.py`, `stodem.in.xml` | 5a: `vote()` now computes `margin_of_victory` for each winner. 5b: `Politician.compute_political_power()` from zone pop, margin, agreement/disagreement ratio. 5c: `Politician.compute_dimension_weights()` from innate sigma inverse. 5d: `govern()` accumulates preference-attraction and aversion-repulsion forces on Pge each step. 5e: Natural spread `sigma += spread_rate/sigma` once per cycle; `spread_rate` added to `<government>` XML and `Government.__init__`. |
| 9 | `build_response_to_well_being()` downstream effects | `citizen.py` | Well-being now drives engagement: `\|well_being\|` shifts all four Gaussian types toward their engaged poles via `_well_being_to_engagement()`. Encapsulated so the richer model (resource, resentment, etc.) can replace the mapping independently. |
| 32 | `Pge.sigma` can go negative under politician forces | `government.py`, `stodem.py` | Added `sigma_floor` to `Government.__init__()` (reads from `<citizens>` XML, same floor as citizen Gaussians). `np.maximum(..., gov.sigma_floor)` applied after each govern step's force accumulation and after the natural spread. Documented in DESIGN.md §7.5.3–§7.5.4. |
| 33 | `political_power` unscaled — proportional to raw zone population | `stodem.py` | In `govern()`, after computing all elected politicians' power, divide each by `total_pop = sum(zone.curr_num_citizens for pol in elected)`. Forces are now dimensionless and world-size independent; relative power distribution is preserved. Documented in DESIGN.md §7.5.1. |
| 34 | Agreement/disagreement ratio unbounded (explodes when `\|disagreement\|` → 0) | `politician.py` | Replaced `agreement / (\|disagreement\| + eps)` with `max(0, agreement + disagreement)`. Since cross-type integrals are non-positive by the theta sign convention, this equals `agreement - \|disagreement\|`, bounded to `[-(N_policy+N_trait), +(N_policy+N_trait)]`. Clamped to 0 so politicians with negative net alignment have zero governing power. Documented in DESIGN.md §7.5.1. |

---

## Active Bugs

No active bugs. See Resolved Items above.

---

## All Tasks Resolved

All implementation tasks have been completed. See the
Resolved Items table above for details on each item.


