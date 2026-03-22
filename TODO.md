# STODEM Implementation Task List

## Summary

The simulation framework is well-structured and the
core data model (Gaussians, zones, patches, citizens,
politicians, government) is largely in place. The
campaign loop skeleton exists and data output
(HDF5/XDMF) works. However, the core interaction
physics — influence shift mechanics, the governing
phase, and several agent strategies — are stubs or
placeholders.

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

---

## Active Bugs

### 19. `build_response_to_politician_influence()` has multiple accumulation bugs — `citizen.py`

The current implementation has three bugs:

1. **List extension instead of per-dimension addition**:
   `self.policy_orien_shift += self.Pcp_Ppp_ol` uses
   Python list `+=`, which *extends* the shift list
   with overlap list elements rather than adding
   values per-dimension. The shift array grows with
   each iteration instead of accumulating numeric
   values.

2. **Not indexed by politician**: Inside the
   `for politician in self.politician_list` loop, the
   code adds the *entire* overlap list
   (`self.Pcp_Ppp_ol`) each iteration rather than the
   current politician's overlap values
   (`self.Pcp_Ppp_ol[pol_idx]`). The loop variable
   `politician` is never used.

3. **Missing absolute values**: The design
   (DESIGN.md §8.6.2) specifies that *absolute values*
   of overlap integrals drive engagement
   (`|I(Pcp,Ppp)|`), but the code adds raw integral
   values without taking absolute values.

**Action**: Rewrite as:
```python
for pol_idx, politician in enumerate(
        self.politician_list):
    self.policy_orien_shift += (
        np.abs(self.Pcp_Ppp_ol[pol_idx])
        + np.abs(self.Pca_Ppa_ol[pol_idx])
        + np.abs(self.Pcp_Ppa_ol[pol_idx])
        + np.abs(self.Pca_Ppp_ol[pol_idx]))
    self.trait_orien_shift += (
        np.abs(self.Tcp_Tpx_ol[pol_idx])
        + np.abs(self.Tca_Tpx_ol[pol_idx]))
```

Also convert shift arrays from Python lists to numpy
arrays in `prepare_for_influence()`.

### 20. `build_response_to_citizen_collective()` has same accumulation bugs — `citizen.py`

Same three issues as #19 applied to the citizen
collective integrals: list extension instead of
per-dimension addition, overlaps not indexed by zone,
and missing absolute values.

**Action**: Fix analogously to #19, indexing by zone
and using absolute values per DESIGN.md §8.6.2.

### 8. Influence shifts accumulated but never applied — `citizen.py`

`prepare_for_influence()` initializes six shift arrays:
`policy_orien_shift`, `policy_pos_shift`,
`policy_stddev_shift`, `trait_orien_shift`,
`trait_pos_shift`, `trait_stddev_shift`. The three
`build_response_to_*()` methods accumulate values into
the orientation shift arrays, but none of the six shift
arrays are ever applied back to the citizen's Gaussian
parameters (`stated_policy_pref`, `stated_policy_aver`,
`stated_trait_pref`, `stated_trait_aver`). The position
and stddev shift arrays are initialized but never
written to at all.

**Action**: Implement an `apply_influence_shifts()`
method on `Citizen` and call it at the end of each
campaign step, after all `build_response_to_*()` calls.
The method should apply shifts to theta, mu, and sigma
per DESIGN.md §8.6.5, enforce sigma_floor per §8.6.5,
apply engagement decay per §8.6.6, and update derived
variables per §8.6.7.

Depends on: #19, #20 (fix accumulation bugs first).

⚠️ **TODO QUESTION**: Should influence shifts be
applied using the susceptibility model (DESIGN.md
§8.6.3) or the force/momentum model (DESIGN.md
§8.6.3)? This choice must be made before position and
spread shifts can be fully implemented. See DESIGN.md
§8.6.9, question 1.

---

## Core Simulation Physics

These tasks implement the interaction mechanics needed
for meaningful simulation results. See DESIGN.md §8
for the full interaction physics specification.

### 21. Implement trait-gates-policy shift mechanics — `citizen.py`

The `build_response_to_politician_influence()` and
`build_response_to_citizen_collective()` methods
currently only accumulate orientation (engagement)
shifts. Per DESIGN.md §8.1 and §8.6.3, they should
also accumulate position (mu) and spread (sigma)
shifts following the trait-gates-policy principle:

- Compute trait_sum for each influence source.
- If positive: shift policy mu and sigma toward
  source.
- If negative: narrow preference sigma, shift
  aversion mu toward source's positions (targeted
  backlash, scaled by `defensive_ratio`).
- Apply susceptibility function S(sigma, theta) to
  modulate shift magnitudes.

Depends on: #19, #20 (fix accumulation bugs first).
See also: DESIGN.md §8.6.3 for full formulas.

### 22. Implement engagement decay — `citizen.py`, `stodem.py`

Per DESIGN.md §8.2 and §8.6.6, every simulation step,
every citizen's theta for every stated Gaussian should
drift toward pi/2 by `engagement_decay_rate`. This is
applied after influence shifts (DESIGN.md §8.6.6).

New parameter needed: `engagement_decay_rate`
(DESIGN.md §8.6.8).

**Action**: Add decay step in `apply_influence_shifts()`
(or as a separate method called after it) and add the
parameter to XML config.

### 23. Expand shift arrays from 6 to 12 — `citizen.py`

Per DESIGN.md §8.6.1, separate arrays are needed for
preference and aversion Gaussians because they respond
differently under negative trait alignment (§8.1). The
current six arrays (three per domain: policy and trait)
must be expanded to twelve (three per Gaussian type:
Pcp, Pca, Tcp, Tca).

Depends on: #19, #20 (fix accumulation bugs first,
then expand).

### 5. Implement `govern()` — `stodem.py`, `government.py`

`govern()` returns immediately with no implementation.
Per DESIGN.md §7.5, it should:

- Update `Government.enacted_policy` based on elected
  politicians' positions using the policy force model
  (DESIGN.md §7.5.1).
- Use political capital weighting
  (DESIGN.md §7.5.2) and zone population weighting
  (DESIGN.md §7.5.3).
- Downstream effects (DESIGN.md §7.5.4) are handled
  automatically by existing overlap integral code
  once `Pge` is updated.

New parameters needed: `governance_rate`.

⚠️ **TODO QUESTION**: The `pander` parameter controls
how much a politician governs based on promises vs.
ideology (DESIGN.md §7.5.1). Currently `pander` is
initialized but never used. Should the govern phase
implementation include pander from the start, or should
it initially assume all politicians govern from their
innate positions?

### 7. Implement `Politician.persuade()` — `politician.py`, `stodem.py`

The method iterates over citizens in the patch with
`pass`. The actual persuasion logic (shifting citizen
Gaussian orientations, positions, and spreads based on
politician-citizen overlap integrals) is described in
DESIGN.md §8.3 and §8.6.2–§8.6.3. `persuade()` is
also never called from the campaign loop.

⚠️ **TODO QUESTION**: The current campaign loop already
calls `build_response_to_politician_influence()` for
each citizen, which is where politician-driven shifts
should accumulate. Should `persuade()` be removed in
favor of expanding
`build_response_to_politician_influence()`, or should
`persuade()` remain as the politician-side entry point
that triggers the citizen-side response?

### 9. Implement `build_response_to_well_being()` downstream effects — `citizen.py`

The method currently only computes
`self.well_being = sum(self.Pci_Pge_ol[0])`. Per
DESIGN.md §7.5.4 and §8.5, well-being should modulate
citizen engagement (orientation shifts) and potentially
policy position shifts. The exact downstream effects
depend on the well-being model chosen (see DESIGN.md
§8.5).

⚠️ **TODO QUESTION**: The well-being model in
DESIGN.md §8.5 is marked as under active design.
Should this task use the current simple implementation
(overlap of ideal policy with enacted policy) for the
downstream engagement modulation, or should it wait
for the full well-being model to be designed?

---

## Feature Completion

### 10. Implement politician move strategies 1+ — `politician.py`

Only strategy 0 (random patch within zone) is coded.
Strategies 1 and above silently do nothing.

⚠️ **TODO QUESTION**: What should strategies 1 and 2
do? The XML defines cumulative probabilities
"0.5,0.75,1.0" for three strategies, but only
strategy 0 has defined behavior. Are there specific
movement patterns intended (e.g., target
high-population patches, move toward favorable
citizens)?

### 10b. Implement politician adapt strategies 1+ — `politician.py`

Only strategy 0 (present innate positions unchanged)
is coded. Strategies 1+ — including pandering and
misrepresentation via `policy_lie`, `trait_lie`, and
`pander` — are unimplemented despite the instance
variables being set in `reset_to_input()`.

**Action**: Define and implement strategies 1+. See
DESIGN.md §6.2 for the existing parameters.

### 11. Use `campaign_strategy` — `politician.py`

`self.campaign_strategy` is set in `__init__` via
`select_strategy()` but is never referenced anywhere.

**Action**: Define what campaign strategies control
and implement their effects, or remove the parameter
if not needed.

### 12. Implement primary election phase — `stodem.py`

The main cycle loop has commented-out calls to
`primary()` and `primary_vote()`. Neither function
exists.

⚠️ **TODO QUESTION**: Is the primary election phase
still in scope? If so, is there a design for how it
differs from the general election (different candidate
pools, intra-party competition)?

### 13. Call `compute_data_range()` — `sim_control.py`, `stodem.py`

`compute_data_range()` computes policy and trait axis
limits for output but is never called. It is fully
implemented.

⚠️ **TODO QUESTION**: Should this be called at
simulation start for visualization setup, or is it
deferred until needed?

---

## Robustness / Architecture

### 16. Clarify overlap integral normalization — `gaussian.py`

The overlap integral formula returns raw values that
are not normalized by the geometric mean of
self-overlaps. Whether values are truly bounded to
[-1, +1] depends on the Gaussian parameters. See
DESIGN.md §4.2.

⚠️ **TODO QUESTION**: Should the integral be
normalized? If so, should normalization be done inside
`Gaussian.integral()` or at the call sites? (See also
DESIGN.md §4.2.)

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
