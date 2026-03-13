# STODEM Code Review: Incomplete Sections and To-Do List

## Summary

The simulation framework is well-structured and the core data model (Gaussians, zones, patches,
citizens, politicians, government) is largely in place. The campaign loop skeleton exists and
data output (HDF5/XDMF) works. However, large portions of the simulation physics are stubs or
placeholders — particularly the influence/persuasion mechanics, the governing phase, the primary
election phase, and several utility functions.

---

## Bugs (Code That Will Fail at Runtime)

### 1. `compute_patch_well_being` called before it is defined — `stodem.py`

`compute_patch_well_being(world)` is called at line 365 inside the campaign `for` loop, but
the function is defined as a nested `def` at line 378, *after* the loop body ends. In Python,
a nested `def` only creates the function when that line executes. Since the loop runs first,
calling the function on line 365 will raise a `NameError`. The function definition needs to
be moved above the loop, or promoted to a module-level function.

### 2. `hdf5.close` is a property reference, not a method call — `stodem.py`

At the end of `main()`, `hdf5.close` reads a property but never calls it. It should be
`hdf5.close()`. Additionally, the `Hdf5` class never defines a `close()` method, so this
will raise an `AttributeError` regardless. A `close()` method that calls `Hdf5.h_fid.close()`
needs to be added to the `Hdf5` class in `output.py`.

### 3. `Citizen.policy_attitude()` references non-existent attribute — `citizen.py`

`policy_attitude()` accesses `self.stated_policy_pref.pos`, but the `Gaussian` class stores
this value as `.mu`, not `.pos`. This will raise an `AttributeError` when called.

### 4. Zone object vs. integer comparison in `vote_for_candidates()` — `citizen.py`

In `vote_for_candidates()`, the check `if (politician.zone != zone_index)` compares
`politician.zone`, which is a `Zone` object, against `zone_index`, which is an integer
drawn from `patch.zone_index`. These types will never be equal, so every politician will
be skipped. The intended comparison is likely against the politician's zone index within
the current zone type.

---

## Incomplete / Stub Implementations

### 5. `govern()` is an empty stub — `stodem.py`

`govern()` returns immediately with no implementation. This is the entire governing phase.
Per the design, it should: update `Government.enacted_policy` based on the elected
politicians' positions, compute citizen well-being based on alignment between
`ideal_policy_pref` and `enacted_policy`, and feed that well-being back into participation
probability and citizen policy shifts for the next cycle.

### 6. `compute_overlap()` is incomplete — `gaussian.py`

The function computes `alpha_1`, `alpha_2`, and `zeta` but then sets `integral = np.pi`
with no further calculation and no `return` statement. The actual integral formula (matching
`Gaussian.integral()`) is not finished. It is also never called anywhere.

### 7. `Politician.persuade()` does nothing — `politician.py`

The method iterates over citizens in the patch with `pass`, then adds a trivial random
value to `world.properties[0].data`. The actual persuasion logic (shifting citizen Gaussian
orientations, positions, and spreads based on politician–citizen overlap integrals) is
described in detail in the design comments but not implemented. `persuade()` is also never
called from the campaign loop.

### 8. Influence shifts are accumulated but never applied — `citizen.py`

`prepare_for_influence()` initializes six shift arrays: `policy_orien_shift`,
`policy_pos_shift`, `policy_stddev_shift`, `trait_orien_shift`, `trait_pos_shift`,
`trait_stddev_shift`. The three `build_response_to_*()` methods accumulate values into the
orientation shifts, but none of the shifts are ever applied back to the citizen's Gaussian
parameters (`stated_policy_pref`, `stated_policy_aver`, `stated_trait_pref`, etc.).
The position and stddev shift arrays are initialized but never written to at all.

### 9. `build_response_to_well_being()` is minimal — `citizen.py`

The method computes `self.well_being` from `Pci_Pge_ol[0]`, but per the design this
well-being value should then modulate the citizen's engagement (orientation shifts),
participation probability, and policy position shifts. None of that downstream effect
is implemented.

### 10. `Politician.move()` only implements strategy 0 — `politician.py`

Only strategy index 0 (random patch within zone) is coded. Strategies 1 and above
silently do nothing. Same applies to `adapt_to_patch()`, which only handles strategy 0
(present innate positions unchanged). Higher strategies — including pandering and
misrepresentation via `policy_lie`, `trait_lie`, and `pander` — are unimplemented
despite the instance variables being set in `reset_to_input()`.

### 11. `Politician.campaign_strategy` is selected but never used — `politician.py`

`self.campaign_strategy` is set in `__init__` via `select_strategy()` but is never
referenced anywhere in the code.

### 12. Primary election phase is absent — `stodem.py`

The main cycle loop has commented-out calls to `primary()` and `primary_vote()`. Neither
function exists. The primary phase is not implemented at all.

### 13. `compute_data_range()` is defined but never called — `sim_control.py`

This method computes policy and trait axis limits for output purposes but is never invoked
from `main()` or anywhere else.

---

## Logic / Design Gaps

### 14. `score_candidates()` ignores `policy_trait_ratio` — `citizen.py`

The design specifies that policy and trait overlap integrals should be weighted against
each other by `self.policy_trait_ratio`. The scoring method sums all integrals equally
without applying this ratio.

### 15. Participation probability is static — `citizen.py`

`self.participation_prob` is drawn once at initialization and never updated. The design
describes six factors that should dynamically influence it (personality alignment, policy
alignment, prior voting history, zone-level agreement, well-being, and whether the
previously supported candidate won). None of these updates are implemented.

### 16. `Gaussian.integral()` result is not normalized — `gaussian.py`

The comment states values range from -1 to +1, but the formula returns the raw integral
`(pi/zeta)^1.5 * exp(-xi*d^2) * cos(theta1) * cos(theta2)`, which is *not* normalized.
Normalization requires dividing by the geometric mean of the two self-overlaps. Whether
this is intentional or an oversight should be clarified.

### 17. Mutable class-level lists in `World` — `world.py`

`World.zone_types`, `World.zones`, `World.citizens`, `World.politicians`, and
`World.properties` are declared as class-level mutable lists (e.g., `zones = []`). If
`World` is ever instantiated more than once in a process (e.g., in tests or parameter
sweeps), these lists will accumulate data across instances rather than resetting. Consider
initializing them in `__init__` instead.

### 18. XDMF/HDF5 step count mismatch if simulation ends early — `output.py`, `stodem.py`

`Xdmf.print_xdmf_xml()` writes entries for all `total_num_steps` time steps up front,
but HDF5 datasets are only created as each step executes. If the simulation terminates
early (e.g., due to an error), the XDMF file will reference HDF5 datasets that do not
exist, making the output unreadable by Paraview.

---

## To-Do List (Prioritized)

### Critical (prevent runtime)
- [ ] Fix `compute_patch_well_being` placement — move definition above the campaign loop or make it a module-level function in `stodem.py`
- [ ] Fix `hdf5.close` → `hdf5.close()` and add a `close()` method to `Hdf5` in `output.py`
- [ ] Fix zone object vs. integer comparison in `Citizen.vote_for_candidates()` in `citizen.py`
- [ ] Fix `Citizen.policy_attitude()` — change `.pos` to `.mu`

### Core Simulation Physics (needed for meaningful results)
- [ ] Apply accumulated influence shifts to citizen Gaussian parameters at the end of each campaign step (`citizen.py`)
- [ ] Implement `Politician.persuade()` and call it from the campaign loop (`politician.py`, `stodem.py`)
- [ ] Implement `govern()` — update `enacted_policy`, compute well-being, update participation probability (`stodem.py`, `government.py`)
- [ ] Implement `build_response_to_well_being()` downstream effects on citizen state (`citizen.py`)
- [ ] Apply `policy_trait_ratio` weighting in `score_candidates()` (`citizen.py`)
- [ ] Update `participation_prob` dynamically during the simulation (`citizen.py`)

### Feature Completion
- [ ] Implement politician move strategies 1+ in `Politician.move()` (`politician.py`)
- [ ] Implement politician adapt strategies 1+ (pandering, misrepresentation) using `policy_lie`, `trait_lie`, `pander` (`politician.py`)
- [ ] Use `campaign_strategy` in campaign behavior (`politician.py`)
- [ ] Implement primary campaign and primary vote phases; add `primary()` and `primary_vote()` functions (`stodem.py`)
- [ ] Complete `compute_overlap()` and integrate it into the workflow, or remove if superseded by `Gaussian.integral()` (`gaussian.py`)
- [ ] Call `compute_data_range()` at simulation start if data axis limits are needed (`stodem.py`)

### Robustness / Design
- [ ] Clarify whether `Gaussian.integral()` should be normalized to [-1, +1]; implement normalization if so (`gaussian.py`)
- [ ] Move mutable class-level list declarations into `World.__init__()` to avoid cross-instance contamination (`world.py`)
- [ ] Address XDMF/HDF5 step count mismatch — either write XDMF incrementally or only after simulation completes (`output.py`, `stodem.py`)
