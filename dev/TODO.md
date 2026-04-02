# STODEM Task List

> **Document hierarchy:** Tasks are organized by the
> level of the design chain they affect: VISION →
> ARCHITECTURE → DESIGN → PSEUDOCODE → CODE. Each
> item cites the relevant document section. For the
> design documents themselves, see `VISION.md`,
> `ARCHITECTURE.md`, `DESIGN.md`, `PSEUDOCODE.md`.

---

## VISION

No pending items.

---

## ARCHITECTURE

No pending items.

---

## DESIGN

- [ ] **#12 — Primary campaign/vote phases**
  (DESIGN §7, deferred). The simulation currently
  runs Campaign → Vote → Govern. The full design
  includes Primary Campaign → Primary Vote before
  the general campaign. Deferred until the general
  cycle is fully validated.

- [ ] **Well-being model expansion** (DESIGN §8.5).
  The current well-being is a simple overlap
  `sum(I(Pci, Pge))`. The candidate model includes
  resource accumulation, perceived satisfaction,
  community fit, policy consistency, policy
  stability, and resentment. Design questions in
  §8.5 must be resolved before implementation.

- [ ] **Ideal policy cylinder direction**
  (DESIGN §12.3). +z (preference) and -z (aversion)
  are taken. If ideal policy ever appears in a
  combined pane with preference and aversion, it
  needs its own convention: a distinct glyph shape,
  a third z-offset, or defer until combined panes
  are actually needed. Currently uses z_pos.

- [ ] **Policy-space view** (DESIGN §12.3, deferred).
  A second XDMF topology where mu values position
  glyphs along policy axes in an abstract coordinate
  space. Deferred until the geographic view is
  validated.

- [ ] **Citizen-politician alignment glyph**
  (DESIGN §12.3, deferred). A composite glyph
  encoding the relationship between a citizen's
  preference and a politician's apparent position,
  showing alignment directly. Deferred until the
  per-agent glyph design is working.

---

## PSEUDOCODE

No pending items.

---

## CODE

- [ ] **Glyph verification re-run** (DESIGN §12.3).
  After the separate-files-per-grid-type redesign,
  needs re-run with quickTest to verify: separate
  XDMF files load correctly, each pane shows
  correct point count, time stepping synchronizes
  across readers.

---

## ARCHIVE

The following items have been resolved. They are
retained here for reference; the `#` numbers are
stable cross-references used throughout DESIGN.md.

| # | Issue | File | Resolution |
|---|-------|------|------------|
| 1 | `compute_patch_well_being` called before defined | `stodem.py` | Moved definition above campaign loop |
| 2 | `hdf5.close` property ref, not method call | `output.py` | Changed to `hdf5.close()`; added `close()` method to `Hdf5` |
| 3 | `Citizen.policy_attitude()` refs `.pos` not `.mu` | `citizen.py` | Changed to `.mu` |
| 4 | Zone object vs. integer comparison in `vote_for_candidates()` | `citizen.py` | Added `zone_index` to `Zone`; filter on both `zone_type` and `zone_index` |
| 6 | `compute_overlap()` incomplete | `gaussian.py` | Superseded by fully implemented `Gaussian.integral()` |
| 14 | `score_candidates()` ignores `policy_trait_ratio` | `citizen.py` | Applied weighting: `w_policy = 0.5 + ratio`, `w_trait = 0.5 - ratio` |
| 15 | Static participation probability | `citizen.py` | Now computed dynamically as `P(vote) = mean(\|cos(theta)\|)` across all stated Gaussians |
| Q4 | No scaling for citizen-collective influence | `citizen.py`, `stodem.in.xml` | `collective_influence_rate` parameter added |
| 24 | Politician Gaussian theta not sign-convention-compliant | `politician.py` | Preference Gaussians initialized to `±1j`; aversion to `(pi-1)j` |
| 25 | `collective_influence_rate` missing from XML files | `stodem.in.xml` | Added to all XML files |
| 26 | Preference Gaussian theta initialized as ±1j | `citizen.py`, `politician.py` | All changed to `np.full(N, 1j)` |
| 27 | Sigma initialization allows negative values | `citizen.py`, `politician.py` | Wrapped with `np.abs()` |
| 28 | `policy_attitude()` dead code | `citizen.py` | Removed |
| 30 | Stale/incomplete comment block in `vote()` | `stodem.py` | Removed |
| 19 | `build_response_to_politician_influence()` accumulation bugs | `citizen.py` | Rewrote with `enumerate`, per-politician indexing, `np.abs()` |
| 20 | `build_response_to_citizen_collective()` same bugs | `citizen.py` | Fixed with #19 |
| 8 | Influence shifts accumulated but never applied | `citizen.py`, `stodem.py` | Implemented `apply_influence_shifts()` with two-phase design |
| 23 | Expand shift arrays from 6 to 12 | `citizen.py` | Twelve named arrays per Gaussian type |
| 21 | Trait-gates-policy shift mechanics | `citizen.py`, `stodem.in.xml` | Full §8.6.3–§8.6.4 physics implemented |
| 22 | Engagement decay not applied | `citizen.py`, `stodem.in.xml` | Proportional decay in `apply_influence_shifts()` |
| 7 | `Politician.persuade()` stub, never called | `politician.py` | Removed; influence computed from citizen side |
| 10 | Politician move strategies 1+ unimplemented | `politician.py` | Strategy 1: teleport to highest-pop patch. Strategy 2: cycle through bottom-half patches |
| 31 | `policy_influence`, `trait_influence`, `pander` obsolete | `politician.py`, `stodem.in.xml` | Removed from all files |
| 13 | `compute_data_range()` never called | `sim_control.py`, `stodem.py` | Called in `main()` after `world.populate()` |
| 17 | Mutable class-level lists/variables | `world.py`, `sim_control.py`, `output.py`, `government.py` | Moved to `__init__()` |
| 18 | XDMF/HDF5 step count mismatch | `output.py`, `stodem.py` | XDMF written after simulation using actual steps |
| 29 | Configurable initial theta distribution | `gaussian.py`, `citizen.py`, `politician.py` | `sample_theta()` helper; XML-driven |
| 10b | Politician adapt strategies 1+ unimplemented | `politician.py` | Strategy 1: pander. Strategy 2: shift away from aversions |
| 11 | `campaign_strategy` unused | `politician.py`, `stodem.in.xml` | Removed |
| 5 | Implement `govern()` (5a–5e) | `stodem.py`, `politician.py`, `government.py` | Full govern phase with political power, dimension weights, forces, spread |
| 9 | `build_response_to_well_being()` downstream effects | `citizen.py` | `\|well_being\|` drives engagement via `_well_being_to_engagement()` |
| 32 | `Pge.sigma` can go negative | `government.py`, `stodem.py` | `sigma_floor` applied after forces and spread |
| 33 | `political_power` unscaled | `stodem.py` | Population-normalized |
| 34 | Agreement/disagreement ratio unbounded | `politician.py` | Replaced with bounded `max(0, agreement + disagreement)` |
