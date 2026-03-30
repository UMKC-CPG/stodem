# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read the DESIGN.md file.

This file will present the high level architectural design of the program.
It will also provide nuanced understanding of specific design decisions.
It is intended to serve as the main thinking space for program design, but it should be intimately linked to the contents of the TODO.md file (discussed next) so that architecture and design decisions have clear connections to specific and actionionable development tasks.

## Read the TODO.md file.

This file describes detailed action items that are oriented toward specific program code and implementation tasks. The intent is for the action items here are derived from and linked to the effective difference between the DESIGN.md document and the current state of the implementation of the program.

## Project Overview

STODEM (Stochastic Democracy Simulation) is a multi-agent based simulation modeling democratic processes. Politicians and citizens interact in a hierarchical geographic world divided into nested zones (districts → states → countries). The simulation uses complex Gaussian representations for policy/trait positions.

## Running the Simulation

```bash
# Set environment variable pointing to RC file directory
export STODEM_RC=/path/to/stodem/.stodem

# Run with defaults (reads stodem.in.xml, outputs stodem.hdf5/xdmf)
python3 src/scripts/stodem.py

# Run with custom input/output
python3 src/scripts/stodem.py -i custom.xml -o output_prefix

# Quick test
cd jobs/quickTest && python3 ../../src/scripts/stodem.py
```

## Build System (CMake)

```bash
mkdir -p build/release && cd build/release
cmake -DCMAKE_BUILD_TYPE=RELEASE ../..
make && make install  # Installs to $STODEM_DIR/bin
```

## Dependencies

- Python 3 with: `lxml`, `numpy`, `h5py`
- CMake 3.1.0+ (for installation)
- Optional: Fortran compiler (gfortran/ifort) for future expansion

## Architecture

The codebase has been refactored from a monolithic `stodem.py` into separate modules under `src/scripts/`:

| Module | Contents |
|---|---|
| `stodem.py` | Entry point, design discussion comments, main simulation loop |
| `settings.py` | `ScriptSettings` — XML + command line config, loads `$STODEM_RC/stodemrc.py` |
| `sim_control.py` | `SimControl`, `SimProperty` — simulation phases and data range computation |
| `world.py` | `World` — main container; computes patch-level well-being, citizen Gaussian stats, and zone-upsampled politician stats via `compute_patch_well_being()`, `compute_patch_gaussian_stats()`, `compute_patch_politician_stats()` |
| `diagnostics.py` | Diagnostic utilities for simulation debugging |
| `zone.py` | `Zone` — geographic region at one hierarchy level |
| `patch.py` | `Patch` — basic grid unit containing citizens |
| `citizen.py` | `Citizen` — agent with Gaussian policy/trait preferences, aversions, ideal positions |
| `politician.py` | `Politician` — agent with innate and external policy/trait positions, strategies |
| `government.py` | `Government` — enacts policies affecting citizen well-being |
| `gaussian.py` | `Gaussian` — complex Gaussian functions with overlap integral computation |
| `output.py` | `Hdf5`, `Xdmf` — standard output; `GlyphHdf5`, `GlyphXdmf`, `write_paraview_script` — cylinder glyph output |
| `random_state.py` | Global `rng` (numpy default_rng, seed=8675309) |

### Simulation Flow

```
main() → for each cycle:
    campaign() → vote() → govern()
```

### Key Mathematical Concepts

**Complex Gaussians**: g(x;σ,μ,θ) = 1/(σ√(2π)) × exp(-(x-μ)²/(2σ²)) × exp(iθ)
- μ (mu): position on policy/trait axis — where the agent stands
- σ (sigma): spread/certainty — strength of attachment to that position
- θ (theta): orientation/engagement — theta=0 fully engaged, theta=π/2 fully apathetic

**Overlap Integral**: Measures alignment between two Gaussians
- I(G1,G2) = (π/ζ)^1.5 × exp(-ξd²) × cos(θ1) × cos(θ2)
- Values range -1 to +1

**Gaussian Notation**:
- Pcp/Pca/Pci: Policy citizen preference/aversion/ideal
- Ppp/Ppa: Policy politician preference/aversion (apparent)
- Pge: Policy government enacted
- Tcp/Tca: Trait citizen preference/aversion
- Tpx: Trait politician external

### Core Interaction Physics

Full architectural design is in `DESIGN.md`. Design
comments also in `stodem.py` (line 226+). Key principles:

- **Trait gates policy**: Trait overlap magnitude determines how much policy positions shift; trait overlap sign determines the type of shift (attraction vs. defensive rigidity)
- **Engagement from |overlap|**: Both agreement and disagreement increase engagement (absolute value of overlap shifts theta toward real)
- **Engagement decay**: Constant decay toward apathy each step (engagement_decay_rate parameter, future-dynamic)
- **Defensive response**: Negative trait alignment causes pref sigma to narrow and aver sigma to broaden (defensive_ratio parameter, future-dynamic)
- **Scoring weights**: policy_trait_ratio (clamped to [-0.5, +0.5]) weights policy vs. trait in candidate scoring
- **Vote probability**: P(vote) = mean(|cos(theta)|) across all stated Gaussians — engagement directly determines turnout

## Configuration (stodem.in.xml)

Key sections:
- `sim_control`: num_cycles, num_campaign_steps, num_govern_steps, data_resolution
- `world`: patch_size, num_policy_dims, num_trait_dims, zone_type_N (hierarchy config)
- `citizens`: policy/trait stddev parameters, policy_trait_ratio
- `politicians`: policy/trait stddevs, influence/lie parameters, strategy probabilities
- `government`: policy position/spread parameters

## Output Files

- `*.hdf5`: Binary simulation data with compression.
  Two groups: `CitizenGeoData` (patch-level citizen
  averages) and `PoliticianGeoData` (zone-upsampled
  politician averages). Fields: `WellBeing`, plus
  mu/sigma/cos_theta for each Gaussian per dimension.
  Citizen fields: `PolicyPref`, `PolicyAver`,
  `IdealPolicy` (per policy dim); `TraitPref`,
  `TraitAver` (per trait dim). Politician fields:
  `InnPolicyPref`, `InnPolicyAver`, `ExtPolicyPref`,
  `ExtPolicyAver` (per policy dim); `InnTrait`,
  `ExtTrait` (per trait dim) — suffixed `_ZT{zt}`.
- `*.xdmf`: XML metadata for Paraview; written after
  simulation completes using actual steps written.
- `*_glyphs/`: Subdirectory containing all glyph
  data files (HDF5 + XDMF). Copy this directory
  for drag-and-drop transfer to another machine.
  - `*_glyphs.hdf5`: Cylinder glyph data for all
    Gaussian types. Groups per type with per-step
    sub-groups: `mu_dim{d}`, `sigma_dim{d}`,
    `cos_theta_dim{d}`, `color_rgb_dim{d}`.
    Geometry under `GlyphGeometry/{group}`.
  - `patches.xdmf`: XDMF for citizen patch grid.
    Simple temporal collection of Uniform grids.
  - `zone_type_{t}.xdmf`: XDMF for politician
    zone type t grid.
  - `government.xdmf`: XDMF for government grid.
- `*_glyphs.py`: Auto-generated pvpython script.
  Loads one XDMF reader per grid type from the
  subdirectory (no ExtractBlock needed). Tunable
  constants: `SIGMA_REFS`, `POPULATION_SCALE`,
  `CYLINDER_RADIUS_SCALE`.
- `command`: Execution log with timestamps

## Known Issues

See `TODO.md` for resolved items. All previously
active tasks have been implemented. Remaining
deferred item:
- Primary campaign/vote phases not implemented
  (TODO #12 — deferred)
