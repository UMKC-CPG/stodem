# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

### Core Classes (src/scripts/stodem.py)

- **ScriptSettings**: Configuration from XML + command line, loads defaults from `$STODEM_RC/stodemrc.py`
- **SimControl**: Manages simulation phases (cycles, campaign steps, govern steps)
- **World**: Main container holding patches, zones, citizens, politicians, government
- **Zone**: Geographic region at one hierarchy level, maintains politician lists and citizen averages
- **Patch**: Basic grid unit containing citizens, computes zone membership
- **Citizen**: Agent with Gaussian policy/trait preferences, aversions, and ideal positions
- **Politician**: Agent with innate and external (presented) policy/trait positions, strategies
- **Government**: Enacts policies affecting citizen well-being
- **Gaussian**: Complex Gaussian functions with overlap integral computation
- **Hdf5/Xdmf**: Output generation for data storage and Paraview visualization

### Simulation Flow

```
main() → for each cycle:
    campaign() → vote() → govern() → (primary phases not yet implemented)
```

### Key Mathematical Concepts

**Complex Gaussians**: g(x;σ,μ,θ) = 1/(σ√(2π)) × exp(-(x-μ)²/(2σ²)) × exp(iθ)
- μ (mu): position on policy axis
- σ (sigma): spread/certainty
- θ (theta): rotation into imaginary axis = apathy/disengagement

**Overlap Integral**: Measures alignment between two Gaussians
- I(G1,G2) = (π/ζ)^1.5 × exp(-ξd²) × cos(θ1) × cos(θ2)
- Values range -1 to +1

**Gaussian Notation**:
- Pcp/Pca/Pci: Policy citizen preference/aversion/ideal
- Ppp/Ppa: Policy politician preference/aversion (apparent)
- Pge: Policy government enacted
- Tcp/Tca: Trait citizen preference/aversion
- Tpx: Trait politician external

## Configuration (stodem.in.xml)

Key sections:
- `sim_control`: num_cycles, num_campaign_steps, num_govern_steps, data_resolution
- `world`: patch_size, num_policy_dims, num_trait_dims, zone_type_N (hierarchy config)
- `citizens`: policy/trait stddev parameters, participation probability
- `politicians`: policy/trait stddevs, influence/lie parameters, strategy probabilities
- `government`: policy position/spread parameters

## Output Files

- `*.hdf5`: Binary simulation data with compression
- `*.xdmf`: XML metadata for Paraview visualization
- `command`: Execution log with timestamps

## Known Issues

- `vote()` function has syntax error at line 1671 (missing colon after `for` statement)
- `govern()` function is a stub (returns immediately)
- Several influence methods contain `pass` statements needing implementation
- Primary campaign/vote phases not implemented
