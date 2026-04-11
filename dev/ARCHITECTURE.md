# STODEM Architecture

This document describes the structural organization of the
STODEM codebase: repository layout, module responsibilities,
dependency relationships, build system, randomness strategy,
and development checkpoint practices. For the scientific
goals and design principles that motivate these choices, see
`VISION.md`. For detailed algorithmic design, see
`DESIGN.md`.

---

## 1. Repository Layout

```
stodem/
  .stodem/              RC files (stodemrc.py, defaults)
  bin/                  Installed copies of Python modules
  build/release/        CMake build artifacts
  jobs/
    quickTest/          Small test case (stodem.in.xml)
    test1/              Larger test case
  src/
    scripts/            Primary source code (Module Map)
  tests/
    conftest.py         Shared pytest fixtures
    unit/               Unit tests (Gaussian math, etc.)
    integration/        Smoke and integration tests
    regression/         Reference output comparisons
  dev/
    VISION.md           Goals and design principles
    ARCHITECTURE.md     This document
    DESIGN.md           Algorithmic design and physics
    PSEUDOCODE.md       Language-agnostic algorithm specs
    TODO.md             Task list organized by level
  CLAUDE.md             AI assistant guidance
  CMakeLists.txt        Build system

Output (generated at runtime):
  {prefix}.hdf5         Regular-grid simulation data
  {prefix}.xdmf         XDMF metadata for ParaView
  {prefix}_glyphs/      Glyph output subdirectory
    {prefix}_glyphs.hdf5  Cylinder glyph HDF5 data
    patches.xdmf          Citizen grid XDMF
    zone_type_{t}.xdmf    Politician grid XDMF
    government.xdmf       Government grid XDMF
  {prefix}_glyphs.py    Generated pvpython script
```

---

## 2. Module Map

All simulation logic lives in `src/scripts/`. Each module
has a single clear responsibility, organized by functional
group.

**Entry Point and Control**

- `stodem.py` — Entry point. Contains `main()`, the
  simulation loop, `campaign()`, `vote()`, `govern()`.
- `settings.py` (`ScriptSettings`) — XML input parsing,
  command-line argument handling, RC file loading.
- `sim_control.py` (`SimControl`, `SimProperty`) —
  Simulation phase counts, total step computation, data
  range computation, property dataclass.

**World and Spatial Structure**

- `world.py` (`World`) — Top-level container. Creates
  patches, zones, citizens, politicians, government.
  Computes patch-level output fields via
  `compute_patch_well_being()`,
  `compute_patch_gaussian_stats()`, and
  `compute_patch_politician_stats()`.
- `zone.py` (`Zone`) — Geographic region at one hierarchy
  level. Maintains politician lists and citizen zone
  averages.
- `patch.py` (`Patch`) — Basic grid unit. Knows its
  (x,y) location, zone membership, and resident citizen
  indices.

**Agents**

- `citizen.py` (`Citizen`) — Agent with Gaussian
  policy/trait preferences, aversions, and ideal
  positions. Computes overlap integrals, influence
  response, and voting logic.
- `politician.py` (`Politician`) — Agent with innate and
  external policy/trait positions. Strategies for
  movement, adaptation, and campaigning.
- `government.py` (`Government`) — Holds enacted policy
  Gaussians that affect citizen well-being.

**Mathematics and Infrastructure**

- `gaussian.py` (`Gaussian`) — Complex Gaussian
  representation with overlap integral computation.
- `random_state.py` — Global `rng` instance (numpy
  `default_rng`, seed=8675309).
- `output.py` (`Hdf5`, `Xdmf`, `GlyphHdf5`,
  `GlyphXdmf`, `write_paraview_script`) — Regular-grid
  and glyph-format output for ParaView visualization.
- `diagnostics.py` — Diagnostic utilities for simulation
  debugging and data logging.

**Visualization (Optional)**

- `policy_space_viz.py` (`PolicySpaceViz`) — Live debug
  visualization of individual agent Gaussians using
  pyqtgraph. Each dimension has two subplots: a real
  projection (cos θ, the engaged component) and an
  imaginary projection (sin θ, the latent/apathetic
  component). Records per-frame state snapshots for
  post-run replay with interactive transport controls.
  Conditionally imported when `-d` / `--debug-viz` is
  active; the simulation never depends on this module.

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
  +-- diagnostics.py
  +-- policy_space_viz.py (optional, -d flag)
        +-- pyqtgraph / Qt
```

All modules that use randomness import the shared `rng`
from `random_state.py` to ensure reproducibility from a
single seed.

---

## 3. Randomness and Reproducibility

All random number generation uses a single shared `rng`
instance (numpy `default_rng` with seed 8675309), imported
from `random_state.py` by every module that needs
randomness. This ensures full reproducibility given the
same seed and execution order.

---

## 4. Build System

CMake is used for installation (copying scripts to
`$STODEM_DIR/bin`) and for future Fortran expansion. The
primary simulation is pure Python and does not require
compilation.

**Dependencies:**
- Python 3 with: `numpy`, `h5py`, `lxml`
- CMake 3.1.0+ (for installation)
- Optional: Fortran compiler (gfortran/ifort) for future
  expansion
- Optional: `pyqtgraph` with Qt binding (`PyQt5`,
  `PyQt6`, or `PySide6`) for `-d` debug visualization
- Testing: `pytest`

**Build commands:**

```bash
mkdir -p build/release && cd build/release
cmake -DCMAKE_BUILD_TYPE=RELEASE ../..
make && make install  # Installs to $STODEM_DIR/bin
```

---

## 5. Development Checkpoints (Git Tags)

Design iterations are recorded as git tags so that any
baseline can be restored if an implementation attempt goes
poorly. Use a tag before starting any significant
implementation effort.

### Creating a checkpoint

```bash
git add <files>
git commit -m "Description of design baseline"
git tag <tag-name>
```

### Retreating to a checkpoint

To restore specific files to a tagged state (stays on the
current branch, files are staged):

```bash
git checkout <tag-name> -- file1 file2 ...
git commit -m "Revert to <tag-name> design baseline"
```

Only the listed files are affected. Other files must be
cleaned up manually if needed.

### Inspecting a checkpoint without changing anything

```bash
git show <tag-name>:path/to/file | less
git diff <tag-name> -- path/to/file
```

### Starting a fresh attempt on a new branch

```bash
git checkout -b retry-branch <tag-name>
```

This creates a new branch rooted at the tag, leaving
`main` untouched.
