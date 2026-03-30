# Glyph Visualization — Implementation TODO

This file translates the design in GLYPH_DESIGN.md into
specific code actions. Items are ordered by dependency.

Do not copy from the old git implementation. Write fresh
from GLYPH_DESIGN.md.

---

## Context: Design Summary

The glyph output writes a separate file,
`stodem_glyphs.hdf5`, with a companion XDMF and a
generated ParaView script. Key design decisions:

- **Cylinder height = population.** Cylinder height encodes
  the number of citizens per patch (citizens) or per zone
  (politicians). Total citizen count (fixed for the run)
  is the normalization reference.

- **Cylinder radius = sigma.** Raw sigma stored in HDF5;
  visual radius controlled by CYLINDER_RADIUS_SCALE in the
  ParaView script. A Calculator filter assembles
  [sigma, sigma, population] for VectorScaleMode.

- **Per-zone-instance politician storage.** Politician
  data stored at zone resolution (one value per zone
  instance per step), not broadcast to patches. Zone
  centroids are the glyph positions.

- **Dynamic district geometry.** zone_type_0 (districts)
  centroid arrays are written per step. Higher zone types
  are static (written once).

- **Multi-grid XDMF.** One `<Grid>` per topology (patches
  and each zone type) within one XDMF Domain.

---

## Resolved Items

TODOs #1–#6 are fully implemented in
`src/scripts/output.py` and `src/scripts/stodem.py`.

The following bugs were found and fixed during
TODO #7 (quickTest verification):

| # | Bug | Fix |
|---|-----|-----|
| B1 | XDMF path hardcoded at generation time; fails when files are copied to another machine. | Generate path at runtime using `os.path.dirname(os.path.abspath(__file__))`. |
| B2 | `ExtractBlockUsingDataAssembly` not exported by `paraview.simple` in all builds. | Add compatibility shim: `try: ExtractBlockUsingDataAssembly / except NameError: ExtractBlockUsingDataAssembly = ExtractBlock`. |
| B3 | Unequal pane widths: `SplitHorizontal` used wrong cell index (`k-1`). | Track right-child cell as `2*cell+2` after each split. |
| B4 | Pane 3 (`citizen_policy_ideal`) invisible: sigma ≈ 0.001–0.014, ~100× smaller than stated policy. Single `CYLINDER_RADIUS_SCALE` cannot span both types. | Add per-group `SIGMA_REFS` normalization (mean sigma per group, computed at generation time). Calculator: `sigma / SIGMA_REFS[group] * CYLINDER_RADIUS_SCALE`. |
| B5 | Time stepping broken: four independent `CollectionType="Temporal"` grids at Domain level; ParaView Time Manager cannot unify them. | Restructured to single outer temporal collection (matching `stodem.xdmf`), spatial collection per step containing all named grids. |
| B6 | Default scales wrong: `POPULATION_SCALE=0.3` gave height ≈ 0.003 for quickTest; `CYLINDER_RADIUS_SCALE=1.0` gave radius up to 3.5. | `POPULATION_SCALE = 0.3 * num_patches` (computed at generation time); `CYLINDER_RADIUS_SCALE = 0.3`. |
| B7 | `Xdmf3ReaderS` does not expose block hierarchy in the GUI (known unresolved bug). `ExtractBlock` could not isolate individual grids; all topologies merged into Block0 (11 points for smallTest: 9 patches + 1 zone + 1 government). | Write separate XDMF files per grid type (`patches.xdmf`, `zone_type_{t}.xdmf`, `government.xdmf`) into a subdirectory alongside the HDF5. Script loads one reader per file; no `ExtractBlock` needed. HDF5 and XDMF files placed in `{outfile}_glyphs/` subdirectory for easy drag-and-drop copying. |

---

## TODO Items

### 1. Confirm sigma values are accessible at output time

**File:** `src/scripts/world.py` (read)

**Action:** Confirm that the sigma fields for each
Gaussian type (e.g., citizen policy pref sigma per
patch per dim) are available in `world` at the time
`GlyphHdf5.add_step()` is called. These are the same
sigma arrays already written by the existing `Hdf5`
class — confirm the access path used there and use
the same path in `GlyphHdf5`.

Note also the `CYLINDER_RADIUS_SCALE` and
`POPULATION_SCALE` constants used in the generated
ParaView script (TODO #5). Initial values of 1.0
are placeholders; tune in the quickTest job
(TODO #7) to determine working values.

---

### 2. Add helper functions to output.py

**File:** `src/scripts/output.py`

**Add imports** at the top: `import numpy as np` and
`import os` if not already present.

**Add these standalone functions:**

**`_hsv_to_rgb_arr(h, s, v)`**: vectorized HSV → float32
RGB conversion. Used when pre-computing per-point color
arrays for the ParaView script.

**`_mu_to_rgb(mu, cos_theta)`**: maps mu and cos_theta
arrays to a pre-computed RGB array. Hue encodes mu via
a sigmoid-based diverging colormap (red = positive,
white/gray = zero, blue = negative). Saturation =
|cos_theta| (engagement). Returns float32 array of
shape (N, 3), values in [0, 1]. Called once per
dimension per Gaussian type group per step; result
written to HDF5 as `color_rgb_dim{d}`.

**`_build_glyph_datasets(settings, world)`**: returns a
list of descriptor objects (namedtuple or dict), one per
Gaussian type group. Each descriptor contains:
  - `group_name`: HDF5 group name string
    (e.g., `"citizen_policy_pref"`)
  - `z_dir`: `"z_pos"` or `"z_neg"`
  - `num_dims`: number of dimensions for this type
  - `zone_type`: None for citizen groups;
    int t for politician zone_type_t groups;
    `"government"` for government_policy

Government policy: `z_dir="z_pos"`,
`zone_type="government"`. All datasets shape (1,)
(or (1, 3) for color_rgb). The `zone_type` value
determines point count and geometry path — no
separate `is_scalar` field needed.

---

### 3. Add GlyphHdf5 class to output.py

**File:** `src/scripts/output.py`

**Public interface:**
```
class GlyphHdf5:
    __init__(settings, world)
    add_step(world, step)
    close()
```

**`__init__`:**
- Open `{settings.outfile}_glyphs.hdf5` exclusive-create.
- Call `_build_glyph_datasets()`, store as `self.datasets`.
- For each descriptor in `self.datasets`, create an HDF5
  group at `{descriptor.group_name}/`.
- Compute and store `self.total_population`: total
  number of citizens across all patches (sum over
  world). Used to normalize all population counts.
- Create `GlyphGeometry/` with subgroups:
  - `patches/` (static)
  - `zone_type_{t}/` for each zone type t.
    zone_type_1 and higher: static (write XYZ here).
    zone_type_0: per-step (write XYZ inside
    Step{i}/ subgroups from add_step).
  - `government/` (static)
- Create `citizen_patch_population/`.
- Create `politician_zone_population_zone_type_{t}/`
  for each zone type t.
- Create `government_population/`.
- Write patch geometry (static):
  - Compute (x, y) centers of all patches in
    flattened row-major order. z = 0 for all points.
  - Write `GlyphGeometry/patches/XYZ`.
    Shape: (num_patches, 3), float32.
- Write static zone centroid geometry for zone_type_1
  and higher:
  - For each zone instance, mean (x, y) of its member
    patches. z = 0. Use `world.zones[t]` list; zone
    instance i is `world.zones[t][i]`.
  - Write `GlyphGeometry/zone_type_{t}/XYZ`.
    Shape: (num_zones[t], 3), float32.
- Write government geometry (static):
  - World centroid: mean (x, y) of all patch centers.
    z = 0. Write `GlyphGeometry/government/XYZ`.
    Shape: (1, 3), float32.

**`add_step(world, step)`:**
- Write district centroid geometry for zone_type_0:
  - Compute centroids from current zone membership.
    z = 0 for all points.
  - Write `GlyphGeometry/zone_type_0/Step{step}/XYZ`.
    Shape: (num_zones[0], 3), float32.
- For each descriptor in `self.datasets`:
  - Create `{group_name}/Step{step}/` subgroup.
  - For each dimension d in range(descriptor.num_dims):
    - Compute mu, sigma, cos_theta arrays for this
      Gaussian type and dimension.
    - For citizen groups (zone_type=None): arrays
      have shape (num_patches,).
    - For politician groups (zone_type=int t): arrays
      have shape (num_zones[t],), one value per zone
      instance, averaged over all politicians in zone.
    - For government (zone_type="government"): compute
      one scalar value per policy dim from the
      government object. Arrays have shape (1,).
    - Write `mu_dim{d}`, `sigma_dim{d}`,
      `cos_theta_dim{d}` as float32 datasets.
    - Compute `rgb = _mu_to_rgb(mu, cos_theta)`.
      Write `color_rgb_dim{d}` as shape (N, 3),
      float32 dataset.
- Write citizen population (normalized):
  `citizen_patch_population/Step{step}/count`
  shape (num_patches,), float32:
  patch_count / self.total_population. Values in
  [0, 1]; sum across patches ≈ 1.0.
- Write politician zone population (normalized):
  `politician_zone_population_zone_type_{t}/
  Step{step}/count` shape (num_zones[t],), float32:
  zone_count / self.total_population. Values in [0, 1].
- Write government population:
  `government_population/Step{step}/count`
  shape (1,), float32: always [1.0].

**`close()`:** flush and close HDF5 file.

---

### 4. Add GlyphXdmf class to output.py

**File:** `src/scripts/output.py`

**Public interface:**
```
class GlyphXdmf:
    __init__(settings, glyph_hdf5)
    write(num_steps)
```

**`write(num_steps)`:**
- Build one XDMF Domain with multiple named Grids,
  one per topology:
  - `patches_grid`: num_patches points
  - `zone_type_0_grid`: num_zones[0] points (per-step
    geometry)
  - `zone_type_{t}_grid` for t >= 1: static geometry
  - `government_grid`: 1 point, static geometry

- Each Grid is a temporal collection:
  `<Grid GridType="Collection" CollectionType="Temporal">`
  containing one inner `<Grid>` per step.

- Each inner Grid contains:
  - `<Geometry GeometryType="XYZ">`: HDF5 reference.
    - patches, zone_type_{t >= 1}, and government:
      same path for all steps (static).
    - zone_type_0: reference
      `GlyphGeometry/zone_type_0/Step{i}/XYZ`
      for each step i.
  - `<Topology Type="Polyvertex" NumberOfElements="{N}">`:
    derive N by reading `dataset.shape[0]` from the
    actual HDF5 dataset at that step — do NOT use a
    cached global constant. This supports future cases
    where num_zones[0] may change between steps.
  - Per dataset per dim, for each Gaussian type group
    active on this topology:
    - `mu_dim{d}`, `sigma_dim{d}`, `cos_theta_dim{d}`:
      `<Attribute Type="Scalar">` (float32)
    - `color_rgb_dim{d}`: `<Attribute Type="Vector">`
      (float32, 3 components — R, G, B in [0, 1])
  - One `<Attribute>` for the population count.
    Use unique XDMF attribute names: `citizen_population`
    in the patches grid, `zone_population` in each
    zone_type grid, and `government_population` in the
    government grid. These names are referenced by the
    Calculator in TODO #5.

- Write to `{settings.outfile}_glyphs.xdmf`.

---

### 5. Add write_paraview_script() to output.py

**File:** `src/scripts/output.py`

**Action:** Generate a pvpython script that builds the
multi-pane ParaView pipeline. The script must be loop-
driven; do not hard-code individual Glyph passes.

**Script constants (tunable at the top):**
```python
POPULATION_SCALE = 0.3       # cylinder height per citizen
CYLINDER_RADIUS_SCALE = 1.0  # cylinder radius multiplier
DIM = 0                      # policy/trait dim to display
ZONE_TYPE = 0                # zone type for politician panes
```

Note: typical sigma values depend on simulation parameters.
CYLINDER_RADIUS_SCALE and POPULATION_SCALE are visualization
parameters only; no data changes needed. Initial values of
1.0 are placeholders; tune in the quickTest job (TODO #7).

**Per-pane pipeline (one pane per Gaussian type group
for dimension DIM):**

1. Load `stodem_glyphs.xdmf`.
2. Select the appropriate grid block (patches grid for
   citizens and government; zone_type_{t} grid for
   politicians of zone type t).
3. Set active time step.
4. **Calculator filter:**
   - Assemble the per-point scale vector. After the
     GlyphTransform, X and Y are the radius axes and
     Z is the height axis, so:
     `iHat*sigma_dim{DIM}*CYLINDER_RADIUS_SCALE
      + jHat*sigma_dim{DIM}*CYLINDER_RADIUS_SCALE
      + kHat*pop*POPULATION_SCALE`
     where `pop` is the XDMF attribute name:
     `citizen_population` for the patches grid,
     `zone_population` for zone_type_{t} grids,
     `government_population` for the government grid.
   - Name the result array `cylinder_scale`.
5. **Cylinder Glyph pass:**
   - Source: Cylinder glyph.
   - GlyphTransform (Scale → Rotate → Translate):
     `Scale = [2, 1, 2]` (normalizes VTK cylinder
     radius 0.5 → 1); `Rotate = [90, 0, 0]` for
     z_pos groups, `[-90, 0, 0]` for z_neg groups;
     `Translate = [0, 0, 0.5]` for z_pos,
     `[0, 0, -0.5]` for z_neg (positions cylinder
     base at the data point, z=0).
   - Scale mode: VectorScaleMode = scale by components;
     scale array = `cylinder_scale`; scale factor = 1.0.
   - Color: use `color_rgb_dim{DIM}` (pre-computed RGB
     Vector attribute) with direct scalar coloring —
     no colormap lookup needed.

6. Apply orthographic top-down camera to all views.
7. Call `Render(view)` per pane, `RenderAllViews()` at
   end.

---

### 6. Integrate into stodem.py

**File:** `src/scripts/stodem.py`

**Action:**
- Instantiate `GlyphHdf5(settings, world)` at startup,
  alongside the existing `Hdf5` instantiation.
- Call `glyph_hdf5.add_step(world, step)` each output
  step, alongside `hdf5.add_dataset()` calls.
- After the simulation loop:
  - Instantiate `GlyphXdmf(settings, glyph_hdf5)` and
    call `.write(num_steps_written)`.
  - Call `write_paraview_script(settings, world)`.
  - Call `glyph_hdf5.close()`.

---

### 7. Test with quickTest job

**Directory:** `jobs/quickTest/`

**Status:** Re-run required after B7 redesign.
Output now goes into `{outfile}_glyphs/`
subdirectory.

**Command:** `python3 ../../src/scripts/stodem.py`

**HDF5 structure (verified ✓ / pending ?):**
- ✓ All Gaussian type groups present with step
  subgroups.
- ✓ `GlyphGeometry/patches/XYZ` shape (81, 3).
- ✓ `GlyphGeometry/zone_type_0/Step0/XYZ` shape
  (num_zones[0], 3).
- ? `GlyphGeometry/zone_type_1/XYZ` present
  (static) and shape (num_zones[1], 3).
- ? `GlyphGeometry/government/XYZ` shape (1, 3).
- ✓ `citizen_patch_population/Step0/count` in
  [0, 1], sum ≈ 1.0.
- ? `politician_zone_population_zone_type_0/
  Step0/count` in [0, 1].
- ? `government_population/Step0/count` = [1.0].
- ✓ sigma > 0 for all datasets.
- ✓ `color_rgb_dim0` shape (N, 3), values in
  [0, 1].

**XDMF (pending re-run with new structure):**
- ? Separate files written to subdirectory.
- ? `patches.xdmf` loads 9 points (smallTest).
- ? `zone_type_0.xdmf` loads 1 point (smallTest).
- ? `government.xdmf` loads 1 point.
- ? Time Manager synchronizes all readers.

**ParaView script:**
- ✓ Three panes equal width.
- ✓ Time stepping via Time Manager verified.
- ? Each pane loads correct point count (no
  merged block).
- ? Final scale factor validation.

---

## Deferred Items

- Policy-space view (separate XDMF with point cloud
  positioned at mu values): deferred until geographic
  view is working and validated.
- Citizen vs. politician alignment composite glyph.
- Trait-space view.
- Open design question: ideal policy cylinder direction
  (z=? in HDF5 tree); deferred until combined panes
  are needed.
