# Glyph-Based Gaussian Visualization Design

## Motivation

The current output stores Gaussian parameters (mu, sigma, cos_theta)
as separate scalar fields per dimension in HDF5. This works for
scalar-field coloring in ParaView but does not lend itself to an
intuitive glyph-based view of the full Gaussian structure.

The primary goal is **scientific understanding**: the simulation has
many interacting parts and we need visualization that supports both
a holistic view and comprehension of details. The design must balance
those two scales simultaneously.

---

## Storage Concepts: Scalars vs. Vectors vs. Glyph-Native

### Scalar fields (current approach)

Each Gaussian parameter is its own HDF5 dataset, e.g.:
  `PolicyPref_mu`, `PolicyPref_sigma`, `PolicyPref_cos_theta`

In XDMF: `<Attribute Type="Scalar">` for each.
In ParaView: loaded independently; user selects which to color by,
and feeds them separately into the Glyph filter via `Scale Array`
and `Orient Array` controls. Flexible but requires manual wiring.

### Vector fields

Parameters packed into a shape-[N, 3] HDF5 array.
In XDMF: `<Attribute Type="Vector">`.
In ParaView: the Glyph filter uses the vector's direction for
orientation and magnitude for scale *automatically* — no Calculator
step needed. This is what "glyph-native" means: data layout directly
matches what the Glyph filter expects.

**Why vectors don't work cleanly here:**
- XDMF vectors are fixed at 3 spatial components.
- With 3–20 policy/trait dimensions, all Gaussian parameters for
  all dimensions cannot fit into one meaningful vector.
- (μ, σ, cos_θ) are not spatial components — treating them as a
  3D direction conflates unrelated quantities.
- The vector approach only makes sense if you have exactly 1–3
  dimensions and want glyphs placed in policy/trait space where
  axes ARE spatial.

### The dual-role of μ across views

The key insight enabling both geographic and policy-space views from
one HDF5 file: the same dataset can appear in two XDMF roles.

- As `<Attribute>` data: scalar colored on a fixed geographic grid
- As `<Geometry>` coordinates: point positions in policy/trait space

So μ values for each dimension define the *positions* of points in
a policy-space view, while σ controls cylinder radius and cos_θ controls
color saturation.
This requires a second XDMF file (different topology: unstructured
point cloud vs. structured grid), but reuses the same HDF5 file
with no data duplication.

---

## Design Decisions (Resolved)

### View targets
Both views are desired:
- Geographic view: glyphs overlaid on the spatial patch grid
- Policy-space view: glyphs positioned at their μ values in an
  abstract policy-axis coordinate space (deferred)
- Trait-space view: same concept for trait dimensions (deferred)

All views served by one HDF5 file; one or two XDMF files.

### Number of dimensions
3–20 policy dims + 3–20 trait dims expected. This rules out
vector-packed storage as the primary layout (too many dims, wrong
semantics). Scalar fields remain the right HDF5 layout; XDMF
handles the interpretation.

### Primary bottleneck / priority
File size and load time are not yet problems — lower tier.
The dominant need is **comprehension**: visualization must support
holistic understanding of many interacting agents and dimensions.

### Multi-pane philosophy
The user will NOT load all data into a single visualization.
Instead: separate ParaView panes, each with its own specific
dataset. Examples:
- One pane: citizen preference averages per patch for policy dim N
- Another pane: politician averages for one zone type, same dim

This approach provides access to granular data on demand without
visual clutter. The design principle is **selective visibility**:
any Gaussian field can be isolated in its own pane or combined
with others deliberately.

Implication for XDMF: all Gaussian fields can coexist in one
XDMF file. ParaView's array-selection dialog at load time lets the
user choose which arrays to activate per pane. Separate XDMF files
are only needed when the topology/geometry differs (e.g., geographic
grid vs. policy-space point cloud).

### Cylinder radius = σ (raw)
Cylinder radius encodes σ directly. As σ grows, the cylinder
widens — a natural visual correspondence between Gaussian
breadth and cylinder girth. σ is stored as a raw value in
HDF5; visual scale is controlled by CYLINDER_RADIUS_SCALE
in the ParaView script.

### Ideal policy cylinders (Pci)
Ideal policy is included with the same cylinder encoding as
preference and aversion. It is treated as any other Gaussian
— available in its own pane or combined deliberately with
others. Direction convention: see open question below.

---

## Glyph Design: Cylinder Encoding

### Spatial orientation
The patch grid occupies the x-y plane. Each cylinder has its
base at z=0 (the patch plane) and its tip along ±z:
- **Preference cylinders**: base at z=0, tip at +z
- **Aversion cylinders**: base at z=0, tip at -z
- **Ideal policy cylinders**: direction TBD (see open
  question below)

This creates a 3D landscape: preferences spike up, aversions
spike down, apathy and scarcity flatten toward the patch
plane. Visually readable at a glance even before inspecting
individual values.

VTK cylinder glyphs are Y-aligned and centered by default.
To produce a base-at-z=0 cylinder, the GlyphTransform must
apply three operations in order (Scale → Rotate → Translate):
1. `Scale = [2, 1, 2]` normalizes VTK cylinder radius
   from 0.5 to 1, so VectorScaleMode receives a unit
   radius for X and Y.
2. `Rotate = [90, 0, 0]` for +z, `[-90, 0, 0]` for
   -z, aligning the cylinder axis with ±Z.
3. `Translate = [0, 0, 0.5]` for +z,
   `[0, 0, -0.5]` for -z, positioning the base at
   the data point (z=0).

Note: the +z/-z distinction between preference and aversion
is important when they appear in the *same* pane. If always
shown in separate panes, both could use +z — but a combined
pane should remain possible, so the convention must be
consistent.

### Magnitude: population
Cylinder height encodes citizen population
normalized by the total world population (fixed
for the run). Counts are divided before writing
to HDF5, giving values in [0, 1]:

- Citizen patch: patch_count / total_population
- Politician zone: zone_count / total_population
- Government: 1.0 (represents 100% by definition)

Without population encoding, all patches appear
equally weighted even when one has hundreds of
citizens and another just a few — a critical
distinction once citizens can move between patches.

All three entity types share the same [0, 1] scale.
`POPULATION_SCALE` has one clear meaning: the height
(in world units) of a glyph representing 100% of
the population. The government glyph at height 1.0
× POPULATION_SCALE is the natural reference maximum.

### Radius: sigma
σ is encoded by the cylinder's radius. A wide
cylinder indicates a broad, uncertain position
(large σ); a narrow cylinder indicates a sharp,
certain position (small σ). This gives the
landscape an immediate visual read: tall narrow
spikes are strong, certain positions; short squat
cylinders are weak, diffuse ones.

  Wide cylinder:   broad, uncertain position (large σ)
  Narrow cylinder: sharp, certain position (small σ)

σ is stored as a raw value in HDF5. Visual radius
scale is controlled by the CYLINDER_RADIUS_SCALE
constant in the ParaView script. Initial value is
1.0; tune in the quickTest job.

In the ParaView script, a Calculator filter
assembles the per-point scale vector
[σ × CYLINDER_RADIUS_SCALE,
 σ × CYLINDER_RADIUS_SCALE,
 population × POPULATION_SCALE]
which is passed to the Glyph filter with
VectorScaleMode = scale by components. After the
GlyphTransform (see "Spatial orientation"), the
cylinder axis is Z-aligned; VectorScaleMode scales
X and Y by the first two components (radius) and Z
by the third (height). Equal X and Y components
keep the cross-section circular.

### Color: μ via hue
Diverging colormap: red (negative μ) → white/gray (μ = 0)
→ blue (positive μ)

The neutral pole at μ = 0 is natural: a citizen with no policy
lean is colorless.

### Saturation: cos_θ (engagement)
Fully engaged (cos_θ = 1, θ = 0): full color saturation.
Fully apathetic (cos_θ = 0, θ = π/2): desaturated toward gray.

This encoding is already established in the codebase (HSV
saturation used for engagement). The neutral-color pole at μ = 0
and the desaturated-gray of apathy coincide naturally — both map
to gray, which is correct.

### Color storage
Per-point RGB is pre-computed from μ and |cos_θ| by
`_mu_to_rgb()` in `output.py` and written to HDF5 as
`color_rgb_dim{d}`, shape (N, 3), float32, values in
[0, 1]. ParaView reads this as a 3-component Vector
attribute and colors the glyph directly without a
colormap lookup table.

### Multi-dimension layout in geographic view
Each policy/trait dimension produces cylinder glyphs per
patch. With 3–20 dims, this risks clutter if all shown
simultaneously. Resolution: multi-pane philosophy handles
this — show one dimension per pane. Stacking in policy-space
view is deferred.

### Politician glyphs
Same cylinder encoding with a distinct color range
(e.g., green hues) to distinguish from citizen glyphs in a
combined pane. Innate vs. apparent positions: dark vs. light
green.

---

## HDF5 File Structure

The glyph output is written to a separate file,
`stodem_glyphs.hdf5`. Variable names used below:

```
  num_patches     = x_num_patches × y_num_patches
                    (total flattened patch count)
  num_policy_dims = number of policy dimensions
  num_trait_dims  = number of trait dimensions
  num_zone_types  = number of zone hierarchy levels
  num_zones[t]    = number of zone instances of
                    zone type t
```

Each Gaussian type group contains one sub-group
per time step (`Step0`, `Step1`, ...). Within each
step, there are 4 × num_dims datasets:
`mu_dim{d}`, `sigma_dim{d}`, `cos_theta_dim{d}`,
`color_rgb_dim{d}`.
sigma is stored raw; visual radius scale is set by
CYLINDER_RADIUS_SCALE in the ParaView script.
color_rgb_dim{d} stores pre-computed HSV-derived
RGB values: shape (N, 3), float32, values in
[0, 1]. Hue encodes mu; saturation encodes
|cos_theta|. Written by `_mu_to_rgb()` at output
time; ParaView uses it directly as a Vector
attribute without a colormap lookup.

```
stodem_glyphs.hdf5
│
├── GlyphGeometry/
│   ├── patches/             (static; written once)
│   │   └── XYZ  shape (num_patches, 3), float32
│   ├── zone_type_0/  (per-step; districts may change)
│   │   ├── Step0/
│   │   │   └── XYZ  shape (num_zones[0], 3), float32
│   │   ├── Step1/
│   │   │   └── (same)
│   │   └── ...
│   ├── zone_type_1/         (static; written once)
│   │   └── XYZ  shape (num_zones[1], 3), float32
│   ├── ...  (zone_type_2 and higher: static)
│   └── government/          (static; written once)
│       └── XYZ  shape (1, 3), float32
│
├── citizen_policy_pref/  (z_pos; num_policy_dims dims)
│   ├── Step0/
│   │   ├── mu_dim0        [num_patches, float32]
│   │   ├── sigma_dim0     [num_patches, float32]
│   │   ├── cos_theta_dim0 [num_patches, float32]
│   │   ├── color_rgb_dim0 shape (num_patches, 3), float32
│   │   ├── mu_dim1        [num_patches, float32]
│   │   ├── sigma_dim1     [num_patches, float32]
│   │   ├── cos_theta_dim1 [num_patches, float32]
│   │   └── color_rgb_dim1 shape (num_patches, 3), float32
│   ├── Step1/
│   │   └── (same datasets)
│   └── ...
│
├── citizen_policy_aver/  (z_neg; num_policy_dims dims)
│   └── (same step/dataset structure)
│
├── citizen_policy_ideal/ (z=?; num_policy_dims dims)
│   └── (same step/dataset structure)
│
├── citizen_trait_pref/   (z_pos; num_trait_dims dims)
│   └── (same step/dataset structure)
│
├── citizen_trait_aver/   (z_neg; num_trait_dims dims)
│   └── (same step/dataset structure)
│
├── citizen_patch_population/        (cylinder height)
│   ├── Step0/
│   │   └── count  shape (num_patches,), float32
│   │              values in [0,1]; sum ≈ 1.0
│   ├── Step1/
│   │   └── (same)
│   └── ...
│
├── politician_innate_policy_pref_zone_type_0/
│   │                  (z_pos; num_policy_dims dims)
│   ├── Step0/
│   │   ├── mu_dim0        [num_zones[0], float32]
│   │   ├── sigma_dim0     [num_zones[0], float32]
│   │   ├── cos_theta_dim0 [num_zones[0], float32]
│   │   ├── color_rgb_dim0 shape (num_zones[0], 3), float32
│   │   └── ...  (repeated for each policy dim)
│   └── ...
│
├── politician_innate_policy_aver_zone_type_0/
│   │                  (z_neg; num_policy_dims dims)
│   └── (same step/dataset structure)
│
├── politician_external_policy_pref_zone_type_0/
│   │                  (z_pos; num_policy_dims dims)
│   └── (same step/dataset structure)
│
├── politician_external_policy_aver_zone_type_0/
│   │                  (z_neg; num_policy_dims dims)
│   └── (same step/dataset structure)
│
├── politician_innate_trait_zone_type_0/
│   │                  (z_pos; num_trait_dims dims)
│   └── (same step/dataset structure)
│
├── politician_external_trait_zone_type_0/
│   │                  (z_neg; num_trait_dims dims)
│   └── (same step/dataset structure)
│
│   (above 6 politician groups repeat for
│    zone_type_1, ...,
│    zone_type_{num_zone_types-1})
│
├── politician_zone_population_zone_type_0/  (cylinder height)
│   ├── Step0/
│   │   └── count  shape (num_zones[0],), float32
│   │              values in [0,1]
│   └── ...
│
│   (above repeats for zone_type_1, ...,
│    zone_type_{num_zone_types-1})
│
├── government_population/               (cylinder height)
│   ├── Step0/
│   │   └── count  shape (1,), float32  (always 1.0)
│   └── ...
│
└── government_policy/  (z_pos; num_policy_dims dims)
    ├── Step0/
    │   ├── mu_dim0        [1, float32]
    │   ├── sigma_dim0     [1, float32]
    │   ├── cos_theta_dim0 [1, float32]
    │   ├── color_rgb_dim0 shape (1, 3), float32
    │   └── ...  (repeated for each policy dim)
    └── ...
```

Groups whose z direction is marked `z=?` are
pending resolution of open design questions below.

---

## Politician Glyph Storage: Per-Zone-Instance

Politician glyphs are stored at zone resolution,
not at patch resolution. Each dataset in a
politician group has shape `[num_zones[t]]`, where
`t` is the zone type index for that group. Array
position `i` corresponds to zone instance `i` of
that zone type — the zone instance index is encoded
implicitly by array position.

This contrasts with the existing scalar output
(`stodem.hdf5`), which broadcasts each politician's
values across all patches in their zone. For glyph
visualization, per-zone-instance storage is more
meaningful: one cylinder per district, one cylinder
per state, rather than the same value repeated
across all patches of a zone.

### Zone centroid geometry

For each zone type, glyph positions are placed at
zone centroids. A zone centroid is the mean (x, y)
position of all patches belonging to that zone
instance. All geometry is stored at z=0; the
cylinder direction (±z) is encoded in the
GlyphTransform (see "Spatial orientation"), not
in the geometry coordinates.

Zone centroid geometry for zone_type_1 and higher
(states, countries) is written once at
initialization — those boundaries never change.
Zone centroid geometry for zone_type_0 (districts)
is written per step, following the same
`Step0/`, `Step1/`, ... structure as the data
groups, because district boundaries may shift
over the course of the simulation.

The ordering of centroids in each
`GlyphGeometry/zone_type_{t}/Step{i}/XYZ`
(or the static `XYZ` for higher zone types) must
exactly match the ordering of zone instances in
the corresponding politician data arrays. The
existing `world.zones[zone_type]` list provides
this ordering: zone instance `i` is at
`world.zones[zone_type][i]`.

### XDMF topology

Citizens and each zone type of politicians have
different point counts (num_patches vs.
num_zones[0], num_zones[1], ...), so they cannot
share a single XDMF Grid entry. Each requires its
own `<Grid>` block. These are written as sibling
`<Grid>` elements within one XDMF `<Domain>`,
giving one XDMF file with multiple named,
selectable grids in ParaView.

For static geometries (patches, zone_type_1 and
higher), all timesteps reference the same HDF5
geometry dataset. For zone_type_0, each timestep's
`<Grid>` references its own
`GlyphGeometry/zone_type_0/Step{i}/XYZ`.

Implementation note: the XDMF writer must derive
the point count for each timestep's `<Topology>`
from the actual HDF5 dataset shape at that step
(e.g., `dataset.shape[0]`), not from a cached
global constant. This ensures correctness if
`num_zones[0]` changes between steps in the
future.

---

## Open Design Questions

### A. Ideal policy cylinder direction
+z (preference) and -z (aversion) are taken. If ideal policy
ever appears in a combined pane with preference and aversion,
it needs its own convention. Options:
- A distinct glyph shape (e.g., sphere or disc)
- Assign a third z-offset direction to separate visually
- Defer until combined panes are actually needed

### B. Policy-space view layout (deferred)
When implemented: each dimension's μ values position glyphs along
that axis. With multiple dims, glyphs stack. Layout details TBD.

---

## Future Ideas

### Citizen vs. politician alignment glyph
A single composite glyph encoding the *relationship* between a
citizen's preference and a politician's apparent position — showing
alignment or misalignment directly rather than requiring the user
to compare two separate glyphs. Deferred until the direct
per-agent glyph design is working.

---

## Transition Notes

Once this design is stable, the relevant decisions will be
merged into `DESIGN.md` and this file removed.
