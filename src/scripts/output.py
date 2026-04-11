import os
import numpy as np
import h5py as h5
from lxml import etree
from collections import namedtuple


class Hdf5():
    """Manages the HDF5 binary data file for
    simulation output.

    HDF5 is a hierarchical data format that stores
    large numerical arrays with optional
    compression. Each simulation property (e.g.,
    "WellBeing") is stored in its own HDF5 group,
    with one dataset per time step inside that
    group (e.g., WellBeing0, WellBeing1, ...).

    The file is opened in exclusive-create mode
    ("x"), which will raise an error if the file
    already exists — preventing accidental
    overwriting of previous simulation results.

    Attributes
    ----------
    h_fid : h5py.File
        The open HDF5 file handle.
    group_gid : dict
        Maps property group names to their HDF5
        group objects.
    dataset_did : dict
        Maps dataset names (e.g., "WellBeing0")
        to their HDF5 dataset objects.
    """

    def __init__(self, settings, world):
        """Create the HDF5 file and set up one
        group per simulation property.

        Parameters
        ----------
        settings : ScriptSettings
            Provides the output filename prefix.
        world : World
            Provides the list of simulation
            properties to create groups for.
        """
        self.h_fid = h5.File(
                f"{settings.outfile}.hdf5", "x")

        self.group_gid = {}
        self.dataset_did = {}

        for p in world.properties:
            if p.group not in self.group_gid:
                self.group_gid[p.group] = (
                    self.h_fid.create_group(
                        f"{p.group}"))

    def add_dataset(self, p, i):
        """Write one time step's data for a
        property into the HDF5 file.

        Parameters
        ----------
        p : SimProperty
            The property to write (contains group
            name, dataset name, and the 2-D numpy
            array of data).
        i : int
            The simulation step index, appended to
            the dataset name to form a unique key
            (e.g., "WellBeing3" for step 3).
        """
        self.dataset_did[f"{p.name}{i}"] = (
                self.group_gid[p.group].create_dataset(f"{p.name}{i}",
                        compression="gzip", data=p.data))

    def close(self):
        """Flush and close the HDF5 file. Must
        be called after the simulation completes
        to ensure all data is written to disk."""
        self.h_fid.close()


class Xdmf():
    """Manages the XDMF metadata file that
    describes the structure of the companion HDF5
    data file for visualization in Paraview.

    XDMF (eXtensible Data Model and Format) is an
    XML-based format that tells Paraview how to
    interpret HDF5 data: what the grid looks like,
    where each time step's data lives inside the
    HDF5 file, and what data type each field is.
    Paraview reads the .xdmf file, which points
    to the .hdf5 file for the actual numbers.

    The XDMF file is written AFTER the simulation
    completes (not incrementally during the run).
    This is a deliberate design choice: the XDMF
    file must reference exactly the same number of
    time steps as were actually written to HDF5.
    If the simulation terminates early (e.g., due
    to an error or convergence), writing XDMF at
    the end with the actual step count ensures
    consistency. Writing it incrementally would
    require either rewriting the file each step or
    risking a mismatch if the simulation crashes.

    Attributes
    ----------
    x : file handle
        The open XDMF output file.
    """

    def __init__(self, settings):
        """Open the XDMF file and write the XML
        header lines. The body of the file is
        written later by print_xdmf_xml().
        """
        self.x = open(
                f"{settings.outfile}.xdmf", "w")
        self.x.write(
                '<?xml version="1.0"'
                ' encoding="utf-8"?>\n')
        self.x.write(
                '<!DOCTYPE Xdmf SYSTEM'
                ' "Xdmf.dtd" []>\n')


    def print_xdmf_xml(self, settings,
                        num_steps, world):
        """Write the XDMF XML body referencing
        exactly num_steps time steps of HDF5 data.

        This method builds an XML tree describing
        a temporal collection of uniform 2-D grids
        (one per time step), where each grid
        contains one or more scalar attributes
        (e.g., WellBeing) that point to datasets
        inside the HDF5 file.

        Must be called after the simulation
        completes so that num_steps equals the
        actual number of HDF5 datasets written.

        Parameters
        ----------
        settings : ScriptSettings
            Provides the output filename prefix
            (used to construct HDF5 dataset paths).
        num_steps : int
            The actual number of simulation steps
            completed. This is sim_control.curr_step,
            NOT sim_control.total_num_steps, to
            handle early termination correctly.
        world : World
            Provides grid dimensions
            (x_num_patches, y_num_patches) and
            the list of simulation properties.
        """

        # Create local variable lists for the
        #   XDMF file.
        timestep_grid = []
        timestep = []
        topology = []
        geometry = []
        geometry_origin = []
        geometry_dxdy = []
        attribute = []
        data_item = []

        # Build the XDMF XML file.

        # The root will be the xdmf tag. We will
        #   need to add some data before the root
        #   into the XML file. Specifically a line:
        #   '<?xml version="1.0" encoding="utf-8"?>'
        #   and a line:
        #   '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>"'.
        #   Note also that "xdmf" is lower case while
        #   the class name is "Xdmf". Finally, note
        #   that the xmlnsxi attribute should really
        #   be xmlns:xi, but python does not like
        #   colons in variable names and when I tried
        #   to add that key-value pair to the
        #   attribute dictionary after the fact it
        #   wouldn't accept it either. (Python called
        #   it a bad key I think.) So, the resolution
        #   is to use the xmlnsxi attribute and then
        #   when the xml file is actually printed I do
        #   a string substitution on xmlnsxi to make
        #   it xmlns:xi. (Kludgy.)
        root = etree.Element("root")
        xdmf = etree.SubElement(root, "Xdmf", Version="3.0",
                xmlnsxi=("[http://www.w3.org/2001/XInclude]"))

        # Make the domain. Only one domain is needed
        #   for the simulation results.
        domain = etree.SubElement(xdmf, "Domain")

        # Create a "grid" that is a collection of
        #   grids, one for each time step.
        time_collection_grid = etree.SubElement(
                domain, "Grid",
                Name="TimeSteps",
                GridType="Collection",
                CollectionType="Temporal")

        # Start the loop for adding time step grids.
        for i in range(num_steps):

            # Add the time step grid.
            timestep_grid.append(etree.SubElement(time_collection_grid,
                "Grid", Name=f"Step{i}", GridType="Uniform"))

            # Make the time step value an integer (arbitrary time duration).
            timestep.append(etree.SubElement(
                timestep_grid[i], "Time", Value=f"{i}.0"))

            # Define the topology for this grid. (Ideally, we would define
            #   a single topology below the "Domain" and then reference it
            #   here. But for some reason it didn't work and so we repeat
            #   it for every time step grid even though it "wastes" space.)
            topology.append(etree.SubElement(timestep_grid[i], "Topology",
                Name="Topo", TopologyType="2DCoRectMesh",
                Dimensions=(f"{world.x_num_patches}"
                            f" {world.y_num_patches}")))

            # Same for the geometry as for the topology.
            geometry.append(etree.SubElement(timestep_grid[i], "Geometry",
                Name="Geom", GeometryType="ORIGIN_DXDY"))

            # The geometry needs two data elements: origin and spatial deltas.
            geometry_origin.append(etree.SubElement(geometry[i], "DataItem",
                NumberType="Float", Dimensions="2", Format="XML"))
            geometry_origin[i].text = "0.0 0.0"
            geometry_dxdy.append(etree.SubElement(geometry[i], "DataItem",
                NumberType="Float", Dimensions="2", Format="XML"))
            geometry_dxdy[i].text = "1.0 1.0"

            # XDMF attributes frame the data items
            #   that point to the actual HDF5 data.
            #   Each attribute / data set in XDMF
            #   corresponds to one monitored property
            #   in our simulation. Thus, because there
            #   will be more than one property and
            #   because they are computed in a time
            #   series we need a list of lists. (One
            #   property list for each member of the
            #   timestep list.)
            # Here, we add an empty attribute list
            #   (and its partner DataItem list) for
            #   this time step into the list of
            #   timestep lists.
            attribute.append([])
            data_item.append([])

            # Now, we add each attribute that will
            #   live on the grid for this timestep
            #   along with its data item that points
            #   to the actual data in the HDF5 file.
            for p in world.properties:
                attribute[i].append(etree.SubElement(timestep_grid[i],
                    "Attribute", Name=f"{p.name}", Center="Node",
                    AttributeType=f"{p.datatype}"))
                data_item[i].append(etree.SubElement(attribute[i][-1],
                        "DataItem", NumberType="Float", Dimensions=(
                        f"{world.x_num_patches} {world.y_num_patches}"),
                        Format="HDF"))
                data_item[i][-1].text = (
                        f"{settings.outfile}.hdf5:/{p.group}/{p.name}{i}")

        # Pretty print the xdmf file to the output
        #   file.
        temp_xml = etree.tostring(
                xdmf, pretty_print=True,
                encoding="utf-8").decode()
        self.x.write(
                temp_xml.replace(
                    "xmlnsxi", "xmlns:xi", 1))
        self.x.close()


# ----------------------------------------------------------
# Glyph output helpers
# ----------------------------------------------------------

_GlyphDataset = namedtuple(
    '_GlyphDataset',
    ['group_name', 'z_dir', 'num_dims',
     'zone_type', 'gaussian_attr'])


def _hsv_to_rgb_arr(h, s, v):
    """Vectorized HSV to float32 RGB.

    Parameters
    ----------
    h, s, v : 1-D float32 arrays, values in [0, 1].

    Returns
    -------
    numpy.ndarray  shape (N, 3), float32.
    """
    h6 = h * 6.0
    i = np.floor(h6).astype(np.int32) % 6
    f = (h6 - np.floor(h6)).astype(np.float32)
    p = (v * (1.0 - s)).astype(np.float32)
    q = (v * (1.0 - f * s)).astype(np.float32)
    t = (v * (1.0 - (1.0 - f) * s)).astype(
        np.float32)
    n = len(h)
    r = np.empty(n, dtype=np.float32)
    g = np.empty(n, dtype=np.float32)
    b = np.empty(n, dtype=np.float32)
    for case, (rv, gv, bv) in enumerate([
            (v, t, p), (q, v, p),
            (p, v, t), (p, q, v),
            (t, p, v), (v, p, q)]):
        m = (i == case)
        r[m] = rv[m]
        g[m] = gv[m]
        b[m] = bv[m]
    return np.stack([r, g, b], axis=1)


def _mu_to_rgb(mu, cos_theta):
    """Per-point float32 RGB from mu and cos(theta).

    Hue encodes mu via sigmoid-based diverging
    colormap: red (positive), white/gray (zero),
    blue (negative). Saturation combines |cos_theta|
    with |2*sigmoid(mu) - 1| so that mu=0 always
    maps to white/gray regardless of engagement.
    Value is always 1.0.

    Parameters
    ----------
    mu, cos_theta : 1-D array-like.

    Returns
    -------
    numpy.ndarray  shape (N, 3), float32.
    """
    mu = np.asarray(mu, dtype=np.float32).ravel()
    ct = np.asarray(
        cos_theta, dtype=np.float32).ravel()
    sig = (1.0 / (1.0 + np.exp(
        -np.clip(mu, -80.0, 80.0)))).astype(
        np.float32)
    h = (2.0 / 3.0 * (1.0 - sig)).astype(
        np.float32)
    s = (np.abs(2.0 * sig - 1.0)
         * np.abs(ct)).astype(np.float32)
    v = np.ones(len(h), dtype=np.float32)
    return _hsv_to_rgb_arr(h, s, v)


def _build_glyph_datasets(settings, world):
    """Return one _GlyphDataset descriptor per
    Gaussian type group, in HDF5 creation order.

    Parameters
    ----------
    settings : ScriptSettings
    world    : World

    Returns
    -------
    list of _GlyphDataset
    """
    ndp = world.num_policy_dims
    ndt = world.num_trait_dims
    nzt = world.num_zone_types
    ds = []
    ds.append(_GlyphDataset(
        'citizen_policy_pref', 'z_pos', ndp,
        None, 'stated_policy_pref'))
    ds.append(_GlyphDataset(
        'citizen_policy_aver', 'z_neg', ndp,
        None, 'stated_policy_aver'))
    ds.append(_GlyphDataset(
        'citizen_policy_ideal', 'z_pos', ndp,
        None, 'ideal_policy_pref'))
    ds.append(_GlyphDataset(
        'citizen_trait_pref', 'z_pos', ndt,
        None, 'stated_trait_pref'))
    ds.append(_GlyphDataset(
        'citizen_trait_aver', 'z_neg', ndt,
        None, 'stated_trait_aver'))
    for t in range(nzt):
        ds.append(_GlyphDataset(
            f'politician_innate_policy_pref'
            f'_zone_type_{t}',
            'z_pos', ndp, t,
            'innate_policy_pref'))
        ds.append(_GlyphDataset(
            f'politician_innate_policy_aver'
            f'_zone_type_{t}',
            'z_neg', ndp, t,
            'innate_policy_aver'))
        ds.append(_GlyphDataset(
            f'politician_external_policy_pref'
            f'_zone_type_{t}',
            'z_pos', ndp, t,
            'ext_policy_pref'))
        ds.append(_GlyphDataset(
            f'politician_external_policy_aver'
            f'_zone_type_{t}',
            'z_neg', ndp, t,
            'ext_policy_aver'))
        ds.append(_GlyphDataset(
            f'politician_innate_trait'
            f'_zone_type_{t}',
            'z_pos', ndt, t,
            'innate_trait'))
        ds.append(_GlyphDataset(
            f'politician_external_trait'
            f'_zone_type_{t}',
            'z_neg', ndt, t,
            'ext_trait'))
    ds.append(_GlyphDataset(
        'government_policy', 'z_pos', ndp,
        'government', 'enacted_policy'))
    return ds


class GlyphHdf5():
    """Manages the glyph HDF5 output file
    (`*_glyphs.hdf5`).

    Writes cylinder glyph data for all Gaussian
    types (citizen, politician, government) at
    geographic positions. Each Gaussian type group
    gets per-step sub-groups containing mu, sigma,
    cos_theta, and pre-computed RGB color arrays
    per policy/trait dimension.

    Attributes
    ----------
    h_fid : h5py.File
        The open HDF5 file.
    datasets : list of _GlyphDataset
        One descriptor per Gaussian type group.
    total_population : int
        Total citizen count; normalizes all
        population values to [0, 1].
    num_zone_types : int
    num_zones : list of int
        num_zones[t] = zone instance count for
        zone type t.
    """

    def __init__(self, settings, world):
        subdir = f"{settings.outfile}_glyphs"
        os.makedirs(subdir, exist_ok=True)
        hdf5_basename = (
            os.path.basename(settings.outfile)
            + "_glyphs.hdf5")
        self.subdir = subdir
        self.hdf5_basename = hdf5_basename
        self.h_fid = h5.File(
            os.path.join(subdir, hdf5_basename),
            "x")
        self.datasets = _build_glyph_datasets(
            settings, world)
        self.total_population = len(world.citizens)
        self.num_zone_types = world.num_zone_types
        self.num_zones = [
            len(world.zones[t])
            for t in range(world.num_zone_types)]

        # Create one HDF5 group per Gaussian type.
        for desc in self.datasets:
            self.h_fid.create_group(desc.group_name)

        # Create GlyphGeometry and subgroups.
        geo = self.h_fid.create_group(
            'GlyphGeometry')
        geo.create_group('patches')
        for t in range(world.num_zone_types):
            geo.create_group(f'zone_type_{t}')
        geo.create_group('government')

        # Create population groups.
        self.h_fid.create_group(
            'citizen_patch_population')
        for t in range(world.num_zone_types):
            self.h_fid.create_group(
                f'politician_zone_population'
                f'_zone_type_{t}')
        self.h_fid.create_group(
            'government_population')

        # Write patch geometry (static).
        nx = world.x_num_patches
        ny = world.y_num_patches
        num_p = nx * ny
        xyz = np.zeros(
            (num_p, 3), dtype=np.float32)
        for i in range(nx):
            for j in range(ny):
                idx = i * ny + j
                xyz[idx, 0] = float(i)
                xyz[idx, 1] = float(j)
        geo_p = self.h_fid['GlyphGeometry/patches']
        geo_p.create_dataset('XYZ', data=xyz)

        # Write static zone centroid geometry for
        # zone_type_1 and higher.
        for t in range(1, world.num_zone_types):
            nz = len(world.zones[t])
            xyz_z = np.zeros(
                (nz, 3), dtype=np.float32)
            for zi, zone in enumerate(
                    world.zones[t]):
                xs = [p.x_location
                      for p in zone.patches]
                ys = [p.y_location
                      for p in zone.patches]
                xyz_z[zi, 0] = float(np.mean(xs))
                xyz_z[zi, 1] = float(np.mean(ys))
            geo_zt = self.h_fid[
                f'GlyphGeometry/zone_type_{t}']
            geo_zt.create_dataset(
                'XYZ', data=xyz_z)

        # Write government geometry (static).
        # World centroid = mean of all patch centers
        # on a regular grid: ((nx-1)/2, (ny-1)/2).
        gov_xyz = np.array(
            [[(nx - 1) / 2.0, (ny - 1) / 2.0, 0.0]],
            dtype=np.float32)
        geo_gov = self.h_fid[
            'GlyphGeometry/government']
        geo_gov.create_dataset(
            'XYZ', data=gov_xyz)

    def _citizen_arrays(self, world, attr, d):
        """Per-patch mu, sigma, cos_theta for
        citizen Gaussian `attr`, dimension d.

        Returns three 1-D float32 arrays of
        length num_patches (x_num_patches *
        y_num_patches), row-major order: patch
        (i, j) at index i*y_num_patches + j.
        """
        ny = world.y_num_patches
        n = world.x_num_patches * ny
        mu    = np.zeros(n, dtype=np.float32)
        sigma = np.zeros(n, dtype=np.float32)
        ct    = np.zeros(n, dtype=np.float32)
        count = np.zeros(n, dtype=np.float32)
        for cit in world.citizens:
            g   = getattr(cit, attr)
            x   = cit.current_patch.x_location
            y   = cit.current_patch.y_location
            idx = x * ny + y
            mu[idx]    += g.mu[d]
            sigma[idx] += g.sigma[d]
            ct[idx]    += g.cos_theta[d]
            count[idx] += 1.0
        safe = np.where(count == 0, 1.0, count)
        return (mu / safe,
                sigma / safe,
                ct / safe)

    def _politician_arrays(
            self, world, attr, zt, d):
        """Per-zone mu, sigma, cos_theta for
        politician Gaussian `attr` of zone_type
        zt, dimension d.

        Returns three 1-D float32 arrays of
        length num_zones[zt], one per zone
        instance.
        """
        nz    = self.num_zones[zt]
        mu    = np.zeros(nz, dtype=np.float32)
        sigma = np.zeros(nz, dtype=np.float32)
        ct    = np.zeros(nz, dtype=np.float32)
        count = np.zeros(nz, dtype=np.float32)
        for pol in world.politicians:
            if pol.zone_type != zt:
                continue
            g = getattr(pol, attr)
            z = pol.zone.zone_index
            mu[z]    += g.mu[d]
            sigma[z] += g.sigma[d]
            ct[z]    += g.cos_theta[d]
            count[z] += 1.0
        safe = np.where(count == 0, 1.0, count)
        return (mu / safe,
                sigma / safe,
                ct / safe)

    def add_step(self, world, step):
        """Write one step's glyph data to HDF5.

        Writes zone_type_0 centroid geometry,
        all Gaussian datasets (mu, sigma,
        cos_theta, color_rgb) per group per dim,
        and normalized population counts.
        """
        ny = world.y_num_patches

        # Write zone_type_0 geometry (per-step).
        nz0  = self.num_zones[0]
        xyz0 = np.zeros(
            (nz0, 3), dtype=np.float32)
        for zi, zone in enumerate(world.zones[0]):
            xs = [p.x_location
                  for p in zone.patches]
            ys = [p.y_location
                  for p in zone.patches]
            xyz0[zi, 0] = float(np.mean(xs))
            xyz0[zi, 1] = float(np.mean(ys))
        geo0 = self.h_fid[
            'GlyphGeometry/zone_type_0']
        s0_grp = geo0.create_group(f'Step{step}')
        s0_grp.create_dataset('XYZ', data=xyz0)

        # Write Gaussian datasets per descriptor.
        for desc in self.datasets:
            step_grp = self.h_fid[
                desc.group_name
            ].create_group(f'Step{step}')
            attr = desc.gaussian_attr
            for d in range(desc.num_dims):
                if desc.zone_type is None:
                    mu, sigma, ct = (
                        self._citizen_arrays(
                            world, attr, d))
                elif desc.zone_type == 'government':
                    g = getattr(
                        world.government, attr)
                    mu = np.array(
                        [g.mu[d]],
                        dtype=np.float32)
                    sigma = np.array(
                        [g.sigma[d]],
                        dtype=np.float32)
                    ct = np.array(
                        [g.cos_theta[d]],
                        dtype=np.float32)
                else:
                    mu, sigma, ct = (
                        self._politician_arrays(
                            world, attr,
                            desc.zone_type, d))
                step_grp.create_dataset(
                    f'mu_dim{d}', data=mu)
                step_grp.create_dataset(
                    f'sigma_dim{d}', data=sigma)
                step_grp.create_dataset(
                    f'cos_theta_dim{d}', data=ct)
                rgb = _mu_to_rgb(mu, ct)
                step_grp.create_dataset(
                    f'color_rgb_dim{d}', data=rgb)

        # Write citizen patch population.
        nx    = world.x_num_patches
        num_p = nx * ny
        pop   = np.zeros(num_p, dtype=np.float32)
        for i in range(nx):
            for j in range(ny):
                idx = i * ny + j
                pop[idx] = float(len(
                    world.patches[i][j]
                    .citizen_list))
        pop /= self.total_population
        cpp_grp = self.h_fid[
            'citizen_patch_population'
        ].create_group(f'Step{step}')
        cpp_grp.create_dataset('count', data=pop)

        # Write politician zone population.
        for t in range(self.num_zone_types):
            nz   = self.num_zones[t]
            zpop = np.zeros(nz, dtype=np.float32)
            for zi, zone in enumerate(
                    world.zones[t]):
                for patch in zone.patches:
                    zpop[zi] += len(
                        patch.citizen_list)
            zpop /= self.total_population
            zpop_grp = self.h_fid[
                f'politician_zone_population'
                f'_zone_type_{t}'
            ].create_group(f'Step{step}')
            zpop_grp.create_dataset(
                'count', data=zpop)

        # Write government population.
        gov_pop = np.ones(1, dtype=np.float32)
        gov_grp = self.h_fid[
            'government_population'
        ].create_group(f'Step{step}')
        gov_grp.create_dataset(
            'count', data=gov_pop)

    def close(self):
        """Flush and close the HDF5 file."""
        self.h_fid.close()


class GlyphXdmf():
    """Writes per-grid-type XDMF files into the
    glyph subdirectory. One file per topology:

    - patches.xdmf      (citizens, 81 points)
    - zone_type_{t}.xdmf (politicians)
    - government.xdmf   (1 point)

    Each file is a simple temporal collection of
    Uniform grids (one per step). This gives
    Xdmf3ReaderS one independent time series per
    file, which it handles correctly. Because
    each file has only one grid type, ParaView
    loads them separately and there is no block
    isolation problem.
    """

    def __init__(self, settings, glyph_hdf5):
        self.settings = settings
        self.hdf5 = glyph_hdf5

    def _write_xdmf_file(
            self, filename, grid_name,
            descs, geom_path_fn,
            pop_attr_name, pop_path_fn,
            num_steps):
        """Write one XDMF file: a temporal
        collection of Uniform grids.

        Parameters
        ----------
        filename      : str
            Full path to the output file.
        grid_name     : str
            Name attribute of the temporal
            collection Grid element.
        descs         : list of _GlyphDataset
            Gaussian groups on this topology.
        geom_path_fn  : callable(int) -> str
            Returns HDF5 path to XYZ for step i.
        pop_attr_name : str
            XDMF Attribute name for population.
        pop_path_fn   : callable(int) -> str
            Returns HDF5 path to population
            count dataset for step i.
        num_steps     : int
        """
        hname = self.hdf5.hdf5_basename
        hf    = self.hdf5.h_fid
        xf = open(filename, 'w')
        xf.write(
            '<?xml version="1.0"'
            ' encoding="utf-8"?>\n')
        xf.write(
            '<!DOCTYPE Xdmf SYSTEM'
            ' "Xdmf.dtd" []>\n')
        root = etree.Element("root")
        xdmf = etree.SubElement(
            root, "Xdmf", Version="3.0",
            xmlnsxi=(
                "[http://www.w3.org/2001/"
                "XInclude]"))
        domain = etree.SubElement(
            xdmf, "Domain")
        temporal = etree.SubElement(
            domain, "Grid",
            Name=grid_name,
            GridType="Collection",
            CollectionType="Temporal")
        for i in range(num_steps):
            geom_path = geom_path_fn(i)
            n = hf[geom_path].shape[0]
            step_grid = etree.SubElement(
                temporal, "Grid",
                Name=f"Step{i}",
                GridType="Uniform")
            etree.SubElement(
                step_grid, "Time",
                Value=f"{i}.0")
            etree.SubElement(
                step_grid, "Topology",
                Type="Polyvertex",
                NumberOfElements=f"{n}")
            geo = etree.SubElement(
                step_grid, "Geometry",
                GeometryType="XYZ")
            geo_di = etree.SubElement(
                geo, "DataItem",
                NumberType="Float",
                Dimensions=f"{n} 3",
                Format="HDF")
            geo_di.text = (
                f"{hname}:/{geom_path}")
            for desc in descs:
                base = (
                    f"{desc.group_name}"
                    f"/Step{i}")
                for d in range(desc.num_dims):
                    for ds_name, atype, dims in [
                        (f"mu_dim{d}",
                         "Scalar", f"{n}"),
                        (f"sigma_dim{d}",
                         "Scalar", f"{n}"),
                        (f"cos_theta_dim{d}",
                         "Scalar", f"{n}"),
                        (f"color_rgb_dim{d}",
                         "Vector", f"{n} 3"),
                    ]:
                        attr_name = (
                            f"{desc.group_name}"
                            f"_{ds_name}")
                        attr_el = etree.SubElement(
                            step_grid, "Attribute",
                            Name=attr_name,
                            Center="Node",
                            AttributeType=atype)
                        di = etree.SubElement(
                            attr_el, "DataItem",
                            NumberType="Float",
                            Dimensions=dims,
                            Format="HDF")
                        di.text = (
                            f"{hname}:/"
                            f"{base}/{ds_name}")
            pop_path = pop_path_fn(i)
            pop_el = etree.SubElement(
                step_grid, "Attribute",
                Name=pop_attr_name,
                Center="Node",
                AttributeType="Scalar")
            pop_di = etree.SubElement(
                pop_el, "DataItem",
                NumberType="Float",
                Dimensions=f"{n}",
                Format="HDF")
            pop_di.text = (
                f"{hname}:/{pop_path}")
        temp_xml = etree.tostring(
            xdmf, pretty_print=True,
            encoding="utf-8").decode()
        xf.write(temp_xml.replace(
            "xmlnsxi", "xmlns:xi", 1))
        xf.close()

    def write(self, num_steps):
        """Write one XDMF file per grid type."""
        subdir = self.hdf5.subdir
        nzt    = self.hdf5.num_zone_types

        # Patches file (citizens).
        cit_descs = [
            d for d in self.hdf5.datasets
            if d.zone_type is None]
        self._write_xdmf_file(
            os.path.join(subdir, 'patches.xdmf'),
            'patches',
            cit_descs,
            lambda i: 'GlyphGeometry/patches/XYZ',
            'citizen_population',
            lambda i: (
                f'citizen_patch_population'
                f'/Step{i}/count'),
            num_steps)

        # Zone-type files (politicians).
        for t in range(nzt):
            pol_descs = [
                d for d in self.hdf5.datasets
                if d.zone_type == t]
            def geom_fn(i, _t=t):
                if _t == 0:
                    return (
                        f'GlyphGeometry'
                        f'/zone_type_0'
                        f'/Step{i}/XYZ')
                return (
                    f'GlyphGeometry'
                    f'/zone_type_{_t}/XYZ')
            def pop_fn(i, _t=t):
                return (
                    f'politician_zone_population'
                    f'_zone_type_{_t}'
                    f'/Step{i}/count')
            self._write_xdmf_file(
                os.path.join(
                    subdir,
                    f'zone_type_{t}.xdmf'),
                f'zone_type_{t}',
                pol_descs,
                geom_fn,
                'zone_population',
                pop_fn,
                num_steps)

        # Government file.
        gov_descs = [
            d for d in self.hdf5.datasets
            if d.zone_type == 'government']
        self._write_xdmf_file(
            os.path.join(
                subdir, 'government.xdmf'),
            'government',
            gov_descs,
            lambda i: (
                'GlyphGeometry/government/XYZ'),
            'government_population',
            lambda i: (
                f'government_population'
                f'/Step{i}/count'),
            num_steps)


def _compute_sigma_ref(world, desc):
    """Mean sigma across all agents for desc.

    Averages over all agents and all dims.
    Used to normalize cylinder radius so that
    all Gaussian types are visually comparable
    regardless of their inherent sigma scale.
    Returns 1.0 if no agents are found or if
    the mean is zero.
    """
    attr = desc.gaussian_attr
    vals = []
    if desc.zone_type is None:
        for cit in world.citizens:
            g = getattr(cit, attr)
            vals.extend(g.sigma.tolist())
    elif desc.zone_type == 'government':
        g = getattr(world.government, attr)
        vals.extend(g.sigma.tolist())
    else:
        t = desc.zone_type
        for pol in world.politicians:
            if pol.zone_type == t:
                g = getattr(pol, attr)
                vals.extend(g.sigma.tolist())
    if not vals:
        return 1.0
    ref = float(np.mean(vals))
    return ref if ref > 0.0 else 1.0


def write_paraview_script(settings, world):
    """Generate a pvpython cylinder glyph script.

    Writes {settings.outfile}_glyphs.py.
    The script loads one XDMF file per grid
    type from the glyph subdirectory, so each
    reader contains exactly the right points.
    No ExtractBlock needed.
    Tune the constants at the top of the
    generated script, then run with pvpython
    or load via File -> Run Script in ParaView.
    """
    datasets = _build_glyph_datasets(
        settings, world)
    subdir_basename = (
        os.path.basename(settings.outfile)
        + '_glyphs')

    # Per-descriptor sigma normalization factors.
    sigma_refs = {
        d.group_name: _compute_sigma_ref(world, d)
        for d in datasets}

    # Default POPULATION_SCALE: gives the average
    # patch (1/num_patches citizens) a height of
    # ~0.3 grid units.
    num_patches = (
        world.x_num_patches
        * world.y_num_patches)
    pop_scale_default = round(
        0.3 * num_patches, 1)

    def _pop_attr(desc):
        if desc.zone_type is None:
            return 'citizen_population'
        if desc.zone_type == 'government':
            return 'government_population'
        return 'zone_population'

    L = []

    def w(*lines):
        L.extend(lines)

    w("# Generated by stodem.py",
      "# Tune constants below and re-run.",
      "from paraview.simple import *",
      "",
      "# ---- Tunable constants ----",
      f"POPULATION_SCALE      = {pop_scale_default}",
      "CYLINDER_RADIUS_SCALE = 0.3",
      "DIM       = 0  # dim to display",
      "ZONE_TYPE = 0  # politician zone type",
      "NUM_ROWS = 2  # grid rows",
      "NUM_COLS = 3  # grid columns",
      "",
      "import os as _os",
      "_here = _os.path.dirname(",
      "    _os.path.abspath(__file__))",
      "_subdir = _os.path.join(",
      f"    _here, {subdir_basename!r})",
      "",
      "# (group_name, z_dir, zone_type, pop_attr)",
      "DESCRIPTORS = [")

    for desc in datasets:
        pa = _pop_attr(desc)
        line = (
            f"    ({desc.group_name!r},"
            f" {desc.z_dir!r},"
            f" {desc.zone_type!r},"
            f" {pa!r}),")
        w(line)

    w("]")

    w("",
      "# Per-group sigma normalization (computed",
      "# at generation time from simulation data).",
      "SIGMA_REFS = {")
    for d in datasets:
        ref = sigma_refs[d.group_name]
        w(f"    {d.group_name!r}: {ref:.6g},")
    w("}",
      "",
      "# Filter to active descriptors.",
      "active = [",
      "    d for d in DESCRIPTORS",
      "    if (d[2] is None",
      "        or d[2] == 'government'",
      "        or d[2] == ZONE_TYPE)",
      "]",
      "if not active:",
      "    raise RuntimeError(",
      "        'No active descriptors.')",
      "active = active[:NUM_ROWS * NUM_COLS]",
      "",
      "",
      "# ---- Compatibility shim ----",
      "# Xdmf3ReaderS uses FileName (singular);",
      "# XDMFReader uses FileNames (plural).",
      "try:",
      "    Xdmf3ReaderS",
      "    def _xdmf_reader(path):",
      "        return Xdmf3ReaderS(",
      "            FileName=[path])",
      "except NameError:",
      "    def _xdmf_reader(path):",
      "        return XDMFReader(",
      "            FileNames=[path])",
      "",
      "# ---- Layout: NUM_ROWS x NUM_COLS grid ----",
      "# Splitting cell c: top/left child = 2*c+1,",
      "#                   bot/right child = 2*c+2.",
      "layout = GetLayout()",
      "row_cells = []",
      "_cell = 0",
      "for _r in range(NUM_ROWS - 1):",
      "    layout.SplitVertical(",
      "        _cell, 1.0 / (NUM_ROWS - _r))",
      "    row_cells.append(2 * _cell + 1)",
      "    _cell = 2 * _cell + 2",
      "row_cells.append(_cell)",
      "leaf_cells = []",
      "for _r_cell in row_cells:",
      "    _cell = _r_cell",
      "    for _c in range(NUM_COLS - 1):",
      "        layout.SplitHorizontal(",
      "            _cell, 1.0 / (NUM_COLS - _c))",
      "        leaf_cells.append(2 * _cell + 1)",
      "        _cell = 2 * _cell + 2",
      "    leaf_cells.append(_cell)",
      "views = [GetActiveView()]",
      "for _hint in leaf_cells[1:]:",
      "    _v = CreateRenderView()",
      "    AssignViewToLayout(",
      "        view=_v, layout=layout,",
      "        hint=_hint)",
      "    views.append(_v)",
      "n = len(views)",
      "",
      "# ---- Load one reader per grid type ----",
      "# Each XDMF file contains exactly one",
      "# grid topology, so no ExtractBlock needed.",
      "def _grid_key(zone_type):",
      "    if zone_type is None:",
      "        return 'patches'",
      "    if zone_type == 'government':",
      "        return 'government'",
      "    return f'zone_type_{zone_type}'",
      "",
      "_readers = {}",
      "for _desc in active:",
      "    _key = _grid_key(_desc[2])",
      "    if _key not in _readers:",
      "        _fname = _os.path.join(",
      "            _subdir, _key + '.xdmf')",
      "        _readers[_key] = _xdmf_reader(",
      "            _fname)",
      "        UpdatePipeline(",
      "            proxy=_readers[_key])",
      "",
      "# ---- Pipeline per descriptor ----",
      "for idx, desc in enumerate(active):",
      "    gname = desc[0]",
      "    z_dir = desc[1]",
      "    zt    = desc[2]",
      "    pop   = desc[3]",
      "    view  = views[idx]",
      "    SetActiveView(view)",
      "",
      "    reader = _readers[_grid_key(zt)]",
      "",
      "    # Programmable Filter per pane.",
      "    # Edit Script in ParaView GUI to",
      "    # switch GROUP, DIM, Z_DIR.",
      "    sigma_ref = SIGMA_REFS[gname]",
      "    _pf_script = (",
      "        'GROUP = %r\\n'",
      "        'DIM   = %d\\n'",
      "        'Z_DIR = %r\\n'",
      "        '# Edit GROUP/DIM/Z_DIR to switch.\\n'",
      "        '# When Z_DIR changes also update\\n'",
      "        '#  GlyphTransform Rotate/Translate\\n'",
      "        '#  z_pos: [90,0,0]  [0,0, 0.5]\\n'",
      "        '#  z_neg: [-90,0,0] [0,0,-0.5]\\n'",
      "        'CYLINDER_RADIUS_SCALE = %g\\n'",
      "        'POPULATION_SCALE      = %g\\n'",
      "        'POP_ATTR  = %r\\n'",
      "        'SIGMA_REF = %g\\n'",
      "        'from paraview.vtk.numpy_interface'",
      "        ' import dataset_adapter as dsa\\n'",
      "        'import numpy as np\\n'",
      "        'inp_vtk = self.GetInput()\\n'",
      "        'out_vtk = self.GetOutput()\\n'",
      "        'out_vtk.ShallowCopy(inp_vtk)\\n'",
      "        'out_vtk.GetPointData().Initialize()\\n'",
      "        'pd = dsa.WrapDataObject(inp_vtk).PointData\\n'",
      "        'sig_name = (GROUP + \"_sigma_dim\""
      " + str(DIM))\\n'",
      "        'rgb_name = (GROUP + \"_color_rgb_dim\""
      " + str(DIM))\\n'",
      "        'sigma = np.array(pd[sig_name])\\n'",
      "        'color = pd[rgb_name]\\n'",
      "        'pop_arr = np.array(pd[POP_ATTR])\\n'",
      "        'r = sigma / SIGMA_REF"
      " * CYLINDER_RADIUS_SCALE\\n'",
      "        'h = pop_arr * POPULATION_SCALE\\n'",
      "        'cyl = np.column_stack("
      "[r, r, h]).astype(np.float32)\\n'",
      "        'out_dsa = dsa.WrapDataObject(out_vtk)\\n'",
      "        'out_dsa.PointData.append("
      "cyl, \"cylinder_scale\")\\n'",
      "        'out_dsa.PointData.append("
      "color, \"color_rgb\")\\n'",
      "    ) % (gname, DIM, z_dir,",
      "         CYLINDER_RADIUS_SCALE,",
      "         POPULATION_SCALE, pop,",
      "         sigma_ref)",
      "    _pf = ProgrammableFilter(Input=reader)",
      "    _pf.Script = _pf_script",
      "",
      "    # Cylinder glyph, scale by components.",
      "    glyph = Glyph(",
      "        Input=_pf, GlyphType='Cylinder')",
      "    glyph.OrientationArray = [",
      "        'POINTS',",
      "        'No orientation array']",
      "    glyph.ScaleArray = [",
      "        'POINTS', 'cylinder_scale']",
      "    glyph.ScaleFactor     = 1.0",
      "    glyph.GlyphMode       = 'All Points'",
      "    glyph.VectorScaleMode = (",
      "        'Scale by Components')",
      "",
      "    # GlyphTransform: Scale->Rotate->Translate.",
      "    gt = glyph.GlyphTransform",
      "    gt.Scale = [2.0, 1.0, 2.0]",
      "    if z_dir == 'z_pos':",
      "        gt.Rotate    = [90.0, 0.0, 0.0]",
      "        gt.Translate = [0.0, 0.0, 0.5]",
      "    else:",
      "        gt.Rotate    = [-90.0, 0.0, 0.0]",
      "        gt.Translate = [0.0, 0.0, -0.5]",
      "",
      "    # Display with pre-computed RGB.",
      "    display = Show(glyph, view)",
      "    ColorBy(display,",
      "        ('POINTS', 'color_rgb'))",
      "    display.MapScalars = 0",
      "",
      "    # Orthographic top-down camera.",
      "    view.CameraParallelProjection = 1",
      "    view.CameraPosition   = [0., 0., 1.]",
      "    view.CameraFocalPoint = [0., 0., 0.]",
      "    view.CameraViewUp     = [0., 1., 0.]",
      "    view.ResetCamera()",
      "    Render(view)",
      "",
      "GetAnimationScene()"
      ".UpdateAnimationUsingDataTimeSteps()",
      "RenderAllViews()",
      "")

    out = f"{settings.outfile}_glyphs.py"
    with open(out, 'w') as pf:
        pf.write('\n'.join(L))
