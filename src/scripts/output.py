import os
import numpy as np
import h5py as h5
from lxml import etree


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
                self.group_gid[
                    p.group].create_dataset(
                    f"{p.name}{i}",
                    compression="gzip",
                    data=p.data))

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
        xdmf = etree.SubElement(
                root, "Xdmf", Version="3.0",
                xmlnsxi=(
                    "[http://www.w3.org/2001/"
                    "XInclude]"))

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
            timestep_grid.append(
                    etree.SubElement(
                        time_collection_grid,
                        "Grid",
                        Name=f"Step{i}",
                        GridType="Uniform"))

            # Make the time step value an integer.
            #   (Arbitrary time duration.)
            timestep.append(
                    etree.SubElement(
                        timestep_grid[i], "Time",
                        Value=f"{i}.0"))

            # Define the topology for this grid.
            #   (Ideally, we would define a single
            #   topology just below the "Domain" and
            #   then reference it here. But for some
            #   reason it didn't work and so we will
            #   just repeat the topology here for
            #   every time step grid even though it
            #   "wastes" space.)
            topology.append(
                    etree.SubElement(
                        timestep_grid[i],
                        "Topology",
                        Name="Topo",
                        TopologyType="2DCoRectMesh",
                        Dimensions=(
                            f"{world.x_num_patches}"
                            f" {world.y_num_patches}"
                        )))

            # Same for the geometry as for the
            #   topology.
            geometry.append(
                    etree.SubElement(
                        timestep_grid[i],
                        "Geometry",
                        Name="Geom",
                        GeometryType="ORIGIN_DXDY"))

            # The geometry needs two data elements,
            #   the origin and the spatial deltas.
            geometry_origin.append(
                    etree.SubElement(
                        geometry[i], "DataItem",
                        NumberType="Float",
                        Dimensions="2",
                        Format="XML"))
            geometry_origin[i].text = "0.0 0.0"
            geometry_dxdy.append(
                    etree.SubElement(
                        geometry[i], "DataItem",
                        NumberType="Float",
                        Dimensions="2",
                        Format="XML"))
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
                attribute[i].append(
                        etree.SubElement(
                            timestep_grid[i],
                            "Attribute",
                            Name=f"{p.name}",
                            Center="Node",
                            AttributeType=(
                                f"{p.datatype}")))
                data_item[i].append(
                        etree.SubElement(
                            attribute[i][-1],
                            "DataItem",
                            NumberType="Float",
                            Dimensions=(
                                f"{world.x_num_patches}"
                                f" {world.y_num_patches}"
                            ),
                            Format="HDF"))
                data_item[i][-1].text = (
                        f"{settings.outfile}"
                        f".hdf5:/{p.group}/"
                        f"{p.name}{i}")

        # Pretty print the xdmf file to the output
        #   file.
        temp_xml = etree.tostring(
                xdmf, pretty_print=True,
                encoding="utf-8").decode()
        self.x.write(
                temp_xml.replace(
                    "xmlnsxi", "xmlns:xi", 1))
        self.x.close()


# ===========================================================
# Glyph output: compact point-cloud visualization
# ===========================================================
#
# Each Gaussian type (citizen policy pref/aver/ideal,
# citizen trait pref/aver, politician innate/external
# policy pref/aver and trait per zone type, government
# enacted policy) is written as a separate named block
# (Polyvertex point cloud) inside one shared HDF5 file.
# A companion XDMF file describes a temporal collection
# of spatial collections (one spatial collection per
# output step, each containing all named blocks).
#
# Point layout: patch (i, j) -> linear index i*Ny + j.
# XYZ[i*Ny+j] = (float(i), float(j), z_offset).
# Three static geometry arrays are written once:
#   z_pos  (+0.25): preference / innate  Gaussians
#   z_neg  (-0.25): aversion  / external Gaussians
#   z_zero ( 0.00): ideal policy, government policy
#
# Per-step arrays per block per dimension d:
#   mu_d{d}          : patch-averaged Gaussian centre
#   inv_sigma_d{d}   : sigma_ref / sigma  (larger =>
#                      sharper / more certain position)
#   cos_theta_d{d}   : |cos(theta)| in [0,1]  (larger
#                      => more engaged / less apathetic)
#
# sigma_ref for each Gaussian type equals the
# *_stddev_stddev XML parameter used to initialise
# that Gaussian's sigma, so inv_sigma ~ 1 for a
# "typical" agent regardless of Gaussian type.
#
# Government values are scalars (one per policy dim)
# replicated to all Np patch points so that the
# spatial layout matches citizens and politicians.


def _hsv_to_rgb_arr(h, s, v):
    """Vectorized HSV -> float32 RGB in [0, 1].

    All inputs are 1-D float32 arrays of equal
    length with values in [0, 1].  Returns a
    float32 array of shape (N, 3).
    """
    h6 = h * 6.0
    i  = np.floor(h6).astype(np.int32) % 6
    f  = (h6 - np.floor(h6)).astype(np.float32)
    p  = v * (1.0 - s)
    q  = v * (1.0 - f * s)
    t  = v * (1.0 - (1.0 - f) * s)
    N  = len(h)
    r = np.empty(N, dtype=np.float32)
    g = np.empty(N, dtype=np.float32)
    b = np.empty(N, dtype=np.float32)
    for case, (rv, gv, bv) in enumerate([
            (v, t, p), (q, v, p),
            (p, v, t), (p, q, v),
            (t, p, v), (v, p, q)]):
        m = (i == case)
        r[m], g[m], b[m] = rv[m], gv[m], bv[m]
    return np.stack([r, g, b], axis=1)


def _mu_to_rgb(mu, cos_theta):
    """Per-point float32 RGB from mu and cos(theta).

    Encodes policy/trait position as hue and
    engagement as saturation.  Brightness is
    always 1.0 so all agents remain visible.

    Hue mapping (Cool-to-Warm two-pole):
      mu << 0  ->  blue  (H = 0.667)
      mu  = 0  ->  white (S = 0, neutral)
      mu >> 0  ->  red   (H = 0.0)
    Position is normalised by sigmoid so mu = 0
    is always the white neutral point regardless
    of the distribution scale.

    Saturation = |cos_theta| in [0, 1]:
      0  ->  grey   (fully apathetic)
      1  ->  vivid  (fully engaged)

    Parameters
    ----------
    mu        : float32 array, shape (N,)
    cos_theta : float32 array, shape (N,)

    Returns
    -------
    float32 array, shape (N, 3), values in [0, 1]
    """
    ct = np.abs(cos_theta.astype(np.float32))
    # Sigmoid normalisation: mu=0 -> t=0.5 (white)
    t = (1.0 / (
        1.0 + np.exp(-mu.astype(np.float64)))
        ).astype(np.float32)
    # Hue: blue pole at t<0.5, red pole at t>0.5
    h = np.where(t < 0.5,
                 np.float32(0.667),
                 np.float32(0.0))
    # Saturation from position: 0 at centre, 1 at extremes
    s_pos = np.where(t < 0.5,
                     1.0 - 2.0 * t,
                     2.0 * t - 1.0
                     ).astype(np.float32)
    s = s_pos * ct
    v = np.ones(len(mu), dtype=np.float32)
    return _hsv_to_rgb_arr(h, s, v)


def _build_glyph_datasets(settings, world):
    """Build the descriptor list for all glyph
    datasets except government.

    Returns a list of 7-tuples:
      (tag, z_key, mu_pat, sig_pat, ct_pat,
       sigma_ref, n_dims)

      tag       : HDF5 group and XDMF grid name
      z_key     : "z_pos" | "z_neg" | "z_zero"
      mu_pat    : property-name format str (use {d})
      sig_pat   : property-name format str (use {d})
      ct_pat    : property-name format str (use {d})
      sigma_ref : reference sigma for normalisation
      n_dims    : number of Gaussian dimensions

    Government is excluded; its data comes from
    world.government rather than world.properties.
    """
    ndp = world.num_policy_dims
    ndt = world.num_trait_dims
    nzt = world.num_zone_types
    cit = settings.infile_dict[1]["citizens"]
    pol = settings.infile_dict[1]["politicians"]

    ds = []

    # --- Citizen policy ---
    ds.append((
        "cit_pol_pref", "z_pos",
        "PolicyPrefMu{d}",
        "PolicyPrefSigma{d}",
        "PolicyPrefCosTheta{d}",
        float(cit["policy_pref_stddev_stddev"]),
        ndp))
    ds.append((
        "cit_pol_aver", "z_neg",
        "PolicyAverMu{d}",
        "PolicyAverSigma{d}",
        "PolicyAverCosTheta{d}",
        float(cit["policy_aver_stddev_stddev"]),
        ndp))
    ds.append((
        "cit_pol_ideal", "z_zero",
        "IdealPolicyMu{d}",
        "IdealPolicySigma{d}",
        "IdealPolicyCosTheta{d}",
        float(cit[
            "ideal_policy_pref_stddev_stddev"]),
        ndp))

    # --- Citizen trait ---
    ds.append((
        "cit_trt_pref", "z_pos",
        "TraitPrefMu{d}",
        "TraitPrefSigma{d}",
        "TraitPrefCosTheta{d}",
        float(cit["trait_pref_stddev_stddev"]),
        ndt))
    ds.append((
        "cit_trt_aver", "z_neg",
        "TraitAverMu{d}",
        "TraitAverSigma{d}",
        "TraitAverCosTheta{d}",
        float(cit["trait_aver_stddev_stddev"]),
        ndt))

    # --- Politician per zone type ---
    for zt in range(nzt):
        sfx = f"_ZT{zt}"
        ds.append((
            f"pol_inn_pol_pref_zt{zt}",
            "z_pos",
            f"InnPolicyPrefMu{{d}}{sfx}",
            f"InnPolicyPrefSigma{{d}}{sfx}",
            f"InnPolicyPrefCosTheta{{d}}{sfx}",
            float(pol[
                "policy_pref_stddev_stddev"]),
            ndp))
        ds.append((
            f"pol_inn_pol_aver_zt{zt}",
            "z_neg",
            f"InnPolicyAverMu{{d}}{sfx}",
            f"InnPolicyAverSigma{{d}}{sfx}",
            f"InnPolicyAverCosTheta{{d}}{sfx}",
            float(pol[
                "policy_aver_stddev_stddev"]),
            ndp))
        ds.append((
            f"pol_ext_pol_pref_zt{zt}",
            "z_pos",
            f"ExtPolicyPrefMu{{d}}{sfx}",
            f"ExtPolicyPrefSigma{{d}}{sfx}",
            f"ExtPolicyPrefCosTheta{{d}}{sfx}",
            float(pol[
                "policy_pref_stddev_stddev"]),
            ndp))
        ds.append((
            f"pol_ext_pol_aver_zt{zt}",
            "z_neg",
            f"ExtPolicyAverMu{{d}}{sfx}",
            f"ExtPolicyAverSigma{{d}}{sfx}",
            f"ExtPolicyAverCosTheta{{d}}{sfx}",
            float(pol[
                "policy_aver_stddev_stddev"]),
            ndp))
        ds.append((
            f"pol_inn_trt_zt{zt}",
            "z_pos",
            f"InnTraitMu{{d}}{sfx}",
            f"InnTraitSigma{{d}}{sfx}",
            f"InnTraitCosTheta{{d}}{sfx}",
            float(pol[
                "trait_innate_stddev_stddev"]),
            ndt))
        ds.append((
            f"pol_ext_trt_zt{zt}",
            "z_neg",
            f"ExtTraitMu{{d}}{sfx}",
            f"ExtTraitSigma{{d}}{sfx}",
            f"ExtTraitCosTheta{{d}}{sfx}",
            float(pol[
                "trait_ext_stddev_stddev"]),
            ndt))

    return ds


def _add_glyph_grid(parent, tag, z_key,
                    n_dims, np_pts,
                    hdf5_file, step_idx):
    """Append one Uniform Polyvertex grid to
    a parent XDMF element.

    Parameters
    ----------
    parent : lxml element
        The containing spatial collection.
    tag : str
        HDF5 group name and XDMF grid name.
    z_key : str
        "z_pos", "z_neg", or "z_zero".
    n_dims : int
        Number of Gaussian dimensions.
    np_pts : int
        Total number of points (Nx * Ny).
    hdf5_file : str
        Filename (not path) of glyph HDF5.
    step_idx : int
        Output step index.
    """
    g = etree.SubElement(
        parent, "Grid",
        Name=tag, GridType="Uniform")
    etree.SubElement(
        g, "Topology",
        TopologyType="Polyvertex",
        NumberOfElements=str(np_pts))
    geom = etree.SubElement(
        g, "Geometry",
        GeometryType="XYZ")
    di = etree.SubElement(
        geom, "DataItem",
        NumberType="Float",
        Dimensions=f"{np_pts} 3",
        Format="HDF")
    di.text = (
        f"{hdf5_file}:"
        f"/GlyphGeometry/{z_key}/XYZ")
    for d in range(n_dims):
        for qty in (
            f"mu_d{d}",
            f"inv_sigma_d{d}",
            f"cos_theta_d{d}",
        ):
            attr = etree.SubElement(
                g, "Attribute",
                Name=qty,
                AttributeType="Scalar",
                Center="Node")
            di = etree.SubElement(
                attr, "DataItem",
                NumberType="Float",
                Dimensions=str(np_pts),
                Format="HDF")
            di.text = (
                f"{hdf5_file}:/{tag}"
                f"/Step{step_idx}/{qty}")
        # Pre-computed RGB colour (float32 vector):
        #   hue = sigmoid(mu), sat = |cos_theta|,
        #   value = 1.0. Used with MapScalars=0.
        cname = f"color_d{d}"
        attr = etree.SubElement(
            g, "Attribute",
            Name=cname,
            AttributeType="Vector",
            Center="Node")
        di = etree.SubElement(
            attr, "DataItem",
            NumberType="Float",
            Precision="4",
            Dimensions=f"{np_pts} 3",
            Format="HDF")
        di.text = (
            f"{hdf5_file}:/{tag}"
            f"/Step{step_idx}/{cname}")


class GlyphHdf5():
    """Manages the HDF5 glyph data file.

    Writes mu, inv_sigma (= sigma_ref/sigma),
    and |cos_theta| for every Gaussian type at
    patch resolution. Each Gaussian type is a
    separate named HDF5 group. Three shared
    static XYZ geometry arrays (z_pos, z_neg,
    z_zero) are written once at init.

    Attributes
    ----------
    h_fid : h5py.File
        Open HDF5 file handle.
    datasets : list of 7-tuples
        Descriptor for every glyph dataset
        except government. See
        _build_glyph_datasets() for the tuple
        field order.
    """

    def __init__(self, settings, world):
        """Create the glyph HDF5 file, build
        dataset descriptors, and write static
        XYZ geometry arrays.
        """
        self.h_fid = h5.File(
            f"{settings.outfile}"
            "_glyphs.hdf5", "x")
        self.datasets = _build_glyph_datasets(
            settings, world)
        gov = settings.infile_dict[1][
            "government"]
        self._gov_sigma_ref = float(
            gov["policy_stddev_stddev"])

        # One HDF5 group per dataset.
        for tag, *_ in self.datasets:
            self.h_fid.create_group(tag)
        self.h_fid.create_group("gov_pol")

        self._write_geometry(world)

    def _write_geometry(self, world):
        """Write three static XYZ point arrays
        shared by all glyph grids.

        point[i*Ny + j] = (i, j, z_offset)
        for i in [0, Nx), j in [0, Ny).
        """
        nx = world.x_num_patches
        ny = world.y_num_patches
        np_pts = nx * ny
        i_vals = np.repeat(
            np.arange(nx, dtype=np.float32),
            ny)
        j_vals = np.tile(
            np.arange(ny, dtype=np.float32),
            nx)
        grp = self.h_fid.create_group(
            "GlyphGeometry")
        for z_key, z_val in (
            ("z_pos",   0.25),
            ("z_neg",  -0.25),
            ("z_zero",  0.0),
        ):
            xyz = np.column_stack([
                i_vals, j_vals,
                np.full(
                    np_pts, z_val,
                    dtype=np.float32),
            ]).astype(np.float32)
            grp.create_dataset(
                f"{z_key}/XYZ",
                data=xyz,
                compression="gzip")

    def add_step(self, world, step_idx):
        """Write all glyph arrays for one step.

        Citizen and politician arrays are read
        from world.properties by name and
        flattened (C-order) to shape (Np,).
        Government values are scalars replicated
        to all Np points.

        inv_sigma is clipped to avoid division
        by zero on empty patches (sigma = 0);
        those points also have cos_theta = 0
        so they render as invisible glyphs.
        """
        props = {
            p.name: p.data
            for p in world.properties}

        for (tag, _z,
             mu_pat, sig_pat, ct_pat,
             sigma_ref, n_dims,
             ) in self.datasets:
            sgrp = self.h_fid[
                tag].create_group(
                f"Step{step_idx}")
            for d in range(n_dims):
                mu = (
                    props[mu_pat.format(d=d)]
                    .ravel()
                    .astype(np.float32))
                sig = (
                    props[sig_pat.format(d=d)]
                    .ravel()
                    .astype(np.float32))
                ct = np.abs(
                    props[ct_pat.format(d=d)]
                    .ravel()
                    .astype(np.float32))
                inv_sig = (
                    np.float32(sigma_ref)
                    / np.maximum(
                        sig,
                        np.float32(1e-9)))
                sgrp.create_dataset(
                    f"mu_d{d}",
                    data=mu,
                    compression="gzip")
                sgrp.create_dataset(
                    f"inv_sigma_d{d}",
                    data=inv_sig,
                    compression="gzip")
                sgrp.create_dataset(
                    f"cos_theta_d{d}",
                    data=ct,
                    compression="gzip")
                sgrp.create_dataset(
                    f"color_d{d}",
                    data=_mu_to_rgb(mu, ct),
                    compression="gzip")

        # Government: replicate scalar to Np.
        np_pts = (
            world.x_num_patches
            * world.y_num_patches)
        gov_ep = (
            world.government.enacted_policy)
        sig_ref = self._gov_sigma_ref
        sgrp = self.h_fid[
            "gov_pol"].create_group(
            f"Step{step_idx}")
        for d in range(world.num_policy_dims):
            mu_v = np.float32(gov_ep.mu[d])
            sig_v = float(max(
                gov_ep.sigma[d], 1e-9))
            ct_v = np.float32(
                abs(gov_ep.cos_theta[d]))
            inv_v = np.float32(
                sig_ref / sig_v)
            sgrp.create_dataset(
                f"mu_d{d}",
                data=np.full(
                    np_pts, mu_v,
                    dtype=np.float32),
                compression="gzip")
            sgrp.create_dataset(
                f"inv_sigma_d{d}",
                data=np.full(
                    np_pts, inv_v,
                    dtype=np.float32),
                compression="gzip")
            sgrp.create_dataset(
                f"cos_theta_d{d}",
                data=np.full(
                    np_pts, ct_v,
                    dtype=np.float32),
                compression="gzip")
            mu_arr = np.full(
                np_pts, mu_v,
                dtype=np.float32)
            ct_arr = np.full(
                np_pts, ct_v,
                dtype=np.float32)
            sgrp.create_dataset(
                f"color_d{d}",
                data=_mu_to_rgb(mu_arr, ct_arr),
                compression="gzip")

    def close(self):
        """Flush and close the glyph HDF5."""
        self.h_fid.close()


class GlyphXdmf():
    """Manages the XDMF metadata file for
    glyph point-cloud visualization.

    Written after the simulation completes
    so the step count matches actual HDF5
    datasets (same rationale as Xdmf).

    Structure: temporal collection of spatial
    collections. Each spatial collection (one
    per output step) contains one Uniform
    Polyvertex grid per Gaussian type.

    Attributes
    ----------
    x : file
        Open XDMF output file handle.
    """

    def __init__(self, settings):
        """Open the glyph XDMF file and write
        the XML header lines.
        """
        self.x = open(
            f"{settings.outfile}"
            "_glyphs.xdmf", "w")
        self.x.write(
            '<?xml version="1.0"'
            ' encoding="utf-8"?>\n')
        self.x.write(
            '<!DOCTYPE Xdmf SYSTEM'
            ' "Xdmf.dtd" []>\n')

    def write(self, settings, num_steps,
              world, glyph_hdf5):
        """Write the full XDMF XML body.

        Parameters
        ----------
        settings : ScriptSettings
        num_steps : int
            Actual steps written
            (sim_control.curr_step).
        world : World
        glyph_hdf5 : GlyphHdf5
            Provides glyph_hdf5.datasets for
            the block descriptor list.
        """
        np_pts = (
            world.x_num_patches
            * world.y_num_patches)
        hdf5_file = (
            f"{settings.outfile}"
            "_glyphs.hdf5")

        root = etree.Element("root")
        xdmf = etree.SubElement(
            root, "Xdmf", Version="3.0",
            xmlnsxi=(
                "[http://www.w3.org/2001/"
                "XInclude]"))
        domain = etree.SubElement(
            xdmf, "Domain")
        t_coll = etree.SubElement(
            domain, "Grid",
            Name="GlyphData",
            GridType="Collection",
            CollectionType="Temporal")

        for i in range(num_steps):
            s_coll = etree.SubElement(
                t_coll, "Grid",
                Name=f"Step{i}",
                GridType="Collection",
                CollectionType="Spatial")
            etree.SubElement(
                s_coll, "Time",
                Value=f"{i}.0")

            for (tag, z_key,
                 _mu, _sig, _ct,
                 _sr, n_dims,
                 ) in glyph_hdf5.datasets:
                _add_glyph_grid(
                    s_coll, tag, z_key,
                    n_dims, np_pts,
                    hdf5_file, i)

            # Government block.
            _add_glyph_grid(
                s_coll, "gov_pol",
                "z_zero",
                world.num_policy_dims,
                np_pts, hdf5_file, i)

        temp_xml = etree.tostring(
            xdmf, pretty_print=True,
            encoding="utf-8").decode()
        self.x.write(
            temp_xml.replace(
                "xmlnsxi", "xmlns:xi", 1))
        self.x.close()


def write_paraview_script(settings, world):
    """Write a pvpython visualization script
    for the glyph output file.

    Embeds simulation-specific parameters at
    generation time. The XDMF path is resolved
    at runtime relative to the script file so
    files can be copied to any machine.

    Three-pane layout (Citizens | Politicians |
    Government). Arrows coloured by pre-computed
    RGB (hue = sigmoid(mu), saturation =
    |cos_theta|, value = 1.0) with no LUT applied
    (MapScalars = 0). Engagment is therefore
    visible as colour saturation, not opacity.

    Parameters
    ----------
    settings : ScriptSettings
        Provides the output filename prefix.
    world : World
        Provides grid dimensions and counts.
    """
    ndp = world.num_policy_dims
    ndt = world.num_trait_dims
    nzt = world.num_zone_types
    nx  = world.x_num_patches
    ny  = world.y_num_patches
    xdmf_abs = os.path.abspath(
        f"{settings.outfile}_glyphs.xdmf")
    script_path = (
        f"{settings.outfile}_paraview.py")
    bn = os.path.basename(script_path)
    xdmf_bn = os.path.basename(xdmf_abs)

    with open(script_path, "w") as fh:

        # --- Header ---
        fh.write(
            f"# Auto-generated by STODEM."
            f" Re-run simulation to regenerate.\n"
            f"# Usage: pvpython {bn}\n"
            f"# Tested with ParaView 6.0.1.\n"
            "#\n"
            "# Three-pane layout:\n"
            "#   Citizens | Politicians | Government\n"
            "# Arrow direction:"
            " +Z=pref/innate, -Z=aver/external,\n"
            "#   +X=ideal/enacted (horizontal).\n"
            "# Arrow length ~ certainty"
            " (sigma_ref / sigma).\n"
            "# Arrow colour: hue=position(mu),"
            " sat=engagement(|cos_theta|).\n"
            "#   blue=negative, white=neutral,"
            " red=positive; grey=apathetic.\n\n"
        )

        # --- Embedded metadata ---
        fh.write(
            "# ---- Simulation metadata"
            " (do not edit) ----\n"
            "# GLYPHS_XDMF is resolved at runtime"
            " relative to this script\n"
            "# so the files can be copied"
            " to any machine.\n"
            f"# Generated from: {xdmf_abs}\n"
            "import os as _os\n"
            "_here = _os.path.dirname("
            "_os.path.abspath(__file__))\n"
            f"GLYPHS_XDMF = _os.path.join("
            f"_here, '{xdmf_bn}')\n"
            f"NUM_POLICY_DIMS = {ndp}\n"
            f"NUM_TRAIT_DIMS  = {ndt}\n"
            f"NUM_ZONE_TYPES  = {nzt}\n"
            f"NX = {nx}\n"
            f"NY = {ny}\n\n"
            "# ---- User-adjustable parameters ----\n"
            "DIM       = 0  # 0..NUM_POLICY_DIMS-1\n"
            "ZONE_TYPE = 0  # 0..NUM_ZONE_TYPES-1\n"
            "GLYPH_SCALE = 0.3\n\n"
        )

        # --- Imports and reader ---
        fh.write(
            "from paraview.simple import *\n\n"
            "reader = Xdmf3ReaderT("
            "FileName=[GLYPHS_XDMF])\n"
            "reader.UpdatePipeline()\n\n\n"
        )

        # --- Helpers ---
        # {name}, {dim}, {_zt} below are literal
        # braces in the generated file: they are
        # in regular Python strings here (not
        # f-strings), so they pass through unchanged
        # and become f-string variables when the
        # generated script runs in pvpython.
        fh.write(
            "def extract(name):\n"
            "    \"\"\"Return ExtractBlock"
            " for one named block.\"\"\"\n"
            "    eb = ExtractBlock(Input=reader)\n"
            "    eb.Selectors = [f'/Root/{name}']\n"
            "    eb.UpdatePipeline()\n"
            "    return eb\n\n\n"
            "def make_glyph(src, z_sign, dim,"
            " scale=GLYPH_SCALE):\n"
            "    \"\"\"Arrow glyph:"
            " +Z (z_sign=+1), -Z (-1), +X (0).\"\"\"\n"
            "    g = Glyph(Input=src,"
            " GlyphType='Arrow')\n"
            "    g.ScaleArray = ['POINTS',"
            " f'inv_sigma_d{dim}']\n"
            "    g.ScaleFactor = scale\n"
            "    g.GlyphMode = 0  # All Points\n"
            "    g.GlyphTransform.Rotate ="
            " [0.0, -90.0 * z_sign, 0.0]\n"
            "    g.UpdatePipeline()\n"
            "    return g\n\n\n"
            "def apply_style(rep, view, dim):\n"
            "    \"\"\"Colour by pre-computed"
            " color_d{dim} (float32 RGB).\n"
            "    Hue = sigmoid(mu): blue=negative,"
            " white=0, red=positive.\n"
            "    Saturation = |cos_theta|:"
            " vivid=engaged, grey=apathetic.\n"
            "    MapScalars=0 uses RGB values"
            " directly without a LUT.\"\"\"\n"
            "    rep.ColorArrayName ="
            " ['POINTS', f'color_d{dim}']\n"
            "    rep.MapScalars = 0\n"
            "    rep.SetScalarBarVisibility("
            "view, False)\n\n\n"
            "def show_pair(view,"
            " pref_src, aver_src, dim):\n"
            "    \"\"\"Show pref (+Z) and"
            " aver (-Z) arrow glyphs.\"\"\"\n"
            "    for src, zs in"
            " ((pref_src, +1), (aver_src, -1)):\n"
            "        g = make_glyph(src, zs, dim)\n"
            "        rep = Show(g, view)\n"
            "        apply_style(rep, view, dim)\n\n\n"
        )

        # --- Layout: 3 horizontal panes ---
        fh.write(
            "# ---- Layout: 3 horizontal panes"
            " ----\n"
            "layout = CreateLayout('STODEM"
            " Glyphs')\n"
            "layout.SplitHorizontal(0, 1.0/3.0)\n"
            "layout.SplitHorizontal(2, 0.5)\n\n"
            "view_cit = CreateView('RenderView')\n"
            "view_pol = CreateView('RenderView')\n"
            "view_gov = CreateView('RenderView')\n"
            "AssignViewToLayout("
            "view=view_cit, layout=layout, hint=1)\n"
            "AssignViewToLayout("
            "view=view_pol, layout=layout, hint=3)\n"
            "AssignViewToLayout("
            "view=view_gov, layout=layout, hint=4)\n\n"
        )

        # --- Camera: orthographic, top-down ---
        fh.write(
            "for _v in (view_cit, view_pol,"
            " view_gov):\n"
            "    _v.CameraPosition ="
            " [NX/2.0, NY/2.0, max(NX,NY)*2.0]\n"
            "    _v.CameraFocalPoint ="
            " [NX/2.0, NY/2.0, 0.0]\n"
            "    _v.CameraViewUp = [0.0, 1.0, 0.0]\n"
            "    _v.CameraParallelProjection = 1\n"
            "    _v.CameraParallelScale ="
            " max(NX, NY) / 2.0\n\n"
        )

        # --- Citizens pane (left) ---
        fh.write(
            "# ---- Citizens pane (left) ----\n"
            "_cpp = extract('cit_pol_pref')\n"
            "_cpa = extract('cit_pol_aver')\n"
            "_cpi = extract('cit_pol_ideal')\n"
            "show_pair(view_cit, _cpp, _cpa, DIM)\n"
            "_g = make_glyph(_cpi, 0, DIM)\n"
            "_r = Show(_g, view_cit)\n"
            "apply_style(_r, view_cit, DIM)\n"
            "Render(view_cit)\n\n"
        )

        # --- Politicians pane (middle) ---
        # {_zt} is a literal brace in the
        # generated file (f-string variable
        # at pvpython runtime).
        fh.write(
            "# ---- Politicians pane (middle)"
            " ----\n"
            "_zt = ZONE_TYPE\n"
            "_pip = extract("
            "f'pol_inn_pol_pref_zt{_zt}')\n"
            "_pia = extract("
            "f'pol_inn_pol_aver_zt{_zt}')\n"
            "_pep = extract("
            "f'pol_ext_pol_pref_zt{_zt}')\n"
            "_pea = extract("
            "f'pol_ext_pol_aver_zt{_zt}')\n"
            "show_pair(view_pol, _pip, _pia, DIM)\n"
            "show_pair(view_pol, _pep, _pea, DIM)\n"
            "Render(view_pol)\n\n"
        )

        # --- Government pane (right) ---
        fh.write(
            "# ---- Government pane (right) ----\n"
            "_gov = extract('gov_pol')\n"
            "_g = make_glyph(_gov, 0, DIM)\n"
            "_r = Show(_g, view_gov)\n"
            "apply_style(_r, view_gov, DIM)\n"
            "Render(view_gov)\n\n"
            "RenderAllViews()\n"
        )
