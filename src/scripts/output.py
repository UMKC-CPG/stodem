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

    Embeds simulation-specific parameters
    (absolute file path, grid dimensions,
    dimension counts) at generation time so
    the script runs without modification.

    Generated script shows a three-pane layout:
      Left   — citizens (policy pref/aver/ideal)
      Middle — politicians (innate+external
               policy pref/aver, one zone type)
      Right  — government enacted policy
    Arrow direction encodes type:
      +Z => preference/innate, -Z => aversion/
      external, +X => ideal/enacted.
    Arrow length ~ certainty (inv_sigma).
    Arrow colour ~ mu (Cool to Warm).
    Arrow opacity ~ engagement (|cos_theta|).

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

    with open(script_path, "w") as fh:

        # --- Header and embedded metadata ---
        fh.write(
            "# Auto-generated by STODEM."
            " Re-run the simulation\n"
            "# to regenerate."
            f" Usage: pvpython {bn}\n"
            "# Tested with ParaView 6.0.1.\n"
            "#\n"
            "# Three-pane layout:\n"
            "#   Citizens | Politicians"
            " | Government\n"
            "# +Z => pref/innate,"
            " -Z => aver/external,\n"
            "#   +X => ideal/enacted"
            " (horizontal).\n"
            "# Arrow length"
            " ~ certainty (sigma_ref/sigma).\n"
            "# Colour ~ mu"
            " (Cool to Warm).\n"
            "# Opacity"
            " ~ engagement (|cos_theta|).\n\n"
        )
        fh.write(
            "# ---- Simulation metadata"
            " (do not edit) ----\n"
            f"GLYPHS_XDMF     ="
            f" r'{xdmf_abs}'\n"
            f"NUM_POLICY_DIMS = {ndp}\n"
            f"NUM_TRAIT_DIMS  = {ndt}\n"
            f"NUM_ZONE_TYPES  = {nzt}\n"
            f"NX = {nx}\n"
            f"NY = {ny}\n\n"
            "# ---- User-adjustable"
            " parameters ----\n"
            "DIM       = 0"
            "  # 0..NUM_POLICY_DIMS-1\n"
            "ZONE_TYPE = 0"
            "  # 0..NUM_ZONE_TYPES-1\n"
            "GLYPH_SCALE = 0.3\n\n"
        )

        # --- Imports and data reader ---
        fh.write(
            "from paraview.simple import *\n\n"
            "reader = Xdmf3ReaderT(\n"
            "    FileName=[GLYPHS_XDMF])\n"
            "reader.UpdatePipeline()\n\n\n"
        )

        # --- Helper: extract one named block
        #   Note: {name} below is a literal
        #   in the generated file (regular
        #   Python string in the generator,
        #   f-string in the generated script).
        fh.write(
            "def extract(name):\n"
            "    \"\"\"Extract one named"
            " block from reader.\"\"\"\n"
            "    eb = ExtractBlock(\n"
            "        Input=reader)\n"
            "    eb.Selectors ="
            " [f'/Root/{name}']\n"
            "    eb.UpdatePipeline()\n"
            "    return eb\n\n\n"
        )

        # --- Helper: make arrow glyph ---
        # {dim} below is a literal brace in
        # the generated file.
        fh.write(
            "def make_glyph(\n"
            "        src, z_sign, dim,\n"
            "        scale=GLYPH_SCALE):\n"
            "    \"\"\"Arrow glyph:"
            " +Z (z_sign=+1),"
            " -Z (-1), +X (0).\"\"\"\n"
            "    g = Glyph(\n"
            "        Input=src,\n"
            "        GlyphType='Arrow')\n"
            "    g.ScaleArray = [\n"
            "        'POINTS',\n"
            "        f'inv_sigma_d{dim}']\n"
            "    g.ScaleFactor = scale\n"
            "    g.GlyphMode = 0"
            "  # All Points\n"
            "    g.GlyphTransform.Rotate"
            " = [\n"
            "        0.0,"
            " -90.0 * z_sign, 0.0]\n"
            "    g.UpdatePipeline()\n"
            "    return g\n\n\n"
        )

        # --- Helper: apply colour + opacity ---
        fh.write(
            "def apply_style("
            "rep, view, dim):\n"
            "    \"\"\"Colour by mu,"
            " opacity by |cos_theta|.\"\"\"\n"
            "    mu_key ="
            " f'mu_d{dim}'\n"
            "    ct_key ="
            " f'cos_theta_d{dim}'\n"
            "    ColorBy(rep,"
            " ('POINTS', mu_key))\n"
            "    lut ="
            " GetColorTransferFunction(\n"
            "        mu_key)\n"
            "    lut.ApplyPreset(\n"
            "        'Cool to Warm', True)\n"
            "    rep.SetScalarBarVisibility(\n"
            "        view, False)\n"
            "    rep.EnableOpacityMapping"
            " = 1\n"
            "    rep.OpacityArrayName ="
            " (\n"
            "        'POINTS', ct_key)\n"
            "    otf ="
            " GetOpacityTransferFunction(\n"
            "        ct_key)\n"
            "    otf.Points = [\n"
            "        0.0, 0.0, 0.5, 0.0,\n"
            "        1.0, 1.0, 0.5, 0.0]\n\n\n"
        )

        # --- Helper: show pref/aver pair ---
        fh.write(
            "def show_pair(\n"
            "        view, pref_src,"
            " aver_src, dim):\n"
            "    \"\"\"Show pref (+Z) and"
            " aver (-Z) glyphs.\"\"\"\n"
            "    for src, zs in (\n"
            "        (pref_src, +1),\n"
            "        (aver_src, -1),\n"
            "    ):\n"
            "        g = make_glyph("
            "src, zs, dim)\n"
            "        rep = Show(g, view)\n"
            "        apply_style("
            "rep, view, dim)\n\n\n"
        )

        # --- Layout: 3 horizontal panes ---
        fh.write(
            "# ---- Layout:"
            " 3 horizontal panes ----\n"
            "layout = CreateLayout(\n"
            "    'STODEM Glyphs')\n"
            "layout.SplitHorizontal(\n"
            "    0, 1.0/3.0)\n"
            "layout.SplitHorizontal(\n"
            "    2, 0.5)\n\n"
            "view_cit = CreateView("
            "'RenderView')\n"
            "view_pol = CreateView("
            "'RenderView')\n"
            "view_gov = CreateView("
            "'RenderView')\n"
            "AssignViewToLayout(\n"
            "    view=view_cit,"
            " layout=layout, hint=1)\n"
            "AssignViewToLayout(\n"
            "    view=view_pol,"
            " layout=layout, hint=3)\n"
            "AssignViewToLayout(\n"
            "    view=view_gov,"
            " layout=layout, hint=4)\n\n"
        )

        # --- Camera: top-down over grid ---
        fh.write(
            "for _v in (\n"
            "        view_cit,"
            " view_pol, view_gov):\n"
            "    _v.CameraPosition = [\n"
            "        NX/2.0, NY/2.0,\n"
            "        max(NX, NY)*2.0]\n"
            "    _v.CameraFocalPoint = [\n"
            "        NX/2.0, NY/2.0, 0.0]\n"
            "    _v.CameraViewUp ="
            " [0.0, 1.0, 0.0]\n"
            "    _v.CameraParallelProjection"
            " = 1\n"
            "    _v.CameraParallelScale"
            " = max(NX, NY) / 2.0\n\n"
        )

        # --- Citizens pane (left) ---
        fh.write(
            "# ---- Citizens pane"
            " (left) ----\n"
            "_cpp ="
            " extract('cit_pol_pref')\n"
            "_cpa ="
            " extract('cit_pol_aver')\n"
            "_cpi ="
            " extract('cit_pol_ideal')\n"
            "show_pair("
            "view_cit, _cpp, _cpa, DIM)\n"
            "_g = make_glyph("
            "_cpi, 0, DIM)\n"
            "_r = Show(_g, view_cit)\n"
            "apply_style("
            "_r, view_cit, DIM)\n"
            "Render(view_cit)\n\n"
        )

        # --- Politicians pane (middle) ---
        # {_zt} below is a literal brace
        # in the generated file.
        fh.write(
            "# ---- Politicians pane"
            " (middle) ----\n"
            "_zt = ZONE_TYPE\n"
            "_pip = extract(\n"
            "    f'pol_inn_pol_pref_zt{_zt}')\n"
            "_pia = extract(\n"
            "    f'pol_inn_pol_aver_zt{_zt}')\n"
            "_pep = extract(\n"
            "    f'pol_ext_pol_pref_zt{_zt}')\n"
            "_pea = extract(\n"
            "    f'pol_ext_pol_aver_zt{_zt}')\n"
            "show_pair("
            "view_pol, _pip, _pia, DIM)\n"
            "show_pair("
            "view_pol, _pep, _pea, DIM)\n"
            "Render(view_pol)\n\n"
        )

        # --- Government pane (right) ---
        fh.write(
            "# ---- Government pane"
            " (right) ----\n"
            "_gov = extract('gov_pol')\n"
            "_g = make_glyph("
            "_gov, 0, DIM)\n"
            "_r = Show(_g, view_gov)\n"
            "apply_style("
            "_r, view_gov, DIM)\n"
            "Render(view_gov)\n\n"
            "RenderAllViews()\n"
        )
