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

