import h5py as h5
from lxml import etree


class Hdf5():

    # Create a class variable for the HDF5 file handle.
    h_fid = 0

    # Create class variable dictionaries that will hold the group and
    #   dataset ids.
    group_gid = {}
    dataset_did = {}

    def __init__(self, settings, world):
        # Create the HDF5 file.
        Hdf5.h_fid = h5.File(f"{settings.outfile}.hdf5", "x")

        # Create a group for each property.
        for p in world.properties:
            Hdf5.group_gid[p.group] = Hdf5.h_fid.create_group(f"{p.group}")

    def add_dataset(self, p, i):
        # Create the HDF5 dataset from the given property.
        Hdf5.dataset_did[f"{p.name}{i}"] = \
                Hdf5.group_gid[p.group].create_dataset(
                f"{p.name}{i}", compression="gzip", data=p.data)


class Xdmf():

    # Create a class variable file handle.
    x = 0

    def __init__(self, settings):
        # Create the actual file handle and write the xml header lines.
        Xdmf.x = open(f"{settings.outfile}.xdmf", "w")
        Xdmf.x.write('<?xml version="1.0" encoding="utf-8"?>\n')
        Xdmf.x.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')


    def print_xdmf_xml(self, settings, sim_control, world):

        # Create local variable lists for the XDMF file.
        timestep_grid = []
        timestep = []
        topology = []
        geometry = []
        geometry_origin = []
        geometry_dxdy = []
        attribute = []
        data_item = []

        # Build the XDMF XML file.

        # The root will be the xdmf tag. We will need to add some data before
        #   the root into the XML file. Specifically a line:
        #   '<?xml version="1.0" encoding="utf-8"?>' and a line:
        #   '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>"'. Note also that "xdmf"
        #   is lower case while then class name is "Xdmf". Finally, note that
        #   the xmlnsxi attribute should really be xmlns:xi, but python does
        #   not like colons in variable names and when I tried to add that
        #   key-value pair to the attribute dictionary after the fact it
        #   wouldn't accept it either. (Python called it a bad key I think.)
        #   So, the resolution is to use the xmlnsxi attribute and then when
        #   the xml file is actually printed I do a string substitution on
        #   xmlnsxi to make it xmlns:xi. (Kludgy.)
        root = etree.Element("root")
        xdmf = etree.SubElement(root, "Xdmf", Version="3.0",
                xmlnsxi="[http://www.w3.org/2001/XInclude]")

        # Make the domain. Only one domain is needed for the simulation results.
        domain = etree.SubElement(xdmf, "Domain")

        # Create a "grid" that is a collection of grids, one for each time step.
        time_collection_grid = etree.SubElement(domain, "Grid",
                Name="TimeSteps", GridType="Collection",
                CollectionType="Temporal")

        # Start the loop for adding time step grids.
        for i in range(sim_control.total_num_steps):

            # Add the time step grid.
            timestep_grid.append(etree.SubElement(time_collection_grid, "Grid",
                    Name=f"Step{i}", GridType="Uniform"))

            # Make the time step value an integer. (Arbitrary time duration.)
            timestep.append(etree.SubElement(timestep_grid[i], "Time",
                    Value=f"{i}.0"))

            # Define the topology for this grid. (Ideally, we would define a
            #   single topology just below the "Domain" and then reference it
            #   here. But for some reason it didn't work and so we will just
            #   repeat the topology here for every time step grid even though
            #   it "wastes" space.)
            topology.append(etree.SubElement(timestep_grid[i], "Topology",
                    Name="Topo", TopologyType="2DCoRectMesh",
                    Dimensions=f"{world.x_num_patches} {world.y_num_patches}"))

            # Same for the geometry as for the topology.
            geometry.append(etree.SubElement(timestep_grid[i], "Geometry",
                    Name="Geom", GeometryType="ORIGIN_DXDY"))

            # The geometry needs two data elements, the origin and the
            #   spatial deltas.
            geometry_origin.append(etree.SubElement(geometry[i], "DataItem",
                    NumberType="Float", Dimensions="2", Format="XML"))
            geometry_origin[i].text="0.0 0.0"
            geometry_dxdy.append(etree.SubElement(geometry[i], "DataItem",
                    NumberType="Float", Dimensions="2", Format="XML"))
            geometry_dxdy[i].text="1.0 1.0"

            # XDMF attributes frame the data items that point to the actual
            #   HDF5 data. Each attribute / data set in XDMF corresponds to
            #   one monitored property in our simulation. Thus, because there
            #   will be more than one property and because they are computed
            #   in a time series we need a list of lists. (One property list
            #   for each member of the timestep list.)
            # Here, we add an empty attribute list (and its partner DataItem
            #   list) for this time step into the list of timestep lists.
            attribute.append([])
            data_item.append([])

            # Now, we add each attribute that will live on the grid for this
            #   timestep along with its data item that points to the actual
            #   data in the HDF5 file.
            for p in world.properties:
                attribute[i].append(etree.SubElement(timestep_grid[i],
                        "Attribute", Name=f"{p.name}", Center="Node",
                        AttributeType=f"{p.datatype}"))
                data_item[i].append(etree.SubElement(attribute[i][-1],
                        "DataItem", NumberType="Float",
                        Dimensions=f"{world.x_num_patches} "
                        f"{world.y_num_patches}", Format="HDF"))
                data_item[i][-1].text = (f"{settings.outfile}.hdf5:/"
                        f"{p.group}/{p.name}{i}")

        # Pretty print the xdmf file to the output file.
        temp_xml = etree.tostring(xdmf, pretty_print=True,
                encoding="utf-8").decode()
        Xdmf.x.write(temp_xml.replace("xmlnsxi", "xmlns:xi", 1))
        Xdmf.x.close()
