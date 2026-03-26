import numpy as np

from sim_control import SimProperty
from patch import Patch
from zone import Zone
from citizen import Citizen
from politician import Politician
from government import Government
from random_state import rng


class World():

    def __init__(self, settings, sim_control):

        # Initialize instance variables.
        self.x_num_patches = 1
        self.y_num_patches = 1
        self.zone_types = []
        self.zones = []
        self.citizens = []
        self.politicians = []
        self.properties = []
        self.government = []
        self.policy_limits = []
        self.trait_limits = []

        # Get the number of policy and trait dimensions.
        self.num_policy_dims = int(
                settings.infile_dict[1][
                    "world"]["num_policy_dims"])
        self.num_trait_dims = int(
                settings.infile_dict[1][
                    "world"]["num_trait_dims"])

        # Get the patch size. (This is mostly just useless at the moment
        #   because the politicians and citizens will move *on the patch
        #   lattice* itself as opposed to occupying real valued space
        #   "within" a patch.)
        self.patch_size = int(
                settings.infile_dict[1][
                    "world"]["patch_size"])

        # Create zone types. (This is a bit rigid now, but maybe it could
        #   be more flexible in the future. Zones are political zones that a
        #   politician may be restricted to working within. I.e., a
        #   politician for a zone only collects votes from citizens within
        #   their designated zone.) Zones must be contiguous and may or may
        #   not be static.

        # Get the total number of zone types.
        self.num_zone_types = int(
                settings.infile_dict[1][
                    "world"]["num_zone_types"])

        # For each zone type, store its initial properties (x,y dimensions
        #   and whether it is static or not). The crucial thing to understand
        #   is that the size is given in units of "sub_units". For the 0th
        #   zone type, the sub_unit is a patch. For the 1st zone type, the
        #   sub_unit is the 0th zone type. For the 2nd zone type, the
        #   sub_unit is the 1st zone type. Etc. I.e., each zone type is
        #   "built" out of units of the next smaller zone type. All zones of
        #   a given type are identical. The picture you should have is that
        #   the three images below overlay eachother so that a given patch
        #   will belong to one zone of each type. I.e., the patch in the
        #   middle might belong to zone index 24 of zone-type 0, zone index
        #   4 of zone type 1, and zone index 0 of zone type 2. (Indices
        #   start counting at 0.)
        #
        #    Zone-Type 0       Zone-Type 1      Zone-Type 2
        #   |-----------|     |-----------|    |-----------|
        #   |+++++++++++|     |   |   |   |    |           |
        #   |+++++++++++|     |---|---|---|    |           |
        #   |+++++++++++|     |   |   |   |    |           |
        #   |+++++++++++|     |---|---|---|    |           |
        #   |+++++++++++|     |   |   |   |    |           |
        #   |-----------|     |-----------|    |-----------|
        #
        # For example, zone type 2 is 3x3 because it is made of a 3x3 grid
        #   of zone type 1 subunits. (That is not a typo. This was the
        #   easiest way to define it.)
        for zone_type in range(self.num_zone_types):
            self.zone_types.append(settings.infile_dict[1]["world"]
                    [f"zone_type_{zone_type}"])
            self.zone_types[zone_type]["x_sub_units"] = \
                    int(self.zone_types[zone_type]["x_sub_units"])
            self.zone_types[zone_type]["y_sub_units"] = \
                    int(self.zone_types[zone_type]["y_sub_units"])
            if (self.zone_types[zone_type]["static"] == "0"):
                self.zone_types[zone_type]["static"] = False
            else:
                self.zone_types[zone_type]["static"] = True

        # Compute the initial number of patches in each dimension (x,y) for
        #   each zone type. With the understanding of the previous paragraph,
        #   we see that the number of patches for a zone depends on the
        #   number of patches that the "one-lower" zone type has.
        for zone_type in range(self.num_zone_types):
            if (zone_type == 0):
                self.zone_types[zone_type]["x_num_patches"] = \
                        self.zone_types[zone_type]["x_sub_units"]
                self.zone_types[zone_type]["y_num_patches"] = \
                        self.zone_types[zone_type]["y_sub_units"]
            else:
                self.zone_types[zone_type]["x_num_patches"] = \
                        self.zone_types[zone_type - 1]["x_num_patches"] * \
                        self.zone_types[zone_type]["x_sub_units"]
                self.zone_types[zone_type]["y_num_patches"] = \
                        self.zone_types[zone_type - 1]["y_num_patches"] * \
                        self.zone_types[zone_type]["y_sub_units"]

        # Compute the total number of patches in each dimension (x,y) for
        #   the World.
        for zone_type in range(self.num_zone_types):
            self.x_num_patches *= self.zone_types[zone_type]["x_sub_units"]
            self.y_num_patches *= self.zone_types[zone_type]["y_sub_units"]
        print (f"There are {self.x_num_patches} in the x-direction.")
        print (f"There are {self.y_num_patches} in the y-direction.")

        # Create patches that fill the world and zones. Each patch needs to
        #   know its "x,y" location to determine what zones it is a part of.
        #   A key side-effect of this step is that each patch will know the
        #   index number of the zones that it is a part of. This is a bit
        #   odd because we have not yet actually created the zones. (That is
        #   done below. We only specified zone types so far.) However,
        #   making the patches aware of their zone index now is very useful
        #   for making the zones later.
        self.patches = [[Patch(settings, i, j, self.x_num_patches,
                self.y_num_patches, self.num_zone_types, self.zone_types)
                for i in range(self.x_num_patches)]
                for j in range(self.y_num_patches)]

        # Create the zones themselves. (Note above that we just created the
        #   *zone types*, not the zones themselves.)

        # First, make an empty list of zones for each zone type. The zones
        #   of each zone type will be stored as a list of zones of that
        #   type. self.zones is an empty list. After this loop, the
        #   list will contain a set of empty lists. Before: []. After:
        #   [[], [], []] with one empty list for each zone type.
        for zone_type in range(self.num_zone_types):
            self.zones.append([])

        # Visit every patch and whenever we find one with a new zone index,
        #   then create an official new zone. If we find a patch that is not
        #   a new zone index then we just add this patch to the list of
        #   patches that belong to this zone.

        # Initialize the current zone index for each zone type.
        curr_zone_index = [-1] * self.num_zone_types

        # Tripley nested loop over the x,y number of patches and number of
        #   types.
        for i in range(len(self.patches)):
            for j in range(len(self.patches[i])):
                for zone_type in range(self.num_zone_types):

                    # Check if we have come across a new zone index for
                    #   this zone type. Note, we are only looking for zone
                    #   indices that are *greater* than any one found before.
                    #   As we traverse the x,y patch grid, we will encounter
                    #   the same zones multiple times so we don't want to
                    #   think of them as being new. (Consider a 5x5 grid of
                    #   zones that are each made of a 5x5 grid of patches.
                    #   There are 25 zones in total. If we are going
                    #   row-by-row through the patches, then we will
                    #   encounter zones 0-4 on the bottom row of patches and
                    #   then 0-4 again in the second from the bottom row,
                    #   etc. Only when we get to patch row #5 (the sixth
                    #   patch row) will we encounter zones 5-9 as we
                    #   traverse over the patches.
                    if (self.patches[i][j].zone_index[zone_type] >
                            curr_zone_index[zone_type]):

                        # If so, update the current zone index.
                        curr_zone_index[zone_type] = \
                                self.patches[i][j].zone_index[zone_type]

                        # Then, create a new zone and add it to the world
                        #   list of zones of this type. Pass curr_zone_index so
                        #   the Zone object knows its own integer index within
                        #   self.zones[zone_type]; this is needed later when
                        #   citizens compare a politician's zone against the
                        #   integer zone indices stored in patch.zone_index.
                        self.zones[zone_type].append(Zone(settings, zone_type,
                                curr_zone_index[zone_type],
                                self.patches[i][j]))
                    else:
                        # Add this patch to the patch list of the current
                        #   zone. It is a convoluted expression. Basically,
                        #   for the current zone type, get the current i,j
                        #   patch and use the zone index of the current zone
                        #   type from that patch to add the current patch to
                        #   that zone. (May need to read that a couple of
                        #   times...)
                        self.zones[zone_type][
                                self.patches[i][j].zone_index[
                                        zone_type]].add_patch(
                                        self.patches[i][j])

        # Define the global property types of the world.
        self.properties.append(SimProperty("CitizenGeoData", "WellBeing",
                "Scalar",
                rng.uniform(size=(self.x_num_patches,
                self.y_num_patches))))
        #self.properties.append(SimProperty("CitizenData", "PolicyPref",
        #        "Scalar", rng.uniform(size=(sim_control.data_resolution,
        #        len(self.citizens)))))


    def repopulate_politicians(self, settings):

        # Make things easy to start.
        # - Every politician who was elected and governed, will run in the
        #   next cycle.
        # - Every politician who lost will be replaced with a new politician.
        # - All politicians have zero votes.
        for politician in self.politicians:
            if (politician.elected == False):
                politician.reset_to_input(settings)
            else:
                politician.elected = False
            politician.reset_votes()


    # Add citizens and politicians into the world.
    def populate(self, settings):

        # Initialize the list of all politicians.
        self.politicians = []

        # Sprout politicians within their geographic boundary. We do this
        #   by visiting every single zone (of every zone type) and creating
        #   a politician in a random patch of the necessary type in that
        #   zone.
        for zone_type in range(self.num_zone_types):
            for zone in self.zones[zone_type]:
                for politician in range(zone.num_politicians):

                    # Select a random patch in the current zone.
                    random_patch = zone.random_patch()

                    # Create a politician and append it to the list of all
                    #   politicians. Note, presently, that the patch is not
                    #   "made aware" of the fact that there is a politician
                    #   associated with it.
                    temp_politician = Politician(settings, zone_type, zone,
                            random_patch)
                    self.politicians.append(temp_politician)

                    # However, we do want the zone to know which politicians
                    #   are competing.
                    zone.add_politician(temp_politician)


        # Create a government for the world.
        self.government = Government(settings)


        # Initialize the list of all citizens. This list will hold the
        #   actual citizen objects.
        self.citizens = []

        # Add citizens in groups to each patch by way of their index number
        #   within the global list. (I.e., each patch will know which
        #   citizens are on that patch.) Only then do we actually create the
        #   citizens and append them to the World list. We do this order so
        #   that when the citizens are created they will know which patch
        #   they are on.

        # Initialize the citizen index counter.
        start_index = 0

        # Get each list (i, recalling that self.patches is a list of
        #   lists) and then visit each patch in the list (i).
        for i in self.patches:  # i is a list of patches.
            for patch in i:  # patch is a patch in the list i.

                # Ask this patch to store the indices of the citizens that
                #   are about to be made.
                (patch.sprout_citizens(start_index, patch.num_citizens))
                start_index += patch.num_citizens

                # Add those citizens to the global (world) list.
                for citizen in range(patch.num_citizens):
                    self.citizens.append(Citizen(settings, patch, self.zones))


    def compute_patch_well_being(self):
        """Aggregate citizen well-being onto the
        2-D patch grid for spatial visualization.

        Each citizen has a scalar well_being value
        computed from the overlap between their
        ideal policy positions and the government's
        enacted policy (see citizen.py
        build_response_to_well_being()). This
        method averages the well-being of all
        citizens on each patch to produce a 2-D
        heatmap that can be visualized in Paraview
        via the HDF5/XDMF output pipeline.

        Patches with no citizens receive a value
        of 0 (the count_grid guard prevents
        division by zero by setting empty-patch
        counts to 1 before dividing).

        The result is written directly into
        self.properties[0].data, which is the
        SimProperty that the HDF5 writer reads
        each step.
        """
        well_being_grid = np.zeros(
            (self.x_num_patches,
             self.y_num_patches))
        count_grid = np.zeros(
            (self.x_num_patches,
             self.y_num_patches))

        for citizen in self.citizens:
            x = citizen.current_patch.x_location
            y = citizen.current_patch.y_location
            well_being_grid[x, y] += (
                citizen.well_being)
            count_grid[x, y] += 1

        # Guard against division by zero for
        #   empty patches.
        count_grid[count_grid == 0] = 1
        self.properties[0].data = (
            well_being_grid / count_grid)


    #def dump_state(self, settings, time_step):
    #
    #    # Record each hdf5 dataset for the current time step.
    #    settings.wellbeing_did.append(
    #            settings.wellbeing_gid.create_dataset(f"{time_step}",
    #            (self.x_num_patches, self.y_num_patches), data=))
    #    settings.policy_alignment_did.append(
    #            settings.policy_alignment_gid.create_dataset(
    #            f"{time_step}",(self.x_num_patches, self.y_num_patches),
    #            data=))
    #    settings.politicians_did.append(
    #            settings.politicians_gid.create_dataset(f"{time_step}",
    #            (self.x_num_patches, self.y_num_patches), data=))
