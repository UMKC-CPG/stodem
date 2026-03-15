import numpy as np

from sim_control import SimProperty
from patch import Patch
from zone import Zone
from citizen import Citizen
from politician import Politician
from government import Government
from random_state import rng


class World():

    # Declare class variables.
    num_policy_dims = 0
    num_trait_dims = 0
    x_num_patches = 1
    y_num_patches = 1
    patch_size = 0
    patches = 0
    zone_types = []
    zones = []
    citizens = []
    politicians = []
    properties = []
    government = []
    policy_limits = []
    trait_limits = []


    def __init__(self, settings, sim_control):

        # Get the number of policy and trait dimensions.
        World.num_policy_dims = int(
                settings.infile_dict[1]["world"]["num_policy_dims"])
        World.num_trait_dims = int(
                settings.infile_dict[1]["world"]["num_trait_dims"])

        # Get the patch size. (This is mostly just useless at the moment
        #   because the politicians and citizens will move *on the patch
        #   lattice* itself as opposed to occupying real valued space
        #   "within" a patch.)
        World.patch_size = int(settings.infile_dict[1]["world"]["patch_size"])

        # Create zone types. (This is a bit rigid now, but maybe it could
        #   be more flexible in the future. Zones are political zones that a
        #   politician may be restricted to working within. I.e., a
        #   politician for a zone only collects votes from citizens within
        #   their designated zone.) Zones must be contiguous and may or may
        #   not be static.

        # Get the total number of zone types.
        World.num_zone_types = int(
                settings.infile_dict[1]["world"]["num_zone_types"])

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
        for zone_type in range(World.num_zone_types):
            World.zone_types.append(settings.infile_dict[1]["world"]
                    [f"zone_type_{zone_type}"])
            World.zone_types[zone_type]["x_sub_units"] = \
                    int(World.zone_types[zone_type]["x_sub_units"])
            World.zone_types[zone_type]["y_sub_units"] = \
                    int(World.zone_types[zone_type]["y_sub_units"])
            if (World.zone_types[zone_type]["static"] == "0"):
                World.zone_types[zone_type]["static"] = False
            else:
                World.zone_types[zone_type]["static"] = True

        # Compute the initial number of patches in each dimension (x,y) for
        #   each zone type. With the understanding of the previous paragraph,
        #   we see that the number of patches for a zone depends on the
        #   number of patches that the "one-lower" zone type has.
        for zone_type in range(World.num_zone_types):
            if (zone_type == 0):
                World.zone_types[zone_type]["x_num_patches"] = \
                        World.zone_types[zone_type]["x_sub_units"]
                World.zone_types[zone_type]["y_num_patches"] = \
                        World.zone_types[zone_type]["y_sub_units"]
            else:
                World.zone_types[zone_type]["x_num_patches"] = \
                        World.zone_types[zone_type - 1]["x_num_patches"] * \
                        World.zone_types[zone_type]["x_sub_units"]
                World.zone_types[zone_type]["y_num_patches"] = \
                        World.zone_types[zone_type - 1]["y_num_patches"] * \
                        World.zone_types[zone_type]["y_sub_units"]

        # Compute the total number of patches in each dimension (x,y) for
        #   the World.
        for zone_type in range(World.num_zone_types):
            World.x_num_patches *= World.zone_types[zone_type]["x_sub_units"]
            World.y_num_patches *= World.zone_types[zone_type]["y_sub_units"]
        print (f"There are {World.x_num_patches} in the x-direction.")
        print (f"There are {World.y_num_patches} in the y-direction.")

        # Create patches that fill the world and zones. Each patch needs to
        #   know its "x,y" location to determine what zones it is a part of.
        #   A key side-effect of this step is that each patch will know the
        #   index number of the zones that it is a part of. This is a bit
        #   odd because we have not yet actually created the zones. (That is
        #   done below. We only specified zone types so far.) However,
        #   making the patches aware of their zone index now is very useful
        #   for making the zones later.
        World.patches = [[Patch(settings, i, j, World.x_num_patches,
                World.y_num_patches, World.num_zone_types, World.zone_types)
                for i in range(World.x_num_patches)]
                for j in range(World.y_num_patches)]

        # Create the zones themselves. (Note above that we just created the
        #   *zone types*, not the zones themselves.)

        # First, make an empty list of zones for each zone type. The zones
        #   of each zone type will be stored as a list of zones of that
        #   type. The World.zones is an empty list. After this loop, the
        #   list will contain a set of empty lists. Before: []. After:
        #   [[], [], []] with one empty list for each zone type.
        for zone_type in range(World.num_zone_types):
            World.zones.append([])

        # Visit every patch and whenever we find one with a new zone index,
        #   then create an official new zone. If we find a patch that is not
        #   a new zone index then we just add this patch to the list of
        #   patches that belong to this zone.

        # Initialize the current zone index for each zone type.
        curr_zone_index = [-1] * World.num_zone_types

        # Tripley nested loop over the x,y number of patches and number of
        #   types.
        for i in range(len(World.patches)):
            for j in range(len(World.patches[i])):
                for zone_type in range(World.num_zone_types):

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
                    if (World.patches[i][j].zone_index[zone_type] >
                            curr_zone_index[zone_type]):

                        # If so, update the current zone index.
                        curr_zone_index[zone_type] = \
                                World.patches[i][j].zone_index[zone_type]

                        # Then, create a new zone and add it to the world
                        #   list of zones of this type. Pass curr_zone_index so
                        #   the Zone object knows its own integer index within
                        #   World.zones[zone_type]; this is needed later when
                        #   citizens compare a politician's zone against the
                        #   integer zone indices stored in patch.zone_index.
                        World.zones[zone_type].append(Zone(settings, zone_type,
                                curr_zone_index[zone_type],
                                World.patches[i][j]))
                    else:
                        # Add this patch to the patch list of the current
                        #   zone. It is a convoluted expression. Basically,
                        #   for the current zone type, get the current i,j
                        #   patch and use the zone index of the current zone
                        #   type from that patch to add the current patch to
                        #   that zone. (May need to read that a couple of
                        #   times...)
                        World.zones[zone_type][
                                World.patches[i][j].zone_index[
                                        zone_type]].add_patch(
                                        World.patches[i][j])

        # Define the global property types of the world.
        self.properties.append(SimProperty("CitizenGeoData", "WellBeing",
                "Scalar",
                rng.uniform(size=(World.x_num_patches,
                World.y_num_patches))))
        #self.properties.append(SimProperty("CitizenData", "PolicyPref",
        #        "Scalar", rng.uniform(size=(sim_control.data_resolution,
        #        len(World.citizens)))))


    def repopulate_politicians(self, settings):

        # Make things easy to start.
        # - Every politician who was elected and governed, will run in the
        #   next cycle.
        # - Every politician who lost will be replaced with a new politician.
        # - All politicians have zero votes.
        for politician in World.politicians:
            if (politician.elected == False):
                politician.reset_to_input(settings)
            else:
                politician.elected = False
            politician.reset_votes()


    # Add citizens and politicians into the world.
    def populate(self, settings):

        # Initialize the list of all politicians.
        World.politicians = []

        # Sprout politicians within their geographic boundary. We do this
        #   by visiting every single zone (of every zone type) and creating
        #   a politician in a random patch of the necessary type in that
        #   zone.
        for zone_type in range(World.num_zone_types):
            for zone in World.zones[zone_type]:
                for politician in range(zone.num_politicians):

                    # Select a random patch in the current zone.
                    random_patch = zone.random_patch()

                    # Create a politician and append it to the list of all
                    #   politicians. Note, presently, that the patch is not
                    #   "made aware" of the fact that there is a politician
                    #   associated with it.
                    temp_politician = Politician(settings, zone_type, zone,
                            random_patch)
                    World.politicians.append(temp_politician)

                    # However, we do want the zone to know which politicians
                    #   are competing.
                    zone.add_politician(temp_politician)


        # Create a government for the world.
        World.government = Government(settings)


        # Initialize the list of all citizens. This list will hold the
        #   actual citizen objects.
        World.citizens = []

        # Add citizens in groups to each patch by way of their index number
        #   within the global list. (I.e., each patch will know which
        #   citizens are on that patch.) Only then do we actually create the
        #   citizens and append them to the World list. We do this order so
        #   that when the citizens are created they will know which patch
        #   they are on.

        # Initialize the citizen index counter.
        start_index = 0

        # Get each list (i, recalling that World.patches is a list of
        #   lists) and then visit each patch in the list (i).
        for i in World.patches:  # i is a list of patches.
            for patch in i:  # patch is a patch in the list i.

                # Ask this patch to store the indices of the citizens that
                #   are about to be made.
                (patch.sprout_citizens(start_index, patch.num_citizens))
                start_index += patch.num_citizens

                # Add those citizens to the global (world) list.
                for citizen in range(patch.num_citizens):
                    World.citizens.append(Citizen(settings, patch, World.zones))


    def compute_patch_well_being(self):
        """Aggregate citizen well-being to the patch grid for visualization."""
        well_being_grid = np.zeros((World.x_num_patches, World.y_num_patches))
        count_grid = np.zeros((World.x_num_patches, World.y_num_patches))

        for citizen in self.citizens:
            x = citizen.current_patch.x_location
            y = citizen.current_patch.y_location
            well_being_grid[x, y] += citizen.well_being
            count_grid[x, y] += 1

        count_grid[count_grid == 0] = 1
        self.properties[0].data = well_being_grid / count_grid


    #def dump_state(self, settings, time_step):
    #
    #    # Record each hdf5 dataset for the current time step.
    #    settings.wellbeing_did.append(
    #            settings.wellbeing_gid.create_dataset(f"{time_step}",
    #            (World.x_num_patches, World.y_num_patches), data=))
    #    settings.policy_alignment_did.append(
    #            settings.policy_alignment_gid.create_dataset(
    #            f"{time_step}",(World.x_num_patches, World.y_num_patches),
    #            data=))
    #    settings.politicians_did.append(
    #            settings.politicians_gid.create_dataset(f"{time_step}",
    #            (World.x_num_patches, World.y_num_patches), data=))
