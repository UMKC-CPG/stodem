#!/usr/bin/env python3

import argparse as ap
import os
import sys
from datetime import datetime

from lxml import etree
from dataclasses import dataclass
import numpy as np
import h5py as h5
import random

# Discussion:

# All aspects of the program are open to modification, revision, elimination, etc.

# This program is a multi agent based simulation. The agents include politicians and
#   citizens. The agents populate a two-dimensional world that is divided into a series
#   of nested user-defined zones. For example, the top-level zone can be considered as
#   a country and within that top level zone is a collection of state zones. Each
#   state zone may be composed of a set of district zones. Etc. The world is tiled with
#   a set of patches such that (ideally) the smallest zone consists of at least a few
#   dozen patches.

# The zones may be classified as fixed or flexible. If a zone is fixed, then its boundaries
#   will remain constant throughout the simulation. If a zone is flexible, then its
#   boundaries may change during the simulation with the constraints that a zone is not
#   allowed to overlap with another zone of the same level and it is not allowed to cross
#   a border of a higher level zone. (I.e., a zone cannot be spread across two higher level
#   zones at the same time.)

# Each zone is home to some (non-zero) positive number of politicians. The number of
#   politicians in a zone can change throughout the course of the simulation.

# Every patch will be a member of all the zones that contain it from the lowest level to
#   the highest. Some variable number of citizens (possibly zero) inhabit each patch.
#   Some variable (possibly zero) number of politicians will inhabit each patch. Most
#   patches will have zero politicians and at least a modest number of citizens in them.

# The simulation runs in an iterative sequence of phases:
#   Campaign -> Vote -> Govern -> Primary Campaign -> Primary Vote -> Campaign -> Etc.
#   Each phase will consist of a finite number of time steps. Depending on the phase,
#   different kinds of activities will happen during each time step.

# Each citizen exists in a state that is defined by the relationship between its internal
#   parameters and the external environment. The internal parameters are expressed in a
#   multi-dimensional complex valued space where each dimension represents either an
#   abstract policy or an emotional attitude. The policy dimensions are unbounded
#   (-infinity, +infinity) with relative positions representing relative degrees of policy
#   opinion. Absolute values on the number line carry no intrinsic meaning so concepts like
#   "extreme" and "centrist" are all relative. That being said, initial values for the
#   Gaussians will be distribute with 0 as the mean value or the point about which the
#   distribution is centered. (This is just for convenience.) The policies themselves are
#   also completely abstract and have no specific meaning.

# For each policy dimension, each citizen maintains two Gaussians: a stated policy position
#   and an ideal policy position. The stated policy policy position is the one that the
#   citizen consciously holds and uses when comparing their policy positions with those
#   of the relevant politicians in preparation for a vote. Similarly, the stated policy
#   positions are used when comparing the citizen's policy positions with those of the
#   government during the governing phase to compute an opinion about the citizen's
#   satisfaction with the performance of the government. Further, the stated policy positions
#   are used when citizens interact with other citizens. On the other hand, the ideal policy
#   position is compared to the government policy positions during the governing phase
#   to compute the citizen's actual well-being. The measure of well-being is used modulate
#   the political temperature of the citizen. Within this dynamic, a variety of phenomena
#   may occur. (See below.)

# In addition to the policy dimensions, each citizen maintains additional internal
#   parameters associated with emotional (personality) attitudes. These operate like the
#   policy positions. The emotional dimensions are unbounded and the positions represent
#   a relative degree of affinity for a certain personality attribute. Unlike the policy
#   positions, citizens only have one position for each dimension. The emotional
#   attitudes only interact with the personalities of politicians and other citizens.
#   The government has no emotional position.

# The emotional and policy positions are not single values but rather are each expressed as
#   unit-area Gaussian functions with one parameter that defines the standard deviation
#   (related to FWHM) and another that defines the position. Further, the Gaussian maintains
#   an orientation understood as a rotation about the real axis into the imaginary axis.
#   Only the projection of the Gaussian onto the real axis plays a role in determining the
#   interaction of the citizen's position with other Gaussians (politician, government, or
#   other citizens). Using this approach, a citizen may have a purely real and positive,
#   purely real and negative, purely imaginary (positive or negative will not matter), or
#   other orientation in between the aforementioned with some fractional real and imaginary
#   projections. When a citizen's position has the same real orientation as that of some
#   other Gaussian (politician, citizen, government), we understand that to represent
#   attraction. When the position has an oppositely directed real orientation compared to
#   some other Gaussian, we understand that to represent repulsion. When a citizen's position
#   has an imaginary orientation, we understand to the signify indifference (neither
#   attraction nor repulsion) to another Gaussian. Recall, only the real projection of a
#   citizen's position (Gaussian) will interact with the real projection of the other
#   Gaussian. Note, presently, positive or negative imaginary values have the same
#   interpretation of indifference.

# The notation for Gaussian functions is as follows:
#   Pcs;n = citizen stated policy Gaussian for dimension n.
#   Pci;n = citizen ideal policy Gaussian for dimension n.
#   Ppa;n = politician apparent policy Gaussian for dimension n.
#   Pg;n  = government enacted policy Gaussian for dimension m.
#   Ec;m = citizen emotional Gaussian for dimension m.
#   Ep;m = politician emotional Gaussian for dimension m.

# The functional form of our complex Gaussian is:
#   g(x;sigma,mu,theta) = 1/(sigma * sqrt(2 pi)) * exp(-(x-mu)^2 / (2 sigma^2)) * exp(i theta)
# where:
#   x = the independent variable (arbitray position on the real number line).
#   sigma = the standard deviation (sigma^2 = the variance).
#   mu = the point on the real number line of maximum amplitude.
#   theta = the orientation of the Gaussian about the real number line into the imaginary axis.
# and other convenient variables are:
#   FWHM = 2 sqrt(2 ln(2)) * sigma.
#   alpha = 1/(2 sigma^2)
#   zeta = alpha_1 + alpha_2 for two Gaussians
#   xi = 1/(2 zeta)
#   d = mu_1 - mu_2 for two Gaussians

# The key relationship between any two Gaussians G1, G2 is their overlap integral:
#   I(G1,G2) = Integral(Re(G1)*Re(G2) dx; -infinity..+infinity)
#   I(G1,G2) = (pi/zeta)^1.5 * exp(-xi * d^2) * cos(theta_1) * cos(theta_2)
#   The numerical value of this integral is between -1 and +1. The maximum value occurs when
#   both Gaussians have exactly equal parameters. The minimum value occurs when both
#   Gaussians have exactly equal parameters except for that theta_1 = 0 or pi/2 and
#   theta_2 = pi/2 or 0 respectively.

# Interactions and their effects are computed as follows:

# All relevant integral pairs are computed:
#   I(Pcs,Ppa), (Pcs,Pg), (Ppa,Pg), I(Pci,Pg), I(Ppa,Pg), I(Ec,Ep)

# The I(Ec,Ep) term defined a viscous frictional force between

# Citizen stated policy position <-> politician apparent policy position.
#  (1) The direct force, I(Pcs,Ppa), is computed for the two Gaussians.
#  (2) A viscous frictional force is determined

# Gaussians move according to F = ma where m is the area projected onto the real axis.

#A rational citizen would thus have
#   alignment between their stated policy positions and their ideal policy positions.
#   However, 


# Create a global random number generator.
rng = np.random.default_rng(seed=8675309)


# Define the main class that holds script data structures and settings.
class ScriptSettings():
    """The instance variables of this object are the user settings that
       control the program. The variable values are pulled from a list
       that is created within a resource control file and that are then
       reconciled with command line parameters."""


    def __init__(self):
        """Define default values for the graph parameters by pulling them
        from the resource control file in the default location:
        $STODEM_RC/stodemrc.py or from the current working directory if a local
        copy of stodemrc.py is present."""

        # Read default variables from the resource control file.
        sys.path.insert(1, os.getenv('STODEM_RC'))
        from stodemrc import parameters_and_defaults
        default_rc = parameters_and_defaults()

        # Assign values to the settings from the rc defaults file.
        self.assign_rc_defaults(default_rc)

        # Parse the command line.
        args = self.parse_command_line()

        # Reconcile the command line arguments with the rc file.
        self.reconcile(args)

        # At this point, the command line parameters are set and accepted.
        #   When this initialization subroutine returns the script will
        #   start running. So, we use this as a good spot to record the
        #   command line parameters that were used.
        self.recordCLP()


    def assign_rc_defaults(self, default_rc):

        # Default filename variables.
        self.infile = default_rc["infile"]
        self.outfile = default_rc["outfile"]


    def parse_command_line(self):
    
        # Create the parser tool.
        prog_name = "stodem"

        description_text = """
Version 0.1
The purpose of this program is to simulate the effect of stochastic voting
on the ability of a democracy to navigate a high-dimensional policy space
and find and exploit the global minimum. The global minimum is the point
with the strongest alignment between the internalized policy preferences of
a population (and its politicians) and the actual (unknown) policies that
lead to positive outcomes for the population.
"""

        epilog_text = """
Please contact Paul Rulis (rulisp@umkc.edu) regarding questions.
Defaults are given in $STODEM_RC/stodemrc.py.
"""

        parser = ap.ArgumentParser(prog = prog_name,
                formatter_class=ap.RawDescriptionHelpFormatter,
                description = description_text, epilog = epilog_text)
    
        # Add arguments to the parser.
        self.add_parser_arguments(parser)

        # Parse the arguments and return the results.
        return parser.parse_args()


    def add_parser_arguments(self, parser):
    
        # Define the input file.
        parser.add_argument('-i', '--infile', dest='infile', type=ascii,
                            default=self.infile, help='Input file name. ' +
                            f'Default: {self.infile}')
    
        # Define the output file prefix.
        parser.add_argument('-o', '--outfile', dest='outfile', type=ascii,
                            default=self.outfile, help='Output file name ' +
                            f'prefix for hdf5 and xdmf. Default: {self.outfile}')
    

    def reconcile(self, args):
        self.infile = args.infile.strip("'")
        self.outfile = args.outfile.strip("'")


    def recordCLP(self):
        with open("command", "a") as cmd:
            now = datetime.now()
            formatted_dt = now.strftime("%b. %d, %Y: %H:%M:%S")
            cmd.write(f"Date: {formatted_dt}\n")
            cmd.write(f"Cmnd:")
            for argument in sys.argv:
                cmd.write(f" {argument}")
            cmd.write("\n\n")


    def read_input_file(self):
        
        tree = etree.parse(self.infile)
        root = tree.getroot()

        def recursive_dict(element):
            return (element.tag, dict(map(recursive_dict, element)) or element.text)

        self.infile_dict = recursive_dict(root)


class SimControl():

    # Declare and initialize the class variables.
    curr_step = 0  # The current overall timestep number.
    num_cycles = 0  # The number of campaign cycles.
    num_campaign_steps = 0  # Number of time steps in a campaign.
    num_govern_steps = 0  # Number of time steps to govern.
    num_primary_campaign_steps = 0  # Number of time steps in a primary campaign.
    total_num_steps = 0  # Total number of simulation steps.


    def __init__(self, settings):
        # Extract simulation control parameters from the xml input file.
        SimControl.num_cycles = int(settings.infile_dict[1]["sim_control"]["num_cycles"])
        SimControl.num_campaign_steps = \
                int(settings.infile_dict[1]["sim_control"]["num_campaign_steps"])
        SimControl.num_govern_steps = \
                int(settings.infile_dict[1]["sim_control"]["num_govern_steps"])
        SimControl.num_primary_campaign_steps = \
                int(settings.infile_dict[1]["sim_control"]["num_primary_campaign_steps"])

        # Compute the total number of simulation steps.
        SimControl.total_num_steps = (SimControl.num_campaign_steps + \
                SimControl.num_govern_steps + SimControl.num_primary_campaign_steps) * \
                SimControl.num_cycles


@dataclass
class SimProperty():
    group : str
    name : str
    datatype : str
    data : np.ndarray


class World():

    # Declare class variables.
    num_policy_dims = 0
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


    def __init__(self, settings):

        # Get the number of policy dimensions.
        World.num_policy_dims = int(settings.infile_dict[1]["world"]["num_policy_dims"])

        # Get the patch size. (This is mostly just useless at the moment because the politicians
        #   and citizens will move *on the patch lattice* itself as opposed to occupying real
        #   valued space "within" a patch.)
        World.patch_size = int(settings.infile_dict[1]["world"]["patch_size"])

        # Create zone types. (This is a bit rigid now, but maybe it could be more flexible in
        #   the future. Zones are political zones that a politician may be restricted to working
        #   within. I.e., a politician for a zone only collects votes from citizens within their
        #   designated zone.) Zones must be contiguous and may or may not be static.

        # Get the total number of zone types.
        World.num_zone_types = int(settings.infile_dict[1]["world"]["num_zone_types"])

        # For each zone type, store its initial properties (x,y dimensions and whether it is
        #   static or not). The crucial thing to understand is that the size is given in units
        #   of "sub_units". For the 0th zone type, the sub_unit is a patch. For the 1st zone
        #   type, the sub_unit is the 0th zone type. For the 2nd zone type, the sub_unit is
        #   the 1st zone type. Etc. I.e., each zone type is "built" out of units of the next
        #   smaller zone type. All zones of a given type are identical. The picture you
        #   should have is that the three images below overlay eachother so that a given patch
        #   will belong to one zone of each type. I.e., the patch in the middle might belong
        #   to zone index 24 of zone-type 0, zone index 4 of zone type 1, and zone index 0 of
        #   zone type 2. (Indices start counting at 0.)
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
        # For example, zone type 2 is 3x3 because it is made of a 3x3 grid of zone type 1
        #   subunits.
        for zone_type in range(World.num_zone_types):
            World.zone_types.append(settings.infile_dict[1]["world"][f"zone_type_{zone_type}"])
            World.zone_types[zone_type]["x_sub_units"] = \
                    int(World.zone_types[zone_type]["x_sub_units"])
            World.zone_types[zone_type]["y_sub_units"] = \
                    int(World.zone_types[zone_type]["y_sub_units"])
            if (World.zone_types[zone_type]["static"] == "0"):
                World.zone_types[zone_type]["static"] = False
            else:
                World.zone_types[zone_type]["static"] = True

        # Compute the initial number of patches in each dimension (x,y) for each zone type.
        #   With the understanding of the previous paragraph, we see that the number of
        #   patches for a zone depends on the number of patches that the "one-lower" zone
        #   type has.
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

        # Compute the total number of patches in each dimension (x,y) for the World.
        for zone_type in range(World.num_zone_types):
            World.x_num_patches *= World.zone_types[zone_type]["x_sub_units"]
            World.y_num_patches *= World.zone_types[zone_type]["y_sub_units"]
        print (f"There are {World.x_num_patches} in the x-direction.")
        print (f"There are {World.y_num_patches} in the y-direction.")

        # Create patches that fill the world and zones. Each patch needs to know its "x,y"
        #   location to determine what zones it is a part of. A key side-effect of this
        #   step is that each patch will know the index number of the zones that it is a
        #   part of. This is a bit odd because we have not yet actually created the zones.
        #   (That is done below. We only specified zone types so far.) However, making the
        #   patches aware of their zone index now is very useful for making the zones later.
        World.patches = [[Patch(settings, i, j, World.x_num_patches, World.y_num_patches,
                World.num_zone_types, World.zone_types)
                for i in range(World.x_num_patches)]
                for j in range(World.y_num_patches)]

        # Create the zones themselves. (Note above that we just created the *zone types*, not
        #   the zones themselves.)

        # First, make an empty list for each zone type. The zones of each zone type will be
        #   stored as a list of zones of that type. The World.zones is an empty list. After
        #   this loop, the list will contain a set of empty lists.
        #   Before: []. After: [[], [], []] with one empty list for each zone type.
        for zone_type in range(World.num_zone_types):
            World.zones.append([])

        # Visit every patch and whenever we find one with a new zone index, then create
        #   an official new zone. If we find a patch that is not a new zone index then
        #   we just add this patch to the list of patches that belong to this zone.

        # Initialize the current zone index for each zone type.
        curr_zone_index = [-1] * World.num_zone_types

        # Tripley nested loop over the x,y number of patches and number of types.
        for i in range(len(World.patches)):
            for j in range(len(World.patches[i])):
                for zone_type in range(World.num_zone_types):

                    # Check if we have come across a new zone index for this zone type. Note,
                    #   we are only looking for zone indices that are *greater* than any one
                    #   found before. As we traverse the x,y patch grid, we will encounter
                    #   the same zones multiple times so we don't want to think of them as
                    #   being new. (Consider a 5x5 grid of zones that are each made of a 5x5
                    #   grid of patches. There are 25 zones in total. If we are going row-by-row
                    #   through the patches, then we will encounter zones 0-4 on the bottom row
                    #   of patches and then 0-4 again in the second from the bottom row, etc.
                    #   Only when we get to row #5 (the sixth row) will we encounter zones 5-9.
                    if (World.patches[i][j].zone_index[zone_type] > curr_zone_index[zone_type]):

                        # If so, update the current zone index.
                        curr_zone_index[zone_type] = World.patches[i][j].zone_index[zone_type]

                        # Then, create a new zone.
                        World.zones[zone_type].append(Zone(settings, zone_type,
                                World.patches[i][j]))
                    else:
                        # Add this patch to the patch list of the current zone.
                        World.zones[zone_type][
                                World.patches[i][j].zone_index[zone_type]].add_patch(
                                        World.patches[i][j])

        # Define the global properties types of the world.
        self.properties.append(SimProperty("WellBeing", "WellBeing", "Scalar",
                rng.uniform(size=(World.x_num_patches, World.y_num_patches))))


    # Add citizens and politicians into the world.
    def populate(self, settings):

        # Initialize the array of all citizens. This list will hold the actual citizen objects.
        World.citizens = []

        # Add citizens in groups to each patch by way of their index number within the global
        #   list. (I.e., each patch will know which citizens are on that patch.) Only then do
        #   we actually create the citizens and append them to the World list. We do this order
        #   so that when the citizens are created they will know which patch they are on.

        # Initialize the citizen index counter.
        start_index = 0

        # Get each list (i) and then visit each patch in the list.
        for i in World.patches:  # i is a list of patches.
            for patch in i:  # patch is a patch in the list i.

                # Ask this patch to store the indices of the citizens that are about to be made.
                (patch.sprout_citizens(start_index, patch.num_citizens))
                start_index += patch.num_citizens

                # Add those citizens to the global (world) list.
                for citizen in range(patch.num_citizens):
                    World.citizens.append(Citizen(settings, patch))


        # Initialize the list of all politicians.
        World.politicians = []

        # Sprout politicians within their geographic boundary. We do this by visiting
        #   every single zone (of every zone type) and creating a politician in a random
        #   patch of the necessary type in that zone.
        for zone_type in range(World.num_zone_types):
            for zone in World.zones[zone_type]:
                for politician in range(zone.num_politicians):

                    # Select a random patch in the current zone.
                    random_patch = zone.random_patch()

                    # Create a politician and append it to the list of all politicians. Note,
                    #   presently, that the patch is not "made aware" of the fact that there is
                    #   a politician associated with it.
                    World.politicians.append(Politician(settings, zone_type, zone, random_patch))


        # Create a government for the world.
        World.government = Government(settings)


    #def dump_state(self, settings, time_step):
    #    
    #    # Record each hdf5 dataset for the current time step.
    #    settings.wellbeing_did.append(settings.wellbeing_gid.create_dataset(f"{time_step}",
    #            (World.x_num_patches, World.y_num_patches), data=))
    #    settings.policy_alignment_did.append(settings.policy_alignment_gid.
    #            create_dataset(f"{time_step}",(World.x_num_patches, World.y_num_patches),
    #                    data=))
    #    settings.politicians_did.append(settings.politicians_gid.create_dataset(f"{time_step}",
    #            (World.x_num_patches, World.y_num_patches), data=))


class Zone():

    def __init__(self, settings, zone_type, patch):
        self.zone_type = zone_type
        self.patches = [patch]

        # Get the statistical parameters for the number of politicians for this zone.
        self.min_politicians = int(settings.infile_dict[1]["world"]
                [f"zone_type_{zone_type}"]["min_politicians"])
        self.max_politicians = int(settings.infile_dict[1]["world"]
                [f"zone_type_{zone_type}"]["max_politicians"])
        self.num_politicians_mean = float(settings.infile_dict[1]["world"]
                [f"zone_type_{zone_type}"]["num_politicians_mean"])
        self.num_politicians_stddev = float(settings.infile_dict[1]["world"]
                [f"zone_type_{zone_type}"]["num_politicians_stddev"])

        # Determine the initial number of politicians for this zone.
        self.num_politicians = int(rng.normal(loc=self.num_politicians_mean,
                scale=self.num_politicians_stddev))
        if (self.num_politicians < self.min_politicians):
            self.num_politicians = self.min_politicians
        elif (self.num_politicians > self.max_politicians):
            self.num_politicians = self.max_politicians


    def add_patch(self, patch):
        self.patches.append(patch)


    def random_patch(self):
        return rng.choice(self.patches)


class Patch():

    def __init__(self, settings, i, j, x_num_patches, y_num_patches, num_zone_types,
            zone_types):

        # Initialize the properties of this patch.
        self.zone_index = []
        self.x_location = i
        self.y_location = j

        # Determine the properties of this patch.

        # Initialize the number of citizens on this patch.
        self.num_citizens = int(settings.infile_dict[1]["patch"]["initial_num_citizens"])

        # Compute the set of zone index numbers for this patch. All patches of a given
        #   zone and zone type will have the same index number. Index numbers are computed
        #   contiguously from 0, and increasing by 1 for each "zone-sized" step in the
        #   x-direction and increasing by the number of zones in a row for each row in the
        #   y-direction.
        for zone_type in range(num_zone_types):
            self.zone_index.append( \
                    (i // zone_types[zone_type]["x_num_patches"]) + \
                    (j // zone_types[zone_type]["y_num_patches"]) * \
                    (x_num_patches // zone_types[zone_type]["x_num_patches"]))


    def sprout_citizens(self, start_index, num_citizens):

        # Add citizens to this patch.
        self.citizen_list = np.array(range(start_index,
                start_index + num_citizens + 1))


class Citizen():
    # Citizens have an innate "emotional" position that can change to align with a
    #   politician.

    # Citizens have a stated policy position for each policy. This is the policy position
    #   that the citizen claims to align with and it will affect their preference for a
    #   particular politician.

    # Citizens have a most-beneficial policy position that the citizen does not directly know.
    #   I.e., the well-being of the citizen will depend on the alignment between governing
    #   policy and this most-beneficial policy, but the stated policy may be quite different
    #   than the most-beneficial policy. For example, if a policy represented "tax rates", it
    #   is hard to know what the best tax rate for an individual should be. If it was set to
    #   zero, the individual would pay nothing, but also likely have no services. If it was
    #   set to 100% they would have no money, but would have many servies. The "correct"
    #   number is not easy for any individual to know and that individual's stated preference
    #   may easily be different than whatever number actually benefits them the most.

    # Citizens have a probability of participation. It is affected (in no order) by:
    #   (1) The emotional alignment between a citizen and a politician.
    #   (2) The cumulative policy position alignment between a citizen and a politician.
    #   (3) Whether or not the citizen voted previously.
    #   (4) The number citizens in the same zone that will vote in agreement with the citizen.
    #   (5) The well-being of the citizen.

    # Citizens have a well-being factor that weights the degree to which they will use emotional
    #   or policy alignment when deciding how to cast their vote.
     

    def __init__(self, settings, patch):

        # Get local names for settings variables.
        num_policy_dims = int(settings.infile_dict[1]["world"]["num_policy_dims"])
        num_emot_dims = int(settings.infile_dict[1]["world"]["num_emot_dims"])

        # Define the initial instance variables of this citizen.
        self.participation_prob = \
                float(settings.infile_dict[1]["citizens"]["participation_prob"])
        self.stated_policy_pos = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["citizens"]["policy_pos_stddev"]),
                size=num_policy_dims)
        if (settings.infile_dict[1]["citizens"]["policy_orientation"] == "imaginary"):
            # Create random integers between 0 and 1 inclusive. Then multiply by 2
            #   and subtract 1 (in that order) to get random values of -1 or 1. Finally,
            #   multiply by the imaginary number (1j) to initialize citizen policy
            #   positions.
            self.stated_policy_orientation = (rng.integers(low=0, high=1, endpoint=True,
                    size=num_policy_dims)*2 - 1) * 1j
        else:
            print("Unknown citizen policy orientation\n")
            exit()
        self.stated_policy_orientation = rng.
        self.ideal_policy_pos = self.stated_policy_pos.copy()
        self.ideal_policy_pos = [x + rng.normal(
                loc=0.0, scale=float(settings.infile_dict[1]["citizens"]
                ["ideal_policy_pos_stddev"])) for x in self.ideal_policy_pos]
        self.policy_consistency = self.policy_alignment()
        self.emot_pos = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["citizens"]
                ["emot_pos_stddev"]),size=num_emot_dims)
        self.policy_emot_ratio = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["citizens"]
                ["policy_emot_ratio_stddev"]))
        self.current_patch_x = patch.x_location
        self.current_patch_y = patch.y_location


    # Compute the relationship between this citizen's stated policy positions and the actual
    #   policies as implemented by the government.
    def policy_attitude(self, government):
        attitude = 0
        for (stated, govern) in zip(self.state_policy_pos, government.policy_pos):
            attitude += 


    # Compute the relationship between this citizen's policy positions and the ideal
    #   (unknown to the citizen) policies that will benefit this citizen the most.
    def policy_alignment(self):
        alignment = 0 # Represents perfect alignment.
        for (stated, ideal) in zip(self.stated_policy_pos, self.ideal_policy_pos):
            alignment += abs(stated - ideal)

        return alignment


class Politician():
    # Politicians have an innate "emotional" position on a one-dimensional spectrum. The
    #   distance between a citizen's emotional position and a politician's emotional position
    #   will influence (1) the ability of the politician to persuade a citizen with respect to
    #   their policy positions; (2) the probability that a citizen will vote for a politician;
    #   (3) the probability that a citizen will allow policy position misrepresentations to go
    #   "unpunished".

    # Politicians have an innate position for each policy dimension. The innate position should
    #   not be understood as a "values" statement in any definite sense. For some politicians,
    #   it may be representative of their "values" but for others it may be better thought of as
    #   their "desire". Presently, the innate position is static for each politician.

    # Politicians have an apparent position for each policy dimension. The apparent position is
    #   what a politician presents to citizens. The apparent position will differ from the innate
    #   position due to factors: (1) The innate position and the average position of targeted
    #   citizens differ, and so the politician presents an apparent position that is closer to
    #   that of the local citizens. (2) The politician is willing to misrepresent their innate
    #   position by an amount in proportion to their propensity to lie/pander and believe that
    #   they will not turn off citizens by being detected.
    
    # It is assumed that citizens may or may not be able to detect (or may not care about) the
    #   difference between an apparent policy position and the innate position that a politician
    #   has. When the apparent 

    def __init__(self, settings, zone_type, zone, patch):

        # Get local names for settings variables.
        num_policy_dims = int(settings.infile_dict[1]["world"]["num_policy_dims"])
        num_emot_dims = int(settings.infile_dict[1]["world"]["num_emot_dims"])

        # Define the initial instance variables of this politician.
        self.policy_pos = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["policy_pos_stddev"]),size=num_policy_dims)
        self.policy_spread = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["policy_pos_stddev"]),size=num_policy_dims)
        if (settings.infile_dict[1]["politicians"]["policy_orientation"] == "uniform"):
            self.policy_orientation = rng.uniform(high=2.0*np.pi, size=num_policy_dims)
        else:
            print("Unknown politician policy orientation\n")
            exit()
        self.emot_pos = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["emot_stddev"]),size=num_emot_dims)
        self.emot_spread = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["emot_pos_stddev"]),size=num_emot_dims)
        self.policy_influence = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["policy_influence_stddev"]))
        self.emot_influence = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["emot_influence_stddev"]))
        self.policy_lie = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["policy_lie_stddev"]))
        self.emot_lie = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["emot_lie_stddev"]))
        self.pander = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["pander_stddev"]))

        # Select strategies for politician activities.
        self.move_strategy = self.select_strategy(settings, "move")
        self.adapt_strategy = self.select_strategy(settings, "adapt")
        self.campaign_strategy = self.select_strategy(settings, "campaign")

        # Assign instance variables from passed initialization parameters.
        self.zone_type = zone_type
        self.zone = zone
        self.patch = patch


    def select_strategy(self, settings, strat_type):
        # Select a strategy using the xml given probabilities. The strategy is selected
        #   according to the cumulative probabilities given in the xml file. For example,
        #   if the cumulative probabilities for three strategies are 0.5,0.75,1.0 then
        #   50% of the time the first strategy is used, and 25% of the time each of the
        #   other strategies is used. To make the selection, we obtain a random number
        #   and look for the first index where the random number is less than one of the
        #   cumulative probabilty values.
        strategy_index = -1
        random_float = rng.uniform()
        strategy_distribution = [float(prob) for prob in settings.infile_dict[1]
                ["politicians"][f"cumul_{strat_type}_strategy_probs"].split(',')]
        for index in range(len(strategy_distribution)):
            if (random_float < strategy_distribution[index]):
                strategy_index = index
                break

        return strategy_index


    def move(self):
        # Move according to the strategy that this politician is following.
        if (self.move_strategy == 0):
            # Select a random patch within the same zone.
            self.patch = self.zone.random_patch()


    def adapt_to_patch(self, world):
        # Set apparent policy positions and emotional positions according to the strategy
        #   that this politician is following.
        if (self.adapt_strategy == 0):
            self.apparent_policy_pos = self.policy_pos.copy()  # List copy.
            self.apparent_emot_pos = self.emot_pos  # Float. No need to copy().


    def persuade(self, world):
        # Iterate over the list of citizens in the patch that the politician is currently on.

        # 

        for citizen in self.patch.citizen_list:
            # Compare each policy position of the citizen with the current apparent policy
            #   position of the politician.
            # 
            # In the explanation given below, mirror symmetry has the same result.
            # If the governing policy is left of the citizen and the politician policy is
            #   left 

        world.properties[0].data = world.properties[0].data + rng.uniform(high=0.01)


class Government():

    # Get local names for settings variables.
    num_policy_dims = int(settings.infile_dict[1]["world"]["num_policy_dims"])

    # Define class variables.
    policy_pos = []
    policy_spread = []
    policy_orientation = []

    def __init__(self, settings):
        Government.policy_pos = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["government"]["policy_pos_stddev"]),
                size=num_policy_dims
        Government.policy_spread = abs(rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["government"]["policy_pos_stddev"]),
                size=num_policy_dims
        if (settings.infile_dict[1]["government"]["policy_orientation"] == "real"):
            # Create random integers between 0 and 1 inclusive. Then multiply by 2
            #   and subtract 1 (in that order) to get random values of -1 or 1.
            Government.policy_orientation = rng.integers(low=0, high=1, endpoint=True,
                    size=int(settings.infile_dict[1]["world"]["num_policy_dims"]))*2 - 1
        else:
            print("Unknown government policy orientation\n")
            exit()


# Mathematical form:
#  g(x;sigma,mu,theta) = 1/(sigma * sqrt(2 pi)) * exp(-(x-mu)^2 / (2 sigma^2)) * exp(i theta)
class Gaussian():
    def __init__(self, expected_value, stddev, angle):
        # Create instance variables.
        mu = expected_value
        sigma = stddev
        theta = angle

    def integral(self, g):
        # Get the real parts of the self and g (given) Gaussians.




class Hdf5():

    # Create a class variable for the HDF5 file handle.
    h_fid = 0

    # Create class variable dictionaries that will hold the group and dataset ids.
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
        Hdf5.dataset_did[f"{p.name}{i}"] = Hdf5.group_gid[p.group].create_dataset(
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

        # The root will be the xdmf tag. We will need to add some data before the root into the
        #   XML file. Specifically a line: '<?xml version="1.0" encoding="utf-8"?>' and a line:
        #   '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>"'. Note also that "xdmf" is lower case while
        #   then class name is "Xdmf". Finally, note that the xmlnsxi attribute should really be
        #   xmlns:xi, but python does not like colons in variable names and when I tried to add
        #   that key-value pair to the attribute dictionary after the fact it wouldn't accept
        #   it either. (Python called it a bad key I think.) So, the resolution is to use the
        #   xmlnsxi attribute and then when the xml file is actually printed I do a string
        #   substitution on xmlnsxi to make it xmlns:xi. (Kludgy.)
        root = etree.Element("root")
        xdmf = etree.SubElement(root, "Xdmf", Version="3.0",
                xmlnsxi="[http://www.w3.org/2001/XInclude]")

        # Make the domain. Only one domain is needed for the simulation results.
        domain = etree.SubElement(xdmf, "Domain")

        # Create a "grid" that is a collection of grids, one for each time step.
        time_collection_grid = etree.SubElement(domain, "Grid", Name="TimeSteps",
                GridType="Collection", CollectionType="Temporal")

        # Start the loop for adding time step grids.
        for i in range(sim_control.total_num_steps):

            # Add the time step grid.
            timestep_grid.append(etree.SubElement(time_collection_grid, "Grid",
                    Name=f"Step{i}", GridType="Uniform"))

            # Make the time step value an integer. (Arbitrary time duration.)
            timestep.append(etree.SubElement(timestep_grid[i], "Time", Value=f"{i}.0"))

            # Define the topology for this grid. (Ideally, we would define a single topology
            #   just below the "Domain" and then reference it here. But for some reason it
            #   didn't work and so we will just repeat the topology here for every time step
            #   grid even though it "wastes" space.)
            topology.append(etree.SubElement(timestep_grid[i], "Topology",
                    Name="Topo", TopologyType="2DCoRectMesh",
                    Dimensions=f"{world.x_num_patches} {world.y_num_patches}"))
     
            # Same for the geometry as for the topology.
            geometry.append(etree.SubElement(timestep_grid[i], "Geometry", Name="Geom",
                    GeometryType="ORIGIN_DXDY"))

            # The geometry needs two data elements, the origin and the spatial deltas.
            geometry_origin.append(etree.SubElement(geometry[i], "DataItem",
                    NumberType="Float", Dimensions="2", Format="XML"))
            geometry_origin[i].text="0.0 0.0"
            geometry_dxdy.append(etree.SubElement(geometry[i], "DataItem",
                    NumberType="Float", Dimensions="2", Format="XML"))
            geometry_dxdy[i].text="1.0 1.0"

            # XDMF attributes frame the data items that point to the actual HDF5 data. Each
            #   attribute / data set in XDMF corresponds to one monitored property in our
            #   simulation. Thus, because there will be more than one property and because
            #   they are computed in a time series we need a list of lists. (One property list
            #   for each member of the timestep list.)
            # Here, we add an empty attribute list (and its partner DataItem list) for this
            #   time step into the list of timestep lists.
            attribute.append([])
            data_item.append([])

            # Now, we add each attribute that will live on the grid for this timestep along
            #   with its data item that points to the actual data in the HDF5 file.
            for p in world.properties:
                attribute[i].append(etree.SubElement(timestep_grid[i], "Attribute",
                        Name=f"{p.name}", Center="Node", AttributeType=f"{p.datatype}"))
                data_item[i].append(etree.SubElement(attribute[i][-1], "DataItem",
                        NumberType="Float",
                        Dimensions=f"{world.x_num_patches} {world.y_num_patches}", Format="HDF"))
                data_item[i][-1].text = f"{settings.outfile}.hdf5:/{p.group}/{p.name}{i}"

        # Pretty print the xdmf file to the output file.
        temp_xml = etree.tostring(xdmf, pretty_print=True, encoding="utf-8").decode()
        Xdmf.x.write(temp_xml.replace("xmlnsxi", "xmlns:xi", 1))
        Xdmf.x.close()


def campaign(sim_control, world, hdf5):

    # Campaign activities per time step:
    #
    # - Politicians move randomly to another patch within the same zone. (Perhaps in the future
    #   politicians could move the more specific locations within their zone.)
    # - Politicians modify their emotional and policy positions according to the new environment.
    # - Citizens modify their emotional and policy positions under the influence of the
    #   politician persuasion efforts.
    # - Citizens modify their emotional and policy positions under the influence of their
    #   well_being.
    # - Citizens modify their emotional and policy positions under the influence of their
    #   fellow citizens.
    # - Citizens make a preliminary assessment about who they will vote for.

    for step in range(sim_control.num_campaign_steps):
        # Make the politicians move within their zones and adapt to local conditions.
        for politician in world.politicians:
            politician.move()
            politician.adapt_to_patch(world)

        for politician in world.politicians:
            politician.persuade(world)
        # Citizens respond to the persuasion efforts, their well_being, and their fellow
        #   citizens.
        #for citizen in world.citizens:
        #    citizen

        # Add current world properties to the HDF5 file.
        print(world.properties[0])
        hdf5.add_dataset(world.properties[0], sim_control.curr_step)

        # Increment the simulation timestep counter.
        sim_control.curr_step += 1
            

def vote(sim_control, world):
    # Each citizen decides whether they will vote or not. The participation probability
    #   is initialized at the average participation rate for eligible voters in the USA.
    # The participation probability will be high when there is strong emotional alignment
    #   between the citizen and any candidate.
    # When emotional alignment to all candidates is weak, then the participation probability
    #   will be low.
    #poor when there is strong misalignment
    #   between the citizen policy positions and the current g
    # Each citizen considers each politician at each zone level and computes a probability
    #   that the citizen will vote for that politician.
    return


def govern(sim_control, world):
    return



def main():

    # Get script settings from a combination of the resource control file
    #   and parameters given by the user on the command line.
    settings = ScriptSettings()
    if (settings.verbose): print ("Settings Made")

    # Read the provided input file.
    settings.read_input_file()
    print ("Input File Read")

    # Initialize the simulation control parameters.
    sim_control = SimControl(settings)
    print ("Control Settings Defined")

    # Initialize the simulation world.
    world = World(settings)
    print ("World Created")
    world.populate(settings)
    print ("World Populated")

    # Once the simulation is ready to start executing, we can create and print the
    #   xdmf xml file and create the hdf5 data file.
    xdmf = Xdmf(settings)
    print ("XDMF Contents Defined")
    xdmf.print_xdmf_xml(settings, sim_control, world)
    print ("XDMF File Written")
    hdf5 = Hdf5(settings, world)
    print ("HDF5 File Created")

    # Start executing the main activities of the program.
    for cycle in range(sim_control.num_cycles):
        print ("cycle = ", cycle)
        campaign(sim_control, world, hdf5)

        vote(sim_control, world)

        govern(sim_control, world)

        #primary(sim_control, world)

        #primary_vote(sim_control, world)


    # Finalize the program activities and quit.
    #hdf5.close


if __name__ == '__main__':
    # Everything before this point was a subroutine definition or a request
    #   to import information from external modules. Only now do we actually
    #   start running the program. The purpose of this is to allow another
    #   python program to import *this* script and call its functions
    #   internally.
    main()
