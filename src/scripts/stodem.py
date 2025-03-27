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
#   a 'country' and within that top level zone is a collection of 'state' zones. Each
#   state zone may be composed of a set of district zones. Etc. The world is tiled with
#   a set of patches such that (ideally) the smallest zone consists of at least a few
#   dozen patches.

# The zones may be classified as fixed or flexible. If a zone is fixed, then its boundaries
#   will remain constant throughout the simulation. If a zone is flexible, then its
#   boundaries may change during the simulation with the constraints that a zone is not
#   allowed to overlap with another zone of the same level and it is not allowed to cross
#   a border of a higher level zone. (I.e., a zone cannot be spread across two higher level
#   zones at the same time such as a single district being part of two states.)

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
#   abstract policy or a personality trait.

# Each dimension should be envisioned as a one-dimensional real number line that extends
#   from -infinity to +infinity. The dimensions do not interact with each other and are
#   independent. I.e., a collection of independent lines as opposed to a high-dimensional
#   Cartesian axis. An orthogonal imaginary axis is connected to each dimension. Absolute
#   values on the number line carry no intrinsic meaning so concepts such as "extreme" and
#   "centrist" are all relative. The policies or personality traits that the dimensions
#   intend to represent are also completely abstract and have no specific meaning.

# The citizen state parameters are used to define a set of one-dimensional Gaussians.
#   Each Gaussian has a mid-point located somewhere on the real axis, and a standard
#   deviation that describes the "spread" of the Gaussian. All Gaussian functions have
#   unit area, but they may be "rotated" about the real axis into the imaginary plane such
#   that the area of the projection of the Gaussian on the real axis is less than one
#   (perhaps even zero). Generally, the Gaussians will always maintain one sign. I.e., if a
#   Gaussian is initialized positive, it will always remain positive and will never
#   rotate through the imaginary plane to become negative (and vice-verse for any
#   Gaussians that are initialized to be negative).

# The Gaussian functions have slightly different interpretations depending on their
#   role, but there are general commonalities. Perhaps the best way to understand them is
#   by example. Consider some policy axis and a Gaussian on it. The center point of the
#   Gaussian defines a "position" with regard to some abstract policy concept. The position
#   is extreme or centrist only by reference to all other Gaussians of the other citizens
#   and politicians. Beause all Gaussians have unit area the standard deviation will
#   establish the range over which that area is spread. The spread speaks to the degree of
#   attachment that the citizen or politician has to a specific policy position. E.g., if
#   the Gaussian is tall and sharp, the implication is that the agent is strongly attached
#   to that specific position. Alternatively, if the Gaussian is spread wide, then the
#   implication is that many "nearby" variations on that policy are "acceptable" to the
#   agent. The degree of rotation out of the real axis controls the total area of the
#   Gaussian on the real axis (i.e., the projection on the real axis). The interpretation
#   is that while an agent may hold a position with respect to some policy concept, the
#   agent is apathetic about that policy concept or that the policy concept is currently
#   not a priority. So, a Gaussian that has been rotated fully into the imaginary axis
#   indicates that the agent does not take this policy topic into account when making
#   a voting decision or campaigning.

# For each policy dimension, each citizen maintains three Gaussians: an ideal policy
#   position, a stated policy preference, and a policy aversion. Presently, the stated
#   policy preference and policy aversion are understood to be public, but not
#   necessarily "widely known". I.e., the opinions of regular citizens tends to only
#   go as far as their social networks allow and thus are public but not broadcast
#   beyond their social network. So, we take these Gaussians to represent the citizens'
#   true opinion and not a public facing presentation of themselves which would require
#   another set of "internal opinion" Gaussians that represent their true opinions, which
#   may or may not even be known to themselves. I.e., we are simplifying the meaning and
#   interpretation of a Gaussian so that just a few can represent the attitude that an
#   agent can have about a policy dimension.

# In any case...

# The stated policy preference is consciously held by the citizen and is used when comparing
#   their policy preferences with those of the relevant politicians in preparation for a
#   vote. Similarly, the stated policy positions are used when comparing the citizen's
#   policy preferences with the enacted policies of the government during the governing phase
#   to quantify the citizen's satisfaction with the performance of the government. Further,
#   the stated policy preferences are used when citizens interact with other citizens.

# The policy aversions work in a similar way to the preferences except that they are
#   considered to be negative valued. The aversions are compared to the apparent policy
#   positions of the relevant politicians to help determine how the citizen will vote.
#   Similarly, the policy aversions are used when comparing the citizen's viewpoint with
#   the enacted policies of the government during the governing phase. Additionally, the
#   policy aversions are used when citizens interact with other citizens.

# The ideal policy positions are, on the other hand, a bit different. They are compared to
#   the government policy positions during the governing phase to compute the citizen's actual
#   well-being. The measure of well-being is used modulate the political temperature of the
#   citizen. Within this dynamic, a variety of phenomena may occur. (See below.)

# In addition to the policy dimensions, there are also personality trait dimensions. Each
#   citizen maintains two Gaussians for each personality trait dimension. One is a stated
#   personality trait affinity and the other is a stated personality trait aversion.
#   Each politician has one personality trait for each dimension. The personality attitudes
#   only interact with the personalities of politicians and other citizens. The government
#   has no personality position.

# The notation for Gaussian functions is as follows:
#   Pcp;n = Policy: citizen stated preference Gaussian for dimension n.
#   Pca;n = Policy: citizen stated aversion Gaussian for dimension n.
#   Pci;n = Policy: citizen ideal Gaussian for dimension n.
#   Ppp;n = Policy: politician apparent preference Gaussian for dimension n.
#   Ppa;n = Policy: politician apparent aversion Gaussian for dimension n.
#   Pge;n = Policy: government enacted policy Gaussian for dimension m.
#   Tcp;m = Trait: citizen stated preference Gaussian for dimension m.
#   Tca;m = Trait: citizen stated aversion Gaussian for dimension m.
#   Tpx;m = Trait: politician personality Gaussian for dimension m.

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
#   Gaussians have exactly equal parameters except for that theta_1 = 0 or pi and
#   theta_2 = pi or 0 respectively. In many cases, theta is restricted to the range
#   0 to pi/2 or pi to pi/2. In those cases the integrals are between similarly constrained
#   Gaussian functions such that the integral maximum to minimum is +1 to 0 or 0 to -1.

# Certain integral groups can be identified:
# The comprehensive citizen-politician integration set is I(Pcp;n,Ppp;n), I(Pca;n,Ppa;n),
#   I(Pcp;n,Ppa;n), and I(Pca;n,Ppp;n) for each policy (n) and I(Tcp;m,Tpx;m) and
#   I(Tca;m,Tpx;m) for each personality trait (m). These integrals may be used seperately
#   or summed. Further, the sum of policy integrals may be added to the sum of trait
#   integrals according to a weighting factor.
#
# Subsets of the comprehensive citizen-politician integration set include:
#   ***Mutual policy preference set: I(Pcp;n,Ppp;n) for each policy (n). By construction,
#   all values will be positive. The minimum for any one integral is zero when either a
#   citizen or a politician has a fully imaginary Gaussian (so that the real projection
#   is zero). The maximum value is 1 for when the two Gaussian exactly overlap.
#
#   ***Mutual policy aversion set: I(Pca;n,Ppa;n) for each policy (n). By construction,
#   all values will be positive (both Gaussians have negative orientation and thus the
#   sign cancels) but otherwise they are the same as the mutual policy preference set.
#
#   ***Policy Disagreement set: I(Pca;n,Ppp;n) and I(Pcp;n,Ppa;n) for each policy (n).
#   By construction, all values will be negative because in each case one Gaussian is
#   positive and one is negative.
#
# The comprehensive citizen-citizen integration set is I(Pcp;n,Pcp;n), I(Pca;n,Pca;n),
#   I(Pcp;n,Pca;n), and I(Pca;n,Pcp;n) for each policy (n) and I(Tcp;m,Tcp;m),
#   I(Tca;m,Tca;m), I(Tcp;m,Tca;m), and I(Tca;m,Tcp;m) for each personality trait (m).
#   Each citizen will perform these integrals between themselves and the average form
#   of each Gaussian within each zone that the citizen inhabits. The average position
#   is simply the average position determined by using all citizens in the zone. The
#   average standard deviation is the actual standard deviation of the positions and
#   the average standard deviation determined by using an average of that value from
#   all citizens in the zone.

# Interactions and their effects are computed as follows:

# Voting phase:
# A citizen decides who to vote for by computing the comprehensive citizen-politician
#   integration set for each eligible politician. The accumulated and weighted (policy vs.
#   trait) sum that has the highest positive value will be the politician that the person
#   votes for. The weighting factor will depend on the "satisfaction" that the citizen has
#   for the current "state of things".

# Campaigning phase:
# A campaign is an interaction between the politicians and the citizens. In principle,
#   however, all parties interact with each other.
# An interaction between a politician and a citizen occurs when they are on the same patch.
#
#   Politician-driven citizen engagement (real vs. imaginary policy and trait positions):
#   The comprehensive citizen-politician integration set for that politician-citizen pair
#   is computed. The degree of overlap of every integral is used to orient the citizen's
#   Gaussian more into the real axis. I.e., if I(Pcp,Ppp) has strong overlap, then the
#   citizen's Pcp Gaussian (for that policy) will turn strongly real. If I(Pca,Ppp) has a
#   strong overlap, then the citizen's Pca Gaussian (for that policy) will turn strongly
#   real. Weak overlap leads to a weak turn. 
#
#   Politician-driven citizen policy shifts and spreads:
#   The sum of trait integrals can influence the positions of the Gaussians for the citizen
#   policy prefrences and aversions. If the sum of trait integrals is positive then the
#   citizen's policy preference Gaussians will move toward the respective policy positions
#   of the politician. Additionally, the Gaussian standard deviation will change to align
#   more with that of the politician. Similarly, for a positive trait integral sum, the
#   citizen's policy aversion Gaussians will change (position and standard deviation) to
#   align more with the politician. If the sum of trait integrals is negative, then the
#   citizen's policy preference Gaussian standard deviations will narrow and the policy
#   aversion Gaussian standard deviations will broaden, however, neither will move. The
#   idead is: if the citizen likes the politician based on trait alignment, then the
#   politician can sway the citizen toward their viewpoint. Alternatively, if the citizen
#   dislikes the politician, then the citizen will not change their own viewpoint preference
#   position but will instead sharpen their viewpoint and broaden their policy aversions.
#
#   Politician-driven citizen trait shifts and spreads:
#   There is no direct driver between a politician and the citizen preferences/aversions
#   of personality traits. I.e., while personality traits drive policy preference and
#   aversion changes in citizens, there is no opposite action whereby citizen personality
#   preferences/aversions are changed by any policy or trait state of a politician.
#   Citizen trait preferences/aversions are influenced by the degree to which other
#   citizens are attracted to or repelled by the politician. So, if a politician is able
#   to "convert" some citizens to their cause, then those citizens will influence other
#   citizens to make them more aligned with the politician. So, the politician influence
#   on citizen trait preference positions and spreads is indirect.
#
#   Citizen-driven citizen policy shifts and spreads:
#   The comprehensive citizen-citizen integration subset for policies is used
#   independently for each policy position/aversion. Based on each integration, the
#   citizen changes their position and standard deviation for each Gaussian.
#   The sum of policy integrals (between the citizen and the average Gaussian forms from
#   each zone) may be either positive or negative. Regardless, each citizen will be
#   attracted toward the average form to a degree. I.e., everyone acclimates to and migrates
#   toward the average community behavior. Consider a few different scenarios: (1) everyone has
#   sharp Gaussians at the same position. (2) everyone has sharp Gaussians at different
#   positions. (3) Broad Gaussians in one position. (4) Broad Gaussians in many different
#   positions.

#   
#   negative->away) 
# Politicians want to get elected.
# Politicians attempt to influence citizens directly and indirectly.
# Direct influence
# When a politician in on the same patch as a citizen, they compute a direct influence
#   in the form of the integrals I(Pcp,Ppp), I(Pca,Ppa), I(Pcp,Ppa), and I(Pca,Ppp). influences a citizen

# Governing phase:
# A citizen is "happy" when:
# (1) There is alignment between their stated policy preferences and the enacted policies
#     of the govenment.
# (2) The politicians that they voted for are in office.
# (3) The "economic environment" is good for the individual. This is measured by the
#     alignment 

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
        #   subunits. (That is not a typo. This was the easiest way to define it.)
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

                        # Then, create a new zone and add it to the world list of zones of this
                        #   type.
                        World.zones[zone_type].append(Zone(settings, zone_type,
                                World.patches[i][j]))
                    else:
                        # Add this patch to the patch list of the current zone. It is a
                        #   convoluted expression. Basically, for the current zone type,
                        #   get the current i,j patch and use the zone index of the current
                        #   zone type from that patch to add the current patch to that zone.
                        #   (May need to read that a couple of times...)
                        World.zones[zone_type][
                                World.patches[i][j].zone_index[zone_type]].add_patch(
                                        World.patches[i][j])

        # Define the global properties types of the world.
        self.properties.append(SimProperty("WellBeing", "WellBeing", "Scalar",
                rng.uniform(size=(World.x_num_patches, World.y_num_patches))))


    def repopulate_politicians(self):

        # Make things easy to start.
        # - Every politician who was elected and governed, will run in the next cycle.
        # - Every politician who lost will be replaced with a new politician.
        # - All politicians have zero votes.
        for politician in World.politicians:
            if (politician.elected == False):
                politician.reset_to_input()
            else:
                politician.elected = False
            politician.reset_votes()


    # Add citizens and politicians into the world.
    def populate(self, settings):

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
                    temp_politician = Politician(settings, zone_type, zone, random_patch)
                    World.politicians.append(temp_politician)

                    # However, we do want the zone to know which politicians are competing.
                    zone.add_politician(temp_politician)


        # Create a government for the world.
        World.government = Government(settings)


        # Initialize the list of all citizens. This list will hold the actual citizen objects.
        World.citizens = []

        # Add citizens in groups to each patch by way of their index number within the global
        #   list. (I.e., each patch will know which citizens are on that patch.) Only then do
        #   we actually create the citizens and append them to the World list. We do this order
        #   so that when the citizens are created they will know which patch they are on.

        # Initialize the citizen index counter.
        start_index = 0

        # Get each list (i, recalling that World.patches is a list of lists) and then visit
        #   each patch in the list (i).
        for i in World.patches:  # i is a list of patches.
            for patch in i:  # patch is a patch in the list i.

                # Ask this patch to store the indices of the citizens that are about to be made.
                (patch.sprout_citizens(start_index, patch.num_citizens))
                start_index += patch.num_citizens

                # Add those citizens to the global (world) list.
                for citizen in range(patch.num_citizens):
                    World.citizens.append(Citizen(settings, patch))


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

        # Each zone maintains a list of the politicians who vie for election in it.
        self.politician_list = []


    def add_politician(self, politician):
        self.politician_list.append(politician)


    def clear_politician_list(self):
        self.politician_list.clear()
        

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
    # Citizens have an innate "personality trait" preferences and aversion that can change
    #   to align with a politician.

    # Citizens have a stated policy position for each policy. This is the policy position
    #   that the citizen claims to align with and it will affect their preference for a
    #   particular politician. Similarly, each citizen has a policy aversion for each.

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
    #   (1) The personality alignment between a citizen and a politician.
    #   (2) The cumulative policy position alignment between a citizen and a politician.
    #   (3) Whether or not the citizen voted previously.
    #   (4) The number of citizens in the same zone that vote in agreement with the citizen.
    #   (5) The well-being of the citizen.

    # Citizens have a well-being factor that weights the degree to which they will use personality
    #   or policy alignment when deciding how to cast their vote.
     

    def __init__(self, settings, patch):

        # Get local names for settings variables.
        num_policy_dims = int(settings.infile_dict[1]["world"]["num_policy_dims"])
        num_trait_dims = int(settings.infile_dict[1]["world"]["num_trait_dims"])

        # Define the initial instance variables of this citizen.

        # Define instance variables given in the input file.
        self.participation_prob = \
                float(settings.infile_dict[1]["citizens"]["participation_prob"])
        self.stated_policy_pref = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["citizens"]["policy_pref_stddev"]),
                size=num_policy_dims)
        self.stated_policy_aver = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["citizens"]["policy_aver_stddev"]),
                size=num_policy_dims)
        if (settings.infile_dict[1]["citizens"]["policy_orientation"] == "imaginary"):
            # Create random integers between 0 and 1 inclusive. Then multiply by 2
            #   and subtract 1 (in that order) to get random values of -1 or 1. Finally,
            #   multiply by the imaginary number (1j) to initialize citizen policy
            #   positions. (Citizens initially have no policy engagement.)
            self.stated_policy_pref_orien= (rng.integers(low=0, high=1, endpoint=True,
                    size=num_policy_dims)*2 - 1) * 1j
            self.stated_policy_aver_orien= (rng.integers(low=0, high=1, endpoint=True,
                    size=num_policy_dims)*2 - 1) * 1j
        else:
            print("Unknown citizen policy orientation\n")
            exit()
        self.ideal_policy_pref = self.stated_policy_pref.copy()
        self.ideal_policy_pref = [x + rng.normal(
                loc=0.0, scale=float(settings.infile_dict[1]["citizens"]
                ["ideal_policy_pref_stddev"])) for x in self.ideal_policy_pref]
        self.policy_consistency = self.policy_alignment()
        self.trait_pref = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["citizens"]
                ["trait_pref_stddev"]),size=num_trait_dims)
        self.trait_aver = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["citizens"]
                ["trait_aver_stddev"]),size=num_trait_dims)
        self.policy_trait_ratio = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["citizens"]
                ["policy_trait_ratio_stddev"]))

        # Initialize instance variables that do not come from the input file.
        self.current_patch = patch
        self.politician_list = []


    def vote(self, preliminary):
        for 


    def clear_politicians(self):
        self.politician_list.clear()


    def compute_all_overlaps(self):
        # Compute the overlaps between the citizen and each relevant politician.
        for politician in self.politician_list:
            # Initialize lists that will hold overlap integral solutions.

            # Obtain the overlap between

        # Compute the overlaps between the citizen and 


    def respond_to_politician_influence(self):
        for politician in self.politician_list:
            # Initialize lists that will hold overlap integral solutions.

            # Obtain the overlap between


    def initialize_one_overlap(self, overlap_list, dimension):
        overlap_list = [[[0 for z in range(self.)] for y in range(self.num_zones)] for x in range(dimension)]


    def initialize_all_overlaps(self):
        # Define instance variables to hold the overlaps. Each is a list of lists.
        #   There will be one list for each dimension. Then, in each of those lists,
        #   there will be one list for each zone. In each zone list there will be
        #   one element for each politician.
        self.stated_policy_pref_overlap = [[] for x in range(self.num_policy_dims)]
        self.stated_policy_aver_overlap = [[] for x in range(self.num_policy_dims)]
        self.ideal_policy_pref_overlap = [[] for x in range(self.num_policy_dims)]
        self.ideal_policy_aver_overlap = [[] for x in range(self.num_policy_dims)]
        self.trait_pref_overlap = [[] for x in range(self.num_trait_dims)]
        self.trait_aver_overlap = [[] for x in range(self.num_trait_dims)]


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


    # Compute the full suite of policy overlaps with each politician.
    def compute_politician_policy_overlap(self, world):
        for pol in (world.politicians):
            for policy in (self.num_policies):




class Politician():
    # Politicians have an innate "personality" position on a one-dimensional spectrum. The
    #   distance between a citizen's personality position and a politician's personality position
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
        num_trait_dims = int(settings.infile_dict[1]["world"]["num_trait_dims"])

        # Define the initial instance variables of this politician obtained from the input file.
        self.reset_to_input()

        # Select strategies for politician activities.
        self.move_strategy = self.select_strategy(settings, "move")
        self.adapt_strategy = self.select_strategy(settings, "adapt")
        self.campaign_strategy = self.select_strategy(settings, "campaign")

        # Assign instance variables from passed initialization parameters.
        self.zone_type = zone_type
        self.zone = zone
        self.patch = patch

        # Initialize any other instance variables to their default value.
        self.elected = False
        self.votes = 0


    def reset_to_input(self, settings):
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
        self.trait_pos = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["trait_stddev"]),size=num_trait_dims)
        self.trait_spread = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["trait_pos_stddev"]),size=num_trait_dims)
        self.policy_influence = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["policy_influence_stddev"]))
        self.trait_influence = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["trait_influence_stddev"]))
        self.policy_lie = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["policy_lie_stddev"]))
        self.trait_lie = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["trait_lie_stddev"]))
        self.pander = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["pander_stddev"]))


    def reset_votes(self):
        self.votes = 0


    def present_to_citizens(self, world):
        for patch in self.zone.patches:
            for citizen in world.citizens[patch.citizen_list]:
                citizen.politician_list.append(self)


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
        # Set apparent policy positions and personality positions according to the strategy
        #   that this politician is following.
        if (self.adapt_strategy == 0):
            self.apparent_policy_pos = self.policy_pos.copy()  # List copy.
            self.apparent_trait_pos = self.trait_pos.copy()  # List copy.


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

    # Define class variables.
    num_policy_dims = 0
    policy_pos = []
    policy_spread = []
    policy_orientation = []

    def __init__(self, settings):
        Government.num_policy_dims = int(settings.infile_dict[1]["world"]["num_policy_dims"])
        Government.policy_pos = rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["government"]["policy_pos_stddev"]),
                size=num_policy_dims
        Government.policy_spread = abs(rng.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["government"]["policy_pos_stddev"]),
                size=num_policy_dims
        Government.policy_orientation = [1] * Government.num_policy_dims # Positive real number


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

    # One-time activities as the start of a campaign.

    # - Create the set of politicians who will be campaigning.
    world.repopulate_politicians()

    # - Ask each citizen to clear their list of known politicians.
    for citizen in world.citizens:
        citizen.clear_politicians()

    # - Present the politicians to the citizens who will vote for them.
    for politician in world.politicians:
        politician.present_to_citizens(world)


    # Campaign activities executed with each time step:
    for step in range(sim_control.num_campaign_steps):

        # - Politicians move randomly to another patch within the same zone. (Perhaps in the
        #   future politicians could move to more specific locations within their zone.)
        for politician in world.politicians:
            politician.move()

        # - Politicians modify their personality and policy positions according to the
        #   new environment.
        for politician in world.politicians:
            politician.adapt_to_patch(world)

        # - Citizens modify their personality and policy positions under the influence of the
        #   politician persuasion efforts.
        for citizen in world.citizens:
            citizen.respond_to_politician_influence()

        # - Citizens modify their personality and policy positions under the influence of their
        #   well_being.
        for citizen in world.citizens:
            citizen.respond_to_well_being()

        # - Citizens modify their personality and policy positions under the influence of their
        #   fellow citizens.
        for citizen in world.citizens:
            citizen.respond_to_citizen_collective()

        # - Citizens make a preliminary assessment about who they will vote for.
        for citizen in world.citizens:
            citizen.rank_candidates()

    # One-time activities at the end of a campaign.
    
    # - Citizens give votes to the politicians.
    for citizen in world.citizens:
        citizen.vote()

    # - Evaluate the votes and determine who was elected.
    for politician in world.politicians:
        politician.update_elected_
    

        # Add current world properties to the HDF5 file.
        print(world.properties[0])
        hdf5.add_dataset(world.properties[0], sim_control.curr_step)

        # Increment the simulation timestep counter.
        sim_control.curr_step += 1
            

def vote(sim_control, world):

    # Each citizen decides whether they will vote or not. The participation probability
    #   is initialized at the average participation rate for eligible voters in the USA.
    #   Then, if they vote, each citizen decides who they will vote and stores the result.
    foreach citizen in (world.citizens):
        citizen.vote()

    # Each politician collects votes from the citizens of their zone.
    foreach politician in (world.politicians):
        politician.count_votes()

    # Examine the votes collected by each politician in a given zone and execute a change
    #   of governance.
    foreach politician in (world.politicians):
        politician.update_status()

    # The participation probability will be high when there is strong personality alignment
    #   between the citizen and any candidate.
    # When personality alignment to all candidates is weak, then the participation probability
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
