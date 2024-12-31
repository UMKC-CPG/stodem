#!/usr/bin/env python3

import argparse as ap
import os
import sys
from datetime import datetime

import numpy as np
import random
import matplotlib.pyplot as pp


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

        # First group of default variables.
        self.infile = default_rc["infile"]

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
    
        # Define the XYZa_list argument.
        parser.add_argument('-i', '--infile', dest='infile', type=ascii,
                            default=self.infile, help='Input file name. ' +
                            f'Default: {self.infile}')
    

    def reconcile(self, args):
        self.infile = args.infile.strip("'")


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

        from lxml import etree
        
        tree = etree.parse(self.infile)
        root = tree.getroot()

        def recursive_dict(element):
            return (element.tag, dict(map(recursive_dict, element)) or element.text)

        self.infile_dict = recursive_dict(root)


class SimControl():

    # Declare and initialize the class variables.
    num_cycles = 0
    campaign_steps_pow = 0
    govern_steps_pow = 0
    primary_campaign_steps_pow = 0
    num_campaign_steps = 0
    num_govern_steps = 0
    num_primary_steps = 0


    def __init__(self, settings):
        SimControl.num_cycles = int(settings.infile_dict[1]["sim_control"]["num_cycles"])
        SimControl.campaign_steps_pow = \
                int(settings.infile_dict[1]["sim_control"]["campaign_steps_pow"])
        SimControl.govern_steps_pow = \
                int(settings.infile_dict[1]["sim_control"]["govern_steps_pow"])
        SimControl.primary_steps_pow = \
                int(settings.infile_dict[1]["sim_control"]["primary_steps_pow"])

        SimControl.num_campaign_steps = 10**SimControl.campaign_steps_pow
        SimControl.num_govern_steps = 10**SimControl.govern_steps_pow
        SimControl.num_primary_steps = 10**SimControl.primary_steps_pow


class World():

    # Declare class variables.
    x_num_patches = 1
    y_num_patches = 1
    patch_size = 0
    patches = 0
    zone_types = []
    zones = []


    def __init__(self, settings):

        # Define the global properties of the world.

        # Get the patch size.
        World.patch_size = int(settings.infile_dict[1]["world"]["patch_size"])

        # Create zone types.
        World.num_zone_types = int(settings.infile_dict[1]["world"]["num_zone_types"])
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

        # Compute the total number of patches in each dimension (x,y).
        for zone_type in range(World.num_zone_types):
            World.x_num_patches *= World.zone_types[zone_type]["x_sub_units"]
            World.y_num_patches *= World.zone_types[zone_type]["y_sub_units"]
        print (f"There are {World.x_num_patches} in the x-direction.")
        print (f"There are {World.y_num_patches} in the y-direction.")

        # Create patches that fill the world and zones.
        World.patches = [[Patch(settings, i, j, World.x_num_patches, World.y_num_patches,
                World.num_zone_types, World.zone_types)
                for i in range(World.x_num_patches)]
                for j in range(World.y_num_patches)]

        # Create the zones.

        # First, make an empty list for each zone type.
        for zone_type in range(World.num_zone_types):
            World.zones.append([])

        # Visit every patch and whenever we find one with a new zone index, then create
        #   an official new zone. If we find a patch that is not a new zone index then
        #   we just add this patch to the list of patches that belong to this zone.
        curr_zone_index = [-1, -1, -1] # Initialize the current zone index.
        for i in range(len(World.patches)):
            for j in range(len(World.patches[i])):
                for zone_type in range(World.num_zone_types):
                    # Check if we have come across a new zone index.
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



    def populate(self, settings):
        # Initialize the array of all citizens.
        World.citizens = []

        # Add the citizens in groups to each patch.
        start_index = 0

        # Get each list (i) and then visit each patch in the list.
        for i in World.patches:
            for patch in i:
                (patch.sprout_citizens(start_index, patch.num_citizens))
                start_index += patch.num_citizens
                for citizen in range(patch.num_citizens):
                    World.citizens.append(Citizen(settings, patch))


        # Initialize the array of all politicians.
        World.politicians = []

        # Sprout politicians within their geographic boundary. We do this by visiting
        #   every single zone (of every zone type) and creating a politician in a random
        #   patch of the necessary type in that zone.
        for zone_type in range(World.num_zone_types):
            for zone in World.zones[zone_type]:
                for politician in range(zone.num_politicians):

                    # Select a random patch in the current zone.
                    random_patch = zone.random_patch()
                    World.politicians.append(Politician(settings, zone_type, zone, random_patch))


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
        self.num_politicians_stddiv = float(settings.infile_dict[1]["world"]
                [f"zone_type_{zone_type}"]["num_politicians_stddiv"])

        # Determine the initial number of politicians for this zone.
        self.num_politicians = int(np.random.normal(loc=self.num_politicians_mean,
                scale=self.num_politicians_stddiv))
        if (self.num_politicians < self.min_politicians):
            self.num_politicians = self.min_politicians
        elif (self.num_politicians > self.max_politicians):
            self.num_politicians = self.max_politicians


    def add_patch(self, patch):
        self.patches.append(patch)

    def random_patch(self):
        return random.choice(self.patches)


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
    #   set to 100% they would lose flexibility, but would have many servies. The "correct"
    #   number is not easy for any individual to know and that individual's stated preference
    #   may easily be different than whatever number actually benefits them the most.

    # Citizens have a probability of participation is affected (in no order) by:
    #   (1) The emotional alignment between a citizen and a politician.
    #   (2) The cumulative policy position alignment between a citizen and a politician.
    #   (3) Whether or not the citizen voted previously.
    #   (4) The number citizens in the same zone that will vote in agreement with the citizen.
    #   (5) The well-being of the citizen.

    # Citizens have a well-being factor that weights the degree to which they will use emotional
    #   or policy alignment when deciding how to cast their vote.
     

    def __init__(self, settings, patch):

        # Define the initial instance variables of this citizen.
        self.participation_prob = float(settings.infile_dict[1]["citizens"]
                                        ["participation_prob"])
        self.policy_pos = np.random.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["citizens"]
                ["policy_stddiv"]),size=10)
        self.emot_pos = np.random.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["citizens"]
                ["emot_stddiv"]),size=1)
        self.policy_emot_ratio = np.random.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["citizens"]
                ["policy_emot_ratio_stddiv"]),size=1)
        self.current_patch_x = patch.x_location
        self.current_patch_y = patch.y_location


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

        # Define the initial instance variables of this politician.
        self.policy_pos = np.random.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["policy_stddiv"]),size=10)
        self.emot_pos = np.random.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["emot_stddiv"]),size=1)
        self.policy_influence = np.random.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["policy_influence_stddiv"]),size=1)
        self.emot_influence = np.random.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["emot_influence_stddiv"]),size=1)
        self.policy_lie = np.random.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["policy_lie_stddiv"]),size=1)
        self.emot_lie = np.random.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["emot_lie_stddiv"]),size=1)
        self.pander = np.random.normal(loc=0.0,
                scale=float(settings.infile_dict[1]["politicians"]
                ["pander_stddiv"]),size=1)
        self.zone_type = zone_type
        self.zone = zone
        self.patch = patch


def campaign(sim_control, world):

    for step in range(sim_control.num_campaign_steps):
        # Make the politicians move within their zones and adapt to local conditions.
        for politician in world.politicians:
            politician.move(world)
            politician.adapt_to_patch(world)

        # Politicians campaign to local citizens.
        #for politician in world.politicians:
            


def vote(sim_control, world):
    return


def govern(sim_control, world):
    return


def main():

    # Get script settings from a combination of the resource control file
    #   and parameters given by the user on the command line.
    settings = ScriptSettings()

    # Read the provided input file.
    settings.read_input_file()

    # Initialize the simulation control parameters.
    sim_control = SimControl(settings)

    # Initialize the simulation world.
    world = World(settings)
    world.populate(settings)
    #print(world.citizens[0].policy_emot_ratio)
    #print(world.politicians[0].pander)

    # Start executing the main activities of the program.
    for cycle in range(sim_control.num_cycles):
        campaign(sim_control, world)

        vote(sim_control, world)

        govern(sim_control, world)

        #primary(sim_control, world)

        #primary_vote(sim_control, world)


    # Finalize the program activities and quit.


if __name__ == '__main__':
    # Everything before this point was a subroutine definition or a request
    #   to import information from external modules. Only now do we actually
    #   start running the program. The purpose of this is to allow another
    #   python program to import *this* script and call its functions
    #   internally.
    main()
