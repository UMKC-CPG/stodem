#!/usr/bin/env python3

# Discussion:

# All aspects of the program are open to modification, revision, elimination, etc.

# This program is a multi agent based simulation. The agents include politicians and
#   citizens. The agents populate a two-dimensional world that is divided into a series
#   of nested user-defined zones. For example, the top-level zone can be considered as
#   a 'country' and within that top level zone is a collection of 'state' zones. Each
#   state zone may be composed of a set of district zones. Etc. The world is tiled with
#   a set of patches such that (ideally) the smallest zone consists of at least a few
#   dozen patches.

# The zones may be classified as fixed or flexible. If a zone is fixed, then its
#   boundaries will remain constant throughout the simulation. If a zone is flexible,
#   then its boundaries may change during the simulation with the constraint that a
#   zone is not allowed to overlap with another zone of the same level and it is not
#   allowed to cross a border of a higher level zone. (I.e., a zone cannot be spread
#   across two higher level zones at the same time such as a single district being
#   part of two states.)

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
#   abstract policy or an abstract personality trait.

# Each dimension should be envisioned as a one-dimensional real number line that extends
#   from -infinity to +infinity. The dimensions do not interact with each other and are
#   independent. An orthogonal imaginary axis is connected to each dimension. Absolute
#   values on the number line carry no intrinsic meaning so concepts such as "extreme"
#   and "centrist" are all relative. The policies or personality traits that the
#   dimensions intend to represent are also completely abstract and have no specific
#   meaning.

# The citizen state parameters are used to define a set of one-dimensional Gaussians.
#   Each Gaussian has a mid-point located somewhere on the real axis, and a standard
#   deviation that describes the spread of the Gaussian. All Gaussian functions have
#   unit area, but they may be "rotated" about the real axis into the imaginary plane
#   such that the area of the projection of the Gaussian on the real axis is less than
#   one (perhaps even zero). Generally, the Gaussians will always maintain one sign.
#   I.e., if a Gaussian is initialized positive, it will always remain positive and
#   will never rotate through the imaginary plane to become negative (and vice-verse
#   for any Gaussians that are initialized to be negative).

# The Gaussian functions have slightly different interpretations depending on their
#   role, but there are general commonalities. Perhaps the best way to understand them is
#   by example. Consider some policy axis and a Gaussian on it. The center point of the
#   Gaussian defines a "position" with regard to some abstract policy concept. The
#   position is extreme or centrist only by reference to all other Gaussians of the
#   other citizens and politicians. Because all Gaussians have unit area, the standard
#   deviation will establish the range over which that area is spread. The spread speaks
#   to the degree of attachment that the citizen or politician has to a specific policy
#   position. E.g., if the Gaussian is tall and sharp, the implication is that the agent
#   is strongly attached to that specific position. Alternatively, if the Gaussian is
#   spread wide, then the implication is that many "nearby" variations on that policy
#   are "acceptable" to the agent. The degree of rotation out of the real axis controls
#   the total area of the Gaussian on the real axis (i.e., the projection on the real
#   axis). The interpretation of a Gaussian that has been rotated out of the real plane
#   and into the imaginary is that while an agent may hold a position with respect to
#   some policy concept, the agent is apathetic about that policy concept or that the
#   policy concept is currently not a priority. So, a Gaussian that has been rotated
#   fully into the imaginary axis indicates that the agent does not take this policy
#   topic into account when making a voting decision or campaigning.

# For each policy dimension, each citizen maintains three Gaussians: an ideal policy
#   position, a stated policy preference, and a stated policy aversion. Presently, the
#   policy preference and policy aversion are understood to be public and available for
#   averaging across all citizens. So, we take these Gaussians to represent the citizens'
#   true opinion and not a self-curated "presentation" of themselves which would require
#   another set of "internal opinion" Gaussians that represent their true opinions. On
#   the other hand, the ideal policy position is only indirectly known to the citizen.
#   Therefore, a citizen can have a stated preference that is inconsistent with the
#   ideal position for the citizen. (Indirect in the sense that a "well-being" is
#   computed based on a comparison of the ideal policy positions and the government's
#   enacted policies. The "well-being" affects the citizen's engagement, tendency to
#   believe politicians, etc.)

# In any case...

# The stated policy preference is consciously held by the citizen and is used when
#   comparing their policy preferences with those of the relevant politicians in
#   preparation for a vote. Similarly, the stated policy positions are used when
#   comparing the citizen's policy preferences with the enacted policies of the
#   government during the governing phase to quantify the citizen's satisfaction with
#   the performance of the government. Further, the stated policy preferences are used
#   when citizens interact with other citizens. Note that all policy preference
#   Gaussians are positively valued.

# The policy aversions work in a similar way to the preferences except that they are
#   considered to be negative valued. The aversions are compared to the apparent policy
#   positions of the relevant politicians to help determine how the citizen will vote.
#   Similarly, the policy aversions are used when comparing the citizen's viewpoint with
#   the enacted policies of the government during the governing phase. Additionally, the
#   policy aversions are used when citizens interact with other citizens.

# The ideal policy positions are, on the other hand, a bit different. They are compared
#   to the government policy positions during the governing phase to compute the
#   citizen's actual well-being. The measure of well-being is used modulate the
#   political temperature of the citizen. Within this dynamic, a variety of phenomena
#   may occur. (See below.) Note, that all ideal policy positions are positive.

# In addition to the policy dimensions, there are also personality trait dimensions. Each
#   citizen maintains two Gaussians for each personality trait dimension. One is a stated
#   personality trait affinity and the other is a stated personality trait aversion.
#   Each politician has one personality trait for each dimension. The personality
#   attitudes only interact with the personalities of politicians and other citizens.
#   The government has no personality position.

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
#   g(x;sigma,mu,theta) = 1/(sigma * sqrt(2 pi)) * exp(-(x-mu)^2 / (2 sigma^2))
#                         * exp(i theta)

# where:

#   x = the independent variable (arbitray position on the real number line).
#   sigma = the standard deviation (sigma^2 = the variance).
#   mu = the point on the real number line of maximum amplitude.
#   theta = the orientation of the Gaussian about the real number line into the
#           imaginary axis.

# and other convenient variables are:

#   FWHM = 2 sqrt(2 ln(2)) * sigma.
#   alpha = 1/(2 sigma^2)
#   zeta = alpha_1 + alpha_2 for two Gaussians
#   xi = 1/(2 zeta)
#   d = mu_1 - mu_2 for two Gaussians

# The key relationship between any two Gaussians G1, G2 is their overlap integral:
#   I(G1,G2) = Integral(Re(G1)*Re(G2) dx; -infinity..+infinity)
#   I(G1,G2) = (pi/zeta)^1.5 * exp(-xi * d^2) * cos(theta_1) * cos(theta_2)
#   The numerical value of this integral is between -1 and +1. The maximum value occurs
#   when both Gaussians have exactly equal parameters. The minimum value occurs when both
#   Gaussians have exactly equal parameters except for that theta_1 = 0 or pi and
#   theta_2 = pi or 0 respectively. In many cases, theta is restricted to the range
#   0 to pi/2 or pi to pi/2. In those cases the integrals are between similarly
#   constrained Gaussian functions such that the integral maximum to minimum is +1 to 0
#   or 0 to -1.

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
#   integration set for each eligible politician. The accumulated and weighted (policy
#   vs. trait) sum that has the highest positive value will be the politician that the
#   person votes for. The weighting factor will depend on the "satisfaction" that the
#   citizen has for the current "state of things".

# Campaigning phase:
# A campaign is an interaction between the politicians and the citizens. Politicians
#   attempt to directly influence citizens, citizens' collective attitudes will
#   influence each other, and politicians may adjust their policies and traits based
#   policy status of the citizens.

# A direct interaction between a politician and a citizen occurs when they share a
#   patch. Interactions cause the citizen Gaussians to shift orientation, position, and
#   spread. The orientation is a measure of the degree to which a citizen is engaged
#   with an issue. Engagement naturally decreases without any influence. Engagement
#   increases due to
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
#   The sum of trait integrals can influence the positions of the Gaussians for the
#   citizen policy prefrences and aversions. If the sum of trait integrals is positive
#   then the citizen's policy preference Gaussians will move toward the respective
#   policy positions of the politician. Additionally, the Gaussian standard deviation
#   will change to align more with that of the politician. Similarly, for a positive
#   trait integral sum, the citizen's policy aversion Gaussians will change (position
#   and standard deviation) to align more with the politician. If the sum of trait
#   integrals is negative, then the citizen's policy preference Gaussian standard
#   deviations will narrow and the policy aversion Gaussian standard deviations will
#   broaden, however, neither will move. The idea is: if the citizen likes the
#   politician based on trait alignment, then the politician can sway the citizen toward
#   their viewpoint. Alternatively, if the citizen dislikes the politician, then the
#   citizen will not change their own viewpoint preference position but will instead
#   sharpen their viewpoint and broaden their policy aversions.
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
#   attracted toward the average form to a degree. I.e., everyone acclimates to and
#   migrates toward the average community behavior. Consider a few different scenarios:
#   (1) everyone has sharp Gaussians at the same position. (2) everyone has sharp
#   Gaussians at different positions. (3) Broad Gaussians in one position. (4) Broad
#   Gaussians in many different positions.

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

#import pudb
#pudb.set_trace()

import numpy as np

from settings import ScriptSettings
from sim_control import SimControl
from world import World
from output import Hdf5, Xdmf


def campaign(sim_control, settings, world, hdf5):

    # One-time activities as the start of a campaign.

    # - Create the set of politicians who will be campaigning.
    world.repopulate_politicians(settings)

    # - Ask each citizen to clear their list of known politicians.
    for citizen in world.citizens:
        citizen.clear_politicians()

    # - Present the politicians to the citizens who will vote for them.
    for politician in world.politicians:
        politician.present_to_citizens(world)

    # Campaign activities executed with each time step:
    for step in range(sim_control.num_campaign_steps):
        print (f"step={step}")

        # - Compute the average properties of the citizens in each zone.
        for zone_type in world.zones:
            for zone in zone_type:
                zone.compute_zone_averages(world)

        # - Politicians move randomly to another patch within the same zone. (Perhaps in the
        #   future politicians could move to more specific locations within their zone.)
        for politician in world.politicians:
            politician.move()
        print ("moved politicians")

        # - Politicians modify their personality and policy positions according to the
        #   new environment.
        for politician in world.politicians:
            politician.adapt_to_patch(world)
        print ("politicians adapted")

        # - Citizens compute all the necessary overlap integrals.
        for citizen in world.citizens:
            citizen.compute_all_overlaps(world)
        print ("overlaps computed")

        # - Citizens prepare to have their state influenced.
        for citizen in world.citizens:
            citizen.prepare_for_influence(world.num_policy_dims, world.num_trait_dims)
        print ("prepped for influence")

        # - Citizens modify their personality and policy positions under the influence of the
        #   politician persuasion efforts.
        for citizen in world.citizens:
            citizen.build_response_to_politician_influence()
        print ("built response to influence")

        # - Citizens modify their personality and policy positions under the influence of their
        #   well_being.
        for citizen in world.citizens:
            citizen.build_response_to_well_being()
        print ("built response to well being")

        # - Citizens modify their personality and policy positions under the influence of their
        #   fellow citizens.
        for citizen in world.citizens:
            citizen.build_response_to_citizen_collective()
        print ("built response to collective")

        # - Citizens make a preliminary assessment about who they will vote for.
        for citizen in world.citizens:
            citizen.score_candidates(world)
        print ("candidates scored")

        # - Aggregate well being to patches for output.
        compute_patch_well_being(world)

        # Add current world properties to the HDF5 file.
        print (world.properties[0])
        hdf5.add_dataset(world.properties[0], sim_control.curr_step)

        # Increment the simulation timestep counter.
        sim_control.curr_step += 1

    # One-time activities at the end of a campaign.
    # None so far...


    def compute_patch_well_being(world):
        """Aggregate citizen well-being to the patch grid for visualization."""
        # Initialize the well-being grid
        well_being_grid = np.zeros((World.x_num_patches, World.y_num_patches))
        count_grid = np.zeros((World.x_num_patches, World.y_num_patches))

        # Accumulate well-being from each citizen
        for citizen in world.citizens:
            x = citizen.current_patch.x_location
            y = citizen.current_patch.y_location
            well_being_grid[x, y] += citizen.well_being
            count_grid[x, y] += 1

        # Average per patch (avoid division by zero)
        count_grid[count_grid == 0] = 1
        world.properties[0].data = well_being_grid / count_grid


def vote(sim_control, world):

    # - Citizens give votes to the politicians.
    for citizen in world.citizens:
        citizen.vote_for_candidates(world)

    # - Evaluate the votes and determine who was elected in each zone.
    for zone_type in range(world.num_zone_types):
        for zone in world.zones[zone_type]:
            top_vote_getter = zone.politician_list[0]
            for politician in zone.politician_list:
                if (politician.votes > top_vote_getter.votes):
                    top_vote_getter = politician

            # Now that the top vote getter for this zone has been determined,
            #   we can assign that politician as the winner. We need to make sure
            #   that the politician knows that they are elected or not elected.
            #   We also need to make sure that the zone knows the elected
            #   politician. This call will do both.
            zone.set_elected_politician(top_vote_getter)

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
    #if (settings.verbose): print ("Settings Made")

    # Read the provided input file.
    settings.read_input_file()
    print ("Input File Read")

    # Initialize the simulation control parameters.
    sim_control = SimControl(settings)
    print ("Control Settings Defined")

    # Initialize the simulation world.
    world = World(settings, sim_control)
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
        campaign(sim_control, settings, world, hdf5)

        vote(sim_control, world)

        govern(sim_control, world)

        #primary(sim_control, world)

        #primary_vote(sim_control, world)


    # Finalize the program activities and quit.
    hdf5.close


if __name__ == '__main__':
    # Everything before this point was a subroutine definition or a request
    #   to import information from external modules. Only now do we actually
    #   start running the program. The purpose of this is to allow another
    #   python program to import *this* script and call its functions
    #   internally.
    main()
