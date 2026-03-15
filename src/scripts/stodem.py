#!/usr/bin/env python3

# Discussion:

# All aspects of the program are open to modification, revision,
#   elimination, etc.

# This program is a multi agent based simulation. The agents include
#   politicians and citizens. The agents populate a two-dimensional world
#   that is divided into a series of nested user-defined zones. For
#   example, the top-level zone can be considered as a 'country' and
#   within that top level zone is a collection of 'state' zones. Each
#   state zone may be composed of a set of district zones. Etc. The world
#   is tiled with a set of patches such that (ideally) the smallest zone
#   consists of at least a few dozen patches.

# The zones may be classified as fixed or flexible. If a zone is fixed,
#   then its boundaries will remain constant throughout the simulation. If
#   a zone is flexible, then its boundaries may change during the
#   simulation with the constraint that a zone is not allowed to overlap
#   with another zone of the same level and it is not allowed to cross a
#   border of a higher level zone. (I.e., a zone cannot be spread across
#   two higher level zones at the same time such as a single district
#   being part of two states.)

# Each zone is home to some (non-zero) positive number of politicians.
#   The number of politicians in a zone can change throughout the course
#   of the simulation.

# Every patch will be a member of all the zones that contain it from the
#   lowest level to the highest. Some variable number of citizens
#   (possibly zero) inhabit each patch. Some variable (possibly zero)
#   number of politicians will inhabit each patch. Most patches will have
#   zero politicians and at least a modest number of citizens in them.

# The simulation runs in an iterative sequence of phases:
#   Campaign -> Vote -> Govern -> Primary Campaign -> Primary Vote ->
#   Campaign -> Etc. Each phase will consist of a finite number of time
#   steps. Depending on the phase, different kinds of activities will
#   happen during each time step.

# Each citizen exists in a state that is defined by the relationship
#   between its internal parameters and the external environment. The
#   internal parameters are expressed in a multi-dimensional complex
#   valued space where each dimension represents either an abstract policy
#   or an abstract personality trait.

# Each dimension should be envisioned as a one-dimensional real number
#   line that extends from -infinity to +infinity. The dimensions do not
#   interact with each other and are independent. An orthogonal imaginary
#   axis is connected to each dimension. Absolute values on the number
#   line carry no intrinsic meaning so concepts such as "extreme" and
#   "centrist" are all relative. The policies or personality traits that
#   the dimensions intend to represent are also completely abstract and
#   have no specific meaning.

# The citizen state parameters are used to define a set of one-
#   dimensional Gaussians. Each Gaussian has a mid-point located
#   somewhere on the real axis, and a standard deviation that describes
#   the spread of the Gaussian. All Gaussian functions have unit area,
#   but they may be "rotated" about the real axis into the imaginary
#   plane such that the area of the projection of the Gaussian on the
#   real axis is less than one (perhaps even zero). Generally, the
#   Gaussians will always maintain one sign. I.e., if a Gaussian is
#   initialized positive, it will always remain positive and will never
#   rotate through the imaginary plane to become negative (and vice-verse
#   for any Gaussians that are initialized to be negative).

# The Gaussian functions have slightly different interpretations depending
#   on their role, but there are general commonalities. Perhaps the best
#   way to understand them is by example. Consider some policy axis and a
#   Gaussian on it. The center point of the Gaussian defines a "position"
#   with regard to some abstract policy concept. The position is extreme
#   or centrist only by reference to all other Gaussians of the other
#   citizens and politicians. Because all Gaussians have unit area, the
#   standard deviation will establish the range over which that area is
#   spread. The spread speaks to the degree of attachment that the citizen
#   or politician has to a specific policy position. E.g., if the
#   Gaussian is tall and sharp, the implication is that the agent is
#   strongly attached to that specific position. Alternatively, if the
#   Gaussian is spread wide, then the implication is that many "nearby"
#   variations on that policy are "acceptable" to the agent. The degree
#   of rotation out of the real axis controls the total area of the
#   Gaussian on the real axis (i.e., the projection on the real axis).
#   The interpretation of a Gaussian that has been rotated out of the
#   real plane and into the imaginary is that while an agent may hold a
#   position with respect to some policy concept, the agent is apathetic
#   about that policy concept or that the policy concept is currently not
#   a priority. So, a Gaussian that has been rotated fully into the
#   imaginary axis indicates that the agent does not take this policy
#   topic into account when making a voting decision or campaigning.

# For each policy dimension, each citizen maintains three Gaussians: an
#   ideal policy position, a stated policy preference, and a stated
#   policy aversion. Presently, the policy preference and policy aversion
#   are understood to be public and available for averaging across all
#   citizens. So, we take these Gaussians to represent the citizens' true
#   opinion and not a self-curated "presentation" of themselves which
#   would require another set of "internal opinion" Gaussians that
#   represent their true opinions. On the other hand, the ideal policy
#   position is only indirectly known to the citizen. Therefore, a
#   citizen can have a stated preference that is inconsistent with the
#   ideal position for the citizen. (Indirect in the sense that a
#   "well-being" is computed based on a comparison of the ideal policy
#   positions and the government's enacted policies. The "well-being"
#   affects the citizen's engagement, tendency to believe politicians,
#   etc.)

# In any case...

# The stated policy preference is consciously held by the citizen and is
#   used when comparing their policy preferences with those of the
#   relevant politicians in preparation for a vote. Similarly, the stated
#   policy positions are used when comparing the citizen's policy
#   preferences with the enacted policies of the government during the
#   governing phase to quantify the citizen's satisfaction with the
#   performance of the government. Further, the stated policy preferences
#   are used when citizens interact with other citizens. Note that all
#   policy preference Gaussians are positively valued.

# The policy aversions work in a similar way to the preferences except
#   that they are considered to be negative valued. The aversions are
#   compared to the apparent policy positions of the relevant politicians
#   to help determine how the citizen will vote. Similarly, the policy
#   aversions are used when comparing the citizen's viewpoint with the
#   enacted policies of the government during the governing phase.
#   Additionally, the policy aversions are used when citizens interact
#   with other citizens.

# The ideal policy positions are, on the other hand, a bit different.
#   They are compared to the government policy positions during the
#   governing phase to compute the citizen's actual well-being. The
#   measure of well-being is used modulate the political temperature of
#   the citizen. Within this dynamic, a variety of phenomena may occur.
#   (See below.) Note, that all ideal policy positions are positive.

# In addition to the policy dimensions, there are also personality trait
#   dimensions. Each citizen maintains two Gaussians for each personality
#   trait dimension. One is a stated personality trait affinity and the
#   other is a stated personality trait aversion. Each politician has one
#   personality trait for each dimension. The personality attitudes only
#   interact with the personalities of politicians and other citizens. The
#   government has no personality position.

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

# The key relationship between any two Gaussians G1, G2 is their overlap
#   integral:
#   I(G1,G2) = Integral(Re(G1)*Re(G2) dx; -infinity..+infinity)
#   I(G1,G2) = (pi/zeta)^1.5 * exp(-xi * d^2) * cos(theta_1) * cos(theta_2)
#   The numerical value of this integral is between -1 and +1. The
#   maximum value occurs when both Gaussians have exactly equal
#   parameters. The minimum value occurs when both Gaussians have exactly
#   equal parameters except for that theta_1 = 0 or pi and theta_2 = pi
#   or 0 respectively. In many cases, theta is restricted to the range 0
#   to pi/2 or pi to pi/2. In those cases the integrals are between
#   similarly constrained Gaussian functions such that the integral
#   maximum to minimum is +1 to 0 or 0 to -1.

# Certain integral groups can be identified:
# The comprehensive citizen-politician integration set is
#   I(Pcp;n,Ppp;n), I(Pca;n,Ppa;n), I(Pcp;n,Ppa;n), and
#   I(Pca;n,Ppp;n) for each policy (n) and I(Tcp;m,Tpx;m) and
#   I(Tca;m,Tpx;m) for each personality trait (m). These integrals may
#   be used seperately or summed. Further, the sum of policy integrals
#   may be added to the sum of trait integrals according to a weighting
#   factor.
#
# Subsets of the comprehensive citizen-politician integration set include:
#   ***Mutual policy preference set: I(Pcp;n,Ppp;n) for each policy (n).
#   By construction, all values will be positive. The minimum for any one
#   integral is zero when either a citizen or a politician has a fully
#   imaginary Gaussian (so that the real projection is zero). The maximum
#   value is 1 for when the two Gaussian exactly overlap.
#
#   ***Mutual policy aversion set: I(Pca;n,Ppa;n) for each policy (n).
#   By construction, all values will be positive (both Gaussians have
#   negative orientation and thus the sign cancels) but otherwise they
#   are the same as the mutual policy preference set.
#
#   ***Policy Disagreement set: I(Pca;n,Ppp;n) and I(Pcp;n,Ppa;n) for
#   each policy (n). By construction, all values will be negative because
#   in each case one Gaussian is positive and one is negative.
#
# The comprehensive citizen-citizen integration set is I(Pcp;n,Pcp;n),
#   I(Pca;n,Pca;n), I(Pcp;n,Pca;n), and I(Pca;n,Pcp;n) for each policy
#   (n) and I(Tcp;m,Tcp;m), I(Tca;m,Tca;m), I(Tcp;m,Tca;m), and
#   I(Tca;m,Tcp;m) for each personality trait (m). Each citizen will
#   perform these integrals between themselves and the average form of
#   each Gaussian within each zone that the citizen inhabits. The average
#   position is simply the average position determined by using all
#   citizens in the zone. The average standard deviation is the actual
#   standard deviation of the positions and the average standard
#   deviation determined by using an average of that value from all
#   citizens in the zone.

# ===========================================================================
# Interactions and their effects
# ===========================================================================
#
# The three Gaussian parameters (mu, sigma, theta) each have a distinct
#   physical interpretation and are affected by different mechanisms:
#
#   mu (position): Where the agent stands on a policy or trait axis.
#     Extreme vs. centrist is relative to all other agents. A citizen's
#     stated preference position may differ from their ideal position.
#
#   sigma (spread/standard deviation): Strength of attachment to a
#     specific position. A sharp, narrow Gaussian implies strong
#     attachment to that specific position. A broad, wide Gaussian
#     implies that many nearby policy variations are acceptable.
#
#   theta (orientation/engagement): Degree of engagement with an issue.
#     theta=0 means the Gaussian is fully real — the citizen is fully
#     engaged with this issue and it factors strongly into decisions.
#     theta=pi/2 means the Gaussian is fully imaginary — the citizen is
#     apathetic about this issue and it does not factor into decisions.
#     The projection onto the real axis (cos(theta)) determines how much
#     weight the issue carries in overlap integrals.
#
# ---------------------------------------------------------------------------
# Fundamental principle: Trait overlap gates policy movement
# ---------------------------------------------------------------------------
#
# A consistent rule governs how much a citizen's policy positions shift
#   in response to any influence source (politician or citizen
#   collective): the *magnitude* of trait alignment determines how much
#   the policy shift is, and the *sign* of trait alignment determines
#   what kind of shift occurs.
#
#   trait_sum = sum of trait overlap integrals with the influence source
#   magnitude = |trait_sum|  --> controls the amount of shift
#   sign = sign(trait_sum)   --> controls the type of shift
#
#   If sign >= 0 (citizen likes the source based on personality):
#     - Policy mu shifts toward the source's mu, proportional to
#       magnitude. The source can sway the citizen toward its viewpoint.
#     - Policy sigma shifts toward the source's sigma, proportional to
#       magnitude.
#
#   If sign < 0 (citizen dislikes the source based on personality):
#     - No preference mu movement. The citizen will not change their
#       stated policy preference position.
#     - Preference sigma narrows, proportional to magnitude. The citizen
#       becomes more rigid and attached to their current position.
#     - Aversion mu shifts toward the source's policy positions,
#       proportional to magnitude multiplied by a defensive_ratio
#       parameter. This is a targeted backlash: the citizen develops a
#       specific aversion to the policies of the disliked source rather
#       than becoming diffusely averse to a broad range of policies.
#     The defensive_ratio is initially 1.0 but is stored as a variable
#       so that it can be made dynamic in the future.
#
#   The physical interpretation: people who feel personality affinity
#     with an influence source are susceptible to adopting that source's
#     policy positions. People who feel personality aversion become
#     defensive — they dig in on their existing preferences and develop
#     a targeted aversion to the disliked source's specific policies.
#
# ---------------------------------------------------------------------------
# Engagement decay
# ---------------------------------------------------------------------------
#
# Engagement naturally decreases without external stimulus. Each
#   simulation step, every citizen's theta for every Gaussian drifts
#   toward pi/2 (fully imaginary / fully apathetic) by a constant
#   amount set by an engagement_decay_rate parameter. Without active
#   campaigning or citizen-citizen interaction, citizens gradually
#   disengage from all issues. This creates a fundamental tension:
#   campaigns must actively maintain engagement, not just create it once.
#   The decay rate is stored as a variable so that it can be made
#   dynamic in the future (e.g., modulated by well-being).
#
# ---------------------------------------------------------------------------
# Voting phase
# ---------------------------------------------------------------------------
#
# A citizen decides who to vote for by computing the comprehensive
#   citizen-politician integration set for each eligible politician. The
#   accumulated and weighted (policy vs. trait) sum that has the highest
#   positive value will be the politician that the person votes for. The
#   weighting between policy and trait components is controlled by the
#   citizen's policy_trait_ratio parameter (clamped to [-0.5, +0.5] so
#   that both weights are non-negative and sum to 1):
#     w_policy = 0.5 + policy_trait_ratio
#     w_trait  = 0.5 - policy_trait_ratio
#     score = w_policy * policy_sum + w_trait * trait_sum
#
# ---------------------------------------------------------------------------
# Campaigning phase
# ---------------------------------------------------------------------------
#
# A campaign is an interaction between politicians and citizens.
#   Politicians attempt to directly influence citizens, citizens'
#   collective attitudes influence each other, and politicians may
#   adjust their policies and traits based on the policy status of
#   the citizens. The campaign consists of multiple time steps. Each
#   step, the following influence mechanisms are computed and the
#   resulting shifts are accumulated, then applied to citizen Gaussians.
#
# --- Politician-driven citizen engagement ---
#
# A direct interaction between a politician and a citizen occurs when
#   they share a patch. The comprehensive citizen-politician integration
#   set is computed for each politician-citizen pair. The *absolute
#   value* of each overlap integral is used to orient the corresponding
#   citizen Gaussian more into the real axis (toward theta=0).
#
#   Specifically, each integral affects the citizen Gaussian it involves:
#     |I(Pcp,Ppp)| --> shifts Pcp theta toward real
#     |I(Pca,Ppa)| --> shifts Pca theta toward real
#     |I(Pcp,Ppa)| --> shifts Pcp theta toward real
#     |I(Pca,Ppp)| --> shifts Pca theta toward real
#     |I(Tcp,Tpx)| --> shifts Tcp theta toward real
#     |I(Tca,Tpx)| --> shifts Tca theta toward real
#
#   Strong overlap (positive or negative) leads to strong engagement.
#   A politician who advocates for a policy you have a strong aversion
#   to will make you engage to fight against them, just as a politician
#   who aligns with your preferences will make you engage in support.
#   Weak overlap leads to weak engagement change.
#
#   The politician's policy_influence and trait_influence parameters
#   scale the magnitude of these engagement shifts.
#
# --- Politician-driven citizen policy position and spread shifts ---
#
# The sum of trait integrals between a citizen and a politician
#   determines how the citizen's policy Gaussians shift, following the
#   fundamental "trait gates policy" principle described above.
#
#   If the citizen likes the politician (positive trait sum):
#     - Citizen policy preference mu shifts toward the politician's
#       apparent policy preference mu.
#     - Citizen policy preference sigma shifts toward the politician's
#       apparent policy preference sigma.
#     - Citizen policy aversion mu shifts toward the politician's
#       apparent policy aversion mu.
#     - Citizen policy aversion sigma shifts toward the politician's
#       apparent policy aversion sigma.
#     All shifts are proportional to |trait_sum| and scaled by the
#       politician's policy_influence parameter.
#
#   If the citizen dislikes the politician (negative trait sum):
#     - No preference mu movement. The citizen will not change their
#       stated policy preference position.
#     - Citizen policy preference sigma narrows (citizen becomes more
#       rigid), proportional to |trait_sum| * policy_influence.
#     - Citizen policy aversion mu shifts toward the politician's
#       apparent policy positions (targeted backlash), proportional to
#       |trait_sum| * policy_influence * defensive_ratio.
#
# --- Politician-driven citizen trait shifts and spreads ---
#
# There is no direct mechanism by which a politician changes a
#   citizen's trait preferences or aversions. While personality traits
#   drive policy changes in citizens, there is no reciprocal action.
#   Citizen trait preferences/aversions are influenced only indirectly,
#   through citizen-citizen interactions. If a politician successfully
#   converts some citizens, those citizens will then influence other
#   citizens' traits through the citizen collective mechanism. Thus,
#   politician influence on citizen traits is indirect and emergent.
#
# --- Citizen-driven citizen policy shifts and spreads ---
#
# Citizens unconditionally acclimatize toward the policy views of their
#   community (zone average). The comprehensive citizen-citizen
#   integration subset for policies is used independently for each
#   policy Gaussian. The rate of this acclimatization follows the same
#   "trait gates policy" principle: the citizen's trait overlap with the
#   zone average trait Gaussians determines how much their policy
#   positions shift toward the zone average policy positions.
#
#   If the citizen has positive trait overlap with the community:
#     - Policy preference and aversion mu shift toward zone average mu.
#     - Policy preference and aversion sigma shift toward zone average
#       sigma.
#     The citizen "fits in" personality-wise and is more susceptible to
#       adopting community policy views.
#
#   If the citizen has negative trait overlap with the community:
#     - No policy preference mu movement.
#     - Policy preference sigma narrows (citizen becomes more rigid).
#     - Policy aversion mu shifts toward the community's policy
#       positions (targeted backlash, scaled by defensive_ratio).
#     The citizen feels alienated and becomes more rigid.
#
# --- Citizen-driven citizen trait shifts and spreads ---
#
# Citizen trait preferences and aversions also acclimatize
#   unconditionally toward the zone average trait values. This is the
#   sole mechanism by which traits change — there is no politician-
#   driven direct trait shift. The rate of trait acclimatization is
#   governed by the citizen-citizen trait overlap integrals themselves
#   (i.e., traits gate their own movement, unlike the cross-domain
#   gating of policy by traits).
#
# --- Citizen-driven citizen engagement ---
#
# Citizen-citizen overlap integrals also affect engagement, following
#   the same rule as politician-driven engagement: the absolute value
#   of each citizen-citizen overlap integral shifts the corresponding
#   citizen Gaussian's theta toward real. Citizens who are surrounded
#   by others with strong (agreeing or disagreeing) positions on an
#   issue will become more engaged with that issue.
#
# ---------------------------------------------------------------------------
# Governing phase (to be elaborated)
# ---------------------------------------------------------------------------
#
# The governing phase updates the government's enacted policy based on
#   elected politicians and computes citizen well-being.
#
# ===========================================================================
# Well-being, resource, and resentment — WORK IN PROGRESS
# ===========================================================================
#
# The design below is exploratory and NOT settled. The relationships
#   between these concepts need further discussion before implementation.
#   What follows captures the current thinking and candidate models.
#
# --- Perceived satisfaction (relatively settled) ---
#
# Perceived satisfaction is the direct overlap between a citizen's
#   stated policy preferences and the government's enacted policies:
#   overlap(Pcp, Pge). This is straightforward to compute and represents
#   how satisfied the citizen *feels* about current governance. A citizen
#   whose stated preferences align with enacted policy feels satisfied,
#   regardless of whether those policies actually benefit them.
#   Perceived satisfaction likely contributes to well-being, but the
#   exact relationship is not yet determined.
#
# --- Resource (candidate concept, under discussion) ---
#
# Resource is an abstract economic stock per citizen — an accumulated
#   material state that serves as the primary driver of well-being.
#   It is motivated by the observation that economic conditions are
#   often the dominant factor in a person's well-being. Key properties:
#
#   Accumulation with inertia: Resource grows or shrinks based on the
#     alignment between a citizen's ideal policy positions and the
#     government's enacted policies, but this happens gradually:
#       resource(t+1) = clamp(resource(t) + α * overlap(Pci, Pge, t),
#                             floor, ceiling)
#     This provides the inertia/lag that prevents instantaneous policy
#     effects. Enacted policy changes the *rate* of resource change,
#     not the resource level directly.
#
#   Asymmetric dynamics: Depletion may be faster than accumulation
#     (α_loss vs α_gain), reflecting the asymmetry between destroying
#     and creating wealth. The exact rates are TBD.
#
#   Floor and ceiling: Resource has hard bounds. A citizen at the floor
#     is in crisis regardless of enacted policy. This creates a
#     nonlinearity — low-resource citizens are more sensitive to policy.
#
#   Diminishing returns: The mapping from resource to well-being
#     contribution is concave (e.g., log or sqrt). Going from 0.1 to
#     0.2 matters enormously; going from 0.8 to 0.9 matters much less.
#
#   Note: The lag in resource naturally produces phase effects — a
#     citizen may have high resource during poor policy (still
#     benefiting from prior good policy) and low resource during good
#     policy (still suffering from prior poor policy).
#
# --- Well-being (candidate model, under discussion) ---
#
# Well-being is a composite scalar per citizen. Candidate inputs:
#
#   well_being = f(
#       resource(t),                  # accumulated material state
#       perceived_satisfaction(t),    # immediate feeling about policy
#       community_fit(t),             # trait overlap with zone averages
#       policy_consistency(t),        # stated vs ideal pref alignment
#       policy_stability(t),          # how much Pge has changed recently
#   )
#
#   The exact functional form is TBD. Brief notes on each input:
#
#   resource: Primary driver. Concave mapping (diminishing returns).
#
#   perceived_satisfaction: Immediate. A citizen can feel satisfied
#     even when policy is objectively harmful, or dissatisfied when
#     policy is objectively beneficial.
#
#   community_fit: Trait overlap with zone averages. Social belonging
#     affects well-being independently of policy and economics.
#
#   policy_consistency: Distance between stated and ideal preferences.
#     A citizen whose stated preferences have drifted far from their
#     ideal (e.g., due to politician or community influence) may
#     experience internal dissonance that erodes well-being.
#
#   policy_stability: Variance of Pge over a rolling window. Rapid
#     policy change is destabilizing regardless of whether the policy
#     is good or bad. Citizens benefit from predictability.
#
# --- Resentment (candidate concept, under discussion) ---
#
# Resentment is a separate scalar driven by the gap between a
#   citizen's resource and the zone average resource. It is NOT
#   folded into well-being because it represents a different kind
#   of political state. Key properties:
#
#   Asymmetric: Being below the zone average generates resentment.
#     Being above the zone average does NOT generate an inverse
#     effect — it may instead produce complacency or disengagement
#     (indifference), which is behaviorally distinct.
#
#   Separate behavioral channels: Resentment could affect:
#     - policy_trait_ratio: Resentful citizens may shift toward
#       weighting traits more heavily, voting on "who they trust"
#       rather than policy positions. This creates an emergent
#       dynamic: inequality → resentment → trait-driven voting →
#       politicians winning on personality over policy.
#     - Aversion intensity and susceptibility to influence.
#     - Trait shifts (who the citizen identifies with).
#
#   Orthogonal to well-being and engagement: A resentful citizen can
#     have moderate well-being (getting by, but others do better).
#     A resentful citizen is likely highly engaged — resentment is
#     motivating, not apathy-inducing.
#
# --- Engagement vs. happiness vs. resentment ---
#
# These are three independent axes:
#   Engagement (theta): Apathy about the political process.
#   Happiness/well-being: Personal material and social state.
#   Resentment: Relative economic standing, socially driven.
#
# An unhappy citizen can be engaged or disengaged. A resentful
#   citizen can have moderate well-being. A disengaged citizen is
#   not necessarily unhappy or resentful — just apathetic.
#   The engagement decay rate should NOT be tied to well-being
#   or resentment directly.

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

        # - Politicians move randomly to another patch within the same zone.
        #   (Perhaps in the future politicians could move to more specific
        #   locations within their zone.)
        for politician in world.politicians:
            politician.move()
        print ("moved politicians")

        # - Politicians modify their personality and policy positions
        #   according to the new environment.
        for politician in world.politicians:
            politician.adapt_to_patch(world)
        print ("politicians adapted")

        # - Citizens compute all the necessary overlap integrals.
        for citizen in world.citizens:
            citizen.compute_all_overlaps(world)
        print ("overlaps computed")

        # - Citizens prepare to have their state influenced.
        for citizen in world.citizens:
            citizen.prepare_for_influence(world.num_policy_dims,
                    world.num_trait_dims)
        print ("prepped for influence")

        # - Citizens modify their personality and policy positions under
        #   the influence of the politician persuasion efforts.
        for citizen in world.citizens:
            citizen.build_response_to_politician_influence()
        print ("built response to influence")

        # - Citizens modify their personality and policy positions under
        #   the influence of their well_being.
        for citizen in world.citizens:
            citizen.build_response_to_well_being()
        print ("built response to well being")

        # - Citizens modify their personality and policy positions under
        #   the influence of their fellow citizens.
        for citizen in world.citizens:
            citizen.build_response_to_citizen_collective()
        print ("built response to collective")

        # - Citizens make a preliminary assessment about who they will vote for.
        for citizen in world.citizens:
            citizen.score_candidates(world)
        print ("candidates scored")

        # - Aggregate well being to patches for output.
        world.compute_patch_well_being()

        # Add current world properties to the HDF5 file.
        print (world.properties[0])
        hdf5.add_dataset(world.properties[0], sim_control.curr_step)

        # Increment the simulation timestep counter.
        sim_control.curr_step += 1

    # One-time activities at the end of a campaign.
    # None so far...



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
            #   we can assign that politician as the winner. We need to
            #   make sure that the politician knows that they are elected
            #   or not elected. We also need to make sure that the zone
            #   knows the elected politician. This call will do both.
            zone.set_elected_politician(top_vote_getter)

    # The participation probability will be high when there is strong
    #   personality alignment between the citizen and any candidate.
    # When personality alignment to all candidates is weak, then the
    #   participation probability will be low.
    #poor when there is strong misalignment
    #   between the citizen policy positions and the current g
    # Each citizen considers each politician at each zone level and
    #   computes a probability that the citizen will vote for that
    #   politician.
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

    # Once the simulation is ready to start executing, we can create and
    #   print the xdmf xml file and create the hdf5 data file.
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
    hdf5.close()


if __name__ == '__main__':
    # Everything before this point was a subroutine definition or a request
    #   to import information from external modules. Only now do we actually
    #   start running the program. The purpose of this is to allow another
    #   python program to import *this* script and call its functions
    #   internally.
    main()
