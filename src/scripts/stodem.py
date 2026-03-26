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
# --- Vote probability ---
#
# Before casting a vote, a citizen must decide whether to participate at
#   all. The probability of voting is determined by the citizen's average
#   engagement across all stated Gaussians (policy preferences, policy
#   aversions, trait preferences, trait aversions):
#
#     P(vote) = mean(|cos(theta)|)  over all stated Gaussians
#
#   A citizen whose Gaussians are mostly real (theta near 0) is highly
#   engaged across many issues and is very likely to vote. A citizen
#   whose Gaussians are mostly imaginary (theta near pi/2) is disengaged
#   and unlikely to vote. This emerges naturally from the engagement
#   mechanics: campaigns and citizen-citizen interactions drive theta
#   toward real, while engagement decay drives theta toward imaginary.
#
#   Future extension: A discriminability term could be included. If a
#   citizen's top candidate score is barely above the second-best, the
#   citizen has weak preference among candidates and may be less
#   motivated to vote. The score gap between the top two candidates
#   could multiply the engagement-based probability.
#
# --- Candidate selection ---
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
#   The politician's policy_persuasion and trait_persuasion
#   parameters scale the magnitude of these engagement shifts.
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
#       politician's policy_persuasion parameter.
#
#   If the citizen dislikes the politician (negative trait sum):
#     - No preference mu movement. The citizen will not change their
#       stated policy preference position.
#     - Citizen policy preference sigma narrows (citizen becomes more
#       rigid), proportional to |trait_sum| * policy_persuasion.
#     - Citizen policy aversion mu shifts toward the politician's
#       apparent policy positions (targeted backlash), proportional to
#       |trait_sum| * policy_persuasion * defensive_ratio.
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
#   This shift is UNCONDITIONAL — there is no sign-gating and no
#     defensive branch for community influence (DESIGN.md §8.6.3).
#     The rate is governed by trait_rate = sum of same-type trait
#     overlaps (pref×pref + aver×aver; cross-terms excluded), which
#     is always >= 0.  Policy preference and aversion mu and sigma
#     always drift toward the zone averages, regardless of whether
#     the citizen's traits align with community norms. A citizen who
#     is trait-opposed to the community still absorbs community
#     norms, just more slowly (smaller trait_rate).
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
from diagnostics import Diagnostics


def campaign(sim_control, settings, world,
             hdf5, diag, cycle):
    """Execute one full campaign phase.

    The campaign phase is where politicians
    interact with citizens, trying to influence
    their policy and trait positions. Citizens
    also influence each other through community
    norms. The campaign runs for
    num_campaign_steps time steps.

    Each step follows a fixed sequence:
    1. Compute zone averages (citizen stats per
       geographic zone).
    2. Politicians move to a new patch.
    3. Politicians adapt their external positions
       to their environment.
    4. Citizens compute overlap integrals with
       all politicians and zone averages.
    5. Citizens prepare shift arrays (12 zero
       arrays, 3 per Gaussian type).
    6. Citizens accumulate politician influence
       shifts (trait-gates-policy mechanics).
    7. Citizens accumulate well-being engagement
       shifts.
    8. Citizens accumulate collective (community)
       influence shifts.
    9. Citizens apply all accumulated shifts in
       one pass (two-phase design).
    10. Citizens score candidates for voting.
    11. Aggregate well-being to patch grid for
        output.
    12. Write data to HDF5.

    Steps 6-8 are the "accumulation phase" and
    step 9 is the "application phase" of the
    two-phase accumulate-then-apply design
    (DESIGN.md §8.6). The order of steps 6-8
    does not matter because all three read from
    the same unchanged Gaussian state.
    """

    # One-time activities at the start of a
    #   campaign.

    # - Create the set of politicians who will
    #   be campaigning.
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

        # - Apply accumulated shifts to citizen Gaussian
        #   parameters. This is the second half of the
        #   two-phase accumulate-then-apply pattern: the
        #   three build_response_to_*() calls above each
        #   added their contributions to per-citizen shift
        #   arrays without touching the Gaussians directly.
        #   Now, in a single pass, those shifts are written
        #   into the actual mu, sigma, and theta parameters.
        #   Doing it this way guarantees that the order of
        #   the three accumulation calls does not affect the
        #   outcome — every source saw the same citizen
        #   state at the top of this step (DESIGN.md §8.6).
        #   Engagement decay is also applied here, so that
        #   citizens who were not reached by any politician
        #   or collective pressure drift back toward apathy.
        #   Finally, derived variables cached inside each
        #   Gaussian (alpha, cos_theta, self_norm) are
        #   refreshed so that the next step's overlap
        #   integrals use the updated parameters.
        #   Must be called BEFORE score_candidates() so
        #   that scoring reflects the citizen's updated
        #   engagement state for this step.
        for citizen in world.citizens:
            citizen.apply_influence_shifts()
        print ("applied influence shifts")

        # - Citizens make a preliminary assessment about who they will vote for.
        for citizen in world.citizens:
            citizen.score_candidates(world)
        print ("candidates scored")

        # - Aggregate well being to patches for output.
        world.compute_patch_well_being()
        world.compute_patch_gaussian_stats()
        world.compute_patch_politician_stats()

        # Add current world properties to the HDF5 file.
        print (world.properties[0])
        for p in world.properties:
            hdf5.add_dataset(
                p, sim_control.curr_step)

        # Log diagnostics (no forces during
        #   campaign).
        diag.log_step(
            sim_control.curr_step,
            cycle, 0, world, None)

        # Increment the simulation timestep counter.
        sim_control.curr_step += 1

    # One-time activities at the end of a campaign.
    # None so far...



def vote(sim_control, world):
    """Execute the election phase: citizens vote,
    winners are determined, and margin of victory
    is computed for each zone.

    The election proceeds in two stages:

    Stage 1 — Citizens cast votes:
      Each citizen decides whether to participate
      (based on their engagement-derived vote
      probability), then casts one vote per zone
      level for their highest-scoring candidate
      (see citizen.vote_for_candidates()).

    Stage 2 — Determine winners:
      For each zone at each hierarchy level
      (district, state, country), find the
      politician with the most votes and compute
      their margin of victory:

        margin = (winner_votes - runner_up_votes)
                 / total_votes

      The margin of victory is a scalar in [0, 1]:
        0 = tied or no votes cast
        1 = unanimous (only one candidate, or all
            votes went to the winner)

      This margin feeds directly into the govern
      phase via compute_political_power(): a
      politician who won by a landslide has more
      governing influence than one who barely won.

    Edge cases:
      - If no votes were cast in a zone (all
        citizens too apathetic), margin = 0.
      - If only one candidate runs in a zone,
        runner_up_votes = 0 and margin = 1 (the
        sole candidate has full mandate).
      - The first politician in the zone's list
        is used as the initial "top" candidate;
        ties are broken by list order (the first
        politician encountered with the top
        score wins).
    """

    # Stage 1: Citizens cast votes.
    for citizen in world.citizens:
        citizen.vote_for_candidates(world)

    # Stage 2: Determine winners per zone.
    for zone_type in range(world.num_zone_types):
        for zone in world.zones[zone_type]:

            # Find the top vote getter, the
            #   runner-up, and total votes.
            top = zone.politician_list[0]
            runner_up_votes = 0
            total_votes = 0
            for politician in zone.politician_list:
                total_votes += politician.votes
                if (politician.votes
                        > top.votes):
                    # Current top becomes the
                    #   new runner-up.
                    runner_up_votes = top.votes
                    top = politician
                elif (politician is not top
                        and politician.votes
                            > runner_up_votes):
                    runner_up_votes = (
                        politician.votes)

            # Compute margin of victory.
            if total_votes > 0:
                top.margin_of_victory = (
                    (top.votes - runner_up_votes)
                    / total_votes)
            else:
                top.margin_of_victory = 0.0

            # Assign the winner. This sets
            #   politician.elected = True and
            #   zone.elected_politician = top.
            zone.set_elected_politician(top)

    return


def govern(sim_control, world, hdf5,
           diag, cycle):
    """Execute the govern phase: elected politicians
    exert forces on the government's enacted policy.

    The govern phase runs for num_govern_steps time
    steps each cycle (DESIGN.md §7.5). During this
    phase, the government's enacted policy Gaussians
    (Pge) are pushed and pulled by the collective
    forces of all elected politicians.

    Overview of the mechanics:

    1. POLITICAL POWER COMPUTATION (once per cycle):
       Each elected politician's political_power is
       computed from three factors:
         - Zone population (larger constituency =
           more power)
         - Margin of victory (stronger mandate =
           more power)
         - Agreement/disagreement ratio (alignment
           with constituents = more power)
       See Politician.compute_political_power().

    2. DIMENSION WEIGHTS (once per cycle):
       Each politician's power is distributed across
       policy dimensions based on their innate sigma
       (narrow sigma = strong opinion = more weight
       on that dimension).
       See Politician.compute_dimension_weights().

    3. FORCE ACCUMULATION (each govern step):
       Two types of forces act on Pge each step:

       Preference-attraction: pulls Pge.mu TOWARD
         each politician's innate preference mu.
         "Politicians push policy toward what they
         believe in."

       Aversion-repulsion: pushes Pge.mu AWAY FROM
         each politician's innate aversion mu.
         "Politicians push policy away from what
         they oppose."

       Both forces are direction-only: np.sign()
       produces +1 or -1, so the force magnitude
       per dimension per step is exactly the
       weighted political power, regardless of how
       far Pge is from the target. This prevents
       extreme positions from generating extreme
       forces and ensures gradual, bounded policy
       movement.

       The same force pattern applies to sigma:
       politicians pull Pge.sigma toward their
       innate preference sigma and away from their
       innate aversion sigma.

    4. WELL-BEING OUTPUT (each govern step):
       After forces are applied, Pge's cached
       integration variables are refreshed and
       each citizen's well-being is recomputed
       from the updated Pci-vs-Pge overlap (via
       citizen.recompute_well_being()). The
       per-patch average is written to HDF5 so
       ParaView can show well-being evolving as
       government policy shifts.

    5. NATURAL SPREAD (once per cycle):
       After all govern steps complete, Pge.sigma
       broadens by spread_rate / sigma. This
       represents institutional entropy: without
       active political maintenance, precise
       policies become vague over time. Narrow
       policies (small sigma) spread faster.

    6. REFRESH CACHED VARIABLES:
       After spread, Pge's integration variables
       are refreshed once more so the next
       campaign phase's overlap integrals are
       correct.
    """
    gov = world.government
    Pge = gov.enacted_policy
    n = world.num_policy_dims

    # Zone averages are current from the last
    #   campaign step. Compute political power
    #   and dimension weights for each elected
    #   politician (once per govern cycle).
    elected = []
    for zone_type in world.zones:
        for zone in zone_type:
            if hasattr(zone, 'elected_politician'):
                pol = zone.elected_politician
                pol.compute_political_power()
                pol.compute_dimension_weights()
                elected.append(pol)

    # Normalize political power by total elected
    #   population so that force magnitudes are
    #   independent of world size
    #   (DESIGN.md §7.5.1).
    total_pop = sum(
        pol.zone.curr_num_citizens
        for pol in elected)
    if total_pop > 0:
        for pol in elected:
            pol.political_power /= total_pop

    # Each govern step: accumulate direction-only
    #   forces from all elected politicians, then
    #   apply to Pge.
    for step in range(sim_control.num_govern_steps):

        # Initialize per-step force accumulators.
        pref_force_mu = np.zeros(n)
        pref_force_sigma = np.zeros(n)
        aver_force_mu = np.zeros(n)
        aver_force_sigma = np.zeros(n)

        for pol in elected:
            pw = pol.political_power

            # Preference attraction: pull Pge
            #   toward politician's innate pref.
            #   sign(target - current) gives the
            #   direction; magnitude comes from
            #   pw * pref_weight.
            w_pref = pw * pol.pref_weight
            pref_force_mu += w_pref * np.sign(
                pol.innate_policy_pref.mu
                - Pge.mu)
            pref_force_sigma += w_pref * np.sign(
                pol.innate_policy_pref.sigma
                - Pge.sigma)

            # Aversion repulsion: push Pge AWAY
            #   from politician's innate aversion.
            #   sign(current - aversion) pushes
            #   Pge in the opposite direction from
            #   the aversion target.
            w_aver = pw * pol.aver_weight
            aver_force_mu += w_aver * np.sign(
                Pge.mu
                - pol.innate_policy_aver.mu)
            aver_force_sigma += w_aver * np.sign(
                Pge.sigma
                - pol.innate_policy_aver.sigma)

        # Apply accumulated forces from all
        #   politicians for this step.
        Pge.mu += pref_force_mu + aver_force_mu
        Pge.sigma += (
            pref_force_sigma + aver_force_sigma)

        # Clamp sigma to the floor to prevent
        #   zero or negative values
        #   (DESIGN.md §7.5.3).
        np.maximum(
            Pge.sigma, gov.sigma_floor,
            out=Pge.sigma)

        # Refresh cached integration variables so
        #   that the Pci vs Pge overlap uses the
        #   just-updated mu and sigma values.
        Pge.update_integration_variables()

        # Recompute citizen well-being from the
        #   updated Pge and write to HDF5. This
        #   lets ParaView show how well-being
        #   evolves as government policy shifts.
        for citizen in world.citizens:
            citizen.recompute_well_being(world)
        world.compute_patch_well_being()
        world.compute_patch_gaussian_stats()
        world.compute_patch_politician_stats()
        for p in world.properties:
            hdf5.add_dataset(
                p, sim_control.curr_step)

        # Log diagnostics with the forces
        #   that were applied this step and the
        #   would-be spread for the current
        #   sigma.
        spread = (
            gov.spread_rate / Pge.sigma)
        diag.log_step(
            sim_control.curr_step,
            cycle, 1, world,
            (pref_force_mu,
             aver_force_mu,
             pref_force_sigma,
             aver_force_sigma,
             spread))

        sim_control.curr_step += 1

    # Natural policy spread: applied once per
    #   cycle. sigma += spread_rate / sigma means
    #   narrow policies spread faster.
    Pge.sigma += gov.spread_rate / Pge.sigma

    # Clamp sigma to the floor after spread
    #   (DESIGN.md §7.5.4).
    np.maximum(
        Pge.sigma, gov.sigma_floor,
        out=Pge.sigma)

    # Refresh cached integration variables so
    #   the next campaign's overlap integrals
    #   use the updated Pge (after spread).
    Pge.update_integration_variables()


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

    # Once the simulation is ready to start executing,
    #   compute the data range for visualization and
    #   create the hdf5 data file. The xdmf file is
    #   written after the simulation completes so that
    #   it references only the HDF5 datasets that were
    #   actually created (handling early termination
    #   and the govern-phase gap correctly).
    sim_control.compute_data_range(settings, world)
    print ("Data Range Computed")
    xdmf = Xdmf(settings)
    print ("XDMF Initialized")
    hdf5 = Hdf5(settings, world)
    print ("HDF5 File Created")

    # Select sample citizens for diagnostic
    #   tracking: first, middle, and last in
    #   the global list (geographic spread).
    n_cit = len(world.citizens)
    sample_idx = [
        0, n_cit // 2, n_cit - 1]
    diag = Diagnostics(
        settings, world, sample_idx)
    print ("Diagnostics Initialized")

    # Start executing the main activities of
    #   the program.
    for cycle in range(sim_control.num_cycles):
        print ("cycle = ", cycle)
        campaign(sim_control, settings,
                 world, hdf5, diag, cycle)

        vote(sim_control, world)

        govern(sim_control, world, hdf5,
               diag, cycle)

    # Finalize: write XDMF using the actual
    #   number of steps completed, then close
    #   HDF5 and diagnostic files.
    xdmf.print_xdmf_xml(
        settings, sim_control.curr_step,
        world)
    print ("XDMF File Written")
    hdf5.close()
    diag.close()


if __name__ == '__main__':
    # Everything before this point was a subroutine definition or a request
    #   to import information from external modules. Only now do we actually
    #   start running the program. The purpose of this is to allow another
    #   python program to import *this* script and call its functions
    #   internally.
    main()
