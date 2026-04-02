# STODEM Vision

## Purpose

STODEM (Stochastic Democracy Simulation) is a multi-agent
simulation that models democratic processes. It investigates
whether stochastic elections can help a democracy navigate a
high-dimensional policy space and find alignment between the
internalized policy preferences of a population and the
actual (unknown) policies that lead to positive outcomes.

The simulation is not tied to any real-world political
system. All policy dimensions and personality traits are
abstract and carry no intrinsic meaning. "Extreme" and
"centrist" are relative labels determined only by the
distribution of agents.

## Goals

1. **Model emergent democratic dynamics.** Politicians,
   citizens, and government interact through simple local
   rules. Complex collective phenomena — polarization,
   gridlock, mandate effects, policy erosion, etc. — may
   emerge without being hard-coded.

2. **Explore stochastic elections.** The core research
   question is whether a probabilistic election (in which
   the winner is chosen according to a probability distribution that is
   defined by vote tallies) produces better long-term policy outcomes
   than deterministic elections.

3. **Separate stated and ideal preferences.** Citizens can
   hold stated preferences that do not serve their actual
   interests. The gap between stated and ideal preferences
   creates the core tension: democracy must navigate this
   gap to find good outcomes.

4. **Provide a teaching and research platform.** The
   codebase is written for clarity and expressiveness so
   that students and researchers can read, modify, and
   extend the simulation.

## Design Principles

These principles are non-negotiable constraints on the
design. All architectural and implementation decisions must
be consistent with them.

1. **Abstract dimensions.** All policy and personality
   trait dimensions are abstract. The simulation makes no
   assumption about what they represent. Meaning emerges
   from agent interactions.

2. **Engagement as a first-class quantity.** Engagement
   (theta) is built into the Gaussian representation
   itself via the complex orientation, making it
   inseparable from the agent's position and spread on
   every issue.

3. **Trait gates policy.** Personality determines
   susceptibility to policy influence. Politicians can
   shift policy views only through citizens who find them
   personally agreeable; alienated citizens become more
   rigid. This creates rich emergent dynamics.

4. **Symmetric engagement from asymmetric alignment.**
   Both agreement and disagreement increase engagement.
   Only indifference (weak overlap) leaves engagement
   unchanged. This prevents the modeling pitfall where
   only positive interactions create participation.

5. **Separation of ideal and stated preferences.** The
   gap between what citizens believe and what actually
   benefits them is central to the simulation's purpose.

6. **Accumulate-then-apply influence.** Influence from
   all sources is accumulated into shift arrays before
   being applied. This prevents order-of-evaluation
   artifacts.
