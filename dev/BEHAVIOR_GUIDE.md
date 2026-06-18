# Behavioral Trends Reference (Visual Analysis Aid)

This is a working reference for reading simulation output: it
maps trends you can *see* in the plots/HDF5 back to the code
mechanism that *produces* them. It is NOT part of the five-level
design chain — it is a validation aid, meant to be revised as we
gain experience reading real runs and hone in on the key
behaviors.

## What you can observe, and where

| Signal | Where to find it |
|---|---|
| Engagement / turnout | `cos_theta` fields; debug `citizens.plot` |
| Stated position (μ) | `mu` fields: `PolicyPref`, `PolicyAver` |
| Certainty / spread (σ) | `sigma` fields |
| Well-being (hidden) | `WellBeing` field |
| Preference alignment | derive: Pcp vs government Pge |
| Aversion alignment | derive: Pca vs government Pge |
| Government policy | government glyph / `pge.plot` |

Underlying quantities: turnout = mean `|cos θ|` across stated
Gaussians; well-being = `I(Pci, Pge)` (ideal vs enacted);
preference alignment = `I(Pcp, Pge)`; aversion alignment =
`I(Pca, Pge)`.

## Trends: what you see → what it means

Each signal below lists the trends you might observe, the code
mechanism that drives them, and how to read the result.

### 1. Turnout / engagement

- **Trends:** (a) rises then holds in a band; (b) cyclic — up in
  campaign, down in govern; (c) monotone decay to apathy;
  (d) pinned at full engagement; (e) collapses to ~0.
- **Mechanism:** campaign pushes (politician + community
  `|overlap|` × definedness × negativity_bias) raise engagement;
  the two government channels act every step; the spread-
  proportional fade (`engagement_decay_rate · σ`) pulls toward
  apathy every step.
- **Reading:** (a)/(b) are healthy — the fade prevents freezing.
  (c) ⇒ fade too strong or pushes too weak (raise
  `govt_engagement_scale` / persuasion, lower
  `engagement_decay_rate`). (d) ⇒ fade too weak. (e) ⇒ pushes far
  too weak.

### 2. Government channel balance (Δengagement during govern)

- **Trends:** net up when the government enacts opposed things;
  net down when it neglects preferences; flat.
- **Mechanism:** `drive_av` (carries negativity_bias ≈ 2) versus
  `drive_pg`; mobilization is ~2× withdrawal by construction.
- **Reading:** correlate govern-phase Δengagement with
  `I(Pca,Pge)` (more negative → more mobilization) and preference
  alignment (lower → more withdrawal). If it never mobilizes, the
  midpoint fraction is too high or the steepness too low.

### 3. Stated position drift (μ)

- **Trends:** converge toward the winning politician; herd toward
  the community average; polarize / split; backlash (Pca moves
  toward a disliked politician's *preference*); frozen.
- **Mechanism:** attraction branch (`trait_sum ≥ 0`) vs defensive
  branch (`trait_sum < 0`, targeted backlash); unconditional
  community drift; susceptibility `S = σ(1−|cos θ|)` gates
  movement — engaged citizens are frozen.
- **Reading:** μ should move *only as citizens disengage*. Frozen
  μ while engagement is high is correct early. If μ never moves
  even after disengagement, `S` is too small or no one disengages.

### 4. Spread / certainty (σ)

- **Trends:** narrowing toward `sigma_floor` (rigidity);
  broadening (open-mindedness); stable.
- **Mechanism:** the defensive branch narrows Pcp toward
  `sigma_floor` under negative trait alignment; attraction /
  community pulls σ toward the source's σ; the `sigma_floor`
  clamp.
- **Reading:** narrowing concentrated on citizens exposed to
  disliked politicians = defensive rigidity working. Universal
  collapse to the floor ⇒ defensive dynamics too aggressive
  (lower `defensive_ratio`).

### 5. Preference-alignment vs well-being GAP  ★

- **Trends:** both rise together; alignment rises while well-being
  stays flat or falls; both fall.
- **Mechanism:** Pcp moves toward politicians / government
  (conscious, manipulable); Pci is fixed (the hidden ideal); the
  government serves *stated* positions, not the ideal.
- **Reading:** the most informative signal. A widening gap means
  citizens feel served (high preference alignment) while
  objectively worse off (low well-being) — manipulation / false
  alignment. Watch this one first.

### 6. Definedness gating (who responds)

- **Trends:** sharp (narrow-σ) citizens swing engagement fast;
  vague ones barely move.
- **Mechanism:** `d = sigma_floor / σ` scales every engagement
  push.
- **Reading:** engagement volatility should track sharpness. If
  vague citizens swing as hard as sharp ones, check `d`.

## A suggested first-look checklist for smallTest

1. **Turnout shape** — settle into a band, cycle with the
   election, run to an extreme, or freeze? (§1)
2. **Movement onset** — does any stated μ move, and only after
   engagement falls? (§3, §1)
3. **The gap** — does preference alignment diverge from
   well-being over the 9 cycles? (§5)
4. **Who responds** — do sharp citizens swing more than vague
   ones? (§6)

With one citizen and two politicians, smallTest is small enough to
trace each of these by hand against the per-step values.

## Revision log

- (initial) Created alongside the engagement redesign v2 (sigmoid
  government channels, per-citizen randomized constants). Refine
  the trend list and promote the key behaviors as we read runs.
