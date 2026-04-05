"""Debug visualization of individual agent Gaussians
in policy/trait space using pyqtgraph 2-D projected
curves (DESIGN §12.6).

Each Gaussian is drawn as a 2-D curve whose vertical
amplitude equals the unit-peak bell shape multiplied
by cos(theta). This projects the 3-D rotated Gaussian
onto the real (engaged) plane. Colour saturation then
encodes |cos(theta)| — engaged agents appear in vivid
colour while apathetic agents fade toward white. The
combination of projected height and colour saturation
gives a clear, at-a-glance picture of both position
and engagement without requiring a 3-D viewport.

Colour key:
  Blue  — citizen stated pref/aver
  Green — citizen ideal policy (hidden ground truth)
  Red   — politician external pref/aver/trait
  Black — government enacted policy

The module is imported only when the -d / --debug-viz
flag is active. The simulation never depends on this
module.
"""

import time

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets


# -----------------------------------------------
# Global pyqtgraph configuration. White background
# with black foreground gives clean contrast for
# the saturation-based engagement encoding and
# keeps axis labels / tick marks legible.
# -----------------------------------------------
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


# -----------------------------------------------
# Base RGB colours for each agent type at full
# engagement (maximum saturation). These are lerped
# toward white as engagement drops.
# -----------------------------------------------
CITIZEN_RGB = (70, 130, 230)
IDEAL_RGB = (50, 170, 50)
POLITICIAN_RGB = (220, 50, 50)
GOVERNMENT_RGB = (0, 0, 0)

# Number of sample points per Gaussian curve. 80
# points gives smooth visual appearance without
# wasting cycles on invisible sub-pixel detail.
N_CURVE_POINTS = 80

# Minimum engagement floor so that fully apathetic
# agents (cos(theta) = 0) still leave a faint
# visible trace rather than vanishing into the
# white background. With this floor even a fully
# apathetic curve retains ~15 % colour saturation.
MIN_ENGAGEMENT = 0.15


# -----------------------------------------------
# Colour computation.
# -----------------------------------------------

def engagement_color(base_rgb, cos_theta):
    """Compute an RGB colour whose saturation
    reflects the engagement level |cos(theta)|.

    The base colour is linearly interpolated toward
    white as engagement drops. A floor of
    MIN_ENGAGEMENT prevents fully apathetic agents
    from becoming invisible against the white plot
    background.

    Parameters
    ----------
    base_rgb : tuple of int
        (R, G, B) base colour at full engagement.
    cos_theta : float
        Cosine of the theta angle. The absolute
        value gives the raw engagement level.

    Returns
    -------
    tuple of int
        (R, G, B) with each channel in [0, 255].
    """
    raw_engagement = abs(cos_theta)
    engagement = (
        MIN_ENGAGEMENT
        + (1.0 - MIN_ENGAGEMENT)
        * raw_engagement)
    fade = 1.0 - engagement
    r = int(
        base_rgb[0]
        + (255 - base_rgb[0]) * fade)
    g = int(
        base_rgb[1]
        + (255 - base_rgb[1]) * fade)
    b = int(
        base_rgb[2]
        + (255 - base_rgb[2]) * fade)
    return (r, g, b)


# -----------------------------------------------
# Gaussian curve computation (PSEUDOCODE §8.3).
# -----------------------------------------------

def projected_curve(mu, sigma, cos_theta,
                    n_points=N_CURVE_POINTS):
    """Compute 2-D projected Gaussian curve points.

    The bell curve amplitude is multiplied by
    cos(theta) to project the 3-D rotated Gaussian
    onto the real (engaged) plane:

        x = linspace(mu - 4*sigma, mu + 4*sigma)
        y = exp(-(x-mu)^2 / (2*sigma^2)) * cos_th

    Unit-peak normalization: the prefactor
    1/(sigma * sqrt(2*pi)) is omitted so that fully
    engaged curves peak at +/-1.0 regardless of
    sigma. Sigma is already visible as curve width
    so peak height would carry no independent
    information.

    Parameters
    ----------
    mu : float
        Centre position on the policy/trait axis.
    sigma : float
        Standard deviation (width) of the bell.
    cos_theta : float
        cos(theta) where theta is the real-valued
        angle extracted from the complex theta
        stored in Gaussian objects. Positive for
        preference curves (above the axis), negative
        for aversion curves (below the axis).
    n_points : int
        Number of sample points along the curve.

    Returns
    -------
    x, y : ndarray, each of shape (n_points,)
    """
    x = np.linspace(
        mu - 4.0 * sigma,
        mu + 4.0 * sigma,
        n_points)
    amplitude = np.exp(
        -(x - mu) ** 2 / (2.0 * sigma ** 2))
    y = amplitude * cos_theta
    return x, y


# -----------------------------------------------
# Main visualization class (PSEUDOCODE §8.1).
# -----------------------------------------------

class PolicySpaceViz:
    """Debug visualization of policy/trait space
    using pyqtgraph 2-D projected curves.

    Two-row subplot layout: the top row shows one
    PlotWidget per policy dimension, the bottom row
    shows one per trait dimension. In each subplot
    Gaussians appear as projected bell curves whose
    height encodes the engaged (real) component and
    whose colour saturation encodes the engagement
    level |cos(theta)|.

    All PlotDataItem objects are pre-created during
    __init__. Each frame update changes only the
    data arrays and pen colours via setData() and
    setPen(), avoiding per-frame widget creation
    and keeping frame rates high.

    Usage (from stodem.py):
        viz = PolicySpaceViz(world, settings)
        ...
        viz.update(step_label)   # each step
        ...
        viz.finalize()           # after loop
    """

    def __init__(self, world, settings):
        """Build the Qt window, create plot widgets
        for every dimension, and pre-allocate curve
        items for every agent Gaussian.

        Parameters
        ----------
        world : World
            Simulation world (read-only access to
            citizens, politicians, government, and
            computed data-range limits).
        settings : ScriptSettings
            Provides viz_delay and dimension counts.
        """
        self.world = world
        self.viz_delay = settings.viz_delay

        # Dimension counts from the simulation.
        self.n_policy = world.num_policy_dims
        self.n_trait = world.num_trait_dims

        # Create the Qt application (or retrieve
        #   the existing one if already running)
        #   and the main window container.
        self.app = pg.mkQApp(
            "STODEM Debug Viz")
        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle(
            "STODEM — Policy/Trait Space")
        grid_layout = QtWidgets.QGridLayout()
        self.window.setLayout(grid_layout)

        # Build a 2-row grid of PlotWidgets.
        #   Row 0: one per policy dimension.
        #   Row 1: one per trait dimension.
        n_cols = max(
            self.n_policy, self.n_trait)
        self.plots = [
            [None] * n_cols
            for _ in range(2)]

        for col in range(self.n_policy):
            plot_w = pg.PlotWidget(
                title=f"Policy Dim {col}")
            plot_w.setLabel(
                'bottom', 'Policy value')
            plot_w.setLabel(
                'left', 'Projected amplitude')
            plot_w.setXRange(
                world.policy_limits[0][col],
                world.policy_limits[1][col])
            plot_w.setYRange(-1.1, 1.1)
            # Thin grey zero-line separating
            #   prefs (above) from aversions
            #   (below).
            plot_w.addLine(
                y=0, pen=pg.mkPen(
                    color=(180, 180, 180),
                    width=1))
            grid_layout.addWidget(
                plot_w, 0, col)
            self.plots[0][col] = plot_w

        for col in range(self.n_trait):
            plot_w = pg.PlotWidget(
                title=f"Trait Dim {col}")
            plot_w.setLabel(
                'bottom', 'Trait value')
            plot_w.setLabel(
                'left', 'Projected amplitude')
            plot_w.setXRange(
                world.trait_limits[0][col],
                world.trait_limits[1][col])
            plot_w.setYRange(-1.1, 1.1)
            plot_w.addLine(
                y=0, pen=pg.mkPen(
                    color=(180, 180, 180),
                    width=1))
            grid_layout.addWidget(
                plot_w, 1, col)
            self.plots[1][col] = plot_w

        # Pre-create PlotDataItem objects for
        #   every agent Gaussian that will be
        #   rendered each frame.
        self._create_all_curves()

        # Show the window and flush an initial
        #   paint so the layout is visible before
        #   the simulation loop begins.
        self.window.show()
        self.app.processEvents()

    # -------------------------------------------
    # Curve pre-creation.
    # -------------------------------------------

    def _create_all_curves(self):
        """Pre-create one PlotDataItem per agent
        Gaussian per dimension.

        Curve items are stored in per-agent dicts
        keyed by (gaussian_type_tag, dim_index).
        Tags used:
          pp = policy pref     pa = policy aver
          ip = ideal policy    tp = trait pref
          ta = trait aver      tr = ext trait
        """
        # --- Citizens ---
        #   Per policy dim: stated pref, stated
        #   aver, ideal policy. Per trait dim:
        #   stated pref, stated aver.
        self.citizen_curves = []
        for _ in self.world.citizens:
            items = {}
            for dim in range(self.n_policy):
                pw = self.plots[0][dim]
                items[('pp', dim)] = pw.plot(
                    pen=pg.mkPen(
                        color=CITIZEN_RGB,
                        width=1))
                items[('pa', dim)] = pw.plot(
                    pen=pg.mkPen(
                        color=CITIZEN_RGB,
                        width=1))
                items[('ip', dim)] = pw.plot(
                    pen=pg.mkPen(
                        color=IDEAL_RGB,
                        width=1))
            for dim in range(self.n_trait):
                pw = self.plots[1][dim]
                items[('tp', dim)] = pw.plot(
                    pen=pg.mkPen(
                        color=CITIZEN_RGB,
                        width=1))
                items[('ta', dim)] = pw.plot(
                    pen=pg.mkPen(
                        color=CITIZEN_RGB,
                        width=1))
            self.citizen_curves.append(items)

        # --- Politicians ---
        #   Per policy dim: ext pref, ext aver.
        #   Per trait dim: ext trait.
        self.politician_curves = []
        for _ in self.world.politicians:
            items = {}
            for dim in range(self.n_policy):
                pw = self.plots[0][dim]
                items[('pp', dim)] = pw.plot(
                    pen=pg.mkPen(
                        color=POLITICIAN_RGB,
                        width=1))
                items[('pa', dim)] = pw.plot(
                    pen=pg.mkPen(
                        color=POLITICIAN_RGB,
                        width=1))
            for dim in range(self.n_trait):
                pw = self.plots[1][dim]
                items[('tr', dim)] = pw.plot(
                    pen=pg.mkPen(
                        color=POLITICIAN_RGB,
                        width=1))
            self.politician_curves.append(items)

        # --- Government ---
        #   One enacted-policy curve per policy
        #   dimension (government has no traits).
        self.government_curves = {}
        for dim in range(self.n_policy):
            pw = self.plots[0][dim]
            self.government_curves[dim] = (
                pw.plot(pen=pg.mkPen(
                    color=GOVERNMENT_RGB,
                    width=2)))

    # -------------------------------------------
    # Per-curve update helper.
    # -------------------------------------------

    def _update_curve(self, curve_item, mu,
                      sigma, theta_imag,
                      base_rgb, width):
        """Recompute one curve's projected shape and
        engagement colour, then push both into the
        pre-existing PlotDataItem.

        Parameters
        ----------
        curve_item : PlotDataItem
            Pre-created pyqtgraph curve object.
        mu : float
            Centre of the Gaussian on the policy
            or trait axis.
        sigma : float
            Standard deviation (width) of bell.
        theta_imag : float
            Imaginary part of the complex theta
            stored in the Gaussian object (the
            real-valued rotation angle in radians).
        base_rgb : tuple of int
            Base colour for this agent type.
        width : float
            Line width in pixels.
        """
        cos_theta = np.cos(theta_imag)
        x_data, y_data = projected_curve(
            mu, sigma, cos_theta)
        color = engagement_color(
            base_rgb, cos_theta)
        curve_item.setData(
            x_data, y_data,
            pen=pg.mkPen(
                color=color, width=width))

    # -------------------------------------------
    # Per-step frame update.
    # -------------------------------------------

    def update(self, step_label):
        """Refresh every curve with the current
        agent state and flush the Qt display.

        Called once per simulation step from both
        the campaign and govern loops in stodem.py.

        Parameters
        ----------
        step_label : str
            Display label for the current step
            (e.g. "Campaign  Cycle 0  Step 3").
        """
        self.window.setWindowTitle(
            f"STODEM — {step_label}")

        # --- Citizens (blue) and ideal (green) -
        for cit_idx, citizen in enumerate(
                self.world.citizens):
            items = self.citizen_curves[cit_idx]

            pref = citizen.stated_policy_pref
            aver = citizen.stated_policy_aver
            ideal = citizen.ideal_policy_pref
            for dim in range(self.n_policy):
                self._update_curve(
                    items[('pp', dim)],
                    pref.mu[dim],
                    pref.sigma[dim],
                    pref.theta[dim].imag,
                    CITIZEN_RGB, 1)
                self._update_curve(
                    items[('pa', dim)],
                    aver.mu[dim],
                    aver.sigma[dim],
                    aver.theta[dim].imag,
                    CITIZEN_RGB, 1)
                self._update_curve(
                    items[('ip', dim)],
                    ideal.mu[dim],
                    ideal.sigma[dim],
                    ideal.theta[dim].imag,
                    IDEAL_RGB, 1)

            tpref = citizen.stated_trait_pref
            taver = citizen.stated_trait_aver
            for dim in range(self.n_trait):
                self._update_curve(
                    items[('tp', dim)],
                    tpref.mu[dim],
                    tpref.sigma[dim],
                    tpref.theta[dim].imag,
                    CITIZEN_RGB, 1)
                self._update_curve(
                    items[('ta', dim)],
                    taver.mu[dim],
                    taver.sigma[dim],
                    taver.theta[dim].imag,
                    CITIZEN_RGB, 1)

        # --- Politicians (red) ---
        for pol_idx, politician in enumerate(
                self.world.politicians):
            items = (
                self.politician_curves[pol_idx])

            pref = politician.ext_policy_pref
            aver = politician.ext_policy_aver
            for dim in range(self.n_policy):
                self._update_curve(
                    items[('pp', dim)],
                    pref.mu[dim],
                    pref.sigma[dim],
                    pref.theta[dim].imag,
                    POLITICIAN_RGB, 1)
                self._update_curve(
                    items[('pa', dim)],
                    aver.mu[dim],
                    aver.sigma[dim],
                    aver.theta[dim].imag,
                    POLITICIAN_RGB, 1)

            trait = politician.ext_trait
            for dim in range(self.n_trait):
                self._update_curve(
                    items[('tr', dim)],
                    trait.mu[dim],
                    trait.sigma[dim],
                    trait.theta[dim].imag,
                    POLITICIAN_RGB, 1)

        # --- Government (black, policy only) ---
        enacted = (
            self.world.government.enacted_policy)
        for dim in range(self.n_policy):
            self._update_curve(
                self.government_curves[dim],
                enacted.mu[dim],
                enacted.sigma[dim],
                enacted.theta[dim].imag,
                GOVERNMENT_RGB, 2)

        # Flush the Qt event loop to paint the
        #   updated curves. Then throttle by
        #   looping on processEvents() until the
        #   delay elapses — this keeps the window
        #   responsive during the wait instead of
        #   blocking on a single sleep() call.
        self.app.processEvents()
        if self.viz_delay > 0:
            deadline = (
                time.time() + self.viz_delay)
            while time.time() < deadline:
                self.app.processEvents()
                time.sleep(0.01)

    # -------------------------------------------
    # Finalize — no-op for interactive mode.
    # -------------------------------------------

    def finalize(self):
        """No-op — retained for interface
        consistency with the stodem.py call site.
        """
        pass
