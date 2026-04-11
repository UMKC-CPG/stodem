"""Debug visualization of individual agent Gaussians
in policy/trait space using pyqtgraph 2-D projected
curves (DESIGN §12.6).

Each dimension has two subplots. The real (cos θ)
subplot projects onto the engaged plane; the
imaginary (sin θ) subplot shows the complementary
latent/apathetic component. Colour saturation
encodes |cos(theta)| in both views so that a given
agent always appears in the same colour across
projections.

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
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


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

def projected_curve(mu, sigma, proj_factor,
                    n_points=N_CURVE_POINTS):
    """Compute 2-D projected Gaussian curve points.

    The bell curve amplitude is multiplied by a
    projection factor — cos(theta) for the real
    subplot, sin(theta) for the imaginary subplot:

        x = linspace(mu - 4*sigma, mu + 4*sigma)
        y = amplitude * proj_factor

    Unit-peak normalization: the prefactor
    1/(sigma * sqrt(2*pi)) is omitted so that
    fully engaged curves peak at +/-1.0 regardless
    of sigma. Sigma is already visible as curve
    width so peak height would carry no independent
    information.

    Parameters
    ----------
    mu : float
        Centre position on the policy/trait axis.
    sigma : float
        Standard deviation (width) of the bell.
    proj_factor : float
        Projection multiplier applied to the bell
        amplitude. Pass cos(theta) for the real
        (engaged) subplot or sin(theta) for the
        imaginary (latent) subplot.
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
    y = amplitude * proj_factor
    return x, y


# -----------------------------------------------
# Main visualization class (PSEUDOCODE §8.1).
# -----------------------------------------------

class PolicySpaceViz:
    """Debug visualization of policy/trait space
    using pyqtgraph 2-D projected curves.

    Four-row subplot layout: each policy and trait
    dimension has a real (cos θ) subplot and an
    imaginary (sin θ) subplot stacked vertically.
    Curve height encodes the projection factor and
    colour saturation encodes |cos(theta)| in both
    views, keeping agent identity consistent.

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
        self._grid_layout = grid_layout

        # Build a 4-row grid of PlotWidgets.
        #   Row 0: policy real  (cos θ)
        #   Row 1: policy imag  (sin θ)
        #   Row 2: trait real   (cos θ)
        #   Row 3: trait imag   (sin θ)
        n_cols = max(
            self.n_policy, self.n_trait)
        self.plots = [
            [None] * n_cols
            for _ in range(4)]

        grey_pen = pg.mkPen(
            color=(180, 180, 180), width=1)

        for col in range(self.n_policy):
            # Real (cos θ) subplot.
            pw = pg.PlotWidget(
                title=(f"Policy Dim {col}"
                       f" (cos \u03b8)"))
            pw.setLabel(
                'bottom', 'Policy value')
            pw.setLabel(
                'left', 'Projected amplitude')
            pw.setXRange(
                world.policy_limits[0][col],
                world.policy_limits[1][col])
            pw.setYRange(-1.1, 1.1)
            pw.addLine(y=0, pen=grey_pen)
            grid_layout.addWidget(pw, 0, col)
            self.plots[0][col] = pw

            # Imaginary (sin θ) subplot.
            pw_i = pg.PlotWidget(
                title=(f"Policy Dim {col}"
                       f" (sin \u03b8)"))
            pw_i.setLabel(
                'bottom', 'Policy value')
            pw_i.setLabel(
                'left', 'Projected amplitude')
            pw_i.setXRange(
                world.policy_limits[0][col],
                world.policy_limits[1][col])
            pw_i.setYRange(0, 1.1)
            grid_layout.addWidget(
                pw_i, 1, col)
            self.plots[1][col] = pw_i

        for col in range(self.n_trait):
            # Real (cos θ) subplot.
            pw = pg.PlotWidget(
                title=(f"Trait Dim {col}"
                       f" (cos \u03b8)"))
            pw.setLabel(
                'bottom', 'Trait value')
            pw.setLabel(
                'left', 'Projected amplitude')
            pw.setXRange(
                world.trait_limits[0][col],
                world.trait_limits[1][col])
            pw.setYRange(-1.1, 1.1)
            pw.addLine(y=0, pen=grey_pen)
            grid_layout.addWidget(pw, 2, col)
            self.plots[2][col] = pw

            # Imaginary (sin θ) subplot.
            pw_i = pg.PlotWidget(
                title=(f"Trait Dim {col}"
                       f" (sin \u03b8)"))
            pw_i.setLabel(
                'bottom', 'Trait value')
            pw_i.setLabel(
                'left', 'Projected amplitude')
            pw_i.setXRange(
                world.trait_limits[0][col],
                world.trait_limits[1][col])
            pw_i.setYRange(0, 1.1)
            grid_layout.addWidget(
                pw_i, 3, col)
            self.plots[3][col] = pw_i

        # Pre-create PlotDataItem objects for
        #   every agent Gaussian that will be
        #   rendered each frame.
        self._create_all_curves()

        # Build a colour/style legend in the top-left policy subplot.
        # Four invisible reference curves — one per agent type — carry
        # the base colour and line width at full engagement, giving the
        # developer vivid anchors to identify faded curves at a glance.
        legend_entries = [
            ("Citizen pref/aver", CITIZEN_RGB, 1),
            ("Ideal policy", IDEAL_RGB, 1),
            ("Politician ext", POLITICIAN_RGB, 1),
            ("Government enacted", GOVERNMENT_RGB, 2),
        ]
        legend_host = self.plots[0][0]
        legend_host.addLegend(offset=(10, 10))
        for label, rgb, width in legend_entries:
            legend_host.plot(
                [], [],
                pen=pg.mkPen(
                    color=rgb, width=width),
                name=label)

        # Show the window and flush an initial
        #   paint so the layout is visible before
        #   the simulation loop begins.
        self.window.show()
        self.app.processEvents()

        # Frame recording list for post-run
        # replay. Each entry is a dict mapping
        # curve identity keys to (mu, sigma,
        # theta_imag) triples (§8.7).
        self.frames = []

    # -------------------------------------------
    # Curve pre-creation.
    # -------------------------------------------

    def _create_all_curves(self):
        """Pre-create one PlotDataItem per agent
        Gaussian per dimension in both the real
        and imaginary subplots.

        Curve items are stored in per-agent dicts
        keyed by (gaussian_type_tag, dim_index).
        Tags ending in '_i' are imaginary items.
        Tags used:
          pp / pp_i = policy pref
          pa / pa_i = policy aver
          ip / ip_i = ideal policy
          tp / tp_i = trait pref
          ta / ta_i = trait aver
          tr / tr_i = ext trait
        """
        # --- Citizens ---
        #   Per policy dim: stated pref, stated aver, ideal policy
        #   (real + imag). Per trait dim: stated pref, stated aver
        #   (real + imag).
        self.citizen_curves = []
        for _ in self.world.citizens:
            items = {}
            for dim in range(self.n_policy):
                pw = self.plots[0][dim]
                pw_i = self.plots[1][dim]
                items[('pp', dim)] = pw.plot(
                    pen=pg.mkPen(color=CITIZEN_RGB, width=1))
                items[('pp_i', dim)] = pw_i.plot(
                    pen=pg.mkPen(color=CITIZEN_RGB, width=1))
                items[('pa', dim)] = pw.plot(
                    pen=pg.mkPen(color=CITIZEN_RGB, width=1))
                items[('pa_i', dim)] = pw_i.plot(
                    pen=pg.mkPen(color=CITIZEN_RGB, width=1))
                items[('ip', dim)] = pw.plot(
                    pen=pg.mkPen(color=IDEAL_RGB, width=1))
                items[('ip_i', dim)] = pw_i.plot(
                    pen=pg.mkPen(color=IDEAL_RGB, width=1))
            for dim in range(self.n_trait):
                pw = self.plots[2][dim]
                pw_i = self.plots[3][dim]
                items[('tp', dim)] = pw.plot(
                    pen=pg.mkPen(color=CITIZEN_RGB, width=1))
                items[('tp_i', dim)] = pw_i.plot(
                    pen=pg.mkPen(color=CITIZEN_RGB, width=1))
                items[('ta', dim)] = pw.plot(
                    pen=pg.mkPen(color=CITIZEN_RGB, width=1))
                items[('ta_i', dim)] = pw_i.plot(
                    pen=pg.mkPen(color=CITIZEN_RGB, width=1))
            self.citizen_curves.append(items)

        # --- Politicians ---
        #   Per policy dim: ext pref, ext aver (real + imag).
        #   Per trait dim: ext trait (real + imag).
        self.politician_curves = []
        for _ in self.world.politicians:
            items = {}
            for dim in range(self.n_policy):
                pw = self.plots[0][dim]
                pw_i = self.plots[1][dim]
                items[('pp', dim)] = pw.plot(
                    pen=pg.mkPen(color=POLITICIAN_RGB, width=1))
                items[('pp_i', dim)] = pw_i.plot(
                    pen=pg.mkPen(color=POLITICIAN_RGB, width=1))
                items[('pa', dim)] = pw.plot(
                    pen=pg.mkPen(color=POLITICIAN_RGB, width=1))
                items[('pa_i', dim)] = pw_i.plot(
                    pen=pg.mkPen(color=POLITICIAN_RGB, width=1))
            for dim in range(self.n_trait):
                pw = self.plots[2][dim]
                pw_i = self.plots[3][dim]
                items[('tr', dim)] = pw.plot(
                    pen=pg.mkPen(color=POLITICIAN_RGB, width=1))
                items[('tr_i', dim)] = pw_i.plot(
                    pen=pg.mkPen(color=POLITICIAN_RGB, width=1))
            self.politician_curves.append(items)

        # --- Government ---
        #   One enacted-policy curve per policy dimension
        #   (real + imag). Government has no traits.
        self.government_curves = {}
        for dim in range(self.n_policy):
            pw = self.plots[0][dim]
            pw_i = self.plots[1][dim]
            self.government_curves[('ge', dim)] = pw.plot(
                pen=pg.mkPen(color=GOVERNMENT_RGB, width=2))
            self.government_curves[('ge_i', dim)] = pw_i.plot(
                pen=pg.mkPen(color=GOVERNMENT_RGB, width=2))

    # -------------------------------------------
    # Per-curve update helper.
    # -------------------------------------------

    def _update_curve(self, curve_item, mu,
                      sigma, proj_factor,
                      cos_theta, base_rgb,
                      width):
        """Recompute one curve's projected shape
        and engagement colour, then push both
        into the pre-existing PlotDataItem.

        Parameters
        ----------
        curve_item : PlotDataItem
            Pre-created pyqtgraph curve object.
        mu : float
            Centre of the Gaussian on the policy
            or trait axis.
        sigma : float
            Standard deviation (width) of bell.
        proj_factor : float
            Projection multiplier: cos(theta) for
            real subplots, sin(theta) for
            imaginary subplots.
        cos_theta : float
            cos(theta), used only for engagement
            colour — the same in both views so
            agent identity is consistent.
        base_rgb : tuple of int
            Base colour for this agent type.
        width : float
            Line width in pixels.
        """
        x_data, y_data = projected_curve(
            mu, sigma, proj_factor)
        color = engagement_color(
            base_rgb, cos_theta)
        curve_item.setData(
            x_data, y_data,
            pen=pg.mkPen(
                color=color, width=width))

    def _update_pair(self, real_item, imag_item,
                     mu, sigma, theta_imag,
                     base_rgb, width):
        """Update both the real and imaginary curve
        items for a single Gaussian. Computes
        cos(theta) and sin(theta) once and calls
        _update_curve() twice.

        Parameters
        ----------
        real_item : PlotDataItem
            Curve in the real (cos θ) subplot.
        imag_item : PlotDataItem
            Curve in the imaginary (sin θ) subplot.
        mu, sigma, theta_imag, base_rgb, width :
            Same as _update_curve; theta_imag is
            the imaginary part of the complex theta
            (real-valued angle in radians).
        """
        cos_th = np.cos(theta_imag)
        sin_th = np.sin(theta_imag)
        self._update_curve(
            real_item, mu, sigma,
            cos_th, cos_th, base_rgb, width)
        self._update_curve(
            imag_item, mu, sigma,
            sin_th, cos_th, base_rgb, width)

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
                self._update_pair(items[('pp', dim)], items[('pp_i', dim)],
                    pref.mu[dim], pref.sigma[dim], pref.theta[dim].imag,
                    CITIZEN_RGB, 1)
                self._update_pair(items[('pa', dim)], items[('pa_i', dim)],
                    aver.mu[dim], aver.sigma[dim], aver.theta[dim].imag,
                    CITIZEN_RGB, 1)
                self._update_pair(items[('ip', dim)], items[('ip_i', dim)],
                    ideal.mu[dim], ideal.sigma[dim], ideal.theta[dim].imag,
                    IDEAL_RGB, 1)

            tpref = citizen.stated_trait_pref
            taver = citizen.stated_trait_aver
            for dim in range(self.n_trait):
                self._update_pair(items[('tp', dim)], items[('tp_i', dim)],
                    tpref.mu[dim], tpref.sigma[dim], tpref.theta[dim].imag,
                    CITIZEN_RGB, 1)
                self._update_pair(items[('ta', dim)], items[('ta_i', dim)],
                    taver.mu[dim], taver.sigma[dim], taver.theta[dim].imag,
                    CITIZEN_RGB, 1)

        # --- Politicians (red) ---
        for pol_idx, politician in enumerate(
                self.world.politicians):
            items = self.politician_curves[pol_idx]

            pref = politician.ext_policy_pref
            aver = politician.ext_policy_aver
            for dim in range(self.n_policy):
                self._update_pair(items[('pp', dim)], items[('pp_i', dim)],
                    pref.mu[dim], pref.sigma[dim], pref.theta[dim].imag,
                    POLITICIAN_RGB, 1)
                self._update_pair(items[('pa', dim)], items[('pa_i', dim)],
                    aver.mu[dim], aver.sigma[dim], aver.theta[dim].imag,
                    POLITICIAN_RGB, 1)

            trait = politician.ext_trait
            for dim in range(self.n_trait):
                self._update_pair(items[('tr', dim)], items[('tr_i', dim)],
                    trait.mu[dim], trait.sigma[dim], trait.theta[dim].imag,
                    POLITICIAN_RGB, 1)

        # --- Government (black, policy only) ---
        enacted = self.world.government.enacted_policy
        for dim in range(self.n_policy):
            self._update_pair(self.government_curves[('ge', dim)],
                self.government_curves[('ge_i', dim)],
                enacted.mu[dim], enacted.sigma[dim],
                enacted.theta[dim].imag, GOVERNMENT_RGB, 2)

        # Record a snapshot of the current state
        # for post-run replay (§8.7).
        self._record_frame(step_label)

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
    # Frame recording (PSEUDOCODE §8.7).
    # -------------------------------------------

    def _record_frame(self, step_label):
        """Record a compact snapshot of the
        current rendering state for post-run
        replay.

        Each snapshot stores the step label and
        the (mu, sigma, theta_imag) triple for
        every agent Gaussian — just enough to
        reproduce the display through the same
        _update_curve() path used during the live
        simulation run.

        Parameters
        ----------
        step_label : str
            Simulation phase and step string
            shown in the title bar during replay.
        """
        snapshot = {"label": step_label}

        for cit_idx, citizen in enumerate(
                self.world.citizens):
            pref = citizen.stated_policy_pref
            aver = citizen.stated_policy_aver
            ideal = citizen.ideal_policy_pref
            for dim in range(self.n_policy):
                snapshot[("cit", cit_idx, "pp", dim)] = (pref.mu[dim],
                    pref.sigma[dim], pref.theta[dim].imag)
                snapshot[("cit", cit_idx, "pa", dim)] = (aver.mu[dim],
                    aver.sigma[dim], aver.theta[dim].imag)
                snapshot[("cit", cit_idx, "ip", dim)] = (ideal.mu[dim],
                    ideal.sigma[dim], ideal.theta[dim].imag)

            tpref = citizen.stated_trait_pref
            taver = citizen.stated_trait_aver
            for dim in range(self.n_trait):
                snapshot[("cit", cit_idx, "tp", dim)] = (tpref.mu[dim],
                    tpref.sigma[dim], tpref.theta[dim].imag)
                snapshot[("cit", cit_idx, "ta", dim)] = (taver.mu[dim],
                    taver.sigma[dim], taver.theta[dim].imag)

        for pol_idx, politician in enumerate(
                self.world.politicians):
            pref = politician.ext_policy_pref
            aver = politician.ext_policy_aver
            for dim in range(self.n_policy):
                snapshot[("pol", pol_idx, "pp", dim)] = (pref.mu[dim],
                    pref.sigma[dim], pref.theta[dim].imag)
                snapshot[("pol", pol_idx, "pa", dim)] = (aver.mu[dim],
                    aver.sigma[dim], aver.theta[dim].imag)

            trait = politician.ext_trait
            for dim in range(self.n_trait):
                snapshot[("pol", pol_idx, "tr", dim)] = (trait.mu[dim],
                    trait.sigma[dim], trait.theta[dim].imag)

        enacted = self.world.government.enacted_policy
        for dim in range(self.n_policy):
            snapshot[("gov", 0, "ge", dim)] = (enacted.mu[dim],
                enacted.sigma[dim], enacted.theta[dim].imag)

        self.frames.append(snapshot)

    # -------------------------------------------
    # Replay mode (PSEUDOCODE §8.8).
    # -------------------------------------------

    def finalize(self):
        """Transition to interactive replay mode.

        After the simulation loop completes, this
        method builds a transport control bar with
        play / pause, step, reverse, scrub, and
        speed controls below the existing plot
        grid. It then enters the Qt event loop so
        the developer can explore the recorded
        simulation history interactively. All
        rendering goes through _update_curve() to
        ensure visual consistency between live
        and replayed frames.
        """
        if not self.frames:
            return

        # --- Replay state -----------------------
        self._current_frame = 0
        self._playing = False
        self._play_direction = 1
        self._speed_mult = 1.0
        # Base timer interval in milliseconds.
        # A floor of 50 ms prevents a runaway
        # timer when viz_delay is set to zero.
        self._base_interval = max(
            int(self.viz_delay * 1000), 50)

        # --- Transport control bar --------------
        # Horizontal row of buttons, a scrubber
        # slider, and status labels. Inserted at
        # grid row 4 below the four plot rows.
        control_bar = QtWidgets.QHBoxLayout()

        back_btn = (
            QtWidgets.QPushButton("<<"))
        self._play_btn = (
            QtWidgets.QPushButton("Play"))
        fwd_btn = (
            QtWidgets.QPushButton(">>"))
        rev_btn = (
            QtWidgets.QPushButton("Rev"))

        self._scrubber = QtWidgets.QSlider(
            QtCore.Qt.Horizontal)
        self._scrubber.setMinimum(0)
        self._scrubber.setMaximum(
            len(self.frames) - 1)
        self._scrubber.setValue(0)

        self._speed_label = (
            QtWidgets.QLabel("1×"))
        self._frame_label = (
            QtWidgets.QLabel(""))

        # Lock the speed and frame labels to
        # fixed widths so the scrubber slider
        # never shifts as digit counts change.
        # Widths are computed from the widest
        # plausible strings using the label's
        # own font metrics.
        speed_metrics = (
            self._speed_label.fontMetrics())
        speed_width = (
            speed_metrics.horizontalAdvance(
                "9999×") + 12)
        self._speed_label.setFixedWidth(
            speed_width)

        frame_metrics = (
            self._frame_label.fontMetrics())
        frame_width = (
            frame_metrics.horizontalAdvance(
                "(C) Cycle 999  Step 9999"
                "  [Frame 99999 / 99999]")
            + 12)
        self._frame_label.setFixedWidth(
            frame_width)

        control_bar.addWidget(back_btn)
        control_bar.addWidget(self._play_btn)
        control_bar.addWidget(fwd_btn)
        control_bar.addWidget(rev_btn)
        control_bar.addWidget(self._scrubber)
        control_bar.addWidget(
            self._speed_label)
        control_bar.addWidget(
            self._frame_label)

        # Insert the control bar spanning all
        # columns below the four plot rows.
        n_cols = max(
            self.n_policy, self.n_trait)
        self._grid_layout.addLayout(
            control_bar, 4, 0, 1, n_cols)

        # --- Auto-play timer --------------------
        self._timer = QtCore.QTimer()
        self._timer.setInterval(
            self._base_interval)
        self._timer.timeout.connect(
            self._on_timer_tick)

        # --- Button signal connections ----------
        back_btn.clicked.connect(
            self._step_backward)
        self._play_btn.clicked.connect(
            self._toggle_play)
        fwd_btn.clicked.connect(
            self._step_forward)
        rev_btn.clicked.connect(
            self._reverse_play)
        self._scrubber.valueChanged.connect(
            self._on_scrubber_change)

        # --- Keyboard shortcuts -----------------
        self._bind_shortcuts()

        # Render the first recorded frame and
        # enter the Qt event loop. The window
        # remains open for interactive replay
        # until the developer closes it.
        self._render_frame(0)
        self.app.exec_()

    # -------------------------------------------
    # Render recorded frame (PSEUDOCODE §8.9).
    # -------------------------------------------

    def _render_frame(self, frame_index):
        """Render a single recorded frame by
        reading stored (mu, sigma, theta_imag)
        triples and calling _update_pair() for
        each Gaussian's real and imaginary curves.
        Uses the identical rendering path as the
        live display to ensure visual consistency
        between live and replayed output.

        Parameters
        ----------
        frame_index : int
            Zero-based index into self.frames.
        """
        snapshot = self.frames[frame_index]
        total = len(self.frames)
        title = (
            f"STODEM — {snapshot['label']}"
            f"  [Frame {frame_index + 1}"
            f" / {total}]")
        self.window.setWindowTitle(title)
        self._frame_label.setText(title)

        # --- Citizens (blue / green) -----------
        for cit_idx in range(len(self.world.citizens)):
            items = self.citizen_curves[cit_idx]
            for dim in range(self.n_policy):
                mu, sig, th = snapshot[("cit", cit_idx, "pp", dim)]
                self._update_pair(items[('pp', dim)], items[('pp_i', dim)],
                    mu, sig, th, CITIZEN_RGB, 1)
                mu, sig, th = snapshot[("cit", cit_idx, "pa", dim)]
                self._update_pair(items[('pa', dim)], items[('pa_i', dim)],
                    mu, sig, th, CITIZEN_RGB, 1)
                mu, sig, th = snapshot[("cit", cit_idx, "ip", dim)]
                self._update_pair(items[('ip', dim)], items[('ip_i', dim)],
                    mu, sig, th, IDEAL_RGB, 1)
            for dim in range(self.n_trait):
                mu, sig, th = snapshot[("cit", cit_idx, "tp", dim)]
                self._update_pair(items[('tp', dim)], items[('tp_i', dim)],
                    mu, sig, th, CITIZEN_RGB, 1)
                mu, sig, th = snapshot[("cit", cit_idx, "ta", dim)]
                self._update_pair(items[('ta', dim)], items[('ta_i', dim)],
                    mu, sig, th, CITIZEN_RGB, 1)

        # --- Politicians (red) -----------------
        for pol_idx in range(len(self.world.politicians)):
            items = self.politician_curves[pol_idx]
            for dim in range(self.n_policy):
                mu, sig, th = snapshot[("pol", pol_idx, "pp", dim)]
                self._update_pair(items[('pp', dim)], items[('pp_i', dim)],
                    mu, sig, th, POLITICIAN_RGB, 1)
                mu, sig, th = snapshot[("pol", pol_idx, "pa", dim)]
                self._update_pair(items[('pa', dim)], items[('pa_i', dim)],
                    mu, sig, th, POLITICIAN_RGB, 1)
            for dim in range(self.n_trait):
                mu, sig, th = snapshot[("pol", pol_idx, "tr", dim)]
                self._update_pair(items[('tr', dim)], items[('tr_i', dim)],
                    mu, sig, th, POLITICIAN_RGB, 1)

        # --- Government (black, policy only) ---
        for dim in range(self.n_policy):
            mu, sig, th = snapshot[("gov", 0, "ge", dim)]
            self._update_pair(self.government_curves[('ge', dim)],
                self.government_curves[('ge_i', dim)], mu, sig, th,
                GOVERNMENT_RGB, 2)

        self.app.processEvents()

    # -------------------------------------------
    # Keyboard shortcut setup.
    # -------------------------------------------

    def _bind_shortcuts(self):
        """Bind keyboard shortcuts for the replay
        transport controls to the main window.

        DESIGN §12.6 shortcut table: Space
        toggles play/pause, Left/Right step one
        frame, Up/Down adjust speed, R starts
        reverse playback, Home/End jump to the
        first and last recorded frames.
        """
        def bind(key, slot):
            """Create a QShortcut on the window
            that fires the given callback."""
            QtWidgets.QShortcut(
                QtGui.QKeySequence(key),
                self.window
            ).activated.connect(slot)

        bind(QtCore.Qt.Key_Space,
             self._toggle_play)
        bind(QtCore.Qt.Key_Left,
             self._step_backward)
        bind(QtCore.Qt.Key_Right,
             self._step_forward)
        bind(QtCore.Qt.Key_R,
             self._reverse_play)
        bind(QtCore.Qt.Key_Home,
             self._jump_to_start)
        bind(QtCore.Qt.Key_End,
             self._jump_to_end)
        bind(QtCore.Qt.Key_Up,
             self._speed_up)
        bind(QtCore.Qt.Key_Down,
             self._slow_down)

    # -------------------------------------------
    # Replay transport callbacks.
    # -------------------------------------------

    def _on_timer_tick(self):
        """Advance or reverse one frame per timer
        tick. Stops playback automatically when
        the first or last frame is reached."""
        next_frame = (
            self._current_frame
            + self._play_direction)
        if next_frame >= len(self.frames):
            self._current_frame = (
                len(self.frames) - 1)
            self._stop_playback()
        elif next_frame < 0:
            self._current_frame = 0
            self._stop_playback()
        else:
            self._current_frame = next_frame
        self._render_frame(
            self._current_frame)
        self._set_scrubber_silent(
            self._current_frame)

    def _toggle_play(self):
        """Toggle between play and pause. Resumes
        in whichever direction (forward or reverse)
        was last active."""
        if self._playing:
            self._stop_playback()
        else:
            self._playing = True
            self._play_btn.setText("Pause")
            self._timer.start()

    def _stop_playback(self):
        """Halt auto-play and reset the play
        button label."""
        self._playing = False
        self._timer.stop()
        self._play_btn.setText("Play")

    def _step_forward(self):
        """Stop playback and advance one frame
        forward."""
        self._stop_playback()
        if (self._current_frame
                < len(self.frames) - 1):
            self._current_frame += 1
        self._render_frame(
            self._current_frame)
        self._set_scrubber_silent(
            self._current_frame)

    def _step_backward(self):
        """Stop playback and step one frame
        backward."""
        self._stop_playback()
        if self._current_frame > 0:
            self._current_frame -= 1
        self._render_frame(
            self._current_frame)
        self._set_scrubber_silent(
            self._current_frame)

    def _reverse_play(self):
        """Start reverse playback. Always sets
        direction to -1 regardless of the current
        play state (DESIGN §12.6)."""
        self._play_direction = -1
        self._playing = True
        self._play_btn.setText("Pause")
        self._timer.start()

    def _speed_up(self):
        """Double the playback speed by halving
        the timer interval. Updates the speed
        label to show the current multiplier."""
        self._speed_mult *= 2.0
        self._timer.setInterval(int(
            self._base_interval
            / self._speed_mult))
        self._speed_label.setText(
            f"{self._speed_mult:.4g}×")

    def _slow_down(self):
        """Halve the playback speed by doubling
        the timer interval. Updates the speed
        label to show the current multiplier."""
        self._speed_mult /= 2.0
        self._timer.setInterval(int(
            self._base_interval
            / self._speed_mult))
        self._speed_label.setText(
            f"{self._speed_mult:.4g}×")

    def _on_scrubber_change(self, value):
        """Handle user-driven scrubber movement.
        Stops playback and jumps directly to the
        selected frame."""
        self._stop_playback()
        self._current_frame = value
        self._render_frame(
            self._current_frame)

    def _set_scrubber_silent(self, value):
        """Move the scrubber thumb without firing
        the valueChanged signal. Blocks signals
        during the update to prevent a feedback
        loop that would immediately stop playback
        by triggering _on_scrubber_change."""
        self._scrubber.blockSignals(True)
        self._scrubber.setValue(value)
        self._scrubber.blockSignals(False)

    def _jump_to_start(self):
        """Jump to the first recorded frame and
        stop any active playback."""
        self._stop_playback()
        self._current_frame = 0
        self._render_frame(0)
        self._set_scrubber_silent(0)

    def _jump_to_end(self):
        """Jump to the last recorded frame and
        stop any active playback."""
        self._stop_playback()
        self._current_frame = (
            len(self.frames) - 1)
        self._render_frame(
            self._current_frame)
        self._set_scrubber_silent(
            self._current_frame)
