class Diagnostics():
    """Manages three columnar diagnostic log files
    written alongside the HDF5/XDMF output.

    Each file has a header row and one space-separated
    data row per simulation step. Floats use scientific
    notation (%.6e). Files are flushed after every row
    so partial results survive crashes.

    Files produced
    --------------
    {prefix}.pge.plot
        Government enacted policy trajectory:
        Pge.mu and Pge.sigma per policy dimension.

    {prefix}.forces.plot
        Govern-phase force balance: per-dimension
        preference-attraction and aversion-repulsion
        forces on Pge.mu and Pge.sigma, plus the
        would-be natural spread. Campaign rows are
        all zeros (phase column distinguishes them).

    {prefix}.citizens.plot
        Full Gaussian state for a small sample of
        citizens: mu, sigma, Im(theta) for each of
        the 5 Gaussians, plus well_being,
        policy_trait_ratio, and participation_prob.

    Attributes
    ----------
    sample_indices : list of int
        Indices into world.citizens for the tracked
        sample citizens.
    pge_file : file handle
        Open handle for the Pge trajectory file.
    forces_file : file handle
        Open handle for the force balance file.
    citizens_file : file handle
        Open handle for the citizen tracking file.
    """

    def __init__(self, settings, world, sample_indices):
        """Open the three diagnostic files and
        write their header rows.

        Parameters
        ----------
        settings : ScriptSettings
            Provides the output filename prefix.
        world : World
            Provides dimension counts.
        sample_indices : list of int
            Indices into world.citizens to track.
        """
        self.sample_indices = sample_indices
        n_pol = world.num_policy_dims
        n_tr = world.num_trait_dims
        prefix = settings.outfile

        self.pge_file = open(f"pge.plot", "w")
        self._write_pge_header(n_pol)

        self.forces_file = open(f"forces.plot", "w")
        self._write_forces_header(n_pol)

        self.citizens_file = open(f"citizens.plot", "w")
        self._write_citizens_header(n_pol, n_tr)


    def _write_pge_header(self, n_pol):
        """Write the header for the Pge file."""
        cols = ["step", "cycle", "phase"]
        for d in range(n_pol):
            cols.append(f"Pge_mu_{d}")
        for d in range(n_pol):
            cols.append(f"Pge_sig_{d}")
        self.pge_file.write(" ".join(cols) + "\n")


    def _write_forces_header(self, n_pol):
        """Write the header for the forces file."""
        cols = ["step", "cycle", "phase"]
        for label in ["pref_fmu", "aver_fmu", "pref_fsig", "aver_fsig"]:
            for d in range(n_pol):
                cols.append(f"{label}_{d}")
            cols.append(f"{label}_sum")
        for d in range(n_pol):
            cols.append(f"spread_{d}")
        self.forces_file.write(" ".join(cols) + "\n")


    def _write_citizens_header(self, n_pol, n_tr):
        """Write the header for the citizens file."""
        cols = ["step", "cycle", "phase"]
        for k in range(len(self.sample_indices)):
            for gname in ["Pcp", "Pca", "Pci"]:
                for param in ["mu", "sig", "th"]:
                    for d in range(n_pol):
                        cols.append(f"c{k}_{gname}_{param}_{d}")
            for gname in ["Tcp", "Tca"]:
                for param in ["mu", "sig", "th"]:
                    for d in range(n_tr):
                        cols.append(f"c{k}_{gname}_{param}_{d}")
            cols.append(f"c{k}_wb")
            cols.append(f"c{k}_ptr")
            cols.append(f"c{k}_pp")
        self.citizens_file.write(" ".join(cols) + "\n")


    def log_step(self, step, cycle, phase, world, forces):
        """Write one row to each diagnostic file.

        Parameters
        ----------
        step : int
            Current sim_control.curr_step.
        cycle : int
            Current cycle index.
        phase : int
            0 = campaign, 1 = govern.
        world : World
            Provides government and citizens.
        forces : tuple or None
            For govern steps: (pref_force_mu,
            aver_force_mu, pref_force_sigma,
            aver_force_sigma, spread_per_dim).
            For campaign steps: None (all force
            columns written as 0.0).
        """
        self._log_pge(step, cycle, phase, world)
        self._log_forces(step, cycle, phase, world, forces)
        self._log_citizens(step, cycle, phase, world)


    def _log_pge(self, step, cycle, phase, world):
        """Write one Pge row."""
        Pge = world.government.enacted_policy
        vals = [str(step), str(cycle), str(phase)]
        for v in Pge.mu:
            vals.append(f"{v:.6e}")
        for v in Pge.sigma:
            vals.append(f"{v:.6e}")
        self.pge_file.write(" ".join(vals) + "\n")
        self.pge_file.flush()


    def _log_forces(self, step, cycle, phase, world, forces):
        """Write one forces row."""
        n = world.num_policy_dims
        vals = [str(step), str(cycle), str(phase)]
        if forces is None:
            n_data = 4 * (n + 1) + n
            for _ in range(n_data):
                vals.append(f"{0.0:.6e}")
        else:
            (pfmu, afmu, pfsig, afsig, spread) = forces
            for arr in [pfmu, afmu, pfsig, afsig]:
                for v in arr:
                    vals.append(f"{v:.6e}")
                vals.append(f"{sum(arr):.6e}")
            for v in spread:
                vals.append(f"{v:.6e}")
        self.forces_file.write(" ".join(vals) + "\n")
        self.forces_file.flush()


    def _log_citizens(self, step, cycle, phase, world):
        """Write one citizens row."""
        vals = [str(step), str(cycle), str(phase)]
        for idx in self.sample_indices:
            cit = world.citizens[idx]
            cit.compute_vote_probability()

            # Policy Gaussians: Pcp, Pca, Pci
            for g in [cit.stated_policy_pref,
                      cit.stated_policy_aver,
                      cit.ideal_policy_pref]:
                for v in g.mu:
                    vals.append(f"{v:.6e}")
                for v in g.sigma:
                    vals.append(f"{v:.6e}")
                for v in g.theta:
                    vals.append(f"{v.imag:.6e}")

            # Trait Gaussians: Tcp, Tca
            for g in [cit.stated_trait_pref,
                      cit.stated_trait_aver]:
                for v in g.mu:
                    vals.append(f"{v:.6e}")
                for v in g.sigma:
                    vals.append(f"{v:.6e}")
                for v in g.theta:
                    vals.append(f"{v.imag:.6e}")

            # Scalars
            wb = cit.well_being
            ptr = cit.policy_trait_ratio
            pp = cit.participation_prob
            vals.append(f"{wb:.6e}")
            vals.append(f"{ptr:.6e}")
            vals.append(f"{pp:.6e}")
        self.citizens_file.write(" ".join(vals) + "\n")
        self.citizens_file.flush()


    def close(self):
        """Close all three diagnostic files."""
        self.pge_file.close()
        self.forces_file.close()
        self.citizens_file.close()
