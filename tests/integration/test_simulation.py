"""Integration / smoke tests for the simulation."""

import os
import sys
import tempfile
import shutil
import pytest


class TestSmokeRun:
    """Run a minimal simulation and verify it
    completes without exceptions and produces
    expected output files."""

    def test_quicktest_runs(self, quicktest_dir):
        """Run the quickTest job for 2 cycles and
        verify HDF5 output exists with expected
        groups."""
        h5py = pytest.importorskip('h5py')

        # Work in a temporary copy so we don't
        # pollute the real job directory.
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = os.path.join(tmpdir, 'job')
            shutil.copytree(quicktest_dir, job_dir)

            # Run the simulation.
            orig_dir = os.getcwd()
            try:
                os.chdir(job_dir)
                import stodem
                stodem.main()
            finally:
                os.chdir(orig_dir)

            # Verify output files exist.
            hdf5_path = os.path.join(
                job_dir, 'stodem.hdf5')
            assert os.path.isfile(hdf5_path), \
                "HDF5 output not created"

            xdmf_path = os.path.join(
                job_dir, 'stodem.xdmf')
            assert os.path.isfile(xdmf_path), \
                "XDMF output not created"

            # Verify HDF5 structure.
            with h5py.File(hdf5_path, 'r') as f:
                assert 'CitizenGeoData' in f, \
                    "Missing CitizenGeoData group"
                assert 'PoliticianGeoData' in f, \
                    "Missing PoliticianGeoData group"
