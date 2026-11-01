# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import pytest
import numpy as np
from unittest import mock
import warnings
import h5py
import os

@pytest.mark.run(before='TestFDSolver2D')
class TestFDSolver2D_Init(object):
    def test_init_boundary_condition_initialization(self, fdsolver, hdf):
        with h5py.File(str(hdf), 'r', libver='latest') as hf:
            grp = hf.require_group(fdsolver.hdf5grp)
            assert np.all(grp[fdsolver.hdf5table][:] == grp.attrs.get('TBC'))

    def test_init_resize_hdf5_temperature_object(self, heateqn, fdsolver, hdf):
        temp_nt = heateqn.nt
        temp_nx = heateqn.nx
        temp_ny = heateqn.ny
        with h5py.File(str(hdf), 'r', libver='latest') as hf:
            grp = hf.require_group(heateqn.hdf5grp)
            assert np.shape(grp[fdsolver.hdf5table]) == (temp_nt, temp_nx, temp_ny)
        heateqn.__init__(nx=2*temp_nx, ny=2*temp_ny, nt=2*temp_nt, hdf5filename=str(hdf), hdf5grp='testing')
        fdsolver.__init__(hdf5filename=str(hdf), hdf5grp='testing')
        with h5py.File(str(hdf), 'r', libver='latest') as hf:
            grp = hf.require_group(heateqn.hdf5grp)
            assert np.shape(grp[fdsolver.hdf5table]) != (temp_nx, temp_ny, temp_nt)

@pytest.mark.run(before='TestFDSolver3D')
class TestFDSolver2D(object):
    @pytest.mark.slow
    def test_format_hdf5table_to_finit(self, fdsolver, hdf):
        fdsolver.explicit_method()
        with h5py.File(str(hdf), 'r', libver='latest') as hf:
            grp = hf.require_group(fdsolver.hdf5grp)
            assert not np.array_equal(grp[fdsolver.hdf5table],
                fdsolver.finit*np.ones([fdsolver.nx, fdsolver.ny]))

        fdsolver._format_hdf5table_to_finit()
        with h5py.File(str(hdf), 'r', libver='latest') as hf:
            grp = hf.require_group(fdsolver.hdf5grp)
            assert np.allclose(grp[fdsolver.hdf5table][...],
                                  grp.attrs['TBC']*np.ones([fdsolver.nx, fdsolver.ny]))

    def test_explicit_method_benchmark_diff_finit(self, benchmark, fdsolver, hdf):
        benchmark(fdsolver.explicit_method)
        with h5py.File(str(hdf), 'r', libver='latest') as hf:
            grp = hf.require_group(fdsolver.hdf5grp)
            assert not np.array_equal(grp[fdsolver.hdf5table],
                                      fdsolver.finit[None, :, :])
            #*np.ones([fdsolver.nt, fdsolver.nx, fdsolver.ny])

    @pytest.mark.slow
    def test_explicit_method_hdf_benchmark_diff_finit(self, fdsolver, benchmark, hdf):
        benchmark(fdsolver.explicit_method_hdf)
        with h5py.File(str(hdf), 'r', libver='latest') as hf:
            grp = hf.require_group(fdsolver.hdf5grp)
            assert not np.array_equal(grp[fdsolver.hdf5table],
                fdsolver.finit*np.ones([fdsolver.nt, fdsolver.nx, fdsolver.ny]))

#    def test_format_hdf5table3d_to_finit3d(self, fdsolver, hdf):
#        fdsolver._format_hdf5table3d_to_finit3d

# @pytest.mark.run(after='TestFDSolver2D')
# class TestFDSolver3D(TestFDSolver2D):
#     pass
