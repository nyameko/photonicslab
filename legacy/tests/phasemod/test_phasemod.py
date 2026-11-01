# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import pytest
import numpy as np
import h5py
import warnings

@pytest.mark.run(before='TestThermalLens')
class TestThermalLensInit(object):
    def test_init_hdf5_resize_shape_of_meshgrids(self, heateqn, phasemod, hdf):
        temp_nx, temp_ny = np.shape(phasemod.xx)
        #print(str(temp_nx, temp_nx))
        assert np.shape(phasemod.yy) == (temp_nx, temp_ny)
        heateqn.__init__(nx=100, ny=100, hdf5filename=str(hdf), hdf5grp='heat')
        phasemod.__init__(hdf5filename=str(hdf), hdf5grp='heat')
        assert np.shape(phasemod.xx) != (temp_nx, temp_ny)
        assert np.shape(phasemod.yy) != (temp_nx, temp_ny)

@pytest.mark.run(after='TestThermalLensInit')
class TestThermalLens(object):
    def test_parabolit_fit_negetive_deg_warning(self, phasemod):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            phasemod.parabolic_fit(deg=-2)
            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "malformed degree" in str(w[-1].message).lower()

    def test_parabolic_fit_malformed_limits_error(self, phasemod):
        with pytest.raises(AttributeError):
            phasemod.parabolic_fit(lim_x=[-0.05, -0.05, 0.05])

    def test_parabolic_fit_out_of_bounds_limits_warning(self, phasemod):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            phasemod.parabolic_fit(lim_x=[-1000, 1000])
            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "parabolic fit" in str(w[-1].message).lower()

    #@pytest.mark.skip(reason="Implementation still pending.")
    def test_parabolic_fit_temperature_higher_deg(self, phasemod, phasemod_comp, hdf):
        phasemod.parabolic_fit()
        phasemod_comp.parabolic_fit(deg=3)
        with h5py.File(str(hdf), 'r', libver='latest') as hf:
            grp = hf.require_group(str(phasemod.hdf5grp +'/polyfit/temperature'))
            grp_comp = hf.require_group(str(phasemod_comp.hdf5grp +'/polyfit/temperature'))
            assert not np.allclose(grp['xpolyval'][:], grp_comp['xpolyval'][:])
            assert not np.allclose(grp['ypolyval'][:], grp_comp['ypolyval'][:])
            assert np.shape(grp['xpolyfit'][:]) != np.shape(grp_comp['xpolyfit'][:])
            assert np.shape(grp['ypolyfit'][:]) != np.shape(grp_comp['ypolyfit'][:])
            assert np.shape(grp['xcov'][:]) != np.shape(grp_comp['xcov'][:])
            assert np.shape(grp['ycov'][:]) != np.shape(grp_comp['ycov'][:])

    def test_phase(self, phasemod, hdf):
        phasemod.phase()
        with h5py.File(str(hdf), 'r', libver='latest') as hf:
            assert not np.allclose(hf[str(phasemod.hdf5grp +"/temperature")][:],
                                      hf[str(phasemod.hdf5grp +"/phase")][:])

class TestModalDecomp(object):
    pass

class TestMSquared(object):
    pass

class TestVQF(TestModalDecomp, TestMSquared):
    pass
