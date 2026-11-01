# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import pytest
import numpy as np
import h5py
import warnings
from skimage.measure import compare_ssim, compare_mse, compare_nrmse
from skimage.io import imread

@pytest.mark.run(before='TestVisualize2D')
class TestVisualize2DInit(object):
    def test_init_resize_meshgrids_clock_Qload(self):
        pass

    def test_init_resize_polyfit_inds_and_polyvals(self):
        pass

    def test_init_src_evol_plt_limit(self):
        pass

@pytest.mark.run(before='TestVisualize2DCompareBaselineImages')
class TestVisualize2D(object):
    pass

@pytest.mark.slow
class TestVisualize2DCompareBaselineImages(object):
    @pytest.mark.parametrize(('mse', 'ssim'), [(0.25, 0.75), (0.01, 0.99)
                                               #pytest.param(0.1, 0.9, marks=pytest.mark.skip),
                                               #pytest.param(0.05, 0.95, marks=pytest.mark.skip),
                                               #pytest.param(0, 1, marks=pytest.mark.skip)],
                                               ],
                             #ids=["75% Match", "90% Match", "95% Match", "Perfect Match"]
                             ids=["75% Match", "99% Match"])
    @pytest.mark.incremental
    def test_steadystate_profiles_gaussian_temperature_compare_baseline_images(self, visualize, img, datadir, mse, ssim):
        visualize.steadystate_profiles(savepath=str(img))
        assert compare_mse(imread(str(datadir.join('test_steadystate_temperature_baseline.png'))),
                           imread(str(img))) <= mse
        assert compare_nrmse(imread(str(datadir.join('test_steadystate_temperature_baseline.png'))),
                             imread(str(img))) <= mse
        assert compare_ssim(imread(str(datadir.join('test_steadystate_temperature_baseline.png'))),
                            imread(str(img)), multichannel=True) >= ssim

@pytest.mark.slow
class TestVisualizePhase2DCompareBaselineImages(TestVisualize2DCompareBaselineImages):
    @pytest.mark.parametrize(('mse', 'ssim'), [(0.25, 0.75),
                                               #pytest.param(0.1, 0.9, marks=pytest.mark.skip),
                                               pytest.param(0.01, 0.99) #, marks=pytest.mark.skip),
                                               #pytest.param(0, 1, marks=pytest.mark.skip)],
                                               ],
                             #ids=["75% Match", "90% Match", "95% Match", "Perfect Match"]
                             ids=["75% Match", "Near-Perfect Match"])
    @pytest.mark.incremental
    def test_steadystate_profiles_gaussian_phase_compare_baseline_images(self, visualize_phase, img, datadir, mse, ssim):
        visualize_phase.steadystate_profiles(savepath=str(img), threshold=0.9)
        assert compare_mse(imread(str(datadir.join('test_steadystate_phase_baseline.png'))),
                           imread(str(img))) <= mse
        assert compare_nrmse(imread(str(datadir.join('test_steadystate_phase_baseline.png'))),
                             imread(str(img))) <= mse
        assert compare_ssim(imread(str(datadir.join('test_steadystate_phase_baseline.png'))),
                            imread(str(img)), multichannel=True) >= ssim

@pytest.mark.slow
class TestVisualizeBeam2DCompareBaselineImages(TestVisualize2DCompareBaselineImages):
    @pytest.mark.parametrize(('mse', 'ssim'), [(0.25, 0.75),
                                               #pytest.param(0.1, 0.9, marks=pytest.mark.skip),
                                               pytest.param(0.05, 0.95)
                                               #pytest.param(0, 1, marks=pytest.mark.skip)],
                                               ],
                             #ids=["75% Match", "90% Match", "95% Match", "Perfect Match"]
                             ids=["75% Match", "95% Match"])
    @pytest.mark.parametrize(('z'), [(10, 20, 30)], ids=["10 cm", "20 cm", "30 cm"])
    @pytest.mark.incremental
    def test_steadystate_profiles_gaussian_beam_compare_baseline_images(self, visualize_beam, img, datadir, mse, ssim, z):
        visualize_beam.steadystate_profiles(z_dist=z, savepath=str(img))
        assert compare_mse(imread(str(datadir.join(str('test_steadystate_beam_baseline_z'+str(z)+'.png')))),
                           imread(str(img))) <= mse
        assert compare_nrmse(imread(str(datadir.join(str('test_steadystate_beam_baseline_z'+str(z)+'.png')))),
                             imread(str(img))) <= mse
        assert compare_ssim(imread(str(datadir.join(str('test_steadystate_beam_baseline_z'+str(z)+'.png')))),
                            imread(str(img)), multichannel=True) >= ssim

    @pytest.mark.parametrize(('mse', 'ssim'), [(0.25, 0.75),
                                               #pytest.param(0.1, 0.9, marks=pytest.mark.skip),
                                               pytest.param(0.05, 0.95)
                                               #pytest.param(0, 1, marks=pytest.mark.skip)],
                                               ],
                             #ids=["75% Match", "90% Match", "95% Match", "Perfect Match"]
                             ids=["75% Match", "95% Match"])
    @pytest.mark.parametrize(('z'), [30, 50], ids=["z = 30 cm", "z = 50 cm"])
    @pytest.mark.incremental
    def test_steadystate_profiles_gaussian_beam_compare_baseline_images(self, visualize_beam, hdf, img, datadir, mse, ssim, z):
        visualize_beam.__init__(l=1, phi=0.3, hdf5filename=str(hdf), hdf5grp='testing')
        visualize_beam.steadystate_profiles(z_dist=z, savepath=str(img))
        assert compare_mse(imread(str(datadir.join(str('test_steadystate_beam_lg_baseline_z'+str(z)+'.png')))),
                           imread(str(img))) <= mse
        assert compare_nrmse(imread(str(datadir.join(str('test_steadystate_beam_lg_baseline_z'+str(z)+'.png')))),
                             imread(str(img))) <= mse
        assert compare_ssim(imread(str(datadir.join(str('test_steadystate_beam_lg_baseline_z'+str(z)+'.png')))),
                            imread(str(img)), multichannel=True) >= ssim

#  LocalWords:  steadystate
