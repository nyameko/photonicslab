# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import pytest
import numpy as np
from unittest import mock
import warnings
import h5py
import os
from skimage.measure import compare_ssim, compare_mse, compare_nrmse
from skimage.io import imread

@pytest.mark.run(before='TestHeatEqn2D')
class TestHeatEqn2D_Init(object):
    def test_init_unknown_gain_medium_exception(self, heateqn_init, hdf):
        with pytest.raises(ValueError):
            heateqn_init.__init__(gain_medium='unknown_material', hdf5filename=str(hdf), hdf5grp='heat')

    def test_init_unknown_gain_medium_exception_yvo4(self, heateqn_init, hdf):
        with pytest.raises(ValueError):
            heateqn_init.__init__(gain_medium='yvo4', hdf5filename=str(hdf), hdf5grp='heat')

    def test_init_bad_dimension_or_step_size(self, heateqn_init, hdf):
        heateqn_init.__init__(nx=-50, ny=-50, nt=-100, Lx=-1, Ly=-1, Lz=-1, hdf5filename=str(hdf),
                              hdf5grp='heat')
        assert np.all([heateqn_init.nt, heateqn_init.nx, heateqn_init.ny,
                       heateqn_init.Lx, heateqn_init.Ly, heateqn_init.Lz])

    def test_init_large_dt_stability_factor(self, heateqn_init, hdf):
        heateqn_init.__init__(hdf5filename=str(hdf), hdf5grp='heat', dt_stability_factor=1.1)
        assert heateqn_init.dt_stability_factor == 0.95

    def test_init_hdf5_resize_recreate_dataset(self, heateqn_init, hdf):
        temp_nx = heateqn_init.nx
        temp_ny = heateqn_init.ny
        assert np.shape(heateqn_init.xx) == (temp_nx, temp_ny)
        heateqn_init.__init__(nx=50, ny=50, hdf5filename=str(hdf), hdf5grp='heat')
        assert np.shape(heateqn_init.xx) != (temp_nx, temp_ny)
        assert np.array_equal(heateqn_init.Qinit, np.zeros(np.shape(heateqn_init.Qinit)))

@pytest.mark.run(before='TestHeatEqn3D')
class TestHeatEqn3D_Init(TestHeatEqn2D_Init):
    @pytest.fixture
    def heateqn_init(self, heateqn3d_init):
        return heateqn3d_init

@pytest.mark.run(before='TestHeatEqn3D')
class TestHeatEqn2D(object):
    def test_add_gaussian(self, heateqn):
        heateqn.remove_sources()
        assert np.array_equal(heateqn.Qinit, np.zeros(np.shape(heateqn.Qinit)))
        heateqn.add_gaussian(q=0.95*heateqn.Qmax)
        assert not np.array_equal(heateqn.Qinit, np.zeros(np.shape(heateqn.Qinit)))

    def test_add_gaussian_for_half_total_integration_time(self, heateqn):
        heateqn.remove_sources()
        assert np.array_equal(heateqn.Qinit, np.zeros(np.shape(heateqn.Qinit)))
        heateqn.add_gaussian(q=0.95*heateqn.Qmax, nt_end=0.5*heateqn.nt)
        assert not np.array_equal(heateqn.Qinit[0:int(np.floor(.5*heateqn.nt)),],
                                  np.zeros(np.shape(heateqn.Qinit[0:int(np.floor(.5*heateqn.nt)),])))
        assert np.array_equal(heateqn.Qinit[int(np.ceil(.5*heateqn.nt)):heateqn.nt,],
                              np.zeros(np.shape(heateqn.Qinit[int(np.ceil(.5*heateqn.nt)):heateqn.nt,])))

    def test_add_flattop(self, heateqn):
        heateqn.remove_sources()
        assert np.array_equal(heateqn.Qinit, np.zeros(np.shape(heateqn.Qinit)))
        heateqn.add_flattop(q=0.95*heateqn.Qmax)
        assert not np.array_equal(heateqn.Qinit, np.zeros(np.shape(heateqn.Qinit)))

    def test_add_flattop_ring(self, heateqn):
        heateqn.remove_sources()
        assert np.array_equal(heateqn.Qinit, np.zeros(np.shape(heateqn.Qinit)))
        heateqn.add_flattop_ring(q=0.95*heateqn.Qmax)
        assert not np.array_equal(heateqn.Qinit, np.zeros(np.shape(heateqn.Qinit)))

    def test_add_flattop_block(self, heateqn):
        heateqn.remove_sources()
        assert np.array_equal(heateqn.Qinit, np.zeros(np.shape(heateqn.Qinit)))
        heateqn.add_flattop_block(q=0.95*heateqn.Qmax)
        assert not np.array_equal(heateqn.Qinit, np.zeros(np.shape(heateqn.Qinit)))
    def test_verify_q_Qmax_limit_warning(self, heateqn):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            heateqn.remove_sources()
            heateqn._verify_q(q=1.1*heateqn.Qmax)
            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "runtime" in str(w[-1].message).lower()

    def test_verify_nt_end_nt_limit_warning(self, heateqn):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            heateqn.remove_sources()
            heateqn._verify_nt_range(0, nt_end=1.1*heateqn.nt)
            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "runtime" in str(w[-1].message).lower()

    def test_verify_nt_start_nt_limit_warning(self, heateqn):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            heateqn.remove_sources()
            heateqn._verify_nt_range(nt_start=1.1*heateqn.nt, nt_end=None)
            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "runtime" in str(w[-1].message).lower()

    def test_verify_nt_range_both_nt_limits_warnings(self, heateqn):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            heateqn.remove_sources()
            heateqn._verify_nt_range(nt_start=1.1*heateqn.nt, nt_end=1.1*heateqn.nt)
            assert len(w) == 2
            assert issubclass(w[-2].category, RuntimeWarning)
            assert "end-time" in str(w[-2].message).lower()
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "start-time" in str(w[-1].message).lower()

    def test_verify_nt_range_inverted_nt_warning(self, heateqn):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            heateqn.remove_sources()
            heateqn._verify_nt_range(nt_start=int(.75*heateqn.nt), nt_end=int(0.5*heateqn.nt))
            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "inverted" in str(w[-1].message).lower()

    def test_remove_sources(self, heateqn):
        heateqn.remove_sources()
        heateqn.add_gaussian()
        assert not np.array_equal(heateqn.Qinit, np.zeros(np.shape(heateqn.Qinit)))
        heateqn.remove_sources()
        assert np.array_equal(heateqn.Qinit, np.zeros(np.shape(heateqn.Qinit)))

    @pytest.mark.skip(reason="Implementation still pending.")
    def test_add_supergaussian_flattop_limit(self, heateqn, heateqn_comp):
        heateqn.remove_sources()
        heateqn.add_gaussian()
        heateqn_comp.remove_sources()
        heateqn_comp.add_supergaussian()
        assert np.allclose(heateqn.Qinit, heateqn_comp.Qinit)

    @pytest.mark.skip(reason="Implementation still pending.")
    def test_add_supergaussian_gaussian_limit(self, heateqn):
        heateqn.remove_sources()
        heateqn.add_flattop()
        temp_flattop = heateqn.Qinit[:]
        heateqn.remove_sources()
        heateqn.add_supergaussian()
        assert np.allclose(heateqn.Qinit, temp_flattop)

    def test_add_source_from_image(self, heateqn, datadir):
        heateqn.remove_sources()
        assert np.array_equal(heateqn.Qinit, np.zeros(np.shape(heateqn.Qinit)))
        heateqn.add_source_from_image(savepath=str(datadir.join('test_doughnut.png')),
                                      q=0.95*heateqn.Qmax)
        assert not np.array_equal(heateqn.Qinit, np.zeros(np.shape(heateqn.Qinit)))

    @pytest.mark.skip(reason="Implementation still pending.")
    def test_add_source_from_hdf5(self, heateqn, hdf):
        assert 0

    def test_add_gaussian_exceed_Qmax_warning(self, heateqn):
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            heateqn.remove_sources()
            heateqn.add_gaussian(q=1.1*heateqn.Qmax)
            #print([w[i].message for i in range(len(w))])
            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "reduced to qmax" in str(w[-1].message).lower()

    def test_add_flattop_exceed_Qmax_warning(self, heateqn):
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('ignore', category=PendingDeprecationWarning)
            heateqn.remove_sources()
            heateqn.add_flattop(q=1.1*heateqn.Qmax)
            #print([w[i].message for i in range(len(w))])
            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "reduced to qmax" in str(w[-1].message).lower()

    def test_add_flattop_ring_exceed_Qmax_warning(self, heateqn):
        with warnings.catch_warnings(record=True) as w:
            heateqn.remove_sources()
            heateqn.add_flattop_ring(q=1.1*heateqn.Qmax)
            #print([w[i].message for i in range(len(w))])
            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "reduced to qmax" in str(w[-1].message).lower()

    def test_add_flattop_block_exceed_Qmax_warning(self, heateqn):
        with warnings.catch_warnings(record=True) as w:
            heateqn.remove_sources()
            heateqn.add_flattop_block(q=1.1*heateqn.Qmax)
            #print([w[i].message for i in range(len(w))])
            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "reduced to qmax" in str(w[-1].message).lower()

    def test_add_gaussian_out_of_bounds_warning(self, heateqn):
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            heateqn.remove_sources()
            heateqn.add_gaussian(center_x=1.1*heateqn.Lx)
            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "outofbounds" in str(w[-1].message).lower()

    def test_add_source_from_image_out_of_bounds_warning(self, heateqn, datadir):
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            warnings.filterwarnings('ignore', category=UserWarning)
            heateqn.remove_sources()
            heateqn.add_source_from_image(savepath=str(datadir.join('test_doughnut.png')),
                                          center_x=1.1*heateqn.Lx)
            #print([w[i].message for i in range(len(w))])
            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "outofbounds" in str(w[-1].message).lower()
            # assert len(w) == 2
            # assert issubclass(w[-2].category, RuntimeWarning)
            # assert "outofbounds" in str(w[-2].message).lower()
            # assert issubclass(w[-1].category, RuntimeWarning)
            # assert "zero-range plot" in str(w[-1].message).lower()

    def test_add_flattop_out_of_bounds_warning(self, heateqn):
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            heateqn.remove_sources()
            heateqn.add_flattop(center_x=1.1*heateqn.Lx)
            assert len(w) == 2
            #print(w[i].message for i in range(len(w)))
            assert issubclass(w[-2].category, RuntimeWarning)
            assert "outofbounds" in str(w[-2].message).lower()
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "zero-range plot" in str(w[-1].message).lower()

    def test_add_flattop_negative_radius_warning(self, heateqn):
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('ignore', category=PendingDeprecationWarning)
            heateqn.remove_sources()
            heateqn.add_flattop(center_r=-0.05)
            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "negative" in str(w[-1].message).lower()

    def test_add_flattop_ring_out_of_bounds_warning(self, heateqn):
        with warnings.catch_warnings(record=True) as w:
            heateqn.remove_sources()
            heateqn.add_flattop_ring(center_x=1.1*heateqn.Lx)
            assert len(w) == 2
            assert issubclass(w[-2].category, RuntimeWarning)
            assert "outofbounds" in str(w[-2].message).lower()
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "zero-range plot" in str(w[-1].message).lower()

    def test_add_flattop_ring_inverted_rings_warning(self, heateqn):
        with warnings.catch_warnings(record=True) as w:
            heateqn.remove_sources()
            heateqn.add_flattop_ring(inner_r=0.1, outer_r=0.05)
            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "inner" in str(w[-1].message).lower()

    def test_add_flattop_ring_negative_inner_r_warning(self, heateqn):
        with warnings.catch_warnings(record=True) as w:
            heateqn.remove_sources()
            heateqn.add_flattop_ring(inner_r=-0.05)
            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "negative" in str(w[-1].message).lower()
            #print([w[i].message for i in range(len(w))])

    def test_add_flattop_ring_negative_outer_r_warning(self, heateqn):
        with warnings.catch_warnings(record=True) as w:
            heateqn.remove_sources()
            heateqn.add_flattop_ring(outer_r=-0.1)
            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "radii" in str(w[-1].message).lower()

    def test_add_flattop_ring_inverted_rings_negative_outer_r_warning(self, heateqn):
        with warnings.catch_warnings(record=True) as w:
            heateqn.remove_sources()
            heateqn.add_flattop_ring(inner_r=0.1, outer_r=-0.05)
            assert len(w) == 2
            assert issubclass(w[-2].category, RuntimeWarning)
            assert "corrected" in str(w[-2].message).lower()
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "inner" in str(w[-1].message).lower()

    def test_add_flattop_block_out_of_bounds_warning(self, heateqn):
        with warnings.catch_warnings(record=True) as w:
            heateqn.remove_sources()
            heateqn.add_flattop_block(upper_right_x=1.1*heateqn.Lx)
            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "outofbounds" in str(w[-1].message).lower()

    def test_add_gaussian_exceed_Qmax_warning_and_out_of_bounds_error(self, heateqn):
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            heateqn.remove_sources()
            heateqn.add_gaussian(center_x=1.1*heateqn.Lx, q=1.1*heateqn.Qmax)
            assert len(w) == 2
            assert issubclass(w[-2].category, RuntimeWarning)
            assert "reduced to qmax" in str(w[-2].message).lower()
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "outofbounds" in str(w[-1].message).lower()

    def test_set_plot_limits_exceed_Qmax_warning(self, heateqn):
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            heateqn.remove_sources()
            heateqn.add_gaussian()
            heateqn.add_gaussian()
            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "runtime" in str(w[-1].message).lower()

    def test_set_plot_limits_zero_range_warning(self, heateqn):
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            heateqn.remove_sources()
            heateqn.add_gaussian(q=0)
            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "runtime" in str(w[-1].message).lower()

    def test_add_gaussian_deprecation_warning_supergaussian(self, heateqn):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            heateqn.remove_sources()
            heateqn.add_gaussian(q=0.95*heateqn.Qmax)
            #print([w[x].message for x in range(len(w))])
            assert len(w) == 1
            assert issubclass(w[-1].category, DeprecationWarning)
            assert "deprecation" in str(w[-1].message).lower()

    def test_add_flattop_deprecation_warning_supergaussian(self, heateqn):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            heateqn.remove_sources()
            heateqn.add_flattop()
            assert len(w) == 1
            assert issubclass(w[-1].category, PendingDeprecationWarning)
            assert "deprecation" in str(w[-1].message).lower()

    def test_plot_source_profile_deprecation_warning(self, heateqn, img):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            heateqn.remove_sources()
            heateqn.add_flattop()
            heateqn.plot_source_profile(str(img))
            assert len(w) == 2
            assert issubclass(w[-1].category, PendingDeprecationWarning)
            assert "deprecation" in str(w[-1].message).lower()
            assert issubclass(w[-2].category, PendingDeprecationWarning)
            assert "deprecation" in str(w[-2].message).lower()

    def test_plot_source_profile_creates_nonempty_png(self, heateqn, img):
        with pytest.deprecated_call():
            heateqn.remove_sources()
            heateqn.add_gaussian()
            os.remove(str(img))
            heateqn.plot_source_profile(str(img))
            assert os.path.getsize(str(img)) > 0

    def test_get_explicit_fd_params(self, heateqn, hdf):
        heateqn.remove_sources()
        heateqn.add_gaussian()
        heateqn.get_explicit_fd_params()
        with h5py.File(str(hdf), 'r', libver='latest') as hf:
            grp = hf.require_group(heateqn.hdf5grp)
            assert grp.attrs.__contains__('sx')
            assert grp.attrs.__contains__('sy')
            assert grp.__contains__('clock')
            assert grp.__contains__('Qload')

    def test_get_explicit_fd_params_2x_Qload_for_Qinit_exceeds_warning_and_update_hdf(self, heateqn, hdf):
        heateqn.remove_sources()
        heateqn.add_gaussian()
        heateqn.get_explicit_fd_params()
        with h5py.File(str(hdf), 'r', libver='latest') as hf:
            grp = hf.require_group(heateqn.hdf5grp)
            temp_Qload = grp['Qload'][:]
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            heateqn.add_gaussian()
            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "current peak" in str(w[-1].message).lower()
        heateqn.get_explicit_fd_params()
        with h5py.File(str(hdf), 'r', libver='latest') as hf:
            grp = hf.require_group(heateqn.hdf5grp)
            assert np.array_equal(2*temp_Qload, grp['Qload'][:])

    def test_get_explicit_fd_params_resize_reassign_update_hdf(self, heateqn, hdf):
        heateqn.remove_sources()
        heateqn.add_gaussian()
        heateqn.get_explicit_fd_params()
        temp_nt, temp_nx, temp_ny = heateqn.nt, heateqn.nx, heateqn.ny
        with h5py.File(str(hdf), 'r', libver='latest') as hf:
            grp = hf.require_group(heateqn.hdf5grp)
            temp_sx = grp.attrs['sx']
            temp_sy = grp.attrs['sy']
            assert np.shape(grp['clock'][:]) == (temp_nt,)
            assert np.shape(grp['Qload'][:]) == (temp_nt, temp_nx, temp_ny)
        heateqn.__init__(nt=150, nx=60, ny=60, hdf5filename=str(hdf), hdf5grp='testing')
        heateqn.remove_sources()
        heateqn.add_gaussian()
        heateqn.get_explicit_fd_params()
        with h5py.File(str(hdf), 'r', libver='latest') as hf:
            grp = hf.require_group(heateqn.hdf5grp)
            assert temp_sx != grp.attrs['sx']
            assert temp_sy != grp.attrs['sy']
            assert np.shape(grp['clock'][:]) != (temp_nt,)
            assert np.shape(grp['Qload'][:]) != (temp_nt, temp_nx, temp_ny)

@pytest.mark.slow
class TestHeatEqn2DInitCompareImageBaselines(object):
    @pytest.mark.parametrize(('mse', 'ssim'), [(0.25, 0.75), (0.01, 0.99)],
                             ids=["75% Match", "Near-Perfect Match"])
    @pytest.mark.incremental
    def test_add_gaussian_compare_baseline_images_default_init(self, heateqn_init, img, datadir, mse, ssim):
        heateqn_init.remove_sources()
        heateqn_init.add_gaussian()
        heateqn_init.plot_source_profile(savepath=str(img))
        assert compare_mse(imread(str(datadir.join('test_add_gaussian_baseline_default_init.png'))),
                           imread(str(img))) <= mse
        assert compare_nrmse(imread(str(datadir.join('test_add_gaussian_baseline_default_init.png'))),
                             imread(str(img))) <= mse
        assert compare_ssim(imread(str(datadir.join('test_add_gaussian_baseline_default_init.png'))),
                            imread(str(img)), multichannel=True) >= ssim

@pytest.mark.slow
class TestHeatEqn2DCompareImageBaselines(object):
    @pytest.mark.parametrize(('mse', 'ssim'), [(0.25, 0.75),
                                               pytest.param(0.1, 0.9, marks=pytest.mark.skip),
                                               pytest.param(0.05, 0.95, marks=pytest.mark.skip),
                                               pytest.param(0, 1, marks=pytest.mark.skip)],
                             ids=["75% Match", "90% Match", "95% Match", "Perfect Match"])
    @pytest.mark.incremental
    def test_add_gaussian_compare_baseline_images(self, heateqn, img, datadir, mse, ssim):
        heateqn.remove_sources()
        heateqn.add_gaussian()
        heateqn.plot_source_profile(savepath=str(img))
        assert compare_mse(imread(str(datadir.join('test_add_gaussian_baseline.png'))),
                           imread(str(img))) <= mse
        assert compare_nrmse(imread(str(datadir.join('test_add_gaussian_baseline.png'))),
                             imread(str(img))) <= mse
        assert compare_ssim(imread(str(datadir.join('test_add_gaussian_baseline.png'))),
                            imread(str(img)), multichannel=True) >= ssim

    @pytest.mark.parametrize(('mse', 'ssim'), [(0.25, 0.75),
                                               pytest.param(0.1, 0.9, marks=pytest.mark.skip),
                                               pytest.param(0.05, 0.95, marks=pytest.mark.skip),
                                               pytest.param(0, 1, marks=pytest.mark.skip)],
                             ids=["75% Match", "90% Match", "95% Match", "Perfect Match"])
    @pytest.mark.incremental
    def test_add_source_from_image(self, heateqn, img, datadir, mse, ssim):
        heateqn.remove_sources()
        heateqn.add_source_from_image(savepath=str(datadir.join('test_doughnut.png')),
                                      q=0.95*heateqn.Qmax)
        heateqn.plot_source_profile(savepath=str(img))
        assert compare_mse(imread(str(datadir.join('test_add_source_from_image_baseline.png'))),
                           imread(str(img))) <= mse
        assert compare_nrmse(imread(str(datadir.join('test_add_source_from_image_baseline.png'))),
                             imread(str(img))) <= mse
        assert compare_ssim(imread(str(datadir.join('test_add_source_from_image_baseline.png'))),
                            imread(str(img)), multichannel=True) >= ssim

class TestHeatEqn3D(TestHeatEqn2D):
    @pytest.fixture
    def heateqn(self, heateqn3d):
        return heateqn3d

    @pytest.fixture
    def heateqn_comp(self, heateqn3d_comp):
        return heateqn3d_comp

    def test_remove_sources(self, heateqn):
        heateqn.remove_sources()
        heateqn.add_normalized_pump_gaussian()
        assert not np.array_equal(heateqn.Qinit3d, np.zeros(np.shape(heateqn.Qinit3d)))
        heateqn.remove_sources()
        assert np.array_equal(heateqn.Qinit3d, np.zeros(np.shape(heateqn.Qinit3d)))

    def test_add_flattop_out_of_bounds_warning(self, heateqn):
        # with warnings.catch_warnings(record=True) as w:
        #     warnings.filterwarnings('ignore', category=DeprecationWarning)
        #     heateqn.remove_sources()
        #     heateqn.add_flattop(center_x=1.1*heateqn.Lx)
        #     assert len(w) == 2
        #     print(w[i].message for i in range(len(w)))
        #     assert issubclass(w[-2].category, RuntimeWarning)
        #     assert "outofbounds" in str(w[-2].message).lower()
        #     assert issubclass(w[-1].category, RuntimeWarning)
        #     assert "zero-range plot" in str(w[-1].message).lower()
        pass

    def test_add_flattop_ring_out_of_bounds_warning(self, heateqn):
        # with warnings.catch_warnings(record=True) as w:
        #     heateqn.remove_sources()
        #     heateqn.add_flattop_ring(center_x=1.1*heateqn.Lx)
        #     assert len(w) == 1
        #     assert issubclass(w[-1].category, RuntimeWarning)
        #     assert "outofbounds" in str(w[-1].message).lower()
        pass

    def test_get_explicit_fd_params(self, heateqn, hdf):
        # heateqn.remove_sources()
        # heateqn.add_normalized_pump_gaussian()
        # heateqn.get_explicit_fd_params()
        # with h5py.File(str(hdf), 'r', libver='latest') as hf:
        #     grp = hf.require_group(heateqn.hdf5grp)
        #     assert grp.attrs.__contains__('sx')
        #     assert grp.attrs.__contains__('sy')
        #     assert grp.__contains__('clock')
        #     assert grp.__contains__('Qload3d')
        pass

    def test_set_plot_limits_zero_range_warning(self, heateqn):
        # with warnings.catch_warnings(record=True) as w:
        #     warnings.filterwarnings('ignore', category=DeprecationWarning)
        #     heateqn.remove_sources()
        #     heateqn.add_normalized_pump_gaussian(q=0)
        #     assert len(w) == 1
        #     assert issubclass(w[-1].category, RuntimeWarning)
        #     assert "runtime" in str(w[-1].message).lower()
        pass

    def test_set_plot_limits_exceed_Qmax_warning(self, heateqn):
        # with warnings.catch_warnings(record=True) as w:
        #     warnings.filterwarnings('ignore', category=DeprecationWarning)
        #     heateqn.remove_sources()
        #     heateqn.add_normalized_pump_gaussian()
        #     heateqn.add_normalized_pump_gaussian()
        #     assert len(w) == 1
        #     assert issubclass(w[-1].category, RuntimeWarning)
        #     assert "runtime" in str(w[-1].message).lower()
        pass

    def test_get_explicit_fd_params_2x_Qload_for_Qinit_exceeds_warning_and_update_hdf(self, heateqn, hdf):
        # heateqn.remove_sources()
        # heateqn.add_normalized_pump_gaussian()
        # heateqn.get_explicit_fd_params()
        # with h5py.File(str(hdf), 'r', libver='latest') as hf:
        #     grp = hf.require_group(heateqn.hdf5grp)
        #     temp_Qload = grp['Qload'][:]
        # with warnings.catch_warnings(record=True) as w:
        #     warnings.filterwarnings('ignore', category=DeprecationWarning)
        #     heateqn.add_normalized_pump_gaussian()
        #     assert len(w) == 3
        #     print(w[i].message for i in range(len(w)))
        #     assert issubclass(w[-1].category, RuntimeWarning)
        #     assert "current peak" in str(w[-1].message).lower()
        # heateqn.get_explicit_fd_params()
        # with h5py.File(str(hdf), 'r', libver='latest') as hf:
        #     grp = hf.require_group(heateqn.hdf5grp)
        #     assert np.array_equal(2*temp_Qload, grp['Qload'][:])
        pass

    def test_get_explicit_fd_params_resize_reassign_update_hdf(self, heateqn, hdf):
        heateqn.remove_sources()
        heateqn.add_normalized_pump_gaussian()
        heateqn.get_explicit_fd_params()
        temp_nt, temp_nx, temp_ny, temp_nz = heateqn.nt, heateqn.nx, heateqn.ny, heateqn.nz
        with h5py.File(str(hdf), 'r', libver='latest') as hf:
            grp = hf.require_group(heateqn.hdf5grp)
            temp_sx = grp.attrs['sx']
            temp_sy = grp.attrs['sy']
            assert np.shape(grp['clock'][:]) == (temp_nt,)
            assert np.shape(grp['Qload3d']) == (temp_nx, temp_ny, temp_nz)
        heateqn.__init__(nt=150, nx=60, ny=60, nz=60, hdf5filename=str(hdf),\
                         hdf5grp='testing3d', cavity_mode_waist=191.5e-6)
        heateqn.remove_sources()
        heateqn.add_normalized_pump_gaussian()
        heateqn.get_explicit_fd_params()
        with h5py.File(str(hdf), 'r', libver='latest') as hf:
            grp = hf.require_group(heateqn.hdf5grp)
            assert temp_sx != grp.attrs['sx']
            assert temp_sy != grp.attrs['sy']
            assert np.shape(grp['clock'][:]) != (temp_nt,)
            assert np.shape(grp['Qload3d']) != (temp_nx, temp_ny, temp_nz)
