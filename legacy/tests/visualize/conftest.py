# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import pytest
import photonicslab
import os
from distutils import dir_util

@pytest.fixture
def visualize_init(phasemod, hdf):
    # TODO and FIXME: Test all test if they should be using phasemod_init and  / or heateqn_init here in these instances.
    phasemod.parabolic_fit()
    return photonicslab.visualize.Visualize2D(hdf5filename=str(hdf), hdf5grp='testing')

@pytest.fixture
def visualize(phasemod, hdf):
    phasemod.parabolic_fit()
    return photonicslab.visualize.Visualize2D(hdf5filename=str(hdf), hdf5grp='testing')

@pytest.fixture
def visualize_comp(phasemod_comp, hdf):
    phasemod_comp.parabolic_fit()
    return photonicslab.visualize.Visualize2D(hdf5filename=str(hdf), hdf5grp='testing_comp')

@pytest.fixture
def visualize_phase(phasemod, hdf):
    phasemod.phase()
    phasemod.parabolic_fit(hdf5table='phase')
    return photonicslab.visualize.VisualizePhase2D(hdf5filename=str(hdf), hdf5grp='testing')

@pytest.fixture
def visualize_phase_comp(phasemod_comp, hdf):
    phasemod_comp.phase()
    phasemod_comp.parabolic_fit(hdf5table='phase')
    return photonicslab.visualize.VisualizePhase2D(hdf5filename=str(hdf), hdf5grp='testing_comp')

@pytest.fixture
def visualize_beam(phasemod, hdf):
    phasemod.phase()
    return photonicslab.visualize.VisualizeBeam2D(hdf5filename=str(hdf), hdf5grp='testing')

@pytest.fixture
def visualize_beam_comp(phasemod_comp, hdf):
    phasemod_comp.phase()
    return photonicslab.visualize.VisualizeBeam2D(hdf5filename=str(hdf), hdf5grp='testing_comp')

@pytest.fixture
def datadir(tmpdir, request):
     """
     FIXME: May need to move this up a level as it is a duplicate here and in heateqn tests.

     References:
     https://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
     """
     filename = request.module.__file__
     test_dir = os.path.join(os.path.dirname(filename), 'baseline_images')
     if os.path.isdir(test_dir):
          dir_util.copy_tree(test_dir, str(tmpdir))
     return tmpdir
