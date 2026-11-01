# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import pytest
import photonicslab
import os
from distutils import dir_util

@pytest.fixture(scope='class')
def heateqn_init(hdf):
     return photonicslab.heateqn.HeatEqn2D(hdf5filename=str(hdf), hdf5grp='heat')

@pytest.fixture(scope='class')
def heateqn3d_init(hdf):
     return photonicslab.heateqn.HeatEqn3D(hdf5filename=str(hdf), hdf5grp='heat3d')

@pytest.fixture
def datadir(tmpdir, request):
     """
     References:
     https://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
     """
     filename = request.module.__file__
     test_dir = os.path.join(os.path.dirname(filename), 'baseline_images')
     if os.path.isdir(test_dir):
          dir_util.copy_tree(test_dir, str(tmpdir))
     return tmpdir
