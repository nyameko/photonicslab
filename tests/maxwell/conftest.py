# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import pytest
import photonicslab
import os
from distutils import dir_util

@pytest.fixture(scope='class')
def maxwelleqn_init(hdf):
    return photonicslab.maxwelleqn.MaxwellEqn(hdf5filename=str(hdf), hdf5grp='em')

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
