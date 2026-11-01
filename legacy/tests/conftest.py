# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import pytest
import photonicslab
import h5py

@pytest.fixture(scope='session')
def hdf(tmpdir_factory):
     file_name = tmpdir_factory.mktemp('testdata').join('test_storage.hdf5')
     return file_name

# FIXME: This fixture is broken, do not use carelessly
@pytest.fixture
def grp(hdf):
    print('Opening hdf5')
    with h5py.File(str(hdf), 'r', libver='latest') as hf:
        # Use a yield here instead of a return to ensure that with will eventually comeback close hdf5 file.
        yield hf.require_group('testing')
    print('Closing hdf5')

@pytest.fixture
def heateqn(hdf):
     return photonicslab.heateqn.HeatEqn2D(nx=50, ny=50, nt=100, hdf5filename=str(hdf), hdf5grp='testing')

@pytest.fixture
def heateqn_comp(hdf):
     return photonicslab.heateqn.HeatEqn2D(nx=50, ny=50, nt=100, hdf5filename=str(hdf), hdf5grp='testing_comp')

# Need to get nested inheritence
@pytest.fixture
def heateqn3d(hdf):
     return photonicslab.heateqn.HeatEqn3D(nx=50, ny=50, nz=50, nt=100, hdf5filename=str(hdf), hdf5grp='testing3d')

@pytest.fixture
def heateqn3d_comp(hdf):
     return photonicslab.heateqn.HeatEqn3D(nx=50, ny=50, nz=50, nt=100, hdf5filename=str(hdf), hdf5grp='testing3d_comp')

@pytest.fixture
def fdsolver(heateqn, hdf):
    heateqn.remove_sources()
    heateqn.add_gaussian()
    heateqn.get_explicit_fd_params()
    return photonicslab.fdsolver.FDSolver2D(hdf5filename=str(hdf), hdf5grp='testing')

@pytest.fixture
def fdsolver_comp(heateqn_comp, hdf):
    heateqn_comp.remove_sources()
    heateqn_comp.add_gaussian()
    heateqn_comp.get_explicit_fd_params()
    return photonicslab.fdsolver.FDSolver2D(hdf5filename=str(hdf), hdf5grp='testing_comp')

@pytest.fixture
def phasemod(fdsolver, hdf):
    fdsolver.explicit_method()
    return photonicslab.phasemod.ThermalLens(hdf5filename=str(hdf), hdf5grp='testing')

@pytest.fixture
def phasemod_comp(fdsolver_comp, hdf):
    fdsolver_comp.explicit_method()
    return photonicslab.phasemod.ThermalLens(hdf5filename=str(hdf), hdf5grp='testing_comp')


@pytest.fixture(scope='session')
def img(tmpdir_factory):
     image_name = tmpdir_factory.mktemp('testdata').join('test_image.png')
     return image_name

def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true",
                     default=False, help="run slow tests")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

def pytest_runtest_makereport(item, call):
    if "incremental" in item.keywords:
        if call.excinfo is not None:
            parent = item.parent
            parent._previousfailed = item

def pytest_runtest_setup(item):
    if "incremental" in item.keywords:
        previousfailed = getattr(item.parent, "_previousfailed", None)
        if previousfailed is not None:
            pytest.xfail("previous test failed (%s)" %previousfailed.name)
