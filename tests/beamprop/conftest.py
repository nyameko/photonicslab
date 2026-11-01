# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import pytest
import photonicslab

@pytest.fixture
def beamprop(phasemod, hdf):
    phasemod.parabolic_fit()
    phasemod.phase()
    return photonicslab.beamprop.BeamProp2D(hdf5filename=str(hdf), hdf5grp='testing')

@pytest.fixture
def beamprop_comp(phasemod_comp, hdf):
    phasemod_comp.parabolic_fit()
    phasemod_comp.phase()
    return photonicslab.beamprop.BeamProp2D(hdf5filename=str(hdf), hdf5grp='testing_comp')
