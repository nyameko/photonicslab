# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import pytest
import numpy as np

@pytest.mark.run(before='TestBeamProp3D')
class TestBeamProp2D(object):
    def test_lg_mode_normalization(self, heateqn, beamprop):
        #heateqn.add_gaussian()
        assert np.max(beamprop.lg_mode(0, 0, 0, 0)[2]) == 1

    @pytest.mark.skip
    def test_gaussian_lg_mode_z_equiv_corresponding_convol(self, beamprop):
        assert np.allclose(beamprop.lg_mode(0, 0, 1, 0)[0],
                           beamprop.convolve(beamprop.lg_mode(0, 0, 0, 0)[0], 1)[0])
        assert np.allclose(beamprop.lg_mode(0, 0, 5, 0)[0],
                           beamprop.convolve(beamprop.lg_mode(0, 0, 0, 0)[0], 5)[0])
        assert np.allclose(beamprop.lg_mode(0, 0, 10, 0)[0],
                           beamprop.convolve(beamprop.lg_mode(0, 0, 0, 0)[0], 10)[0])
        assert np.allclose(beamprop.lg_mode(0, 0, 20, 0)[0],
                           beamprop.convolve(beamprop.lg_mode(0, 0, 0, 0)[0], 20)[0])

class TestBeamProp3D(TestBeamProp2D):
    pass
    # @pytest.fixture()
    # def beamprop(self, BeamProp3D):
    #     return BeamProp3D()
