# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""Stress-strain equations for calculating the off-diagonal electrical permittivity tensor components responsible for the induced birefringence effects.

This serves as an initialisation class to generate the initial heat source configuration and to
define the material properties of the element to be heated.

Todo:
    * Generate phase profiles.
    * Extend to 3D case.
    * Consider effects of thermal expansion and thermal diffusivity.
    * Move constructor details of dndT, nbc seedwl=1064 etc to beamprop phase profile section.
      Or write get method here.
    * Consolidate gain media properties into materials database.
"""

import os
import warnings
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
#import skimage
#import imageio

from photonicslab.config import *
from photonicslab.isotropy import GRINLens

class Birefringence(GRINLens):
    """Naturally occurring birefringence effects.

    """

    def __init__(self):
        pass

class ThermalStress(Birefringence):
    """Physically stress-strain (eg Pinched Optics)induced birefringent deformations.

    """

    def __init__(self):
        pass

class MechStress(ThermalStress):
    def __init__(self):
        pass
