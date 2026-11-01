# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""Geometry module for analysing / inducing end-face bulging and fracture effects."""

from photonicslab.config import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import warnings

from photonicslab.anisotropy import MechStress

class ThermalExpansion(MechStress):
    """Linear expansion due to change in temperature used to model end-face bulging effects.

    .. math::

        \\delta l = \\alpha l_{0} \\delta t

    Attributes:
        dl (:obj:`float`): Elongation [cm].
        alpha (:obj:`float`): Temperature expansion coefficient [cm/mK].
        l0 (:obj:`float`): Initial length [cm].
        dt (:obj:`float`): Temperature gradient [K].

    """

    def __init__(self):
        pass

class FractureAnalysis(ThermalExpansion):
    def __init__(self):
        pass
