# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""Adaptive Optics Phase modulation and de/multiplexing module for analyzing beams and correcting hologram generation."""

from photonicslab.config import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import warnings

class Propagation(object):
    def __init__(self):
        pass

class FeaturePrediction(Propagation):
    """Analytical prediction of output seed propagated to an arbitrary plane."""
    def __init__(self):
        pass

class PreserveGaussian(FeaturePrediction):
    def __init__(self):
        pass

class PreserveSuperGaussian(PreserveGaussian):
    def __init__(self):
        pass

class PreserveLGMode(PreserveGaussian):
    def __init__(self):
        pass
