# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""Adaptive Optics Phase modulation and de/multiplexing module for analyzing beams and correcting hologram generation."""

from photonicslab.config import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import warnings

from photonicslab.demultiplex import WavlenDeMux

class Monochromatic(WavlenDeMux):
    def __init__(self):
        pass

class Chromatic(Monochromatic):
    def __init__(self):
        pass

class ZernickePoly(Monochromatic):
    def __init__(self):
        pass
