# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""In many laser gain media the rate equations can be solved analytically and the saturated gain coefficient can be written as:

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

class ScalarGainCoeff(object):
    def __init__(self):
        pass

class VectorGainCoeff(ScalarGainCoeff):
    def __init__(self):
        pass

class DynamicRateEqn(VectorGainCoeff):
    def __init__(self):
        pass
