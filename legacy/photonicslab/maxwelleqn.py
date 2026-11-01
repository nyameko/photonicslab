# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""Finite difference solver for the three-dimensional fully vectorial and temporal paraxial solution to Maxwell's equation using the explicit and / or implicit methods.

This serves as an initialisation class to generate the initial amplification or seed configuration and to
define the gain medium properties of the material to be lased.

Todo:
    * Basic amplification
    * Definitions for mirrors and free space
    * Non-linear effects - self-focusing, vector beam generations
    * Thin Disk Lasers, Side Pumped, Random Lasers and Nanoparticles, Spiral Phase Plates
    * Validation with published data and virtuallab

References:
    * Shu et. al. 10 August 2007, Applied Optics Vol. 46, No. 23.
    * Asoubar et. al 1 JUn 2015, Optics Express Vol 23. No. 11
"""

import os
import warnings
import h5py
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import numpy as np
#import skimage
#import imageio

from photoncislab.config import *

class MaxwellEqn(object):
    """Three-dimensional paraxial solution to Maxwell's equation initialisation object.

    Need to create additional variables for cavity length and cavity definitions. And separate them from gain medium description.

    Args:
        nx (int): Number of steps along x-direction of cross-sectional profile.
        ny (int): Number of steps along y-direction of cross-sectional profile.
        nt (int): Number of time steps for temporal evolution.
        Lx (float): Cross-sectional length along x-direction [cm].
        Ly (float): Cross-sectional length along y-direction [cm].
        Lz (float): Axial length along z-direction [cm].
        Tbc (float): Boundary condition temperature [K].
        Qmax (float): Maximum power of heat source [W/cm^3].
        gain_medium (str): Crystal to use for thermal lensing and beam shaping.
        dt_stability_factor (float): Should be less than 1, can try to increase to speed up convergence.
        hdf5filename (str): Filename of the HDF5 data object.
        hdf5grp (str): Group name of the HDF5 group object.

    Raises:
        ValueError: If no valid gain medimum given.
        ValueError: If stability criterion factor provided will not converge explicit method.

    Attributes:
        dx (:obj:`float`): Spacing along the x-direction [cm].
        dy (:obj:`float`): Spacing along the y-direction [cm].
        dt (:obj:`float`): Time step increments [s].
        rho (:obj:`float`): Density of gain medium [g/cm^3].
        kx (:obj:`float`): Thermal conductivity coefficient along x-direction [W/(cm K)].
        ky (:obj:`float`): Thermal conductivity coefficient along y-direction [W/(cm K)].
        cp (:obj:`float`): Specific heat [J/(g K)].
        dndT (:obj:`float`): Thermal coefficient of refractive index at 1064nm (seed wavelength) [K^-1].
        nbc (:obj:`float`): Index of refraction at the boundary condition temperature.
        xx (:obj:`~numpy.ndarray`): Two-dimensional meshgrid array of x-coordinates used for plotting.
        yy (:obj:`~numpy.ndarray`): Two-dimensional meshgrid array of y-coordinates used for plotting.
        Qinit (:obj:`~numpy.ndarray`): Initial heat source distribution for infinitesimal 2D plane, in time-slices of nt.
        _vmin (:obj:`float`): Private attribute used by plotting function.
        _vmax (:obj:`float`): Private attribute used by plotting function.

    .. seealso::
        * Notes section of :func:`~numpy.meshgrid` for ij indexing parameter.

    .. todo::
        * Make sure to catch things like nd:ylf etc, don't want to pass on nd alone in exceptions.
        * Consider using kwargs for methods with more than6 parameters, eg see numpy.meshgrid docs.
        * Complete add_supergaussian() and replace add_gaussian and add_flattop with calls to add_supergaussian in their respective limits.
        * Thermally induced properties, index of refraction, epsilon,

    References:
    """
    def __init__(self, gain_='yag', hdf5filename='hdf5db/storage.hdf5', hdf5grp='nb1', hdf5table='Maxwell'):
        self.hdf5filename = str(hdf5filename)
        self.hdf5table = str(hdf5table)
        self.hdf5grp = str(hdf5grp)
        self.dsetfsol = str( self.hdf5grp + '/' + self.hdf5table)
        self.nt, self.nx, self.ny, self.nz, self.Lx, self.Ly, self.Lz =\
            get_hdf5_data(self.hdf5filename, self.hdf5grp)

        # Electromagnetic Permitivity and (Ginzburg-Landau), Magnetic Susceptability for polarization
        self.epsilon = np.zeros([nx, ny, nz, 3, 3])
        self.chi = np.zeros([nx, ny, nz, 3, 3])

        with h5py.File(self.hf5filename, 'a', libver='latest') as hf:
            grp = hf.require_group(self.hdf5grp)

            self.Lz = float(np.abs(Lz))
        self.dx = self.Lx/(self.nx-1)
        self.dy = self.Ly/(self.ny-1)
        self.Tbc = float(Tbc)
        self.Qmax = float(Qmax)

        self._vmin = 0
        self._vmax = self.Qmax
        self.num_q = 0

        try:
            if gain_medium.lower() in ['ylf', 'nd:ylf', 'nd-ylf', 'ndylf']:
                self.rho = 3.99
                self.kx, self.ky = 0.063, 0.063
                self.cp = 0.79
                self.dndT = 7.86e-6
                self.nbc = 1.4538
                self.gain_medium = str('Nd:YLF')
            elif gain_medium.lower() in ['yag', 'nd:yag', 'nd-yag', 'ndyag']:
                self.rho = 4.55
                self.kx, self.ky = 0.112, 0.112
                self.cp = 0.59
                self.dndT = 7.8e-6
                self.nbc = 1.8197
                self.gain_medium = str('Nd:YAG')
            else:
                raise ValueError("Unknown gain medium.")
        except ValueError as unknown_gain:
            print("ValuerError: {0}".format(unknown_gain))
            raise unknown_gain

        try:
            if dt_stability_factor >= 1.0:
                raise ValueError("Explicit finite difference method stability criterion too large.")
        except ValueError as large_stability_criterion:
            print(large_stability_criterion)
            self.dt_stability_factor = 0.95
        else:
            self.dt_stability_factor = float(dt_stability_factor)
        finally:
            self.dt = dt_stability_factor*(.5*self.rho*self.cp*np.min([self.dx**2, self.dy**2])
                                           /np.max([self.kx, self.ky]))

        try:
            if (self.Lx <= 0 or self.Ly <= 0 or self.Lz <= 0):
                raise ValueError("Invalid dimensions.")
            elif (self.nx <= 0 or self.ny <= 0 or self.nt <= 0):
                raise ValueError("Invalid number of steps.")
        except ValueError as bad_constr:
            print("ValueError: {0}".format(bad_constr))
        else:
            self.xx, self.yy = np.meshgrid(np.linspace(-self.Lx/2, self.Lx/2, self.nx),
                                           np.linspace(-self.Ly/2, self.Ly/2, self.ny), indexing='ij')

        self.hdf5filename = str(hdf5filename)
        self.hdf5grp = str(hdf5grp)
        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            # if str(self.hdf5grp) in hf:
            #     del hf[str(self.hdf5grp)]
            grp = hf.require_group(self.hdf5grp)
            grp.attrs.create('material', self.gain_medium, dtype=h5py.special_dtype(vlen=str))
            grp.attrs.create('density', self.rho)
            grp.attrs.create('thermal_conductivity', [self.kx, self.ky])
            grp.attrs.create('specific_heat', self.cp)
            grp.attrs.create('dndT', self.dndT)
            grp.attrs.create('nBC', self.nbc)
            grp.attrs.create('TBC', self.Tbc)
            grp.attrs.create('nt', self.nt)
            grp.attrs.create('dt', self.dt)
            grp.attrs.create('nx', self.nx)
            grp.attrs.create('ny', self.ny)
            grp.attrs.create('Lx', self.Lx)
            grp.attrs.create('Ly', self.Ly)
            grp.attrs.create('Lz', self.Lz)
            grp.attrs.create('dx', self.dx)
            grp.attrs.create('dy', self.dy)
            if 'xx' not in grp:
                grp.create_dataset('xx', shape=(self.nx, self.ny), dtype=np.dtype(float).type,
                                   data=self.xx, maxshape=(MAX_NX, MAX_NY))
            else:
                grp['xx'].resize((self.nx, self.ny))
                grp['xx'][...] = self.xx
            if 'yy' not in grp:
                grp.create_dataset('yy', shape=(self.nx, self.ny), dtype=np.dtype(float).type,
                                   data=self.yy, maxshape=(MAX_NX, MAX_NY))
            else:
                grp['yy'].resize((self.nx, self.ny))
                grp['yy'][...] = self.yy

        self.Qinit = np.zeros([self.nt, self.nx, self.ny])

        pass
