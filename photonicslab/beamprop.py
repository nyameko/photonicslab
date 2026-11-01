# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""Electromagnetic beam propagation in two and three dimensions."""

import numpy as np
from scipy.special import genlaguerre
#from scipy.signal import convolve2d
from scipy.signal import fftconvolve
import h5py
from photonicslab.config import *

#from sklearn.preprocessing import normalize

# Seealso:
# import numpy.polynomial.laguerre as lg
# import scipy.special.eval_genlaguerre as eval_lg

import matplotlib.pyplot as plt
import time

class BeamProp2D(object):
    """Two-dimensional electromagnetic beam propagation code.

    Args:
        wavelen (float): Wavelength of the incident seed beam [cm].
        waist (float): Location along z-direction of seed beam waist [cm].
        flens (float): (Not used) Focal length of lens added after crystal [cm].

    Raises:
        ValueError: If no valid gain medimum given.
        ValueError: If stability criterion factor provided will not converge explicit method.

    Attributes:
        xx (:obj:`~numpy.ndarray`): Two-dimensional meshgrid arrays for y-coordinates used for propagation.
        yy (:obj:`~numpy.ndarray`): Two-dimensional meshgrid arrays for y-coordinates used for propagation.
        rr_sd (:obj:`~numpy.ndarray`): Meshgrid array for cross-sectional profile points along spatial-domain radial direction [m].
        rr_fd (:obj:`~numpy.ndarray`): Meshgrid array for cross-sectional profile points along Fourier-domain radial direction [m].
        wavenum (:obj:`float`): Spatial frequency of seed [rad/m].

    Todo:
        * Add hg and other modes
        * Overload addition operator for object oriented superposition and interaction of modes.
        * Write propagation output to hdf5
        * Consolidate units to SI or some other standard.
        * Seed to become a class instance variable, i.e. `self.seed = gen_LG`
        * Use decorator function @property to set and retrieve values
        * Consolidate focal lengths, Rayleigh range and focus due to lens.
        * Instantiate seed and seed_area class objects
        * TODO: Note that everything's been made relative to cm

    .. seealso::
        * Module :func:`numpy.convolve` Returns the discrete, linear convolution of two one-dimensional sequences.
        * Method :func:`scipy.signal.fftconvolve` Convole two N-dimensional arrays using FFT
        * Method :func:`scipy.signal.convolve` Convolve two N-dimensional arrays.
    """

    def __init__(self, wavelen=1.064e-4, waist=191.5e-4, flens=5e-1, hdf5filename='hdf5db/storage.hdf5', \
                 hdf5grp='nb1'):
        self.wavelen = float(wavelen)
        self.waist = float(waist)
        self.hdf5filename = str(hdf5filename)
        self.hdf5grp = str(hdf5grp)
        with h5py.File(self.hdf5filename, 'r', libver='latest') as hf:
            grp = hf.require_group(self.hdf5grp)
            self.xx = grp['xx'][:]
            self.yy = grp['yy'][:]
            self.dxdy = grp.attrs.get('dx')*grp.attrs.get('dy')

        # self.xx, self.yy = np.meshgrid(np.linspace(-self.Lx/2, self.Lx/2, self.nx),
        #                                np.lisps(-self.Ly/2, self.Ly/2, self.ny), indexing='ij')
        self.rr_sd = np.sqrt(self.xx**2 + self.yy**2)
        #self.rr_sd = np.sqrt((self.xx**2 + self.yy**2)*self.dxdy)
        self.rr_fd = np.sqrt((self.xx*(1/(np.shape(self.xx)[0]*(np.abs(self.xx[0,0]-self.xx[1,0]))**2)))**2
                            +(self.yy*(1/(np.shape(self.yy)[1]*(np.abs(self.yy[0,0]-self.yy[0,1]))**2)))**2)

        self.wavenum = 2*np.pi/self.wavelen

        self.flens = flens
        self.focus = np.exp((1j*self.wavenum*self.rr_sd**2)/(2*self.flens))
        self.normalized_LG = np.zeros(np.shape(self.xx))

    def lg_mode(self, p, l, z, phi):
        """Generate amplitude for initial Laguerre-Gauss mode seed.

        Arguments:
            p (int, p>=0): Radial mode index denoting number of radial nodes.
            l (int): Azimuthal mode index denoting number of intertwined helices.
            z (float): Propagation distance along optical axis [cm].
            phi (float): Vortex phase term [rad].

        Returns:
            list: List of :obj:`~numpy.ndarray` of :math:`LG_p^l` modes, normalised :math:`LG_p^l` modes and normalized intensity distributions :math:`I`.

        Attributes:
            zR (:obj:`float`): Rayleigh range.
            width (:obj:`float`): Beam width :math:`w(z) = \\omega_{0}\\sqrt{1+\\left(\\frac{z}{z_R}\\right)}`.
            LG (:obj:`~numpy.ndarray`): 2D cross-sectional profile for Laguerre-Gauss mode.
            normalized_LG (:obj:`~numpy.ndarray`): 2D cross-sectional profile for normalized Laguerre-Gauss mode.
            normalized_I (:obj:`~numpy.ndarray`): 2D cross-sectional profile for normalized intensity distribution.

        Todo:
            * Try / Except blocks for zerodivision or valueerror if waist=0 and for p and l indices.
            * Normalise variables using sklearn.
            * FIXME: Wavelength dependency of index of refraction :math:`n(\\lambda)`

        References:
            * `L. Allen, M.J. Padgett Optics Communications 184 (2000) 67-71 <https://www.sciencedirect.com/science/journal/00304018/184/1-4>`_

        .. seealso::
            * method :func:`sklearn.preprocessing.normalize`

        """

        self.zR = np.pi*self.waist**2/self.wavelen
        self.width = self.waist*np.sqrt(1+(z/self.zR)**2)
        LG = (np.sqrt((2*np.math.factorial(p) / (np.pi*np.math.factorial(p + np.abs(l))))) \
              * 1/self.width * (self.rr_sd*np.sqrt(2)/self.width)**np.abs(l) \
              * np.exp(-(self.rr_sd/self.width)**2) \
              * genlaguerre(p, np.abs(l))(2*self.rr_sd**2/self.width**2) \
              * np.exp(-0.5j*self.wavenum*self.rr_sd**2*z/(z**2+self.zR**2)) \
              #* np.exp(-1j*l*phi # -1j*phi)
              * np.exp(-1j*l*(np.arctan2(self.yy, self.xx)+np.pi/2) -1j*phi) \
              * np.exp(-1j*(2*p+np.abs(l)+1)*np.arctan(z/self.zR)) \
             )
        normalized_LG = LG/np.sqrt(np.sum(np.abs(LG)**2))
        #self.total_I = np.sqrt(np.sum(np.absolute(LG)**2))
        normalized_I = np.abs(normalized_LG**2)/np.max(np.abs(normalized_LG**2))
        return [LG, normalized_LG, normalized_I]

    def convolve(self, e0, z):
        """Two-dimensional Convolution function.

        Args:
            e0 (~numpy.ndarray): Incident field for convolution.
            z (float): Propagation distance along z-direction [m].

        Returns:
            [propagation, normalized_I] (list): The propagation function and normalized intensity profile.

        Attributes:
            propagation (:obj:`~numpy.ndarray`): Convolution of incident field e0 and propagated field at z.
            normalized_I (:obj:`~numpy.ndarray`): 2D cross-sectional profile for normalized intensity distribution.

        References:
            * PG 57 from Goodman, Fourier Optics.
            * Chap 4: Fundamentals of Photonics

        .. seealso::
            * :func:`scipy.signal.convolve2d` Convolve two 2-dimensional arrays.
            * :func:`scipy.signal.convolve` Convolve :math:`T_{N}` 3-dimensional :obj:`~numpy.ndarray`.

        Todo:
            * Consolidate class function return values into instance variables using setter
              and getter decorators.

        """
        #if not np.allclose(0, z, 1e-3, 1e-3):
        #    propagation = fftconvolve(e0, np.exp(-1j*np.pi/z/self.wavelen*self.rr_sd**2), mode='same')
        #else:
        #    propagation = e0
        propagation = fftconvolve(e0, np.exp(-1j*np.pi/z/self.wavelen*self.rr_sd**2), mode='same')
        norm_prop = propagation/np.sqrt(np.sum(np.abs(propagation)**2))
        #norm_prop = propagation/(np.sqrt(np.sum(np.absolute(e0)**2)))
        #normalized_I = np.abs(norm_prop**2)/np.max(np.abs(norm_prop)**2)
        normalized_I = np.abs(norm_prop**2)/(np.max(np.abs((e0/np.sqrt(np.sum(np.abs(e0)**2)))**2)))
        #normalized_I = np.abs(norm_prop**2)/(np.max(np.abs((norm_prop/np.sqrt(np.sum(np.absolute(e0)**2)))**2)))
        #normalized_I = np.abs(propagation)**2
        return [propagation, normalized_I]

    def modulate(self, seed, hdf5table='phase'):
        """Calculate phase transmission function and modulate incident beam.

        Args:
            seed (~numpy.ndarray): Beam cross-sectional profile to modulate.

        Returns:
            (:obj:`numpy.ndarray`): Phase modulated beam.

        Todo:
            * Define seed as class instance variable `self.seed = gen_LG()`.
            * FIXME: Test if e0 is 3-dimensional i.e. [t,x,y]
            * TODO: Fix ordering of indices [x, y, t]
            * TODO: Fix for 2D vs 3D invariant

        """
        with h5py.File(self.hdf5filename, 'r', libver='latest') as hf:
            grp = hf.require_group(self.hdf5grp)
            phasemap = grp[hdf5table][:]
        #    print(np.shape(seed))
        #    print(np.shape(phasemap))
        #return seed[None, ...]*np.exp(1j*phasemap)
        return seed*np.exp(1j*np.remainder(phasemap, 2*np.pi))

class BeamProp3D(BeamProp2D):
    """Three-dimensional electromagnetic beam propagation code.

    Use beamprop2d for each z along axial path, and self re-seed.

    Todo:
        * TODO: Add functions for mirrors and output couplers etc.
        * TODO: Note that everything's been made relative to cm
        * TODO: 3D Propagation means having beam properties across crystal.
    """

    def __init__(self, wavelen=1.064e-4, waist=191.5e-4, p=0, l=0, z=0, phi=0, hdf5filename='hdf5db/storage.hdf5', \
                 hdf5grp='nb8', hdf5table_phase='phase3d', hdf5subgrp_beam='beam3d', hdf5table_intensity='intensity3d'):

        super().__init__(wavelen=wavelen, waist=waist, hdf5filename=hdf5filename, hdf5grp=hdf5grp)
        self.LG, self.normalized_LG, self.normalized_I = super().lg_mode(p, l, z, phi)

        self.hdf5subgrp = str(hdf5subgrp_beam)

        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            grp = hf.require_group(str(self.hdf5grp))

            zR_in_gain = False
            if self.zR < grp.attrs.get('Lz'):
                zR_in_gain = True
                self.z_ind_zR = (np.abs(grp['zzz'][0, 0, :][:] - self.zR).argmin())

            subgrp = grp.require_group(str(self.hdf5subgrp))
            subgrp.attrs.create('wavelength', self.wavelen)
            subgrp.attrs.create('waist', self.waist)
            subgrp.attrs.create('width', self.width)
            subgrp.attrs.create('zR', self.zR)
            subgrp.attrs.create('p', p)
            subgrp.attrs.create('l', l)
            subgrp.attrs.create('phi', phi)
            subgrp.attrs.create('z', z)
            #nt, nz, dz = grp.attrs.get('nt'), grp.attrs.get('nz'), grp.attrs.get('dz')
            # FIXME: Complex numbers and hdf5
            # if str(hdf5table_beam) not in grp:
            #     grp.create_dataset(str(hdf5table_beam), shape=np.shape(grp[str(hdf5table_phase)]), \
            #                        dtype=np.dtype(complex).type, maxshape=(MAX_NT, MAX_NX, MAX_NY, MAX_NZ))
            # else:
            #     grp[str(hdf5table_beam)].resize((np.shape(grp[str(hdf5table_phase)])))

            if str(hdf5table_intensity) not in subgrp:
                subgrp.create_dataset(str(hdf5table_intensity), shape=np.shape(grp[str(hdf5table_phase)]), \
                                   dtype=np.dtype(float).type, maxshape=(MAX_NT, MAX_NX, MAX_NY, MAX_NZ))
            else:
                subgrp[str(hdf5table_intensity)].resize((np.shape(grp[str(hdf5table_phase)])))

            if str(hdf5table_intensity + "zR") not in subgrp:
                subgrp.create_dataset(str(hdf5table_intensity + "zR"), \
                                   shape=[grp.attrs.get('nt'), grp.attrs.get('nx'), grp.attrs.get('ny')], \
                                   dtype=np.dtype(float).type, maxshape=(MAX_NT, MAX_NX, MAX_NY))
            else:
                subgrp[str(hdf5table_intensity + "zR")].resize(([grp.attrs.get('nt'), grp.attrs.get('nx'), grp.attrs.get('ny')]))

            self.temp_beam = np.zeros([grp.attrs.get('nt'), grp.attrs.get('nx'), grp.attrs.get('ny')]).astype(np.complex)

            for i_t in range(grp.attrs.get('nt')):
                #subgrp[str(hdf5table_beam)][t, :, :, 0] = self.LG[:, :]
                self.temp_beam[:, :, i_t] = self.LG[:, :]
                normalized_beam = np.zeros([grp.attrs.get('nx'), grp.attrs.get('ny')])
                for i_z in range(grp.attrs.get('nz')):
                    # self.temp_beam[:, :, t] = fftconvolve(subgrp[str(hdf5table_beam)][t, :, :, z,][:], \
                    #                                  np.exp(-1j*np.pi/dz/self.wavelen*self.rr_sd**2), \
                    #                                  mode='same')
                    # FIXME: TODO: Use super().convolve() Note that there is a modulation due to the crystal
                    self.temp_beam[i_t] = fftconvolve(self.temp_beam[i_t], \
                                                       np.exp(-1j*np.pi/grp.attrs.get('dz')/self.wavelen*self.rr_sd**2), \
                                                       mode='same') * np.exp(1j*grp[str(hdf5table_phase)][i_t, :, :, i_z][:])
                    normalized_beam = self.temp_beam[i_t] / np.sqrt(np.sum(np.abs(self.temp_beam[i_t])**2))
                    #normalized_beam = self.temp_beam[i_t, :, :] / (np.sum(np.abs(self.temp_beam[i_t, :, :])))
                    subgrp[str(hdf5table_intensity)][i_t, :, :, i_z] = np.abs(normalized_beam**2) / np.max(np.abs(normalized_beam**2))

                if zR_in_gain:
                    subgrp[str(hdf5table_intensity + "zR")][i_t] = subgrp[str(hdf5table_intensity)][i_t, :, :, self.z_ind_zR][:]

                else:
                    subgrp[str(hdf5table_intensity + "zR")][i_t] = super().convolve(self.temp_beam[i_t], self.zR)[1]

    def convolve(self, z, hdf5table_prop_intensity='intensity3d_prop'):
        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            grp = hf.require_group(self.hdf5grp)
            subgrp = grp.require_group(self.hdf5subgrp)
            if str(hdf5table_prop_intensity) not in subgrp:
                subgrp.create_dataset(str(hdf5table_prop_intensity), \
                                   shape=[grp.attrs.get('nt'), grp.attrs.get('nx'), grp.attrs.get('ny')], \
                                   dtype=np.dtype(float).type, maxshape=(MAX_NT, MAX_NX, MAX_NY))
            else:
                subgrp[str(hdf5table_prop_intensity)].resize(([grp.attrs.get('nt'), grp.attrs.get('nx'), grp.attrs.get('ny')]))

            for i_t in range(grp.attrs.get('nt')):
                subgrp[str(hdf5table_prop_intensity)][i_t] = super.convolve(self.temp_beam[i_t, :, :, -1], z)[1]

            #subgroup[str(hdf5table_prop_intensity + 'lgmode')]

#  LocalWords:  Laguerre Seealso ndarrays
