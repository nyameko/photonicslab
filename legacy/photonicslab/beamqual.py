# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""Laser beam quality measures."""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

from photonicslab.config import *

from photonicslab.optomech import ParaxialWaveEqn

#from sklearn.preprocessing import normalize

# Seealso:
# import numpy.polynomial.laguerre as lg
# import scipy.special.eval_genlaguerre as eval_lg

import time

class MSquared(ParaxialWaveEqn):
    """Determine the :math:`M^2` beam quality factor from simulated or experimental propagation data.

    Args:
        wavelen (float): Wavelength of the incident seed beam [cm].
        waist (float): Location along z-direction of seed beam waist [cm].
        p (int, p>=0): Radial mode index denoting number of radial nodes.
        l (int): Azimuthal mode index denoting number of intertwined helices.
        z (float): Propagation distance along optical axis [cm].
        phi (float): Vortex phase term [rad].
        hdf5filename (:class:`~h5py:File`): Filename of the HDF5 database.
        hdf5grp (:class:`~h5py:Group`): Group name of the HDF5 group object, with the thermo-optic and opto-mechanical physics data.
        hdf5_phase (:class:`~h5py:Dataset`): HDF5 dataset structure name for phase modulation object
        hdf5subgrp (:class:`~h5py:Group`): Subgroup name of the HDF5 subgroup object to store beam metrics.
        hdf5subset_intensity (:class:`~h5py:Dataset`): HDF5 dataset name storing temporal intensity function.
        hdf5subset_field (:class:`~h5py:Dataset`): HDF5 data structure name storing temporal complex field.

    Attributes:
        parentgrp (:class:`~h5py:Group`): HDF5 group object with amplification / gain thermal-opto-mechanical lensing / phase modulation data.
        grp (:class:`~h5py:Group`): HDF5 subgroup object for beam properties.
        fseed (:obj:`~numpy.ndarray`): Incident `LG_{lp}^\phi(z)` mode seed beam profile. Default is a Gaussian.
        fseed_mod (:obj:`~numpy.ndarray`): Phase modulated incident seed beam profile.

    Todo:
        * TODO: This class should inherit from optomech.ParaxialWaveEqn
        * TODO: Additionally there's the convolution based beam propagation, and then the fem-bpm
    """

    def __init__(self, wavelen=1.064e-4, waist=191.5e-4, flens=5e-1, p=0, l=0, z=0, phi=0, hdf5filename='hdf5db/storage.hdf5', \
                 hdf5grp='nb11', hdf5set_phase='phase', hdf5subgrp='seed_qual', hdf5subset_intensity='intensity', \
                 hdf5subset_field='field'):

        super().__init__(wavelen, waist, flens, p, l, z, phi, hdf5filename, hdf5grp, hdf5subgrp, hdf5subset_intensity, hdf5subset_field)

        self.hdf5set_phase = str(hdf5set_phase)

        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            parentgrp = hf.require_group(self.hdf5grp)
            grp = parentgrp.require_group(self.hdf5subgrp)
            self.Lx = parentgrp.attrs.get('Lx')
            self.Ly = parentgrp.attrs.get('Ly')
            self.Lz = parentgrp.attrs.get('Lz')
            # For simulating regular Gaussian
            #self.fseed_mod = np.asarray([self.fseed[0] for t in range(parentgrp.attrs.get('nt'))], dtype=np.dtype(complex).type)
            if str(hdf5set_phase) in parentgrp:
                self.fseed_mod = super().modulate(self.fseed[0], hdf5table=self.hdf5set_phase)

    def measure_focal_length(self, z_start, z_end, z_start_y=None, z_end_y=None, x_centroid=0.0, y_centroid=0.0, \
                             max_iter=250, atol=1e-1, debug=False):
        """Use binary / bisection search to determine `z_{0x}` and `z_{0y}` i.e. axial focal lengths in both x and y for astigmatic beam profiles.

        Args:
            z_start (float): Propagation start range for measurements.
            z_end (float): Propagation stop range for measurements.
            z_start_y (float): Propagation start range for measurements.
            z_end_y (float): Propagation stop range for measurements.
            y_centroid (float): y value corresponding to peak transverse intensity along x
            x_centroid (float): x value corresponding to peak transverse intensity along y
            max_iter (int): Maximum number of convolutions.
            atol (float): Absolute tolerance, determines stopping condition when z_start and z_end frames are sufficiently close.
            debug (bool): Prints debugging information during execution.

        Attributes:
            parentgrp (:class:`~h5py:Group`): HDF5 group object with amplification / gain thermal-opto-mechanical lensing / phase modulation data.
            grp (:class:`~h5py:Group`): HDF5 subgroup object for beam properties.
            count (int): Search loop iteration counter.
            z_mid (float): Propagation midrange for binary search \ bisection method.
            fseed_mod_z_start (:obj:`~numpy.ndarray`): Search boundary for transverse intensity profile.
            fseed_mod_z_mid (:obj:`~numpy.ndarray`): Search boundary for transverse intensity profile.
            fseed_mod_z_endface (:obj:`~numpy.ndarray`): Search boundary for transverse intensity profile.

        Returns:
            (list): Focal lengths `z_{0x}` and `z_{0y}`, and the corresponding field and intensity profiles are stored in HDF5 database for calculating `M^2` beam quality measure.

        Todo:
            * TODO: Generalize logic to find 2 peaks independently in x and y. (Double the searches though)
            * TODO: Exception handling.

        """

        # while z_satart < z_end
        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            parentgrp = hf.require_group(self.hdf5grp)
            grp = parentgrp.require_group(self.hdf5subgrp)

            count = 0
            while (np.abs(z_start - z_end) >= atol) and (count < max_iter):
                z_mid = float(z_start + z_end)/2

                fseed_mod_z_mid_x = super().convolve(self.fseed_mod[-1], z_mid)[1][:, np.abs(self.yy[0, :] - y_centroid).argmin()]

                fseed_mod_z_start_x = super().convolve(self.fseed_mod[-1], z_start)[1][:, np.abs(self.yy[0, :] - y_centroid).argmin()]
                fseed_mod_z_end_x = super().convolve(self.fseed_mod[-1], z_end)[1][:, np.abs(self.yy[0, :] - y_centroid).argmin()]

                if debug:
                    print("Loop iteration {0}".format(int(count)))
                    print("Peak intensity z_start_x {0:.6} at {1:.6} cm".format(np.max(fseed_mod_z_start_x), float(z_start)))
                    print("Peak intensity z_mid_x {0:.6} at {1:.6} cm".format(np.max(fseed_mod_z_mid_x), float(z_mid)))
                    print("Peak intensity z_end_x {0:.6} at {1:.6} cm".format(np.max(fseed_mod_z_end_x), float(z_end)))

                if (np.max(fseed_mod_z_mid_x) > np.max(fseed_mod_z_start_x)):
                    z_start = float(z_start + z_mid)/2
                if (np.max(fseed_mod_z_mid_x) > np.max(fseed_mod_z_end_x)):
                    z_end = float(z_end + z_mid)/2

                count+=1

            count = 0
            # Need some exception handling here
            if (z_start_y is None) and (z_end_y is None):
                z_start_y = z_start
                z_mid_y = z_mid
                z_end_y = z_end
            else:
                while (np.abs(z_start_y - z_end_y) >= atol) and (count < max_iter):
                    z_mid_y = float(z_start_y + z_end_y)/2

                    fseed_mod_z_mid_y = super().convolve(self.fseed_mod[-1], z_mid_y)[1][np.abs(self.xx[:, 0] - x_centroid).argmin(), :]

                    fseed_mod_z_start_y = super().convolve(self.fseed_mod[-1], z_start_y)[1][np.abs(self.xx[:, 0] - x_centroid).argmin(), :]
                    fseed_mod_z_end_y = super().convolve(self.fseed_mod[-1], z_end_y)[1][np.abs(self.xx[:, 0] - x_centroid).argmin(), :]

                    if debug:
                        print("Loop iteration {0}".format(int(count)))
                        print("Peak intensity z_start_y {0:.6} at {1:.6} cm".format(np.max(fseed_mod_z_start_y), float(z_start_y)))
                        print("Peak intensity z_mid_y {0:.6} at {1:.6} cm".format(np.max(fseed_mod_z_mid_y), float(z_mid_y)))
                        print("Peak intensity z_end_y {0:.6} at {1:.6} cm".format(np.max(fseed_mod_z_end_y), float(z_end_y)))

                    if (np.max(fseed_mod_z_mid_y) > np.max(fseed_mod_z_start_y)):
                        z_start_y = float(z_start_y + z_mid_y)/2
                    if (np.max(fseed_mod_z_mid_y) > np.max(fseed_mod_z_end_y)):
                        z_end_y = float(z_end_y + z_mid_y)/2

                    count+=1

            grp.attrs.create('z0x', float((z_start + z_mid + z_end)/3))
            grp.attrs.create('z0y', float((z_start_y + z_mid_y + z_end_y)/3))

            if str(self.hdf5subset_intens + '_0x') not in grp:
                grp.create_dataset(str(self.hdf5subset_intens + '_0x'), shape=(parentgrp.attrs.get('nt'), parentgrp.attrs.get('nx'), \
                                   parentgrp.attrs.get('ny')), dtype=np.dtype(float).type, maxshape=(MAX_NT, MAX_NX, MAX_NY))
            else:
                grp[str(self.hdf5subset_intens + '_0x')].resize([parentgrp.attrs.get('nt'), parentgrp.attrs.get('nx'), parentgrp.attrs.get('ny')])

            if str(self.hdf5subset_field + '_0x') not in grp:
                grp.create_dataset(str(self.hdf5subset_field + '_0x'), shape=(parentgrp.attrs.get('nt'), parentgrp.attrs.get('nx'), \
                                   parentgrp.attrs.get('ny')), dtype=np.dtype(complex).type, maxshape=(MAX_NT, MAX_NX, MAX_NY))
            else:
                grp[str(self.hdf5subset_field + '_0x')].resize([parentgrp.attrs.get('nt'), parentgrp.attrs.get('nx'), parentgrp.attrs.get('ny')])

            grp[str(self.hdf5subset_field) + '_0x'][...] = np.asarray([super().convolve(self.fseed_mod[t], grp.attrs.get('z0x'))[0] \
                                                                       for t in range(np.shape(self.fseed_mod)[0])], dtype=np.dtype(complex).type)
            grp[str(self.hdf5subset_intens) + '_0x'][...] = np.asarray([super().convolve(self.fseed_mod[t], grp.attrs.get('z0x'))[1] \
                                                                        for t in range(np.shape(self.fseed_mod)[0])], dtype=np.dtype(float).type)

            if str(self.hdf5subset_intens + '_0y') not in grp:
                grp.create_dataset(str(self.hdf5subset_intens + '_0y'), shape=(parentgrp.attrs.get('nt'), parentgrp.attrs.get('nx'), \
                                   parentgrp.attrs.get('ny')), dtype=np.dtype(float).type, maxshape=(MAX_NT, MAX_NX, MAX_NY))
            else:
                grp[str(self.hdf5subset_intens + '_0y')].resize([parentgrp.attrs.get('nt'), parentgrp.attrs.get('nx'), parentgrp.attrs.get('ny')])

            if str(self.hdf5subset_field + '_0y') not in grp:
                grp.create_dataset(str(self.hdf5subset_field + '_0y'), shape=(parentgrp.attrs.get('nt'), parentgrp.attrs.get('nx'), \
                                   parentgrp.attrs.get('ny')), dtype=np.dtype(complex).type, maxshape=(MAX_NT, MAX_NX, MAX_NY))
            else:
                grp[str(self.hdf5subset_field + '_0y')].resize([parentgrp.attrs.get('nt'), parentgrp.attrs.get('nx'), parentgrp.attrs.get('ny')])

            grp[str(self.hdf5subset_field) + '_0y'][...] = np.asarray([super().convolve(self.fseed_mod[t], grp.attrs.get('z0y'))[0] \
                                                                       for t in range(np.shape(self.fseed_mod)[0])], dtype=np.dtype(complex).type)
            grp[str(self.hdf5subset_intens) + '_0y'][...] = np.asarray([super().convolve(self.fseed_mod[t], grp.attrs.get('z0y'))[1] \
                                                                        for t in range(np.shape(self.fseed_mod)[0])], dtype=np.dtype(float).type)

            return [grp.attrs.get('z0x'), grp.attrs.get('z0y')]

    def measure_waist_first_null(self, x_centroid=0.0, y_centroid=0.0, debug=False):
        """Determine `w_{0x}` and `w_{0y}` using first nulls i.e. transverse beam widths in both x and y for astigmatic beam profiles.

        Args:
            y_centroid (float): y value corresponding to peak transverse intensity along x.
            x_centroid (float): x value corresponding to peak transverse intensity along y.
            debug (bool): Prints debugging information during execution.

        Returns:
            (list): [w0x, w0y] Beam waists in x and y transverse planes.

        Raises:
            KeyError: HDF5 dataset for transverse intensity at focal length missing.
            UserWarning: Beam cross section intensity profile non-symmetrical about peak value.
            IndexError: Unable to find beam waist. I.e. Search did not start near peak intensity.
            UserWarning: Warns of astigmatic beams widths.

        Attributes:
            parentgrp (:class:`~h5py:Group`): HDF5 group object with amplification / gain thermal-opto-mechanical lensing / phase modulation data.
            grp (:class:`~h5py:Group`): HDF5 subgroup object for beam properties.
            ~w0x: (float): Beam radius in transverse x plane.
            ~w0x_peak_idx (int): Index for peak intensity value along x transverse plane.
            ~w0x_left_idx (int): Index searching for intensity turning point value along x transverse plane.
            ~w0x_right_idx (int): Index searching for intensity turning point value along x transverse plane.
            ~w0y: (float): Beam radius in transverse y plane.
            ~w0y_peak_idx (int): Index for peak intensity value along y transverse plane.
            ~w0y_left_idx (int): Index searching for intensity turning point value along y transverse plane.
            ~w0y_right_idx (int): Index searching for intensity turning point value along y transverse plane.

        """

        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            parentgrp = hf.require_group(self.hdf5grp)
            grp = parentgrp.require_group(self.hdf5subgrp)

            try:
                if (str(self.hdf5subset_intens + '_0x') not in grp) or (str(self.hdf5subset_intens + '_0y') not in grp):
                    raise UserWarning("Focal length unknown. First run measure_astigmatic_focal_length().")
            except UserWarning as err:
                raise KeyError("HDF5 Dataset Missing {0}".format(err)) from err

            w0x = 0.0
            w0x_peak_idx = (grp[str(self.hdf5subset_intens + '_0x')][-1][np.abs(self.yy[0, :] - y_centroid).argmin(), :]).argmax()
            w0x_left_idx = w0x_peak_idx
            w0x_right_idx = w0x_peak_idx

            try:
                while (grp[str(self.hdf5subset_intens + '_0x')][-1][np.abs(self.yy[0, :] - y_centroid).argmin(), w0x_left_idx] \
                       > grp[str(self.hdf5subset_intens + '_0x')][-1][np.abs(self.yy[0, :] - y_centroid).argmin(), w0x_left_idx-1]):
                    if debug:
                        print("w0x_left_idx: {0}".format(w0x_left_idx))
                    w0x_left_idx-=1
                while (grp[str(self.hdf5subset_intens + '_0x')][-1][np.abs(self.yy[0, :] - y_centroid).argmin(), w0x_right_idx] \
                       > grp[str(self.hdf5subset_intens + '_0x')][-1][np.abs(self.yy[0, :] - y_centroid).argmin(), w0x_right_idx+1]):
                    if debug:
                        print("w0x_right_idx: {0}".format(w0x_right_idx))
                    w0x_right_idx+=1
                if np.abs((grp[str(self.hdf5subset_intens + '_0x')][-1][np.abs(self.yy[0, :] - y_centroid).argmin(), :]).argmax() - w0x_left_idx) \
                   != np.abs((grp[str(self.hdf5subset_intens + '_0x')][-1][np.abs(self.yy[0, :] - y_centroid).argmin(), :]).argmax() - w0x_right_idx):
                    raise UserWarning("Non-symmetric beam waist.")

            except IndexError as err:
                raise IndexError("You are not at a focus - out of bounds error. {0}".format(err)) from err
            except UserWarning as astigmatic_beam_warning:
                warnings.warn("AstigmaticBeamWarning: {0}".format(astigmatic_beam_warning), UserWarning)
            finally:
                w0x = (np.abs(self.xx[w0x_peak_idx, 0] - self.xx[w0x_left_idx, 0]) + np.abs(self.xx[w0x_peak_idx, 0] - self.xx[w0x_right_idx, 0]))/2

            w0y = 0.0
            w0y_peak_idx = (grp[str(self.hdf5subset_intens + '_0y')][-1][np.abs(self.xx[:, 0] - x_centroid).argmin(), :]).argmax()
            w0y_left_idx = w0y_peak_idx
            w0y_right_idx = w0y_peak_idx

            try:
                while (grp[str(self.hdf5subset_intens + '_0y')][-1][np.abs(self.xx[:, 0] - x_centroid).argmin(), w0y_left_idx] \
                       > grp[str(self.hdf5subset_intens + '_0y')][-1][np.abs(self.xx[:, 0] - x_centroid).argmin(), w0y_left_idx-1]):
                    if debug:
                        print("w0y_left_idx: {0}".format(w0y_left_idx))
                    w0y_left_idx-=1
                while (grp[str(self.hdf5subset_intens + '_0y')][-1][np.abs(self.xx[:, 0] - x_centroid).argmin(), w0y_right_idx] \
                       > grp[str(self.hdf5subset_intens + '_0y')][-1][np.abs(self.xx[:, 0] - x_centroid).argmin(), w0y_right_idx+1]):
                    if debug:
                        print("w0y_right_idx: {0}".format(w0y_right_idx))
                    w0y_right_idx+=1
                if np.abs(w0y_peak_idx - w0y_left_idx) != np.abs(w0y_peak_idx - w0y_right_idx):
                    raise UserWarning("Non-symmetric beam waist.")

            except IndexError:
                raise IndexError("You are not at a focus - out of bounds error. {0}".format(err)) from err
            except UserWarning as astigmatic_beam_warning:
                warnings.warn("AstigmaticBeamWarning: {0}".format(astigmatic_beam_warning), UserWarning)
            finally:
                w0y = (np.abs(self.yy[0, w0y_peak_idx] - self.yy[0, w0y_left_idx]) + np.abs(self.yy[0, w0y_peak_idx] - self.yy[0, w0y_right_idx]))/2

            try:
                if w0y != w0x:
                    raise UserWarning("Astigmatic beam widths")
            except UserWarning as astigmatic_beam_warning:
                warnings.warn("AstigmaticBeamWarning: {0}".format(astigmatic_beam_warning), UserWarning)
            finally:
                grp.attrs.create('w0x', w0x)
                grp.attrs.create('w0y', w0y)

            if debug:
                print("Rayleigh range in x {0:.3}".format(np.pi*grp.attrs.get('w0x')**2/self.wavelen))
                print("Rayleigh range in y {0:.3}".format(np.pi*grp.attrs.get('w0y')**2/self.wavelen))

            return [grp.attrs.get('w0x'), grp.attrs.get('w0y')]

    def measure_waist_fwhm(self, x_centroid=0.0, y_centroid=0.0, debug=False):
        """Determine `w_{0x}` and `w_{0y}` using FULL WIDTH HALF MAXIMUM definition i.e. transverse beam widths in both x and y for astigmatic beam profiles.

        Args:
            y_centroid (float): y value corresponding to peak transverse intensity along x.
            x_centroid (float): x value corresponding to peak transverse intensity along y.
            debug (bool): Prints debugging information during execution.

        Returns:
            (list): [w0x, w0y] Beam waists in x and y transverse planes.

        Raises:
            KeyError: HDF5 dataset for transverse intensity at focal length missing.
            UserWarning: Beam cross section intensity profile non-symmetrical about peak value.
            IndexError: Unable to find beam waist. I.e. Search did not start near peak intensity.
            UserWarning: Warns of astigmatic beams widths.

        Attributes:
            parentgrp (:class:`~h5py:Group`): HDF5 group object with amplification / gain thermal-opto-mechanical lensing / phase modulation data.
            grp (:class:`~h5py:Group`): HDF5 subgroup object for beam properties.
            w0x: (float): Beam radius in transverse x plane.
            w0x_peak_idx (int): Index for peak intensity value along x transverse plane.
            w0x_left_idx (int): Index searching for intensity turning point value along x transverse plane.
            w0x_right_idx (int): Index searching for intensity turning point value along x transverse plane.
            w0y: (float): Beam radius in transverse y plane.
            w0y_peak_idx (int): Index for peak intensity value along y transverse plane.
            w0y_left_idx (int): Index searching for intensity turning point value along y transverse plane.
            w0y_right_idx (int): Index searching for intensity turning point value along y transverse plane.

        """

        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            parentgrp = hf.require_group(self.hdf5grp)
            grp = parentgrp.require_group(self.hdf5subgrp)

            try:
                if (str(self.hdf5subset_intens + '_0x') not in grp) or (str(self.hdf5subset_intens + '_0y') not in grp):
                    raise KeyError("Focal length unknown. First run measure_astigmatic_focal_length().")

                tempIntens_x = (grp[str(self.hdf5subset_intens + '_0x')][-1][np.abs(self.yy[0, :] - y_centroid).argmin(), :])
                w0x_peak_idx = tempIntens_x.argmax()
                w0x_left_idx = np.abs((0.5*tempIntens_x[w0x_peak_idx] - tempIntens_x[0:w0x_peak_idx])).argmin()
                w0x_right_idx = w0x_peak_idx + np.abs((0.5*tempIntens_x[w0x_peak_idx] - tempIntens_x[w0x_peak_idx:parentgrp.attrs.get('nx')])).argmin()
                w0x = (np.abs(self.xx[w0x_peak_idx, 0] - self.xx[w0x_left_idx, 0]) + np.abs(self.xx[w0x_peak_idx, 0] - self.xx[w0x_right_idx, 0]))/2

                tempIntens_y = (grp[str(self.hdf5subset_intens + '_0x')][-1][:, np.abs(self.xx[:, 0] - x_centroid).argmin()])
                w0y_peak_idx = tempIntens_y.argmax()
                w0y_left_idx = np.abs((0.5*tempIntens_y[w0y_peak_idx] - tempIntens_y[0:w0y_peak_idx])).argmin()
                w0y_right_idx = w0y_peak_idx + np.abs((0.5*tempIntens_y[w0y_peak_idx] - tempIntens_y[w0y_peak_idx:parentgrp.attrs.get('ny')])).argmin()
                w0y = (np.abs(self.yy[0, w0y_peak_idx] - self.yy[0, w0y_left_idx]) + np.abs(self.yy[0, w0y_peak_idx] - self.yy[0, w0y_right_idx]))/2

                if debug:
                    print("w0x_peak_idx: {0}".format(w0x_peak_idx))
                    print("w0x_left_idx: {0}".format(w0x_left_idx))
                    print("w0x_right_idx: {0}".format(w0x_right_idx))

                    print("w0y_peak_idx: {0}".format(w0y_peak_idx))
                    print("w0y_left_idx: {0}".format(w0y_left_idx))
                    print("w0y_right_idx: {0}".format(w0y_right_idx))

                if np.abs(w0y_peak_idx - w0y_left_idx) != np.abs(w0y_peak_idx - w0y_right_idx) or \
                   np.abs(w0x_peak_idx - w0x_left_idx) != np.abs(w0x_peak_idx - w0x_right_idx) or \
                   w0y != w0x:
                    raise UserWarning("Non-symmetric beam waist.")

            except UserWarning as astigmatic_beam_warning:
                warnings.warn("AstigmaticBeamWarning: {0}".format(astigmatic_beam_warning), UserWarning)

            except KeyError as err:
                raise KeyError("HDF5 Dataset Missing {0}".format(err)) from err

            finally:
                grp.attrs.create('w0x', w0x)
                grp.attrs.create('w0y', w0y)

            if debug:
                print("Rayleigh range in x {0:.3}".format(np.pi*grp.attrs.get('w0x')**2/self.wavelen))
                print("Rayleigh range in y {0:.3}".format(np.pi*grp.attrs.get('w0y')**2/self.wavelen))

            return [grp.attrs.get('w0x'), grp.attrs.get('w0y')]

    def measure_waist_variance(self, x_centroid=0.0, y_centroid=0.0, debug=False):
        """Determine `w_{0x}` and `w_{0y}` using statistical variance definition i.e. transverse beam widths in both x and y for astigmatic beam profiles.

        Args:
            y_centroid (float): y value corresponding to peak transverse intensity along x.
            x_centroid (float): x value corresponding to peak transverse intensity along y.
            debug (bool): Prints debugging information during execution.

        Returns:
            (list): [w0x, w0y] Beam waists in x and y transverse planes.

        Raises:
            KeyError: HDF5 dataset for transverse intensity at focal length missing.
            UserWarning: Beam cross section intensity profile non-symmetrical about peak value.
            IndexError: Unable to find beam waist. I.e. Search did not start near peak intensity.
            UserWarning: Warns of astigmatic beams widths.

        Attributes:
            parentgrp (:class:`~h5py:Group`): HDF5 group object with amplification / gain thermal-opto-mechanical lensing / phase modulation data.
            grp (:class:`~h5py:Group`): HDF5 subgroup object for beam properties.

        """

        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            parentgrp = hf.require_group(self.hdf5grp)
            grp = parentgrp.require_group(self.hdf5subgrp)

            try:
                if (str(self.hdf5subset_intens + '_0x') not in grp) or (str(self.hdf5subset_intens + '_0y') not in grp):
                    raise UserWarning("Focal length unknown. First run measure_astigmatic_focal_length().")
            except UserWarning as err:
                raise KeyError("HDF5 Dataset Missing {0}".format(err)) from err
            finally:
                grp.attrs.create('w0x', np.sqrt(4*(np.sum((self.xx - x_centroid)**2 * grp[str(self.hdf5subset_intens + '_0x')]) \
                                                 / np.sum(grp[str(self.hdf5subset_intens + '_0x')]))))
                grp.attrs.create('w0y', np.sqrt(4*(np.sum((self.yy - y_centroid)**2 * grp[str(self.hdf5subset_intens + '_0y')]) \
                                                 / np.sum(grp[str(self.hdf5subset_intens + '_0y')]))))
                ##grp.attrs.create('w0x', 4*(np.sum((self.xx - x_centroid)**2 * grp[str(self.hdf5subset_intens + '_0x')]) \
                ##                                 / np.sum(grp[str(self.hdf5subset_intens + '_0x')])))
                #grp.attrs.create('w0x', 2*(np.var(grp[str(self.hdf5subset_intens + '_0x')]) \
                #                                 / np.sum(grp[str(self.hdf5subset_intens + '_0x')])))
                ##grp.attrs.create('w0y', 4*(np.sum((self.yy - y_centroid)**2 * grp[str(self.hdf5subset_intens + '_0y')]) \
                ##                                 / np.sum(grp[str(self.hdf5subset_intens + '_0y')])))
                #grp.attrs.create('w0y', 4*(np.var(grp[str(self.hdf5subset_intens + '_0y')]) \
                #                                 / np.sum(grp[str(self.hdf5subset_intens + '_0y')])))
                #grp.attrs.create('w0x', 2*(np.var(grp[str(self.hdf5subset_intens + '_0x')], axis=0) / np.sum(grp[str(self.hdf5subset_intens + '_0x')])))
                #grp.attrs.create('w0y', 2*(np.var(grp[str(self.hdf5subset_intens + '_0y')], axis=1) / np.sum(grp[str(self.hdf5subset_intens + '_0y')])))

            if debug:
                print("Rayleigh range in x {0:.3}".format(np.pi*grp.attrs.get('w0x')**2/self.wavelen))
                print("Rayleigh range in y {0:.3}".format(np.pi*grp.attrs.get('w0y')**2/self.wavelen))

            return [grp.attrs.get('w0x'), grp.attrs.get('w0y')]

    def measure_quality(self, z_start=80, z_end=200, num_measurements=12, x_centroid=0.0, \
                        y_centroid=0.0, deg=2, debug=False, z0x=None, z0y=None, w0x=None, w0y=None):
        """Calculates the field, intensity, variance and finally M-Squared of the beam between the given ranges.

        The beam can also be astigmatic

        Args:
            z_start (float): Propagation start range for measurements.
            z_end (float): Propagation stop range for measurements.
            num_measurements (int): Number of measurements to integrate over.
            x0 (float):
            y0 (float):

        Todo:
            * TODO: Calculate beam centroid for xbar/x0 and ybar/y0, for off axis beams, maybe.
            * TODO: Add exception handling for focal length and waist detection.
            * TODO: Debug  1D array side h5py dataset

        References:
            * `A. E. Siegman, "Howto (maybe) measure laser beam quality" OSA Annual Meeting, 1997 <https://web.archive.org/web/20110604095354/http://www.stanford.edu/~siegman/beams_and_resonators/beam_quality_tutorial_osa.pdf>`_
        """

        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            parentgrp = hf.require_group(self.hdf5grp)
            grp = parentgrp.require_group(self.hdf5subgrp)

            if str('variance_x') not in grp:
                grp.create_dataset(str('variance_x'), shape=(num_measurements, 1), dtype=np.dtype(float).type, maxshape=(MAX_NZ, 1))
            else:
                grp[str('variance_x')].resize([num_measurements, 1])

            if str('variance_y') not in grp:
                grp.create_dataset(str('variance_y'), shape=(num_measurements, 1), dtype=np.dtype(float).type, maxshape=(MAX_NZ, 1))
            else:
                grp[str('variance_y')].resize([num_measurements, 1])

            if str('MSquared_x') not in grp:
                grp.create_dataset(str('MSquared_x'), shape=(num_measurements, 1), dtype=np.dtype(float).type, maxshape=(MAX_NZ, 1))
            else:
                grp[str('MSquared_x')].resize([num_measurements, 1])

            if str('MSquared_y') not in grp:
                grp.create_dataset(str('MSquared_y'), shape=(num_measurements, 1), dtype=np.dtype(float).type, maxshape=(MAX_NZ, 1))
            else:
                grp[str('MSquared_y')].resize([num_measurements, 1])

            if str('MSquared_x_fit') not in grp:
                grp.create_dataset(str('MSquared_x_fit'), shape=(3, 1), dtype=np.dtype(float).type)
            if str('MSquared_y_fit') not in grp:
                grp.create_dataset(str('MSquared_y_fit'), shape=(3, 1), dtype=np.dtype(float).type)

            if z0x is None:
                z0x = grp.attrs.get('z0x')
            if z0y is None:
                z0y = grp.attrs.get('z0y')
            if w0x is None:
                w0x = grp.attrs.get('w0x')
            if w0y is None:
                w0y = grp.attrs.get('w0y')

            self.z_range = np.linspace(z_start, z_end, num_measurements)
            if debug:
                print("Wavelength: {0}".format(grp.attrs.get(str('wl'))))
                print("Focal length in x: {0}".format(grp.attrs.get(str('z0x'))))
                print("Focal length in y: {0}".format(grp.attrs.get(str('z0y'))))
                print("Beam waist in x: {0}".format(grp.attrs.get(str('w0x'))))
                print("Beam waist in y: {0}".format(grp.attrs.get(str('w0y'))))
                print("Rayleigh range in x {0:.3}".format(np.pi*4*grp.attrs.get('w0x')**2/self.wavelen))
                print("Rayleigh range in y {0:.3}".format(np.pi*4*grp.attrs.get('w0y')**2/self.wavelen))


            for iz in range(num_measurements):
                # Generate phase modulated beam for =all times= steady state.
                # Calculate variance in x and y
                # TODO: Just noww: Delete temporal loop
                # TODO: Fix assignment below
                # TODO: Test negative sign in propagation
                # TODO: Get tempIntens separately in x and y.
                tempIntens = super().convolve(self.fseed_mod[-1], -1*self.z_range[iz])[1]
                #grp[str('variance_x')][iz] = (np.var(tempIntens) / np.sum(tempIntens))
                # grp[str('variance_x')][iz] = (4*(np.sum((self.xx - x_centroid)**2 * tempIntens \
                #                            * parentgrp.attrs.get('dx') * parentgrp.attrs.get('dy')) \
                #                            / np.sum(tempIntens * parentgrp.attrs.get('dx') * parentgrp.attrs.get('dy'))))
                # grp[str('variance_y')][iz] = (4*(np.sum((self.yy - y_centroid)**2 * tempIntens \
                #                            * parentgrp.attrs.get('dx') * parentgrp.attrs.get('dy')) \
                #                            / np.sum(tempIntens * parentgrp.attrs.get('dx') * parentgrp.attrs.get('dy'))))

                #grp[str('variance_x')][iz] = np.sqrt(4*np.sum((self.xx - x_centroid)**2 * tempIntens) / np.sum(tempIntens))
                #grp[str('variance_y')][iz] = np.sqrt(4*np.sum((self.yy - y_centroid)**2 * tempIntens) / np.sum(tempIntens))
                grp[str('variance_x')][iz] = np.sqrt(4*np.sum((self.xx - x_centroid)**2 * tempIntens) / np.sum(tempIntens))
                grp[str('variance_y')][iz] = np.sqrt(4*np.sum((self.yy - y_centroid)**2 * tempIntens) / np.sum(tempIntens))

                # grp[str('variance_y')][iz] = (np.var(tempIntens) )
                #grp[str('variance_y')][iz] = (4*np.sum((self.yy - y_centroid)**2 * tempIntens) / np.sum(tempIntens))
                #grp[str('MSquared_x')][iz] = np.sqrt( np.abs((grp[str('variance_x')][iz])**2 - w0x**2 ) \
                #                                      / ((grp.attrs.get('wl')/(np.pi*w0x))**2 * (self.z_range[iz] - z0x)**2) )
                grp[str('MSquared_x')][iz] = np.sqrt( np.abs((grp[str('variance_x')][iz])**2 - w0x**2 ) \
                                                      / ((grp.attrs.get('wl')/(np.pi*w0x))**2 * (self.z_range[iz] - z0x)**2) )
                # grp[str('MSquared_x')][iz] = np.sqrt( np.abs((2*grp[str('variance_x')][iz])**2 - grp.attrs.get('w0x')**2 ) \
                #                                       / ((grp.attrs.get('wl')/(np.pi*grp.attrs.get('w0x')))**2 * (self.z_range[iz] - grp.attrs.get('z0x'))**2) )
                grp[str('MSquared_y')][iz] = np.sqrt( np.abs((grp[str('variance_y')][iz])**2 - w0y**2 ) \
                                                      / ((grp.attrs.get('wl')/(np.pi*w0y))**2 * (self.z_range[iz] - z0y)**2) )
                if debug:
                    print("Axial distance: {0}".format(self.z_range[iz]))
                    print("Variance_x: {0}".format(grp[str('variance_x')][iz]))
                    print("Variance_y: {0}".format(grp[str('variance_y')][iz]))
                    print("MSquared_x: {0}".format(grp[str('MSquared_x')][iz]))
                    print("MSquared_y: {0}".format(grp[str('MSquared_y')][iz]))

            [m2x_a, m2x_b, m2x_c,] = np.polyfit(self.z_range, (grp[str('variance_x')][:])**2, deg=deg)
            [m2y_a, m2y_b, m2y_c,] = np.polyfit(self.z_range, (grp[str('variance_y')][:])**2, deg=deg)

            #grp[str('MSquared_x_fit')] = np.polyfit(self.z_range, grp[str('variance_x')], deg=2)
            #grp[str('MSquared_y_fit')] = np.polyfit(self.z_range, grp[str('variance_y')], deg=2)

            Mx = np.pi/grp.attrs.get('wl')*np.sqrt(m2x_a*m2x_c - (m2x_b**2/4))
            My = np.pi/grp.attrs.get('wl')*np.sqrt(m2y_a*m2y_c - (m2y_b**2/4))
            return [np.mean(grp[str('MSquared_x')]), np.mean(grp[str('MSquared_y')]), Mx, My]

    def plot_quality_profile(self, savepath='images/quality.pang'):
        """Plot the current cross-sectional source profile from the heat load object.

        Args:
            z (float): Source attenuation distance along axial direction and within crystal, :math:`z=0` is the pump end-face.
            time_step (int): Integer placeholder of temporal indices.

        Todo:
            * TODO: Frame images for zy0 and wy0 as well in title.

        """

        warnings.warn("PendingDeprecationWarning: All plotting and imaging will be handled by\
        visualisation module.", PendingDeprecationWarning)

        AXIS_TICK_FONTSIZE = 4 # -2.2
        LABEL_FONTSIZE = 4.5 # -.5

        plt.close('all')
        plt.clf()
        plt.cla()
        plt.figure(dpi=MAX_DPI)
        plt.rc('xtick', labelsize=AXIS_TICK_FONTSIZE)
        plt.rc('ytick', labelsize=AXIS_TICK_FONTSIZE)
        plt.title("Beam quality measures")

        _fig = plt.figure()
        _ax1 = _fig.add_subplot(2, 1, 1)
        _ax2 = _fig.add_subplot(2, 1, 2)
        #_ax3 = _fig.add_subplot(2, 2, 3)
        #_ax4 = _fig.add_subplot(2, 2, 4)

        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            parentgrp = hf.require_group(self.hdf5grp)
            grp = parentgrp.require_group(self.hdf5subgrp)

            _fig.suptitle("Beam quality for z_x0 = {0:.4}, w_x0 = {1:.4}".format(grp.attrs.get('z0x'), grp.attrs.get('w0x')), fontsize=LABEL_FONTSIZE+1.0)
            #_fig.set_tight_layout(True)
            _fig.subplots_adjust(wspace=0.25, hspace=0.4)

            _ax1.set_xlabel('Z [cm]', fontsize=LABEL_FONTSIZE)
            _ax1.set_ylabel('w_x [cm]', fontsize=LABEL_FONTSIZE)
            _ax1.set_title('Waist along transverse x', fontsize=LABEL_FONTSIZE)
            _quad1 = _ax1.plot(self.z_range, (grp[str('variance_x')][:])**2)

            _ax2.set_xlabel('Z [cm]', fontsize=LABEL_FONTSIZE)
            _ax2.set_ylabel('w_y [cm]', fontsize=LABEL_FONTSIZE)
            _ax2.set_title('Waist along transverse y', fontsize=LABEL_FONTSIZE)
            _quad2 = _ax2.plot(self.z_range, (grp[str('variance_y')][:])**2)

            # _ax3.set_xlabel('M_x^{2}', fontsize=LABEL_FONTSIZE)
            # _ax3.set_ylabel('Z [cm]', fontsize=LABEL_FONTSIZE)
            # _ax3.set_title('Quality along transverse x', fontsize=LABEL_FONTSIZE)
            # _quad3 = _ax3.plot(self.z_range, grp[str('MSquared_x')][:])

            # _ax4.set_xlabel('M_y^2', fontsize=LABEL_FONTSIZE)
            # _ax4.set_ylabel('Z [cm]', fontsize=LABEL_FONTSIZE)
            # _ax4.set_title('Waist along transverse y', fontsize=LABEL_FONTSIZE)
            # _quad4 = _ax4.plot(self.z_range, grp[str('MSquared_y')][:])

        plt.savefig(savepath, dpi=MAX_DPI)

    def lens_func(self, e0, f):
        """Modulate beam profile with lens of focal length."""
        #self.fseed *= np.exp(0.5j*self.wavenum*self.rr_sd**2/f)
        #self.fseed_mod *= np.exp(0.5j*self.wavenum*self.rr_sd**2/f)
        e0 *= np.exp(0.5j*self.wavenum*self.rr_sd**2/f)
        return e0

    def convolve(self, z):
        [self.fseed[0], self.fseed[2]] = super().convolve(self.fseed[0], z)

    def plot_second_focal_plane(self, flens, zpts=10, z1=None, z2=None, savepath='images/second_focal_plane.png'):
        """Simple relay lens system that propagates an image a distance z1, then it passes through lens of focal length flens, after which it is images at distance z1+z2.

        Args:
            flens (float): Focal length of lens element.
            zpts (int): Number of integration points along optical axis. Default 10.
            z1 (float): Propagation distance before lens. Default flens.
            z2 (float): Propagation distance after lens. Default flens.
            savepath (str): Filename to save image to.

        Returns:
            PNG (:obj:`~numpy.ndarray`): Four quandrant intensity profile image stored in savepath.

        Attributes:
            intensity2f (:obj:`~numpy.ndarray`): Original beam projected to lens and imaged to final focal length.
            zstep (:obj:`~numpy.linspace`): Axial points for integration of beam propagation.
            xxx, yyy, zzz (:obj:`~numpy.meshgrid`): Utility :obj:`~numpy.ndarray` objects for plotting purposes.
            intensity_profile3d (:obj:`~numpy.ndarray`): Three-dimensional axial beam profile.
            add_lens (bool): Identifies if beam has propagated past lens.
            lensbeam1f (:obj:`~numpy.ndarray`): Complex beam profile / amplitude immediately after lens.

        Todo:
            * TODO: Figure out a way to consolidate visualizations

        """

        if z1 is None:
            z1 = float(flens)
        if z2 is None:
            z2 = z1

        # TODO: This is very cute, DOCUMENT PROPERLY
        # Generate lg mode at z1, apply lens function, then propagate to z2
        intensity2f = super().convolve(self.lens_func(super().lg_mode(self.p, self.l, self.z+z1, self.phi)[0], flens), z2)[1]

        # Then check when \ where lens function is applied, and plot over a linspace in 3d
        zstep, delta_step = np.linspace(self.z, self.z+z1+z2, zpts, retstep=True)
        xxx, yyy, zzz = np.meshgrid(self.xx[0], self.yy.T[0], zstep, indexing='ij')
        intensity_profile3d = np.zeros([self.nx, self.ny, zpts])

        # list comprehension with if statement for > or < position of lens
        #intensity_profile_3d = np.asarray([super.convolve(self.fseed, z_step[i]) if z_step[i] < z1 \
        #                                   else self.convolve
        #                                   for i in range(zpts)], dtype=np.dtype(float).type)
        add_lens = False
        tempfseed = self.fseed[0]
        intensity_profile3d[:, :, 0] = self.fseed[2]
        for i in range(1, zpts):
            #if zstep[i] < z1:
            if zstep[i] < (self.z + z1):
                #[tempfseed, intensity_profile3d[:, :, i]] = super().convolve(tempfseed, delta_step)
                intensity_profile3d[:, :, i] = super().convolve(self.fseed[0], zstep[i])[1]
            else:
                if not add_lens:
                    lens_iter = i
                    add_lens = True
                    if self.z == -1*z1:
                        lensbeam1f = self.lens_func(super().lg_mode(self.p, self.l, self.z+z1, self.phi)[0], flens)
                    else:
                        lensbeam1f = self.lens_func(super().convolve(self.fseed[0], self.z+z1)[0], flens)
                    #tempfseed = self.lens_func(tempfseed, flens)

                #lensbeam1f, intensity_profile3d[:, :, i] = super().convolve(lensbeam1f, delta_step)
                #tempfseed, intensity_profile3d[:, :, i] = super().convolve(tempfseed, delta_step)
                intensity_profile3d[:, :, i] = super().convolve(lensbeam1f, zstep[i])[1]

        warnings.warn("PendingDeprecationWarning: All plotting and imaging will be handled by \
        visualisation module.", PendingDeprecationWarning)

        AXIS_TICK_FONTSIZE = 4 # -2.2
        LABEL_FONTSIZE_3D = 4
        LABEL_FONTSIZE = 4.5 # -.5

        plt.close('all')
        plt.clf()
        plt.cla()
        plt.figure(dpi=MAX_DPI)
        plt.rc('xtick', labelsize=AXIS_TICK_FONTSIZE)
        plt.rc('ytick', labelsize=AXIS_TICK_FONTSIZE)
        plt.title("Initial normalized intensity profile of EM field")

        _fig = plt.figure()
        _ax1 = _fig.add_subplot(2, 2, 1)
        _ax2 = _fig.add_subplot(2, 2, 2)
        _ax3 = _fig.add_subplot(2, 2, 3, projection='3d')
        _ax4 = _fig.add_subplot(2, 2, 4, projection='3d')
        _fig.suptitle("Second focal plane", fontsize=LABEL_FONTSIZE+1.0)
        #_fig.set_tight_layout(True)
        _fig.subplots_adjust(wspace=0.25, hspace=0.4)

        #######################
        _ax1.set_xlabel('X [cm]', fontsize=LABEL_FONTSIZE)
        _ax1.set_ylabel('Y [cm]', fontsize=LABEL_FONTSIZE)
        _ax1.set_title('Input beam intensity profile z = {0:.3f}cm'.format(zstep[0]), fontsize=LABEL_FONTSIZE)
        _quad1 = _ax1.pcolormesh(self.xx, self.yy, intensity_profile3d[:, :, 0], cmap='gnuplot2', shading='gouraud')

        # #######################
        # _ax2.set_xlabel('X [cm]', fontsize=LABEL_FONTSIZE)
        # _ax2.set_ylabel('Y [cm]', fontsize=LABEL_FONTSIZE)
        # _ax2.set_title('Intensity profile on lens at z = {0:3f}cm'.format(zstep[lens_iter]), fontsize=LABEL_FONTSIZE)
        # _quad2 = _ax2.pcolormesh(self.xx, self.yy, intensity_profile3d[:, :, lens_iter], cmap='gnuplot2', shading='gouraud')

        #######################
        _ax2.set_xlabel('X [cm]', fontsize=LABEL_FONTSIZE)
        _ax2.set_ylabel('Y [cm]', fontsize=LABEL_FONTSIZE)
        _ax2.set_title('Intensity profile at second focal plane 2f, z = {0:.3f}cm'.format(zstep[-1]), fontsize=LABEL_FONTSIZE)
        _quad2 = _ax2.pcolormesh(self.xx, self.yy, intensity_profile3d[:, :, -1], cmap='gnuplot2', shading='gouraud')

        # #######################
        _ax3.set_xlabel('X [cm]', fontsize=LABEL_FONTSIZE_3D)
        #_ax3.set_xlim(-0.6*self.waist, 0.6*self.waist)
        _ax3.set_ylabel('Z [cm]', fontsize=LABEL_FONTSIZE_3D)
        #_ax3.set_ylim(1.1*self.z, 1.1*(self.z+z1+z2))
        #self._ax3.set_zlabel('Temperature [$^\circ$K]', fontsize=LABEL_FONTSIZE_3D)
        #_ax3.set_zlim(0, 1)
        #print(np.shape(self.fevolmap3d))
        print(np.shape(xxx))
        _quad3 = _ax3.plot_surface(xxx.T[:, int(round(self.ny/2)), :], zzz.T[:, int(round(self.ny/2)), :], \
                                   intensity_profile3d.T[:, int(round(self.ny/2)), :], \
                                   alpha=0.8, cmap='gnuplot2', vmin=0, vmax=1)
        # _cset3 = _ax3.contourf(xxx.T[:, int(round(self.ny/2)), :], zzz.T[:, int(round(self.ny/2)), :], \
        #                        intensity_profile3d.T[:, int(round(self.ny/2)), :], \
        #                        zdir='x', offset=0, cmap='gnuplot2', vmin=0, vmax=1)
        # _cset3 = _ax3.contourf(xxx.T[:, int(round(self.ny/2)), :], zzz.T[:, int(round(self.ny/2)), :], \
        #                        intensity_profile3d.T[:, int(round(self.ny/2)), :], \
        #                        zdir='y', offset=1, cmap='gnuplot2', vmin=0, vmax=1)
        # _cset3 = _ax3.contourf(xxx.T[:, int(round(self.ny/2)), :], zzz.T[:, int(round(self.ny/2)), :], \
        #                        intensity_profile3d.T[:, int(round(self.ny/2)), :], \
        #                        zdir='z', offset=0, cmap='gnuplot2', vmin=0, vmax=1)

        _ax4.set_xlabel('Y [cm]', fontsize=LABEL_FONTSIZE_3D)
        #_ax4.set_xlim(-0.6*self.waist, 0.6*self.waist)
        _ax4.set_ylabel('Z [cm]', fontsize=LABEL_FONTSIZE_3D)
        #_ax4.set_ylim(1.1*self.z, 1.1*(self.z+z1+z2))
        #self._ax4.set_zlabel('Temperature [$^\circ$K]', fontsize=LABEL_FONTSIZE_3D)
        #_ax4.set_zlim(0, 1.)
        #print(np.shape(self.fevolmap3d))
        _quad4 = _ax4.plot_surface(yyy[int(round(self.nx/2)), :, :], zzz[int(round(self.nx/2)), :, :], \
                                   intensity_profile3d[int(round(self.nx/2)), :, :], alpha=0.3, cmap='gnuplot2')
        _cset4 = _ax4.contourf(yyy[int(round(self.nx/2)), :, :], zzz[int(round(self.nx/2)), :, :], \
                               intensity_profile3d[int(round(self.nx/2)), :, :], zdir='x', offset=-0.5*self.waist, cmap='gnuplot2')
        _cset4 = _ax4.contourf(yyy[int(round(self.nx/2)), :, :], zzz[int(round(self.nx/2)), :, :], \
                               intensity_profile3d[int(round(self.nx/2)), :, :], zdir='y', offset=-0.5*self.waist, cmap='gnuplot2')
        _cset4 = _ax4.contourf(yyy[int(round(self.nx/2)), :, :], zzz[int(round(self.nx/2)), :, :], \
                               intensity_profile3d[int(round(self.nx/2)), :, :], zdir='z', offset=0, cmap='gnuplot2')

        cax = _fig.add_axes([0.95, 0.1, 0.01, 0.75])
        _fig.colorbar(_quad1, cax=cax).set_label('Intensity')

        plt.savefig(savepath, dpi=MAX_DPI)

    def plot_fourth_focal_plane(self, flens1, flens2=None, z1=None, z2=None, z3=None):
        """Relay lens system that propagates an image a distance z1, then it passes through a lens of focal length"""
        pass

class ModalDecomp(MSquared):
    """Modal decomposition and phase de/multiplexing class.

    Todo:
        * set __getter__ to return CGH
    """

    pass


from photonicslab.demultiplex import PolDeMux

class VQF(PolDeMux):
    """Determine the :math:`VQF` or vector quality factor from simulated or experimental propagation data.

    """

    pass
