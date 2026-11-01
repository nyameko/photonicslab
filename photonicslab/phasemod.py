# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""Phase modulation and de/multiplexing module for analyzing beams and hologram generation."""

from photonicslab.config import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import warnings

from photonicslab.isotropy import GRINLens

class ThermalLens(GRINLens):
    """Used in parabolic thermal lensing analysis of heat loads.

    Args:
        hdf5filename (str): Filename of the HDF5 file object to be created or read from / appended to.
        hdf5grp (str): ame of the HDF5 Group object to apply the FD to.

    Attributes:
        xx (:obj:`~numpy.ndarray`): Meshgrid array description along x-direction.
        yy (:obj:`~numpy.ndarray`): Meshgrid array description along y-direction.
    """

    def __init__(self, hdf5filename='hdf5db/storage.hdf5', hdf5grp='nb1'):
        self.hdf5filename = str(hdf5filename)
        self.hdf5grp = str(hdf5grp)
        self.max_nz_const = False
        with h5py.File(self.hdf5filename, 'r', libver='latest') as hf:
            grp = hf.require_group(self.hdf5grp)
            self.xx = grp['xx'][:]
            self.yy = grp['yy'][:]

    def parabolic_fit(self, deg=2, hdf5table='temperature', lim_x=[-0.05, 0.05], lim_y=[-0.05, 0.05]):
        """Used to determine parabolic fit co-efficients for a thermal lens.

        Uses numpy polyval and polyfit to determine coefficients and polynomials.

        Args:
            deg (int): Degree of fitting polynomial.
            hdf5table (str): Dataset name of the HDF5 object to read. I.e. /nb1/polyfit/temperature/(attr:x_ind, xpolyval, )
            lim_x (numpy.ndarray): Two element array with :math:`x \\in [a, b]` limits along x-domain.
            lim_y (numpy.ndarray): Two element array with :math:`y \\in [a, b]` limits along y-domain.

        Todo:
            * Exception handling tests for limits etc...
            * Size tests for MAXSHAPE in create_dataset and deg size restrictions
        """

        self.dsetfsol = str( self.hdf5grp + '/' + hdf5table)
        with h5py.File(self.hdf5filename, 'r', libver='latest') as hf:
            # FIXME: store [xx, yy] meshgrid in hdf5 file (Only one slice per direction)
            #self.z_ind = (np.abs(zz[0,0,:] - z).argmin())
            #self.t_ind = (np.abs(clock-t).argmin())

            # self.xx = hf[self.dsetxx][ 0, lim_x[0]:lim_x[1]]
            _fevolmap = hf[self.dsetfsol][:]
            try:
                if (np.shape(lim_x) != (2,)) or (np.shape(lim_y) != (2,)):
                    #raise InputError("Malformed parabolic fit limits.")
                    raise AttributeError(self.parabolic_fit, "Malformed parabolic fit limits.")
                if (hf[self.hdf5grp].attrs['Lx']/2 < np.max(np.abs(lim_x))):
                    raise ValueError("Parabolic fit range out of bounds.")
                if (hf[self.hdf5grp].attrs['Ly']/2 < np.max(np.abs(lim_y))):
                    raise ValueError("Parabolic fit range out of bounds.")
            except ValueError as outofboundslimits:
                lim_x=[hf[self.hdf5grp].attrs['Lx']*0.1, hf[self.hdf5grp].attrs['Lx']*0.1]
                lim_y=[hf[self.hdf5grp].attrs['Ly']*0.1, hf[self.hdf5grp].attrs['Ly']*0.1]
                warnings.warn("RuntimeWarning: {0}".format(outofboundslimits), RuntimeWarning)
            #except InputError as badnumlimits:

        deg = int(deg)
        try:
            if deg <= 0:
                raise ValueError("Malformed degree of fitting polynomial.")
        except ValueError as baddegree:
            deg = np.abs(deg)
            warnings.warn("RuntimeWarning: {0}".format(baddegree), RuntimeWarning)

        x_ind_a = (np.abs(self.xx[:,0] - lim_x[0]).argmin())
        x_ind_b = (np.abs(self.xx[:,0] - lim_x[1]).argmin())
        y_ind_a = (np.abs(self.yy[0,:] - lim_y[0]).argmin())
        y_ind_b = (np.abs(self.yy[0,:] - lim_y[1]).argmin())

        _xpolyfit = np.empty([np.shape(_fevolmap)[0], deg+1])
        _xpolyval = np.empty([np.shape(_fevolmap)[0], x_ind_b-x_ind_a])
        _ypolyfit = np.empty([np.shape(_fevolmap)[0], deg+1])
        _ypolyval = np.empty([np.shape(_fevolmap)[0], y_ind_b-y_ind_a])

        for t in range(np.shape(_fevolmap)[0]):
            [_xpolyfit[t][0:deg+1], *_xcov] = np.polyfit(self.xx[x_ind_a:x_ind_b,
                                                         round(np.shape(_fevolmap)[2]/2)],
                                                         _fevolmap[t, x_ind_a:x_ind_b,
                                                         round(np.shape(_fevolmap)[2]/2)],
                                                         deg, cov=True)
            [_ypolyfit[t][0:deg+1], *_ycov] = np.polyfit(self.yy[round(np.shape(_fevolmap)[1]/2),
                                                             y_ind_a:y_ind_b],
                                                     _fevolmap[t, round(np.shape(_fevolmap)[1]/2),
                                                               y_ind_a:y_ind_b], deg, cov=True)
            _xpolyval[t] = np.polyval(_xpolyfit[t], self.xx[x_ind_a:x_ind_b,
                                                            round(np.shape(_fevolmap)[2]/2)])
            _ypolyval[t] = np.polyval(_ypolyfit[t], self.yy[round(np.shape(_fevolmap)[1]/2),
                                                            y_ind_a:y_ind_b])

        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            if str(self.hdf5grp +'/polyfit/'+ str(hdf5table)) not in hf:
                grp = hf.require_group(str(self.hdf5grp +'/polyfit/'+ str(hdf5table)))
                grp.create_dataset('xpolyfit', \
                                   shape=np.shape(_xpolyfit), \
                                   dtype=np.dtype(float).type, \
                                   data=_xpolyfit, \
                                   maxshape=(MAX_NT, deg+1))
                grp.create_dataset('xcov', shape=np.shape(_xcov), \
                                   dtype=np.dtype(float).type, \
                                   data=_xcov)
                grp.create_dataset('xpolyval', \
                                   shape=np.shape(_xpolyval), \
                                   dtype=np.dtype(float).type,
                                   data=_xpolyval, \
                                   maxshape=(MAX_NT, MAX_NX))
                grp.create_dataset('ypolyfit', \
                                   shape=np.shape(_ypolyfit), \
                                   dtype=np.dtype(float).type, \
                                   data=_ypolyfit, \
                                   maxshape=(MAX_NT, deg+1))
                grp.create_dataset('ycov', \
                                   shape=np.shape(_ycov), \
                                   dtype=np.dtype(float).type, \
                                   data=_ycov)
                grp.create_dataset('ypolyval', \
                                   shape=np.shape(_ypolyval), \
                                   dtype=np.dtype(float).type, \
                                   data=_ypolyval, \
                                   maxshape=(MAX_NT, MAX_NY))
            else:
                grp = hf.require_group(str(self.hdf5grp +'/polyfit/'+ str(hdf5table)))
                grp['xpolyfit'].resize((np.shape(_xpolyfit)))
                grp['xpolyfit'][...] = _xpolyfit
                grp['xcov'][...] = _xcov
                grp['xpolyval'].resize((np.shape(_xpolyval)))
                grp['xpolyval'][...] = _xpolyval
                grp['ypolyfit'].resize((np.shape(_ypolyfit)))
                grp['ypolyfit'][...] = _ypolyfit
                grp['ycov'][...] = _ycov
                grp['ypolyval'].resize((np.shape(_ypolyval)))
                grp['ypolyval'][...] = _ypolyval

            grp.attrs.create('x_ind', [x_ind_a, x_ind_b])
            grp.attrs.create('y_ind', [y_ind_a, y_ind_b])

    def phase(self, hdf5table='temperature', hdf5dset_phase='phase', hdf5dset_phase_collapsed='phase_test', wavelen=1.064e-4):
        """Generate transient cross-sectional phase profiles.

        Opens HDF5 database containing temporal temperature profiles and generates
        corresponding transient phase profiles. Phase profile dataset is only created
        if one does not already exist. Otherwise the values are simply overwritten.

        Args:
            hdf5dset_temperature (str): `/Path/to/` HDF5 subgroup and dataset for temperature profiles.
            hdf5dset_phase (str): /Path/to/ HDF5 subgroup and dataset for phase profiles.
            wavelen (float): Beam wavelength [cm].

        Todo:
            * TODO: Test integration constant. Test sign of dndt
            * TODO: Change in index of refraction per time. i.e. dndt -> at boundaries or recalculate relative to previous time
            * Refactor phase / index of refraction definitions and where to pass wavelength.
        """

        with h5py.File(str(self.hdf5filename), 'a', libver='latest') as hf:
            fevolmap = hf[str(self.hdf5grp + '/' + hdf5table)]
            grp = hf.require_group(self.hdf5grp)
            if str(hdf5dset_phase) not in grp:
                if self.max_nz_const:
                    grp.create_dataset(str(hdf5dset_phase), shape=np.shape(fevolmap),
                                       dtype=np.dtype(float).type, maxshape=(MAX_NT, MAX_NX, MAX_NY, MAX_NZ))
                elif not self.max_nz_const:
                    grp.create_dataset(str(hdf5dset_phase), shape=np.shape(fevolmap),
                                       dtype=np.dtype(float).type, maxshape=(MAX_NT, MAX_NX, MAX_NY))
            else:
                grp[str(hdf5dset_phase)].resize((np.shape(fevolmap)))
            #phasemap = grp.require_dataset(str(hdf5dset_phase), shape=np.shape(fevolmap),
            #                                dtype=np.dtype(float).type)
            #phasemap[...] = 2*np.pi/(wavelen)*(fevolmap*grp.attrs['dndT'] + grp.attrs['integral_const'])
            # FIXME: Test sign of dndt
            # grp[str(hdf5dset_phase)][...] = 2*np.pi/wavelen*(grp.attrs['nBC'] +
            #                                 (fevolmap - grp.attrs['TBC'])*grp.attrs['dndT'])
            grp[str(hdf5dset_phase)][...] = np.remainder(2*np.pi/wavelen*(grp.attrs['nBC'] +
                                                        (grp.attrs['TBC'] - fevolmap[...])*grp.attrs['dndT']), 2*np.pi)

            # 2D Collapsed phasemap, integrated over z-axis
            if str(hdf5dset_phase_collapsed) not in grp:
                grp.create_dataset(str(hdf5dset_phase_collapsed), shape=(grp.attrs.get('nt'), grp.attrs.get('nx'), grp.attrs.get('ny')),
                                   dtype=np.dtype(float).type, maxshape=(MAX_NT, MAX_NX, MAX_NY))
            else:
                grp[str(hdf5dset_phase_collapsed)].resize([grp.attrs.get('nt'), grp.attrs.get('nx'), grp.attrs.get('ny')])
            # TODO: verify direction of summation (i.e. along z-axis, last axis -1), test against different values for nx, ny, nz, nt
            for i_t in range(grp.attrs.get('nt')):
                grp[str(hdf5dset_phase_collapsed)][i_t] = np.mean(grp[str(hdf5dset_phase)][i_t][:, :, 1:grp.attrs['nz']-2], -1)
                #grp[str(hdf5dset_phase_collapsed)][i_t] = np.remainder(np.sum(grp[str(hdf5dset_phase)][i_t][...]*grp.attrs['dz'], -1), 1)
                #grp[str(hdf5dset_phase_collapsed)][i_t] = np.remainder(np.sum(grp[str(hdf5dset_phase)][i_t]*grp.attrs.get('dz'), -1), 2*np.pi)
                #grp[str(hdf5dset_phase_collapsed)][i_t] = np.remainder(np.sum(grp[str(hdf5dset_phase)][i_t], -1), 2*np.pi)

    def conjugate_phase(self):
        pass

class ThermalLens3D(ThermalLens):
    """Three-dimensional thermal lensing analysis of heat loads.

    """

    def __init__(self, hdf5filename='hdf5db/storage.hdf5', hdf5grp='nb1'):
        #super().__init__(hdf5filename, hdf5grp)
        self.hdf5filename = str(hdf5filename)
        self.hdf5grp = str(hdf5grp)
        self.max_nz_const = True
        with h5py.File(self.hdf5filename, 'r', libver='latest') as hf:
            grp = hf.require_group(self.hdf5grp)
            self.xxx = grp['xxx'][:]
            self.yyy = grp['yyy'][:]
            self.zzz = grp['zzz'][:]

    def phase_heat(self, hdf5table='temperature3d', hdf5dset_phase='phase3d', hdf5dset_phase_collapsed='phase2d', wavelen=1.064e-4):
        """Generate three-dimensional transient cross-sectional phase profiles.

        Opens HDF5 database containing temporal temperature profiles and generates
        corresponding transient phase profiles. Phase profile dataset is only created
        if one does not already exist. Otherwise the values are simply overwritten.

        Args:
            hdf5table (str): `/Path/to/` HDF5 subgroup and dataset for temperature profiles.
            hdf5dset_phase (str): /Path/to/ HDF5 subgroup and dataset for phase profiles.
            wavelen (float): Beam wavelength [cm].

        Todo:
            * Test integration constant.
            * Refactor phase / index of refraction definitions and where to pass wavelength.
            * TODO: Check phase calculation.
            * TODO: Need to keep this DRY - only MAX_NZ gets appended to hdf5_create_dataset
        """

        dsetfsol = str(self.hdf5grp + '/' + hdf5table)
        with h5py.File(str(self.hdf5filename), 'a', libver='latest') as hf:
            fevolmap = hf[dsetfsol]
            grp = hf.require_group(self.hdf5grp)
            if str(hdf5dset_phase) not in grp:
                grp.create_dataset(str(hdf5dset_phase), shape=np.shape(fevolmap),
                                   dtype=np.dtype(float).type, maxshape=(MAX_NT, MAX_NX, MAX_NY, MAX_NZ))
            else:
                grp[str(hdf5dset_phase)].resize((np.shape(fevolmap)))
            #phasemap = grp.require_dataset(str(hdf5dset_phase), shape=np.shape(fevolmap),
            #                                dtype=np.dtype(float).type)
            #phasemap[...] = 2*np.pi/(wavelen)*(fevolmap*grp.attrs['dndT'] + grp.attrs['integral_const'])
            # for i in range(grp.attrs['nt']):
            #    for j in range(grp.attrs['nz']):
            #        grp[str(hdf5dset_phase)][:, :, j, i] = np.remainder(2*np.pi/wavelen*(grp.attrs['nBC'] +
            #                                                              (fevolmap[:, :, j, i] - grp.attrs['TBC'])*grp.attrs['dndT']), 2*np.pi)

            grp[str(hdf5dset_phase)][...] = np.remainder(2*np.pi/wavelen*(grp.attrs['nBC'] +
                                                                         (fevolmap[...] - grp.attrs['TBC'])*grp.attrs['dndT']), 2*np.pi)


            # 2D Collapsed phasemap, integrated over z-axis
            if str(hdf5dset_phase_collapsed) not in grp:
                grp.create_dataset(str(hdf5dset_phase_collapsed), shape=(grp.attrs.get('nt'), grp.attrs.get('nx'), grp.attrs.get('ny')),
                                   dtype=np.dtype(float).type, maxshape=(MAX_NT, MAX_NX, MAX_NY))
            else:
                grp[str(hdf5dset_phase_collapsed)].resize([grp.attrs.get('nt'), grp.attrs.get('nx'), grp.attrs.get('ny')])
            # TODO: verify direction of summation (i.e. along z-axis), test against different values for nx, , ny, nz
            for i_t in range(grp.attrs.get('nt')):
                #grp[str(hdf5dset_phase_collapsed)][i_t] = np.remainder(np.sum(grp[str(hdf5dset_phase)][i_t], -1), 2*np.pi)
                #grp[str(hdf5dset_phase_collapsed)][i_t] = np.remainder(np.sum(grp[str(hdf5dset_phase)][i_t], -1), 2*np.pi)
                grp[str(hdf5dset_phase_collapsed)][i_t] = np.mean(grp[str(hdf5dset_phase)][i_t], -1)

    def conjugate_phase(self):
        pass

class OptoMechLens3D(ThermalLens3D):
    """Three-dimensional stress and strain lensing analysis of restricted thermal expansion or contraction and applied physically deforming torques forces.

    """
    def __init__(self):
        pass

    def phase_stress(self, hdf5table='temperature3d', hdf5dset_phase='phase3d'):
        pass

    def phase_strain(self, hdf5table='temperature3d', hdf5dset_phase='phase3d'):
        pass

class SLM(OptoMechLens3D):
    """Generate Spatial Light Modulator Phase Screen"""

    def __init__(self):
        pass

    def generate_screen(self):
        pass

class DM(OptoMechLens3D):
    """Deformable Mirror"""

    def __init__(self):
        pass
