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
import matplotlib.pyplot as plt
import numpy as np

from photonicslab.config import *

matplotlib.use('Agg')
#import skimage
#import imageio

class GRINLens(object):
    """Graded Index lenslike medium where the refractive index varies with pace coordinates.

    Args:
        nx (int): Number of steps along x-direction of cross-sectional profile.
        ny (int): Number of steps along y-direction of cross-sectional profile.
        nz (int): Number of steps along z-direction of axial profile.
        Lx (float): Cross-sectional length along x-direction [cm].
        Ly (float): Cross-sectional length along y-direction [cm].
        Lz (float): Axial length along z-direction [cm].
        hdf5filename (str): Filename of the HDF5 data object.
        hdf5grp (str): Group name of the HDF5 group object.

    Attributes:
        dx (:obj:`float`): Spacing along the x-direction [cm].
        dy (:obj:`float`): Spacing along the y-direction [cm].
        xx (:obj:`~numpy.ndarray`): Two-dimensional meshgrid array of x-coordinates used for plotting and calculations.
        yy (:obj:`~numpy.ndarray`): Two-dimensional meshgrid array of y-coordinates used for plotting and calculations.
        xxx (:obj:`~numpy.ndarray`): Three-dimensional meshgrid array of x-coordinates used for plotting and calculations.
        yyy (:obj:`~numpy.ndarray`): Three-dimensional meshgrid array of y-coordinates used for plotting and calculations.
        zzz (:obj:`~numpy.ndarray`): Three-dimensional meshgrid array of z-coordinates used for plotting and calculations.

    Todo:
        * TODO: Consolidate all the various classes and methods (i.e. only have a single visualization abstraction)

    """

    def __init__(self, nx=200, ny=200, nz=200, Lx=0.4, Ly=0.4, Lz=2., hdf5filename='hdf5db/storage.hdf5', hdf5grp='nb14'):
        #super().__init__(nx, ny, nz, nt, Lx, Ly, Lz, Tbc, Qmax, gain_medium, hdf5filename, hdf5grp, dt_stability_factor)

        self.nx = int(np.abs(nx))
        self.ny = int(np.abs(ny))
        self.nz = int(np.abs(nz))
        self.Lx = float(np.abs(Lx))
        self.Ly = float(np.abs(Ly))
        self.Lz = float(np.abs(Lz))
        self.dx = self.Lx/(self.nx)
        self.dy = self.Ly/(self.ny)
        self.dz = self.Lz/(self.nz)

        xx, yy = np.meshgrid(np.linspace(-1*self.Lx/2, self.Lx/2, self.nx),\
                                    np.linspace(-1*self.Ly/2, self.Ly/2, self.ny),\
                                    indexing='ij')

        xxx, yyy, zzz = np.meshgrid(np.linspace(-1*self.Lx/2, self.Lx/2, self.nx),\
                                    np.linspace(-1*self.Ly/2, self.Ly/2, self.ny),\
                                    np.linspace(0, self.Lz, self.nz), indexing='ij')


        self.hdf5filename = str(hdf5filename)
        self.hdf5grp = str(hdf5grp)

        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            grp = hf.require_group(self.hdf5grp)
            grp.attrs.create('nx', self.nx)
            grp.attrs.create('ny', self.ny)
            grp.attrs.create('nz', self.nz)
            grp.attrs.create('Lx', self.Lx)
            grp.attrs.create('Ly', self.Ly)
            grp.attrs.create('Lz', self.Lz)
            grp.attrs.create('dx', self.dx)
            grp.attrs.create('dy', self.dy)
            grp.attrs.create('dz', self.dz)

            if 'xx' not in grp:
                grp.create_dataset('xx', shape=(self.nx, self.ny), dtype=np.dtype(float).type, \
                                   data=xx, maxshape=(MAX_NX, MAX_NY))
            else:
                grp['xx'].resize((self.nx, self.ny))
                grp['xx'][...] = xx

            if 'yy' not in grp:
                grp.create_dataset('yy', shape=(self.nx, self.ny), dtype=np.dtype(float).type, \
                                   data=yy, maxshape=(MAX_NX, MAX_NY))
            else:
                grp['yy'].resize((self.nx, self.ny))
                grp['yy'][...] = yy

            if 'xxx' not in grp:
                grp.create_dataset('xxx', shape=(self.nx, self.ny, self.nz), dtype=np.dtype(float).type, \
                                   data=xxx, maxshape=(MAX_NX, MAX_NY, MAX_NZ))
            else:
                grp['xxx'].resize((self.nx, self.ny, self.nz))
                grp['xxx'][...] = xxx

            if 'yyy' not in grp:
                grp.create_dataset('yyy', shape=(self.nx, self.ny, self.nz), dtype=np.dtype(float).type, \
                                   data=yyy, maxshape=(MAX_NX, MAX_NY, MAX_NZ))
            else:
                grp['yyy'].resize((self.nx, self.ny, self.nz))
                grp['yyy'][...] = yyy

            if 'zzz' not in grp:
                grp.create_dataset('zzz', shape=(self.nx, self.ny, self.nz), dtype=np.dtype(float).type, \
                                   data=zzz, maxshape=(MAX_NX, MAX_NY, MAX_NZ))
            else:
                grp['zzz'].resize((self.nx, self.ny, self.nz))
                grp['zzz'][...] = zzz

            # Strain Tensor
            if 'eps' not in grp:
                grp.create_dataset('eps', shape=(3, 3, self.nx, self.ny, self.nz), dtype=np.dtype(float).type, \
                                   data=np.zeros([3, 3, self.nx, self.ny, self.nz]), maxshape=(3, 3, MAX_NX, MAX_NY, MAX_NZ))
            else:
                grp['eps'].resize((3, 3, self.nx, self.ny, self.nz))
                grp['eps'][...] = np.zeros([3, 3, self.nx, self.ny, self.nz])

            # Stress Tensor
            if 'sig' not in grp:
                grp.create_dataset('sig', shape=(3, 3, self.nx, self.ny, self.nz), dtype=np.dtype(float).type, \
                                   data=np.zeros([3, 3, self.nx, self.ny, self.nz]), maxshape=(3, 3, MAX_NX, MAX_NY, MAX_NZ))
            else:
                grp['sig'].resize((3, 3, self.nx, self.ny, self.nz))
                grp['sig'][...] = np.zeros([3, 3, self.nx, self.ny, self.nz])

    def index_distr(self, n0=1.82, b=None):
        """Add the index distribution of the GRaded-INdex of refraction lens.

        Args:
            n0 (float): Original / natural refractive index with gain.
            b (float): Numerical parameter used in calculation of GRaded INdex of refraction lens formulation. Use None for constant index :math:`n_0`.

        Attributes:
            indexN (:obj:`~numpy.ndarray`): Two-dimensional meshgrid array of graded index of refraction, (repeated along axial or z-coordinate).

        Todo:
            * TODO: Unique n0x and n0y formulation for natural birefringence.
            * TODO: For the entire class need to consider placing material properties / constants in subgrps.
        """

        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            grp = hf.require_group(self.hdf5grp)
            grp.attrs.create('b', b)
            grp.attrs.create('n0', n0)

            if b is not None:
                indexN = np.zeros([self.nx, self.ny])
                indexN = n0*(1 - 2 * (grp['xx'][...]**2 + grp['yy'][...]**2)/(float(b))**2)
            else:
                indexN = n0*np.ones([self.nx, self.ny])

            grp['eps'][0, 0, ...] += (indexN[:, :, None])**2
            grp['eps'][1, 1, ...] += (indexN[:, :, None])**2

            indexN_3D = np.repeat(indexN[:, :, None], self.nz, axis=2)
            if 'indexN' not in grp:
                grp.create_dataset('indexN', shape=(self.nx, self.ny, self.nz), dtype=np.dtype(float).type, \
                                   data=indexN_3D, maxshape=(MAX_NX, MAX_NY, MAX_NZ))
            else:
                grp['indexN'].resize((self.nx, self.ny, self.nz))
                grp['indexN'][...] = indexN_3D

    def get_GRIN_focal_len(self):
        """Analytical calculation for the focal length and distance from the medium to the closet principal plane.

        Attributes:
            f (float): Focal length.
            h (float):

        Todo:
            * TODO: Add tests from literature as unit tests.

        References:
            * `H. Kogelnik: Imaging of Optical Modes - Resonators with Lenses <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6773625>`_
        """

        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            grp = hf.require_group(self.hdf5grp)

            f = grp.attrs.get('b') / (2 * grp.attrs.get('n0') * np.sin(2*grp.attrs.get('Lz')*1e-2 / grp.attrs.get('b')))
            h = grp.attrs.get('b') / (2 * grp.attrs.get('n0')) * np.tan(grp.attrs.get('Lz')*1e-2 / grp.attrs.get('b'))

            grp.attrs.create('f', f)
            grp.attrs.create('h', h)
            return f, h

    def remove_strain(self):

        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            grp = hf.require_group(self.hdf5grp)
            grp['eps'][...] = np.zeros([3, 3, self.nx, self.ny, self.nz])

    def remove_stress(self):

        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            grp = hf.require_group(self.hdf5grp)
            grp['sig'][...] = np.zeros([3, 3, self.nx, self.ny, self.nz])

    def add_supergaussian_displacement_field(self, amplitude=0.1, center_i=0.0, center_j=0.0, plane_k=0.0, width_i=0.1, width_j=0.1, decay_k=10, \
                                             power_i=1.0, power_j=1.0, axis_k='x', axis_k_dir='+'):
        """Adds exponentially decaying, normally distributed or Gaussian displacement field vector for generation and calculation of non-zero infinitesimal strain tensor.

        Todo:
            * TODO: Dirt like function or example as well.
            * TODO: Add torsional / rotating initial fields
            * TODO: Differential geometry / continuum mechanics to solve equations of motion for rigorous treatment of deformation / .

        """

        pass

    def plot_source_fields(self, projection='z', plane='None', savepath='images/source_field.png'):
        """Plot the source fields impinging on gain / lens material directed in a particular projection, cut along a specific plane.

        Plot intensity map slice material in quad1, then same thing with vector fields in quads 2-4.

        Args:
            projection (str): Projection along which to demonstrate vector field. Default is 'z'.
            plane (int): Along direction of projection, which plane to bisect medium. Default is None, which uses mid-plane.
            savepath (str): Filename to save image file to.

        Todo:
            * Vector drawing of displacement fields on particular planes.

        """

        pass

    def thermal_stress(self, E, aplha):
        """Generate stress tensor due to thermal expansion or the restriction thereof.

        Args:
            E (float): Young's modulus of elasticity.
            alpha (float): Coefficient of thermal expansion.

        """

        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            grp = hf.require_group(self.hdf5grp)
            #grp['sig'][...] = np.zeros([3, 3, self.nx, self.ny, self.nz])


    def linear_displacement_strain(self):
        """Generate infinitesimal or Cauchy's strain tensor from deformations involving displacement vectors and displacement gradients much much smaller than unity.

        Generalized Strain-displacement equations for linearized infinitesimally small deformations.

        """

        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            grp = hf.require_group(self.hdf5grp)
            #grp['eps'][...] = np.zeros([3, 3, self.nx, self.ny, self.nz])

    def effective_strain(self, E, v):
        """Apply generalized Hook's law to determine overall effect of all stress and strain contributions for isotropic materials in terms of Young's modulus and Poisson's ratio fro the material.

        Args:
            E (float): Young's modulus of elasticity. TODO: Fix these duplicates or maybe define in constructor.
            v (float): Poisson's ratio.

        """
        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            grp = hf.require_group(self.hdf5grp)
            grp['eps'][0,0, ...] = np.zeros([self.nx, self.ny, self.nz])
