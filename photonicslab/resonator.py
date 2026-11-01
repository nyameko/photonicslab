# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""Finite difference solver for the two-dimensional heat equation using the explicit or implicit methods.

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
import scipy.sparse as sp
#import skimage
#import imageio

from photonicslab.config import *
from photonicslab.amplifier import EndPumped
from photonicslab.amplifier import SidePumped

class SingleMode(EndPumped):
    """Paraxial Wave Equations in laser gain medium with pump on.

    This is the pseudo steady-state single pass solution.

    Todo:
        * TODO: Consolidate all __init__ method.
        * TODO: Need to incorporate this into the resonator model.
        * TODO: Registering material variables k0 and n0

    """

    def __init__(self, hdf5filename='hdf5db/storage.hdf5', hdf5grp='nb8', hdf5heat='temperature3d', hdf5stress='stress3d', \
                 hdf5subgrp='nb8_1', wavelength=1.064e-6):
        self.hdf5filename = str(hdf5filename)
        self.hdf5grp = str(hdf5grp)
        self.hdf5heat = str(hdf5heat)
        self.hdf5stress = str(hdf5stress)
        self.hdf5subgrp = str(hdf5subgrp)
        with h5py.File(self.hdf5filename, 'r', libver='latest') as hf:
            grp = hf.require_group(self.hdf5grp)
            self.Sxy = grp.attrs.get('dx')**2*grp.attrs.get('dy')**2
            self.Sxz = grp.attrs.get('dx')**2*grp.attrs.get('dz')
            self.Syz = grp.attrs.get('dy')**2*grp.attrs.get('dz')
            self.Sxyz = grp.attrs.get('dx')**2*grp.attrs.get('dy')**2*grp.attrs.get('dz')
            # TODO: Get k0 and n0, get gain and eps
            # self.vectorize(grp)
            self.nx, self.ny, self.nz = grp.attrs.get('nx'), grp.attrs.get('ny'), grp.attrs.get('nz')
            self.k0, self.n0 = grp.attrs.get('k'), grp.attrs.get('nBC')

    def vectorize(self):
        """Internal function used to convert 3D arrays into 1D vectors.

        .. math::

            \\text{numbering}_{i, j, k} = k \\times n_{x} \\times n_{y} + j \\times n_{x} + i

        Args:
            grp (:obj:`~h5py.Group`): Open HDF5 Group instance.

        Returns:
            numbering (:obj:`~numpy.ndarray`): One-dimensional vector mapping coordinates [i, j, k] -> number.

        Todo:
            * Accelerate this with :obj:`numba.jit`.
        """
        # Maybe convert to list comprehension
        # Careful with empty, could see blowup...?
        num = 0
        #numbering = np.zeros([grp.attrs.get('nx'), grp.attrs.get('ny'), grp.attrs.get('nz')])
        numbering = np.zeros([self.nx, self.ny, self.nz])
        # x = np.zeros([self.nt, self.nx*self.ny*self.nz])
        # b = np.zeros([self.nt, self.nx*self.ny*self.nz])
        # y = np.zeros([self.nt, self.nx*self.ny*self.nz])
        #@time_it
        #numbering_list = [(k*self.nx*self.ny,j*self.nx,i) for i in range(self.nx) for j in range(self.ny) for k in range(self.nz)]
        # Then use sum to get the totals and ... funny indexing - Dont get too clever :-)
        # can instead replace number array with (k*ny*nx + j*nx + i)
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    numbering[i, j, k] = num
                    num += 1
        #return numbering.astype(int)
        self.numbering = numbering.astype(int)

    @time_it
    def implicit_method(self, rtol=1e-03, atol=1e-05, method='spsolve', sp_format='csc'):
        """Uses scipy sparse matrices to solve band diagonal matrix Ax = y linear equation.

        .. math::
            QA_{x} = A_{y}

        * Generate three dimensional grid-point numbering array.
        * Vectorize the 3D arrays into linear vectors :math:`\\vec{x}`.
        * Generate the Sparse Diagonal Matrices A and B.
        * Use :obj:`~scipy.sparse.linalg` routines to solve :math:`\\vec{b} = B\\vec{x} + \\vec{y}`.
        * Finally, use :obj:`~scipy.sparse.linalg.spsolve` the system of equations :math:`\\vec{x} = A^{-1}\\vec{b}` for all times.

        Args:
            rtol (float): Relative tolerance for convergence criterion test (Numpy default is 1e-05).
            atol (float): Absolute tolerance for convergence criterion test (Numpy default is 1e-08).
            method (str): Linear Algebra method. One of {spsolve, bicg, bicgstab, cg, cgs, gmres, lgmres, minres, qmr, gcrotmk}
            sp_format (str): Sparse matrix format. One of {csc, csr}

        .. seealso::
            :obj:`~scipy.sparse.linalg.spsolve`, :obj:`~scipy.sparse.linalg.gmres`, :obj:`~scipy.sparse.linalg.lgmres`, and :obj:`~scipy.sparse.linalg.bicgstab`
            :obj:`~numpy.allclose`

        References:
            * `SO Tridiagonal Block Matrix with SciPy Sparse <https://stackoverflow.com/questions/44641996/traditional-block-matrix-using-sip-sparse>`_
            * `SO Tridiagonal Matrix Using SciPy Sparse Diags <https://stackoverflow.com/questions/50292546/scipy-tridiagonal-matrix-using-scipy-sparse-diags>`_
            * `Consider storing hdf5file in memory <https://www.pytables.org/cookbook/inmemory_hdf5_files.html>`_

        Todo:
            * TODO: Accelerate nested for loops using :obj:`numba.jit`
            * TODO: Test using diagonal / banded matrix creation methods and solvers.
            * TODO: Test other linear algebra solvers :obj:`scipy.sparse.linalg`.
            * TODO: Test dense conversion and solutions with :obj:`numpy.linalg.solve` and / or :obj:`scipy.linalg.solve`.
            * TODO: Make the construction of the seven band tridiagonal matrices more concise.
            * TODO: Consolidate with fdsolver.py.implicit_method routines as well.
        """

        Ax_vec = np.zeros(self.nx*self.ny*self.nz)
        Ax_vec_new = np.zeros(self.nx*self.ny*self.nz)
        Ay_vec = np.zeros(self.nx*self.ny*self.nz)
        Ay_vec_new = np.zeros(self.nx*self.ny*self.nz)
        Ax_dense = np.zeros([self.nx*self.ny*self.nz, self.nx*self.ny*self.nz])
        Ay_dense = np.zeros([self.nx*self.ny*self.nz, self.nx*self.ny*self.nz])

        # TODO: Check indexing here and FIXME: with respect to broadcasting 3D -> 2D nx and ny, but keep the same for now
        # Above fixed with numbering with respect to [x, y, z]
        # TODO: sp.sparse.diags add dynamic spatial gain and permeability terms g and eps.
        # Linear order operator, compressed sparse row and compressed sparse column matrices
        # coordinates: [x-1, x+1, y-1, y+1, z-1, z+1] = [-1, +1, -nx, +nx, -nx*ny, nx*ny]
        # TODO: Compare csr and csc performance
        # TODO: Test sp.diags and sp.spdiags
        LO_Ay = sp.diags([-2*self.Syz, 2*self.Syz, -2*self.Sxz, 2*self.Sxz, -1j*self.k0*self.n0*self.Sxy, 1j*self.k0*self.n0*self.Sxy,\
                          2*self.Sxyz*self.k0**2*self.n0**2 + 4*(self.Sxz + self.Syz)],\
                         [-1, 1, -1*self.nx, self.nx, -1*self.nx*self.ny, self.nx*self.ny, 0], \
                         shape=(self.nx*self.ny*self.nz, self.nx*self.ny*self.nz), format=str(sp_format), dtype=np.dtype(complex).type)

        LO_Ax = sp.diags([-2*self.Syz, 2*self.Syz, -2*self.Sxz, 2*self.Sxz, -1j*self.k0*self.n0*self.Sxy, 1j*self.k0*self.n0*self.Sxy,\
                          2*self.Sxyz*self.k0**2*self.n0**2 + 4*(self.Sxz + self.Syz)],\
                         [-1, 1, -1*self.nx, self.nx, -1*self.nx*self.ny, self.nx*self.ny, 0], \
                         shape=(self.nx*self.ny*self.nz, self.nx*self.ny*self.nz), format=str(sp_format), dtype=np.dtype(complex).type)
        # TODO: 27/11/18 and 4 months later...
        # 1. Write out mathematics in full,
        # 2. Clear idea of Ax, Ay, Bx etc...
        # 3. Verify linear algebra and how you vectorize / linearize matrices
        # 4. Is the mathematics necessary in the z-direction?
        # 5. Use these results to fix everything else.

        for i in range(1, self.nx-2):
            for j in range(1, self.ny-2):
                for k in range(0, self.nz-2):
                    Ax_dense[self.numbering[i, j, k], self.numbering[i, j, k]] = 0
                    Ax_dense[self.numbering[i, j, k], self.numbering[i, j, k]] = 0
                    Ax_dense[self.numbering[i, j, k], self.numbering[i, j, k]] = 0
                    Ax_dense[self.numbering[i, j, k], self.numbering[i, j, k]] = 0
                    Ax_dense[self.numbering[i, j, k], self.numbering[i, j, k]] = 0
                    Ax_dense[self.numbering[i, j, k], self.numbering[i, j, k]] = 0

class SingleModeSide(SidePumped):
    def __init__(self):
        pass
