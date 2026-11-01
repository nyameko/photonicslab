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
from photonicslab.optomech import ParaxialWaveEqn
# TODO: Replace line below with super()
from photonicslab.beamprop import BeamProp2D
#import skimage
#import imageio

from photonicslab.config import *

class EndPumped(ParaxialWaveEqn):
    """Paraxial Wave Equations in laser gain medium with pump on.

    This is the pseudo steady-state single pass solution.

    Args:
        wavelen (float): Wavelength of the incident seed beam [cm].
        waist (float): Location along z-direction of seed beam waist [cm].
        flens (float): (Not used) Focal length of lens added after crystal [cm].
        hdf5filename (str): Filename of the HDF5 data object.
        hdf5grp (str): Group name of the HDF5 group object.


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
        * TODO: Consolidate all __init__ method.
        * TODO: Need to incorporate this into the resonator model.
        * TODO: Registering material variables k0 and n0

    """

    def __init__(self, wavelength=1.064e-4,  waist=191.5e-4, p=0, l=0, z=0, phi=0, hdf5filename='hdf5db/storage.hdf5', hdf5grp='nb12', \
                 hdf5temperature='temperature3d', hdf5gain='gain', hdf5strain='strain', hdf5stress='stress', hdf5subgrp='nb12'):
        self.hdf5filename = str(hdf5filename)
        self.hdf5grp = str(hdf5grp)
        self.hdf5temperature = str(hdf5temperature)
        self.hdf5strain = str(hdf5strain)
        self.hdf5stress = str(hdf5stress)
        self.hdf5gain = str(hdf5gain)
        self.hdf5subgrp = str(hdf5subgrp)
        self.wavelen = float(wavelength)
        self.waist = float(waist)
        # TODO: Replace line below with super()
        self.fseed = BeamProp2D(wavelen=self.wavelen, waist=self.waist, hdf5filename=self.hdf5filename,
                                hdf5grp=self.hdf5grp).lg_mode(p, l, z, phi)
        self.k0 = 2*np.pi/self.wavelen
        with h5py.File(self.hdf5filename, 'r', libver='latest') as hf:
            grp = hf.require_group(self.hdf5grp)
            self.Sxy = grp.attrs.get('dx')**2*grp.attrs.get('dy')**2
            self.Sxz = grp.attrs.get('dx')**2*grp.attrs.get('dz')
            self.Syz = grp.attrs.get('dy')**2*grp.attrs.get('dz')
            self.Sxyz = grp.attrs.get('dx')**2*grp.attrs.get('dy')**2*grp.attrs.get('dz')
            self.rx = -0.5j*grp.attrs.get('dz') / (self.k0*grp.attrs.get('nBC')*grp.attrs.get('dx')**2)
            self.ry = -0.5j*grp.attrs.get('dz') / (self.k0*grp.attrs.get('nBC')*grp.attrs.get('dy')**2)
            self.rq = -0.5j*grp.attrs.get('dz')*self.k0 / (grp.attrs.get('nBC'))
            # TODO: Get k0 and n0, get gain and eps
            # self.vectorize(grp)
            self.nx, self.ny, self.nz = grp.attrs.get('nx'), grp.attrs.get('ny'), grp.attrs.get('nz')
            self.n0 = grp.attrs.get('nBC')

            if str('Ax') not in grp:
                grp.create_dataset(str('Ax'), shape=[self.nx, self.ny, self.nz], dtype=np.dtype(complex).type, maxshape=(MAX_NX, MAX_NY, MAX_NZ))
            else:
                grp[str('Ax')].resize([self.nx, self.ny, self.nz])
            if str('Ay') not in grp:
                grp.create_dataset(str('Ay'), shape=[self.nx, self.ny, self.nz], dtype=np.dtype(complex).type, maxshape=(MAX_NX, MAX_NY, MAX_NZ))
            else:
                grp[str('Ay')].resize([self.nx, self.ny, self.nz])

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

        # for i in range(1, self.nx-2):
        #     for j in range(1, self.ny-2):``
        #         for k in range(0, self.nz-2):
        #             Ax_dense[self.numbering[i, j, k], self.numbering[i, j, k]] =
        #             Ax_dense[self.numbering[i, j, k], self.numbering[i, j, k]] =
        #             Ax_dense[self.numbering[i, j, k], self.numbering[i, j, k]] =
        #             Ax_dense[self.numbering[i, j, k], self.numbering[i, j, k]] =
        #             Ax_dense[self.numbering[i, j, k], self.numbering[i, j, k]] =
        #             Ax_dense[self.numbering[i, j, k], self.numbering[i, j, k]] =

    @time_it
    def adi_method(self, rtol=1e-04, atol=1e-06, neumann_BC=0.01):
        """Uses Douglas and Rachford Alternating Direction Implicit Method to solve Ax = Bx + \\vec{y} linear equation.

        .. math::
            A\\vec{x}^{n+1} = B\\vec{x}^{n} + y^{n}

        * Generate three dimensional grid-point numbering array.
        * Vectorize the 3D arrays into linear vectors :math:`\\vec{x}`.
        * Generate the Sparse Diagonal Matrices A and B.
        * Use :obj:`~scipy.sparse.linalg` routines to solve :math:`\\vec{b} = B\\vec{x} + \\vec{y}`.
        * Finally, use :obj:`~scipy.sparse.linalg.spsolve` the system of equations :math:`\\vec{x} = A^{-1}\\vec{b}` for all times.

        Args:
            rtol (float): Relative tolerance for convergence criterion test (Numpy default is 1e-05).
            atol (float): Absolute tolerance for convergence criterion test (Numpy default is 1e-08).
            method (string): Linear Algebra method. One of {spsolve, bicg, bicgstab, cg, cgs, gmres, lgmres, minres, qmr, gcrotmk}
            neumann_BC (float): Neumann Boundary Condition for constant flux set through fictitious BC's.

        Todo:
            * TODO: Fix Neumann BC on crystal end-faces with ADI method.
            * TODO: Document notes on this procedure and diagram for internal mesh-points in []a,b,c,d] tdma_solver
            * Type of ADI method mathematical formulation.
            * Consider method of Fairweather and Mitchell.

        """

        # Set LHS matrix coefficients
        ax, bx, cx = -0.5*self.rx*np.ones(self.nx-2), (1 + self.rx)*np.ones(self.nx-2), -0.5*self.rx*np.ones(self.nx-2)
        ay, by, cy = -0.5*self.ry*np.ones(self.ny-2), (1 + self.ry)*np.ones(self.ny-2), -0.5*self.ry*np.ones(self.ny-2)

        fsol_Ax = np.zeros([self.nx, self.ny])
        fsol_Ax_x = np.zeros([self.nx, self.ny])
        fsol_Ax_new = np.zeros([self.nx, self.ny])

        fsol_Ay = np.zeros([self.nx, self.ny])
        fsol_Ay_x = np.zeros([self.nx, self.ny])
        fsol_Ay_new = np.zeros([self.nx, self.ny])

        Ax = np.zeros([self.nx, self.ny, self.nz])
        Ay = np.zeros([self.nx, self.ny, self.nz])

        # Generated lg-mode, need to check intensity and polarization etc
        # Should this be equated to original or normalized lg-mode
        # TODO: Inserting functions directly
        Ax[:, :, 0] = self.fseed[0]
        Ay[:, :, 0] = self.fseed[0]

        eps_xx = np.zeros([self.nx, self.ny, self.nz])
        eps_yy = np.zeros([self.nx, self.ny, self.nz])
        eps_xy = np.zeros([self.nx, self.ny, self.nz])
        eps_yx = np.zeros([self.nx, self.ny, self.nz])
        gain = np.zeros([self.nx, self.ny, self.nz])

        clock = 0
        z_step = 0
        # TODO: Try to non-dimensionalize y and x, such that BC = z and they become sparse vectors
        # TODO: Test to see if data in ram changes before it's written to hdf5
        # TODO: Might be necessary to have two with h5py statements
        convergence = False
        #self._format_hdf5table3d_to_finit3d
#        print("DEBUGGING: _format_hdf5table3d_to_finit3d")
        while (z_step < self.nz-1) and (convergence is False):
#            print("DEBUGGING: ADI Loop: {0}".format(clock))
            with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
                grp = hf.require_group(self.hdf5grp)
                # x = np.asarray([grp[self.hdf5table3d][clock, i, j, k] for i in range(self.nx) for j in range(self.ny)
                #                 for k in range(self.nz)])
                # y = np.asarray([grp['Qload3d'][clock, i, j, k] for i in range(self.nx) for j in range(self.ny)
                #                 for k in range(self.nz)])
                # # Neumann BC's note sign of start and end-face terms
                # y[numbering[:, :, 0]] += -2*self.sz*grp.attrs.get('dz')*neumann_BC
                # y[numbering[:, :, self.nz-1]] += 2*self.sz*grp.attrs.get('dz')*neumann_BC
                # # TODO: Test and fix below, may be broken
                # # TODO: test against b = np.dot(A.toarray(), x)
                # # TODO: Also test against b.data += y
                # # TODO: Have options for linalg routines
                # b = A.dot(x) + y
                fsol_Ax = grp['Ax'][..., z_step]
                fsol_Ax_x = grp['Ax'][..., z_step]
                fsol_Ax_new = grp['Ax'][..., z_step]

                fsol_Ay = grp['Ay'][..., z_step]
                fsol_Ay_x = grp['Ay'][..., z_step]
                fsol_Ay_new = grp['Ay'][..., z_step]

                d_Ax_x = np.zeros(self.nx-2)
                d_Ay_x = np.zeros(self.nx-2)
                for j in range(1, self.ny-1):
                    #d = np.zeros(self.nx-2)
                    for i in range(1, self.nx-1):
                        d_Ax_x[i-1] = 0.5*self.rx*fsol_Ax[i-1, j] + 0.5*self.rx*fsol_Ax[i+1, j] \
                            + self.ry*fsol_Ax[i, j-1] + self.ry*fsol_Ax[i, j+1] \
                            + (1 - self.rx - 2*self.ry)*fsol_Ax[i, j] \
                            + self.rq*(1j*self.n0/self.k0*gain(i, j, z_step) + eps_xx(i, j, z_step) - self.n0**2)*fsol_Ax[i, j] \
                            + self.rq*eps_xy(i, j, z_step)*fsol_Ay[i, j]
                        d_Ay_x[i-1] = 0.5*self.rx*fsol_Ay[i-1, j] + 0.5*self.rx*fsol_Ay[i+1, j] \
                            + self.ry*fsol_Ay[i, j-1] + self.ry*fsol_Ay[i, j+1] \
                            + (1 - self.rx - 2*self.ry)*fsol_Ay[i, j] \
                            + self.rq*(1j*self.n0/self.k0*gain(i, j, z_step) + eps_yy(i, j, z_step) - self.n0**2)*fsol_Ay[i, j] \
                            + self.rq*eps_yx(i, j, z_step)*fsol_Ax[i, j]

                    d_Ax_x[0] += 0.5*self.rx*fsol_Ax[0, j]
                    d_Ax_x[-1] += 0.5*self.rx*fsol_Ax[-1, j]
                    d_Ay_x[0] += 0.5*self.rx*fsol_Ay[0, j]
                    d_Ay_x[-1] += 0.5*self.rx*fsol_Ay[-1, j]

                    fsol_Ax_x[1:self.nx-1, j] = self.tdma_solver(ax, bx, cx, d_Ax_x)[...]
                    fsol_Ay_x[1:self.nx-1, j] = self.tdma_solver(ax, bx, cx, d_Ay_x)[...]

                d_Ax_y = np.zeros(self.ny-2)
                d_Ay_y = np.zeros(self.ny-2)
                for i in range(1, self.nx-1):
                    #d = np.zeros(self.nx-2)
                    for j in range(1, self.ny-1):
                        d_Ax_y[j-1] = 0.5*self.rx*fsol_Ax[i-1, j] + 0.5*self.rx*fsol_Ax[i+1, j] \
                            + 0.5*self.rx*fsol_Ax_x[i-1, j] + self.rx*fsol_Ax_x[i, j] + 0.5*self.rx*fsol_Ax_x[i+1, j] \
                            + self.ry*fsol_Ax[i, j-1] + self.ry*fsol_Ax[i, j+1] \
                            + (1 - self.rx - self.ry)*fsol_Ax[i, j] \
                            + self.rq*(1j*self.n0/self.k0*gain(i, j, z_step) + eps_xx(i, j, z_step) - self.n0**2)*fsol_Ax[i, j] \
                            + self.rq*eps_xy(i, j, z_step)*fsol_Ay[i, j]
                        d_Ay_y[j-1] = 0.5*self.rx*fsol_Ay[i-1, j] + 0.5*self.rx*fsol_Ay[i+1, j] \
                            + 0.5*self.rx*fsol_Ay_x[i-1, j] + self.rx*fsol_Ay_x[i, j] + 0.5*self.rx*fsol_Ay_x[i+1, j] \
                            + self.ry*fsol_Ay[i, j-1] + self.ry*fsol_Ay[i, j+1] \
                            + (1 - self.rx - self.ry)*fsol_Ay[i, j] \
                            + self.rq*(1j*self.n0/self.k0*gain(i, j, z_step) + eps_yy(i, j, z_step) - self.n0**2)*fsol_Ay[i, j] \
                            + self.rq*eps_yx(i, j, z_step)*fsol_Ax[i, j]

                    d_Ax_y[0] += 0.5*self.ry*fsol_Ax[i, 0]
                    d_Ax_y[-1] += 0.5*self.ry*fsol_Ax[i, -1]
                    d_Ay_y[0] += 0.5*self.ry*fsol_Ay[i, 0]
                    d_Ay_y[-1] += 0.5*self.ry*fsol_Ay[i, -1]

                    fsol_Ax_new[i, 1:self.ny-1] = self.tdma_solver(ay, by, cy, d_Ax_y)[...]
                    fsol_Ay_new[i, 1:self.ny-1] = self.tdma_solver(ay, by, cy, d_Ay_y)[...]

                z_step+=1

                # TODO: Time it and test if an additional with h5py statement must come here
                # TODO: Verify proper assignment of values i.e. explicit ellipses to copy values and not pointer!
                #grp[self.hdf5table3d][..., clock] = fsol_new[...] + grp['Qload3d'][..., clock-1]
                grp['Ax'][..., z_step] = fsol_Ax_new[...]
                grp['Ay'][..., z_step] = fsol_Ay_new[...]
                #print("Convergence Test: {0}".format(np.allclose(fsol_new, fsol, rtol, atol)))
                #if (np.allclose(fsol_new, fsol, rtol, atol)):
                    #convergence = True
                #if (convergence) or (z_step == self.nz-1):
                #    grp.attrs.create('time', clock)

class SidePumped(ParaxialWaveEqn):
    def __init__(self):
        pass

class MultiMode(EndPumped):
    def __init__(self):
        pass
