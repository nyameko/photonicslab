# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""Finite difference solver for the two-dimensional opto-mechanical Paraxial Wave Equation equation using the explicit or implicit methods for isotropic media.

This serves as an initialisation class to generate the initial heat source configuration and to
define the material properties of the element to be heated.

Todo:
    * TODO: Extend to 3D beamProp case for axially varying beams.

"""

import os
import warnings
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from photonicslab.beamprop import BeamProp2D
from photonicslab.fdsolver import FDSolver3D
#import skimage
#import imageio

from photonicslab.config import *

class ParaxialWaveEqn(BeamProp2D, FDSolver3D):
    """Paraxial Wave Equations in lenslike  media.

    This is the pseudo steady-state single pass solution.

    Args:
        wavelen (float): Wavelength of the incident seed beam [cm].
        waist (float): Location along z-direction of seed beam waist [cm].
        flens (float): (Not used) Focal length of lens added after crystal [cm].
        hdf5filename (str): Filename of the HDF5 data object.
        hdf5grp (str): Group name of the HDF5 group object.


    Raises:
        ValueError: If no valid gain medimum given. Anisotropy modules needs to be run first.
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
        * TODO: Registering material variables k0 and n0 and vector case n0x and n0y
        * TODO: Induced stress and strain, in the same way the gain depends on the field.
        * TODO: Double check and verify generation of initializer for Ax and Ay

    """

    def __init__(self, wavelen=1.064e-4,  waist=191.5e-4, flens=5e-1, p=0, l=0, z=0, phi=0, hdf5filename='hdf5db/storage.hdf5', hdf5grp='nb12', \
                 hdf5subgrp='tem00', hdf5subset_intensity='intensity', hdf5subset_field='field', hdf5subset_gain='gain', hdf5subset_strain='strain', \
                 hdf5subset_stress='stress'):

        super().__init__(wavelen, waist, flens, hdf5filename, hdf5grp)

        # TODO: Get rid of all subset details.
        self.hdf5subgrp = str(hdf5subgrp)
        self.hdf5subset_strain = str(hdf5subset_strain)
        self.hdf5subset_stress = str(hdf5subset_stress)
        self.hdf5subset_gain = str(hdf5subset_gain)
        self.hdf5subset_intens = str(hdf5subset_intensity)
        self.hdf5subset_field = str(hdf5subset_field)

        self.p = int(p)
        self.l = int(l)
        self.z = float(z)
        self.phi = float(phi)

        self.fseed = super().lg_mode(p, l, z, phi)
        self.Ax = 2**(-0.5)*self.fseed[0]
        self.Ay = 2**(-0.5)*self.fseed[0]

        self.k0 = 2*np.pi/self.wavelen
        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            parentgrp = hf.require_group(self.hdf5grp)
            grp = parentgrp.require_group(self.hdf5subgrp)

            grp.attrs.create('waist', self.waist)
            grp.attrs.create('wl', self.wavelen)
            grp.attrs.create('p', self.p)
            grp.attrs.create('l', self.l)
            grp.attrs.create('z', self.z)
            grp.attrs.create('phi', self.phi)

            self.Sxy = parentgrp.attrs.get('dx')**2*parentgrp.attrs.get('dy')**2
            self.Sxz = parentgrp.attrs.get('dx')**2*parentgrp.attrs.get('dz')
            self.Syz = parentgrp.attrs.get('dy')**2*parentgrp.attrs.get('dz')
            self.Sxyz = parentgrp.attrs.get('dx')**2*parentgrp.attrs.get('dy')**2*parentgrp.attrs.get('dz')
            self.rx = -0.5j*parentgrp.attrs.get('dz') / (self.k0*parentgrp.attrs.get('n0')*parentgrp.attrs.get('dx')**2)
            self.ry = -0.5j*parentgrp.attrs.get('dz') / (self.k0*parentgrp.attrs.get('n0')*parentgrp.attrs.get('dy')**2)
            self.rq = -0.5j*parentgrp.attrs.get('dz')*self.k0 / (parentgrp.attrs.get('n0'))
            # TODO: Get k0 and n0, get gain and eps
            # self.vectorize(parentgrp)
            self.nx, self.ny, self.nz = parentgrp.attrs.get('nx'), parentgrp.attrs.get('ny'), parentgrp.attrs.get('nz')
            self.xx, self.yy = parentgrp.attrs.get('xx'), parentgrp.attrs.get('yy')
            self.xxx, self.yyy, self.zzz = parentgrp.attrs.get('xxx'), parentgrp.attrs.get('yyy'), parentgrp.attrs.get('zzz')
            self.n0 = parentgrp.attrs.get('n0')

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
            * TODO: Free space should be built in here. Need to reinit self.nz, self.z_step, self.fseed

        """

        # Set LHS matrix coefficients
        ax, bx, cx = -0.5*self.rx*np.ones(self.nx-2), (1 + self.rx)*np.ones(self.nx-2), -0.5*self.rx*np.ones(self.nx-2)
        ay, by, cy = -0.5*self.ry*np.ones(self.ny-2), (1 + self.ry)*np.ones(self.ny-2), -0.5*self.ry*np.ones(self.ny-2)

        fsol_Ax = np.zeros([self.nx, self.ny], dtype=complex)
        fsol_Ax_x = np.zeros([self.nx, self.ny], dtype=complex)
        fsol_Ax_new = np.zeros([self.nx, self.ny], dtype=complex)

        fsol_Ay = np.zeros([self.nx, self.ny], dtype=complex)
        fsol_Ay_x = np.zeros([self.nx, self.ny], dtype=complex)
        fsol_Ay_new = np.zeros([self.nx, self.ny], dtype=complex)

        Ax = np.zeros([self.nx, self.ny, self.nz], dtype=complex)
        Ay = np.zeros([self.nx, self.ny, self.nz], dtype=complex)

        # TODO: Generated lg-mode, need to check intensity and polarization etc
        # TODO: Should this be equated to original or normalized lg-mode, or half thereof...?
        # TODO: Inserting functions directly
        #Ax[:, :, 0] = self.fseed[0]
        #Ay[:, :, 0] = self.fseed[0]
        Ax[:, :, 0] = self.Ax
        Ay[:, :, 0] = self.Ay

        clock = 0
        z_step = 0
        # TODO: Try to non-dimensionalize y and x, such that BC = z and they become sparse vectors
        # TODO: Test to see if data in ram changes before it's written to hdf5
        # TODO: Might be necessary to have two with h5py statements
        convergence = False
        #self._format_hdf5table3d_to_finit3d
        #print("DEBUGGING: _format_hdf5table3d_to_finit3d")
        while (z_step < self.nz-1) and (convergence is False):
            #print("DEBUGGING: ADI Loop: {0}".format(clock))
            with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
                parentgrp = hf.require_group(self.hdf5grp)
                grp = parentgrp.require_group(self.hdf5subgrp)
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

                eps_xx = parentgrp['eps'][0, 0, :, :, z_step]
                eps_xy = parentgrp['eps'][0, 1, :, :, z_step]
                eps_yy = parentgrp['eps'][1, 1, :, :, z_step]
                eps_yx = parentgrp['eps'][1, 0, :, :, z_step]

                d_Ax_x = np.zeros(self.nx-2, dtype=complex)
                d_Ay_x = np.zeros(self.nx-2, dtype=complex)
                for j in range(1, self.ny-1):
                    #d = np.zeros(self.nx-2)
                    for i in range(1, self.nx-1):
                        d_Ax_x[i-1] = 0.5*self.rx*fsol_Ax[i-1, j] + 0.5*self.rx*fsol_Ax[i+1, j] \
                            + self.ry*fsol_Ax[i, j-1] + self.ry*fsol_Ax[i, j+1] \
                            + (1 - self.rx - 2*self.ry)*fsol_Ax[i, j] \
                            + self.rq*(eps_xx[i, j] - self.n0**2)*fsol_Ax[i, j] \
                            + self.rq*eps_xy[i, j]*fsol_Ay[i, j]
                        d_Ay_x[i-1] = 0.5*self.rx*fsol_Ay[i-1, j] + 0.5*self.rx*fsol_Ay[i+1, j] \
                            + self.ry*fsol_Ay[i, j-1] + self.ry*fsol_Ay[i, j+1] \
                            + (1 - self.rx - 2*self.ry)*fsol_Ay[i, j] \
                            + self.rq*(eps_yy[i, j] - self.n0**2)*fsol_Ay[i, j] \
                            + self.rq*eps_yx[i, j]*fsol_Ax[i, j]

                    d_Ax_x[0] += 0.5*self.rx*fsol_Ax[0, j]
                    d_Ax_x[-1] += 0.5*self.rx*fsol_Ax[-1, j]
                    d_Ay_x[0] += 0.5*self.rx*fsol_Ay[0, j]
                    d_Ay_x[-1] += 0.5*self.rx*fsol_Ay[-1, j]

                    fsol_Ax_x[1:self.nx-1, j] = self.tdma_solver(ax, bx, cx, d_Ax_x)[...]
                    fsol_Ay_x[1:self.nx-1, j] = self.tdma_solver(ax, bx, cx, d_Ay_x)[...]

                d_Ax_y = np.zeros(self.ny-2, dtype=complex)
                d_Ay_y = np.zeros(self.ny-2, dtype=complex)
                for i in range(1, self.nx-1):
                    #d = np.zeros(self.nx-2)
                    for j in range(1, self.ny-1):
                        d_Ax_y[j-1] = 0.5*self.rx*fsol_Ax[i-1, j] + 0.5*self.rx*fsol_Ax[i+1, j] \
                            + 0.5*self.rx*fsol_Ax_x[i-1, j] + self.rx*fsol_Ax_x[i, j] + 0.5*self.rx*fsol_Ax_x[i+1, j] \
                            + self.ry*fsol_Ax[i, j-1] + self.ry*fsol_Ax[i, j+1] \
                            + (1 - self.rx - self.ry)*fsol_Ax[i, j] \
                            + self.rq*(eps_xx[i, j] - self.n0**2)*fsol_Ax[i, j] \
                            + self.rq*eps_xy[i, j]*fsol_Ay[i, j]
                        d_Ay_y[j-1] = 0.5*self.rx*fsol_Ay[i-1, j] + 0.5*self.rx*fsol_Ay[i+1, j] \
                            + 0.5*self.rx*fsol_Ay_x[i-1, j] + self.rx*fsol_Ay_x[i, j] + 0.5*self.rx*fsol_Ay_x[i+1, j] \
                            + self.ry*fsol_Ay[i, j-1] + self.ry*fsol_Ay[i, j+1] \
                            + (1 - self.rx - self.ry)*fsol_Ay[i, j] \
                            + self.rq*(eps_yy[i, j] - self.n0**2)*fsol_Ay[i, j] \
                            + self.rq*eps_yx[i, j]*fsol_Ax[i, j]

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

        self.Ax = fsol_Ax_new
        self.Ay = fsol_Ay_new

    @time_it
    def convolve(self, z):
        """Two-dimensional Freznel diffraction convolution function using field current state of object at distance z."""
        return super().convolve(self.Ax, z), super().convolve(self.Ay, z)

    def mirror_phase_factor(self, curvature=30):
        """Incorporates phase factor on beam components due to mirror.

        Args:
            curvature(float): Radius of curvature of mirror surface.

        """

    def surface_deformation_phase_factor(self, dz=0.01, x_0=0.0, y_0=0.0, width_x=0.1, width_y=0.1, power_x=1.0, power_y=1.0):
        """Generate the surface displacement field :math:`\delta z(x,y)` and calculate the phase factor influence on fields :math:`A_x` and :math:`A_y`.

        Args:
            dz (float): Amplitude of the displacement field [Default value: 0.01].
            x_0 (float): x co-ordinates for the center of surface displacement field. [Default value: 0.0].
            y_0 (float): y co-ordinates for the center of surface displacement field. [Default value: 0.0].
            width_x (float) :Gaussian width in x direction.
            width_y (float) :Gaussian width in y direction.
        """
        pass

    def free_space_implicit(self):
        pass


    @time_it
    def free_space_adi_method(self, rtol=1e-04, atol=1e-06, neumann_BC=0.01):
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
            * TODO: Steps in z, delta z etc for amplifiers.

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

        # TODO: Generated lg-mode, need to check intensity and polarization etc
        # TODO: Should this be equated to original or normalized lg-mode, or half thereof...?
        # TODO: Inserting functions directly
        Ax[:, :, 0] = self.fseed[0]
        Ay[:, :, 0] = self.fseed[0]

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
                            + self.rq*(eps_xx(i, j, z_step) - self.n0**2)*fsol_Ax[i, j] \
                            + self.rq*eps_xy(i, j, z_step)*fsol_Ay[i, j]
                        d_Ay_x[i-1] = 0.5*self.rx*fsol_Ay[i-1, j] + 0.5*self.rx*fsol_Ay[i+1, j] \
                            + self.ry*fsol_Ay[i, j-1] + self.ry*fsol_Ay[i, j+1] \
                            + (1 - self.rx - 2*self.ry)*fsol_Ay[i, j] \
                            + self.rq*(eps_yy(i, j, z_step) - self.n0**2)*fsol_Ay[i, j] \
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
                            + self.rq*(eps_xx(i, j, z_step) - self.n0**2)*fsol_Ax[i, j] \
                            + self.rq*eps_xy(i, j, z_step)*fsol_Ay[i, j]
                        d_Ay_y[j-1] = 0.5*self.rx*fsol_Ay[i-1, j] + 0.5*self.rx*fsol_Ay[i+1, j] \
                            + 0.5*self.rx*fsol_Ay_x[i-1, j] + self.rx*fsol_Ay_x[i, j] + 0.5*self.rx*fsol_Ay_x[i+1, j] \
                            + self.ry*fsol_Ay[i, j-1] + self.ry*fsol_Ay[i, j+1] \
                            + (1 - self.rx - self.ry)*fsol_Ay[i, j] \
                            + self.rq*(eps_yy(i, j, z_step) - self.n0**2)*fsol_Ay[i, j] \
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

    def plot_field_in_medium(self, savepath='images/field_profile.png', z_ind=-1):
        """Plot the current cross-sectional source profile from the heat load object.

        This method calls the parent / ancestor / super method to do the actual plotting of figures,
        at the desired z-axial point.

        Args:
            savepath (str): Filename to save image file to.
            z (float): Source attenuation distance along axial direction and within crystal, :math:`z=0` is the pump end-face.
            time_step (int): Integer placeholder of temporal indices.

        Attributes:
            _tempQinit (:obj:`~numpy.ndarray`): Private member function for storing Qinit while it is
        exponentially attenuated by distance z.

        Todo:
            * Verify whether we really need 3D meshgrid profiles or if 2D ones will suffice.
        """

        #try:
        #    if int(time_step) > self.nt:
        #        raise ValueError("Invalid time index provided.")
        #except ValueError as t_out_of_bounds:
        #    time_step = int(self.nt)
        #    print(t_out_of_bounds)

        #print('Pump side total input power: {0:.2f} W'.format(self.Ptot_in))
        #print('Output coupler total power: {0:.2f} W'.format(self.Ptot_out))
        # try:
        #     if (z < 0) or (z > self.Lz):
        #         raise ValueError("Invalid attenuation distance provided.")
        # except ValueError as z_out_of_bounds:
        #     z = self.Lz
        #     print(z_out_of_bounds)

        warnings.warn("PendingDeprecationWarning: All plotting and imaging will be handled by\
        visualisation module.", PendingDeprecationWarning)

        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            parentgrp = hf.require_group(self.hdf5grp)
            grp = parentgrp.require_group(self.hdf5subgrp)
            I_x = np.square(np.abs(grp['Ax'][...]))
            I_y = np.square(np.abs(grp['Ay'][...]))

        AXIS_TICK_FONTSIZE = 4 # -2.2
        LABEL_FONTSIZE = 4.5 # -.5

        plt.close('all')
        plt.clf()
        plt.cla()
        plt.figure(dpi=MAX_DPI)
        plt.rc('xtick', labelsize=AXIS_TICK_FONTSIZE)
        plt.rc('ytick', labelsize=AXIS_TICK_FONTSIZE)
        plt.title("Initial normalized intensity profile of pump EM field")

        _fig = plt.figure()
        _ax1 = _fig.add_subplot(4, 2, 1)
        _ax2 = _fig.add_subplot(4, 2, 2)
        _ax3 = _fig.add_subplot(4, 2, 3, projection='3d')
        _ax4 = _fig.add_subplot(4, 2, 4, projection='3d')
        _ax5 = _fig.add_subplot(4, 2, 5)
        _ax6 = _fig.add_subplot(4, 2, 6)
        _ax7 = _fig.add_subplot(4, 2, 7, projection='3d')
        _ax8 = _fig.add_subplot(4, 2, 8, projection='3d')
        _fig.suptitle("Vector field profile over gain medium", fontsize=LABEL_FONTSIZE+1.0)
        #_fig.set_tight_layout(True)
        _fig.subplots_adjust(wspace=0.25, hspace=0.4)

        #######################
        _ax1.set_xlabel('X [cm]', fontsize=LABEL_FONTSIZE)
        _ax1.set_ylabel('Y [cm]', fontsize=LABEL_FONTSIZE)
        _ax1.set_title('Pump-side end face Ax field', fontsize=LABEL_FONTSIZE)
        _quad1 = _ax1.pcolormesh(self.xx, self.yy, I_x[:, :, 0], cmap='gnuplot2', shading='gouraud')

        #######################
        _ax2.set_xlabel('X [cm]', fontsize=LABEL_FONTSIZE)
        _ax2.set_ylabel('Y [cm]', fontsize=LABEL_FONTSIZE)
        _ax2.set_title('Output coupler side end face Ax field'.format(z_ind), fontsize=LABEL_FONTSIZE)
        _quad2 = _ax2.pcolormesh(self.xx, self.yy, I_x[:, :, z_ind], cmap='gnuplot2', shading='gouraud')

        #######################
        _ax3.set_xlabel('X [cm]', fontsize=LABEL_FONTSIZE)
        _ax3.set_xlim(-1*self.Lx/2, self.Lx/2)
        _ax3.set_ylabel('Z [cm]', fontsize=LABEL_FONTSIZE)
        _ax3.set_ylim(0, 1.1*self.Lz)
#        _ax3.set_zlabel('Q', fontsize=LABEL_FONTSIZE)
#        _ax3.set_zlim(self._vmin, self._vmax)
        _quad3 = _ax3.plot_surface(self.xxx.T[:, round(self.ny/2), :], \
                                   self.zzz.T[:, round(self.ny/2), :], \
                                   I_x, \
                                   alpha=0.3, cmap='gnuplot2')
        _cset3 = _ax3.contourf(self.xxx.T[:, round(self.ny/2), :], \
                               self.zzz.T[:, round(self.ny/2), :], \
                               self.Qinit3d.T[:, round(self.ny/2), :]/(self.nx*self.ny), \
                               zdir='x', offset=-0.6*self.Lx, cmap='gist_yarg')
        _cset3 = _ax3.contourf(self.xxx.T[:, round(self.ny/2), :], \
                               self.zzz.T[:, round(self.ny/2), :], \
                               self.Qinit3d.T[:, round(self.ny/2), :]/(self.nx*self.ny), \
                               zdir='y', offset=1.1*self.Lz, cmap='gist_gray')
        _cset3 = _ax3.contourf(self.xxx.T[:, round(self.ny/2), :], \
                               self.zzz.T[:, round(self.ny/2), :], \
                               self.Qinit3d.T[:, round(self.ny/2), :]/(self.nx*self.ny), \
                               zdir='z', offset=0, cmap='gnuplot2')

        #######################
        _ax4.set_xlabel('Y [cm]', fontsize=LABEL_FONTSIZE)
        _ax4.set_xlim(-0.5*self.Ly, 0.5*self.Ly)
        _ax4.set_ylabel('Z [cm]', fontsize=LABEL_FONTSIZE)
        _ax4.set_ylim(0, 1.1*self.Lz)
#        _ax4.set_title('Q', fontsize=LABEL_FONTSIZE)
#        _ax4.set_zlim(self._vmin, self._vmax)
        _quad4 = _ax4.plot_surface(self.yyy[round(self.nx/2), :, :], \
                                   self.zzz[round(self.nx/2), :, :], \
                                   I_x, \
                                   alpha=0.3, cmap='gnuplot2')
        _cset4 = _ax4.contourf(self.yyy[round(self.nx/2), :, :], \
                               self.zzz[round(self.nx/2), :, :], \
                               self.Qinit3d[round(self.nx/2), :, :]/(self.nx*self.ny), \
                               zdir='x', offset=-0.6*self.Ly, cmap='gist_yarg')
        _cset4 = _ax4.contourf(self.yyy[round(self.nx/2), :, :], \
                               self.zzz[round(self.nx/2), :, :], \
                               self.Qinit3d[round(self.nx/2), :, :]/(self.nx*self.ny), \
                               zdir='y', offset=1.1*self.Lz, cmap='gist_gray')
        _cset4 = _ax4.contourf(self.yyy[round(self.nx/2), :, :], \
                               self.zzz[round(self.nx/2), :, :], \
                               self.Qinit3d[round(self.nx/2), :, :]/(self.nx*self.ny), \
                               zdir='z', offset=0, cmap='gnuplot2')

        #cax = _fig.add_axes([0.85, 0.1, 0.03, 0.8])
        cax = _fig.add_axes([0.95, 0.1, 0.01, 0.75])
        _fig.colorbar(_quad1, cax=cax).set_label('Q')

        plt.savefig(savepath, dpi=MAX_DPI)


class NonParaxialWaveEqn(ParaxialWaveEqn):
    def __init__(self):
        pass
