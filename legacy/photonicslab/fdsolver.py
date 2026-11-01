# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""Finite difference solver for a general, second-order differential equations.

Module uses finite differences to solve second order differential equations, saving the output to hdf5
format for each time step to optimise memory utilization. This module can be easily expanded to higher-order
equation with even differing partial differentiation terms.

Todo:
    * Separate out terms into :math:`u, u_{t}, u_{x}, u_{xx}, u_{y}, u_{yy}` etc for readability
    * Add implicit method
    * Extend to 3D
"""

import numpy as np
import h5py
#import scipy
#from scipy import sparse as sps
# TODO: FIXME: Importing and exception handling.
# from sps.linalg import spsolve as solve

import scipy.sparse as sps
from scipy.sparse.linalg import spsolve as solve

from photonicslab.config import *
#from numba import jit, f8

class FDSolver2D(object):
    """Two-dimensional finite-difference solver object.

    This employs a simple forward time, central step approximation.

    Args:
        nt (int): Number of time steps for temporal evolution.
        finit (:obj:`~numpy.ndarray`): Cross-sectional profile of initial system solution / state.
        Fdrive (:obj:`~numpy.ndarray`): Cross-sectional profile of initial system driving / damping force.
        sx (float): Coefficient for approximation to forward-time, central step along x-direction.
        sy (float): Coefficient for approximation to forward-time, central step along y-direction.
        hdf5filename (str): Filename of the HDF5 file object to be created or read from / appended to.
        hdf5table (str): Dataset name of the HDF5 group/dataset object to apply the FD to.

    References:
        * `Lecture notes from GEOL557 Numerical Modelling of Earth Systems by Becker and Kaus (2016) <http://www-udc.ig.utexas.edu/external/becker/teaching-557.html>`_

    """

    def __init__(self, hdf5filename='hdf5db/storage.hdf5', hdf5grp='nb1', hdf5table='temperature'):
        self.hdf5filename = str(hdf5filename)
        self.hdf5table = str(hdf5table)
        self.hdf5grp = str(hdf5grp)
        self.dsetfsol = str( self.hdf5grp + '/' + self.hdf5table)
        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            grp = hf.require_group(self.hdf5grp)
            self.nt = grp.attrs.get('nt')
            self.sx = grp.attrs.get('sx')
            self.sy = grp.attrs.get('sy')
            self.nx = grp.attrs.get('nx')
            self.ny = grp.attrs.get('ny')
            self.finit = grp.attrs['TBC']*np.ones([grp.attrs['nx'], grp.attrs['ny']])
            #self.finit = grp.attrs.get('TBC')*np.ones([grp.attrs.get('nx'), grp.attrs.get('ny')])
            self.Fdrive = grp['Qload'][:]
            if self.dsetfsol not in hf:
                grp.create_dataset(self.hdf5table, (self.nt, self.finit.shape[0], self.finit.shape[1]),
                                   dtype=np.dtype(float).type, maxshape=(MAX_NT, MAX_NX, MAX_NY))
            else:
                grp[self.hdf5table].resize((self.nt, self.finit.shape[0], self.finit.shape[1]))
            grp[self.hdf5table][...] = self.finit[None, :, :]

    def _format_hdf5table_to_finit(self):
        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            grp = hf.require_group(self.hdf5grp)
            grp[self.hdf5table][...] = self.finit[None, :, :]

    def explicit_method(self, tol_clock=None):
        """Explicit forward-time, central step finite-difference method.

        Writes solution to finite-difference approximation out to HDF5 file.

        Args:
            tol_clock (bool): Steady-state convergence criteria satisfied.

        Todo:
            * Add check for convergence and reshape fsol
        """

        self._format_hdf5table_to_finit()
        fsol = []
        fsol = self.finit
        # TODO: Only need to assign BC's to temp
        clock = 0
        while (clock < self.nt-1) and (tol_clock is None):
            clock+=1
            temp = []
            temp = self.finit
            # TODO: Only need BC assignments here temp[0,:] = self.TBC etc...
            for i in range(1, self.nx-2):
                for j in range(1, self.ny-2):
                    temp[i, j] = fsol[i, j]\
                        + self.sx*(fsol[i, j-1] -2*fsol[i, j] + fsol[i, j+1])\
                        + self.sy*(fsol[i-1, j] -2*fsol[i, j] + fsol[i+1, j])\
                        + self.Fdrive[clock, i, j]

            with h5py.File(self.hdf5filename, 'a') as hf:
                hf[self.dsetfsol][clock, ...] = temp
            fsol = []
            fsol = temp

    def explicit_method_hdf(self, tol_clock=None):
        """Explicit forward-time, central step finite-difference method directly using HDF5.

        Writes solution to finite-difference approximation out to HDF5 file.

        Args:
            tol_clock (bool): Steady-state convergence criteria satisfied.

        Todo:
            * Add check for convergence and reshape fsol
        """

        self._format_hdf5table_to_finit()
        clock = 0
        while (clock < self.nt-1) and (tol_clock is None):
            with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
                grp = hf.require_group(self.hdf5grp)
                fsol = grp[self.hdf5table]
                for i in range(1, self.nx-2):
                    for j in range(1, self.ny-2):
                        fsol[clock+1, i, j, ...] = fsol[clock, i, j]\
                          + self.sx*(fsol[clock, i, j-1] -2*fsol[clock, i, j] + fsol[clock, i, j+1])\
                          + self.sy*(fsol[clock, i-1, j] -2*fsol[clock, i, j] + fsol[clock, i+1, j])\
                          + self.Fdrive[clock, i, j]
            clock+=1

    def cuda_implicit_method(self):
        """Tri-diagonal matrix solver using NVIDIA CUDA gpu-array elements objects."""
        pass

    def cuda_adi_method(self):
        pass

class FDSolver3D(FDSolver2D):
    """Three-dimensional finite-difference solver object.

    This solver works in a similar manner to it's base class, moreover it efficiently uses the 2D profiles
    for the initial and driving functions in generating the corresponding 3D profiles which are manipulated
    using hdf5 files only.

    Args:
        sz (float): Coefficient for approximation to forward-time, central step along z-direction.
        nz (int): Number of spatial steps along z-direction.
        alpha (float): Exponential decay constant for absorption of driving force along (axial) z-direction.
        dz (float): Step size along (axial) z-direction, used for damping calculation together with alpha.
        hdf5grp (str): HDF5 group name.
        hdf5tabledrive (str): Dataset name of the HDF5 (excluding group) object to store driving force to.

    Todo:
        * TODO: Document "book-keeping" issues, mapping 3D temperature array to 1D vector.
        * Need to move hdf5 databases to group format and manage their creation / deletion in base and child constructors.
        * Dereference hdf5 as variable: i.e. dset2 =grp.create_dataset("qinit") or Tsol = hf[fdsol/temperature]
        * Implement stopping condition when working in region with a source / temperature change.
        * Need to verify order and arrangement of indices, CHECK shape assignments.
        * Resolve circular generation and dependencies of Fdrive, meshgrids, ranges and linspaces.
        * Current configuration does not easily permit implementation of time-dependent sources.

    References:
        * `SO Broadcast 2D Array to 3D <https://stackoverflow.com/questions/32171917/copy-2d-array-into-3rd-dimension-n-times-python>`_
        * `SO CUDA Accelerate Sparse Matrix by Vector Product <https://stackoverflow.com/questions/49019189/how-can-i-accelerate-a-sparse-matrix-by-dense-vector-product-currently-implemen>`_
    """

    def __init__(self, hdf5filename='hdf5db/storage.hdf5', hdf5grp='nb1', hdf5table='temperature'):
        super().__init__(hdf5filename=hdf5filename, hdf5grp=hdf5grp, hdf5table=hdf5table)
        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            grp = hf.require_group(self.hdf5grp)
            self.sz = grp.attrs.get('sz')
            self.nz = grp.attrs.get('nz')
            self.dz = grp.attrs.get('dz')
            self.TBC = grp.attrs.get('TBC')
            self.finit3d = grp.attrs.get('TBC')*np.ones([self.nx, self.ny, self.nz])
            # TODO: FIXME: Temporal pump / driving source.
            #self.FDrive3d = grp['Qload3d'][:, :, :, 0]
            #self.FDrive3d = grp['Qload3d'][..., 0]
            # TODO: FIXME: Qload3d is not temporal
            self.FDrive3d = grp['Qload3d'][...]
            self.dsetfsol3d = str(self.dsetfsol + '3d')
            self.hdf5table3d = str(self.hdf5table + '3d')
            if self.dsetfsol3d not in hf:
                grp.create_dataset(self.hdf5table3d, (self.nt, self.nx, self.ny, self.nz), dtype=np.dtype(float).type,
                                   maxshape=(MAX_NT, MAX_NX, MAX_NY, MAX_NZ))
            else:
                grp[self.hdf5table3d].resize((self.nt, self.nx, self.ny, self.nz))
            grp[self.hdf5table3d][...] = self.finit3d[None, ...]

    def _format_hdf5table3d_to_finit3d(self):
        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            grp = hf.require_group(self.hdf5grp)
            grp[self.hdf5table3d][...] = self.finit3d[None, ...]

    def _numbering(self, i, j, k):
        return i*self.ny*self.nz + j*self.nz + k

    def vectorize(self):
        """Internal function used to convert 3D arrays into 1D vectors.

        .. math::

            \\text{numbering}_{i, j, k} = k \\times n_{x} \\times n_{y} + j \\times n_{x} + i

        Returns:
            numbering (:obj:`~numpy.ndarray`): One-dimensional vector mapping coordinates [i, j, k] -> number.

        Todo:
            * Accelerate this with :obj:`numba.jit`.
        """
        # Maybe convert to list comprehension
        # Careful with empty, could see blowup...?
        num = 0
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
        return numbering.astype(int)

    @time_it
    def implicit_method(self, rtol=1e-01, atol=1e-03, method='spsolve', neumann_BC=0.1):
        """Uses scipy sparse matrices to solve seven band diagonal matrix Ax = Bx + \\vec{y} linear equation.

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
            method (str): Linear Algebra method. One of {spsolve, bicg, bicgstab, cg, cgs, gmres, lgmres, minres, qmr, gcrotmk}
            neumann_BC (float): Neumann Boundary Condition for constant flux set through fictitious BC's.

        .. seealso::
            :obj:`~scipy.sparse.linalg.spsolve`, :obj:`~scipy.sparse.linalg.gmres`, :obj:`~scipy.sparse.linalg.lgmres`, and :obj:`~scipy.sparse.linalg.bicgstab`
            :obj:`~numpy.allclose`

        References:
            * `SO Tridiagonal Block Matrix with SciPy Sparse <https://stackoverflow.com/questions/44641996/tridiagonal-block-matrix-using-scipy-sparse>`_
            * `SO Tridiagonal Matrix Using SciPy Sparse Diags <https://stackoverflow.com/questions/50292546/scipy-tridiagonal-matrix-using-scipy-sparse-diags>`_
            * `Consider storing hdf5file in memory <https://www.pytables.org/cookbook/inmemory_hdf5_files.html>`_

        Todo:
            * TODO: Accelerate nested for loops using :obj:`numba.jit`
            * TODO: Test using diagonal / banded matrix creation methods and solvers.
            * TODO: Test other linear algebra solvers :obj:`scipy.sparse.linalg`.
            * TODO: Test dense conversion and solutions with :obj:`numpy.linalg.solve` and / or :obj:`scipy.linalg.solve`.
            * TODO: Make the construction of the seven band tridiagonal matrices more concise.
        """

        try:
            direct_method = False
            ##import scipy.sparse as sps
            if (method == 'bicg'):
                from sps.linalg import bicg as solve
            elif (method == 'bicgstab'):
                from sps.linalg import bicgstab as solve
            elif (method == 'cg'):
                from sps.linalg import cg as solve
            elif (method == 'cgs'):
                from sps.linalg import cgs as solve
            elif (method == 'gmres'):
                from sps.linalg import gmres as solve
            elif (method == 'lgmres'):
                from sps.linalg import lgmres as solve
            elif (method == 'minres'):
                from sps.linalg import lmres as solve
            elif (method == 'qmr'):
                from sps.linalg import lmres as solve
            elif (method == 'gcrotmk'):
                from sps.linalg import lmres as solve
            else:
                raise ValueError("Implicit method solver does not exist.")
        except ValueError as default_method:
            ##import scipy.sparse as sps
            ##from sps.linalg import spsolve as solve
            #from sps.linalg import spsolve as solve
            #import scipy.sparse.linalg.spsolve as solve
            direct_method = True
            if (method != 'spsolve'):
                print("BadSolverImport: {0}".format(default_method))

        # TODO: FIXME: Importing and exception handling.
        #from sps.linalg import spsolve as solve

        numbering = self.vectorize()
        x = np.zeros(self.nx*self.ny*self.nz)
        xnew = np.zeros(self.nx*self.ny*self.nz)
        b = np.zeros(self.nx*self.ny*self.nz)
        y = np.zeros(self.nx*self.ny*self.nz)

        #A = sps.csr_matrix((self.nz*self.ny*self.nx, self.nz*self.ny*self.nx), dtype=np.dtype(float).type)
        #B = sps.csr_matrix((self.nz*self.ny*self.nx, self.nz*self.ny*self.nx), dtype=np.dtype(float).type)
        #A_dense = np.zeros([self.nz*self.ny*self.nx, self.nz*self.ny*self.nx], dtype=np.dtype(float).type)
        A_dense = np.zeros([self.nx*self.ny*self.nz, self.nx*self.ny*self.nz])
        B_dense = np.zeros([self.nx*self.ny*self.nz, self.nx*self.ny*self.nz])

        for i in range(1, self.nx-2):
            for j in range(1, self.ny-2):
                for k in range(self.nz):
                    #print(numbering[i, j, k])
                    A_dense[numbering[i, j, k], numbering[i, j, k]] = 1 + 2*self.sx + 2*self.sy + 2*self.sz
                    A_dense[numbering[i, j, k], numbering[i-1, j, k]] = -1*self.sx
                    A_dense[numbering[i, j, k], numbering[i+1, j, k]] = -1*self.sx
                    A_dense[numbering[i, j, k], numbering[i, j-1, k]] = -1*self.sy
                    A_dense[numbering[i, j, k], numbering[i, j+1, k]] = -1*self.sy

                    B_dense[numbering[i, j, k], numbering[i, j, k]] = 1 - 2*self.sx - 2*self.sy - 2*self.sz
                    B_dense[numbering[i, j, k], numbering[i-1, j, k]] = self.sx
                    B_dense[numbering[i, j, k], numbering[i+1, j, k]] = self.sx
                    B_dense[numbering[i, j, k], numbering[i, j-1, k]] = self.sy
                    B_dense[numbering[i, j, k], numbering[i, j+1, k]] = self.sy
                    if (k == 0):
                        A_dense[numbering[i, j, k], numbering[i, j, k+1]] = -2*self.sz
                        B_dense[numbering[i, j, k], numbering[i, j, k+1]] = 2*self.sz
                    # Neumann Boundary Conditions for Crystal End-Faces
                    elif (k == self.nz-1):
                        A_dense[numbering[i, j, k], numbering[i, j, k-1]] = -2*self.sz
                        B_dense[numbering[i, j, k], numbering[i, j, k-1]] = 2*self.sz
                    elif (k != 0) and (k != self.nz-1):
                        # Only need an else here actually
                        A_dense[numbering[i, j, k], numbering[i, j, k-1]] = -1*self.sz
                        A_dense[numbering[i, j, k], numbering[i, j, k+1]] = -1*self.sz
                        B_dense[numbering[i, j, k], numbering[i, j, k-1]] = self.sz
                        B_dense[numbering[i, j, k], numbering[i, j, k+1]] = self.sz

        # Dirchelet Boundary Conditions
        # Diagonals = 1, and there are no off-diagonal elements
        # TODO: Test performance of the two assignment methods below
        # A_dense[numbering[0, :, :], numbering[0, :, :]] = 1
        # A_dense[numbering[-1, :, :], numbering[-1, :, :]] = 1
        # B_dense[numbering[0, :, :], numbering[0, :, :]] = 1
        # B_dense[numbering[-1, :, :], numbering[-1, :, :]] = 1

        # A_dense[numbering[:, 0, :], numbering[:, 0, :]] = 1
        # A_dense[numbering[:, -1, :], numbering[:, -1, :]] = 1
        # B_dense[numbering[:, 0, :], numbering[:, 0, :]] = 1
        # B_dense[numbering[:, -1, :], numbering[:, -1, :]] = 1
        for k in range(self.nz):
            for j in range (self.ny):
                A_dense[numbering[0, j, k], numbering[0, j, k]] = 1
                A_dense[numbering[-1, j, k], numbering[-1, j, k]] = 1
                B_dense[numbering[0, j, k], numbering[0, j, k]] = 1
                B_dense[numbering[-1, j, k], numbering[-1, j, k]] = 1
            for i in range (self.nx):
                A_dense[numbering[i, 0, k], numbering[i, 0, k]] = 1
                A_dense[numbering[i, -1, k], numbering[i, -1, k]] = 1
                B_dense[numbering[i, 0, k], numbering[i, 0, k]] = 1
                B_dense[numbering[i, -1, k], numbering[i, -1, k]] = 1

        #A_dense.eliminate_zeros()
        #B_dense.eliminate_zeros()

        #A = sps.dia_matrix(A_dense)
        #B = sps.dia_matrix(B_dense)

        A = sps.csr_matrix(A_dense)
        B = sps.csr_matrix(B_dense)

        clock = 0
        # TODO: Try to non-dimensionalize y and x, such that BC = z and they become sparse vectors
        # TODO: Test to see if data in ram changes before it's written to hdf5
        # TODO: Might be necessary to have two with h5py statements
        convergence = False

        # FIXME: Remove this when you implement temporal sources
        # y = np.asarray([grp['Qload3d'][i, j, k] \
        #                 for i in range(self.nx) \
        #                     for j in range(self.ny) \
        #                         for k in range(self.nz)])

        while (clock < self.nt-1) and (convergence is False):
            with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
                grp = hf.require_group(self.hdf5grp)
                x = np.asarray([grp[self.hdf5table3d][clock, i, j, k] \
                                for i in range(self.nx) \
                                    for j in range(self.ny) \
                                        for k in range(self.nz)])
                y = np.asarray([grp['Qload3d'][i, j, k] \
                                for i in range(self.nx) \
                                    for j in range(self.ny) \
                                        for k in range(self.nz)])
                # FIXME: Temporal component of Qload3d
                #y = np.asarray([grp['Qload3d'][clock, i, j, k] \
                # y = np.asarray([grp['Qload3d'][i, j, k] \
                #                 for i in range(self.nx) \
                #                     for j in range(self.ny) \
                #                         for k in range(self.nz)])
                # Neumann BC's note sign of start and end-face terms
                y[numbering[:, :, 0]] += -2*self.sz*grp.attrs.get('dz')*neumann_BC
                y[numbering[:, :, self.nz-1]] += 2*self.sz*grp.attrs.get('dz')*neumann_BC
                # TODO: Test and fix below, may be broken
                # TODO: test against b = np.dot(A.toarray(), x)
                # TODO: Also test against b.data += y
                # TODO: Have options for linalg routines
                b = A.dot(x) + y
                if (direct_method):
                    xnew = sps.linalg.spsolve(A, b)
                    info = 0
                else:
                    xnew, info = sps.linalg.spsolve(A, b, x)

                try:
                    if info < 0:
                        raise ValueError("Illegal input or breakdown in sparse linear algebra solver.")
                    elif info > 0:
                        raise ValueError("Convergence to tolerance not achieved, number of iterations.")
                except ValueError as convergence_error:
                    print("ConvergenceError: {0}".format(convergence_error))
                    raise convergence_error
                else:
                    clock+=1
                    # TODO: Fix this broadcast of numbering
                    xnew_numbering = np.zeros([self.nx, self.ny, self.nz])
                    for i in range(self.nx):
                        for j in range(self.ny):
                            for k in range(self.nz):
                                xnew_numbering[i, j, k] = xnew[i*self.ny*self.nz + j*self.nz + k]

                # TODO: Time it and test if an additional with h5py statement must come here
                # TODO: Verify proper assignment of values i.e. explicit ellipses to copy values and not pointer!
                grp[self.hdf5table3d][clock, ...] = xnew_numbering[...]
                if (np.allclose(x, xnew, rtol, atol)):
                    convergence = True
                if (convergence) or (clock == self.nt-1):
                    grp.attrs.create('time', clock)

        ######################

    @time_it
    def implicit_cupy_method(self):
        pass

    @time_it
    def implicit_pycuda_method(self):
        pass

    @time_it
    def implicit_tf_method(self):
        pass

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
        #ax0, bx0, cx0 = (1 - self.sx), 2*self.sx, -1*self.sx
        #axN, bxN, cxN = -1*self.sx, 2*self.sx, (1 - self.sx)
        ax, bx, cx = -1*self.sx*np.ones(self.nx-2), (1 + 2*self.sx)*np.ones(self.nx-2), -1*self.sx*np.ones(self.nx-2)
        #ax[0], bx[0], cx[0] = (1 - self.sx), 2*self.sx, -1*self.sx
        #ax[-1], bx[-1], cx[-1] = -1*self.sx, 2*self.sx, (1 - self.sx)

        ay, by, cy = -1*self.sy*np.ones(self.ny-2), (1 + 2*self.sy)*np.ones(self.ny-2), -1*self.sy*np.ones(self.ny-2)
        #ay[0], by[0], cy[0] = (1 - self.sy), 2*self.sy, -1*self.sy
        #ay[-1], by[-1], cy[-1] = -1*self.sy, 2*self.sy, (1 - self.sy)

        az, bz, cz = -1*self.sz*np.ones(self.nz+4), (1 + 2*self.sz)*np.ones(self.nz+4), -1*self.sz*np.ones(self.nz+4)
        #az, bz, cz = -1*self.sz*np.ones(self.nz), (1 + 2*self.sz)*np.ones(self.nz), -1*self.sz*np.ones(self.nz)
        #az[0], bz[0], cz[0] = (1 - self.sz), 2*self.sz, -1*self.sz
        #az[-1], bz[-1], cz[-1] = -1*self.sz, 2*self.sz, (1 - self.sz)

        xxx = np.zeros([self.nx, self.ny, self.nz])

        fsol = np.ones([self.nx, self.ny, self.nz+6])
        fsol_x = np.ones([self.nx, self.ny, self.nz+6])
        fsol_y = np.ones([self.nx, self.ny, self.nz+6])
        fsol_new = np.ones([self.nx, self.ny, self.nz+6])
        fdrive = np.zeros([self.nx, self.ny, self.nz+6])
        #fsol_x_BC = np.zeros([self.nx, self.ny, 2])

        clock = 0
        # TODO: Try to non-dimensionalize y and x, such that BC = z and they become sparse vectors
        # TODO: Test to see if data in ram changes before it's written to hdf5
        # TODO: Might be necessary to have two with h5py statements
        convergence = False
        self._format_hdf5table3d_to_finit3d
        while (clock < self.nt-1) and (convergence is False):
            with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
                grp = hf.require_group(self.hdf5grp)
                # x = np.asarray([grp[self.hdf5table3d][clock, i, j, k] for i in range(self.nx) for j in range(self.ny)
                #                 for k in range(self.nz)])
                # y = np.asarray([grp['Qload3d'][clock, i, j, k] for i in range(self.nx) for j in range(self.NY)
                #                 for k in range(self.nz)])
                # # Neumann BC's note sign of start and end-face terms
                # y[numbering[:, :, 0]] += -2*self.sz*grp.attrs.get('dz')*neumann_BC
                # y[numbering[:, :, self.nz-1]] += 2*self.sz*grp.attrs.get('dz')*neumann_BC
                # # TODO: Test and fix below, may be broken
                # # TODO: test against b = np.dot(A.toarray(), x)
                # # TODO: Also test against b.data += y
                # # TODO: Have options for linalg routines
                # b = A.dot(x) + y
                #fsol_x_BC[..., 0] = grp[self.hdf5table3d][clock, :, :, 0] + grp['Qload3d'][:, :, 0]
                #fsol_x_BC[..., -1] = grp[self.hdf5table3d][clock, :, :, -1] + grp['Qload3d'][:, :, -1]
                fsol[:, :, :] = grp.attrs['TBC']
                fsol[:, :, 3:self.nz+3] = grp[self.hdf5table3d][clock, ...] + grp['Qload3d'][...]
                #fsol[:, :, 0] = grp[self.hdf5table3d][clock, ..., 0] + grp['Qload3d'][..., 0]
                fsol[:, :, 2] = grp[self.hdf5table3d][clock, ..., 0] + grp['Qload3d'][..., 0]
                fsol[:, :, 1] = grp[self.hdf5table3d][clock, ..., 0] + 0.5*grp['Qload3d'][..., 0]
                fsol[:, :, 0] = grp[self.hdf5table3d][clock, ..., 0] + 0.25*grp['Qload3d'][..., 0]
                #fsol[:, :, -1] = grp[self.hdf5table3d][clock, ..., -1] + grp['Qload3d'][..., -1]
                #fsol[1:self.nx+1, 1:self.ny+1, -1] = grp[self.hdf5table3d][clock, ..., -1] + grp['Qload3d'][..., -1]
                fsol[:, :, -1] = grp[self.hdf5table3d][clock, ..., -1] + 0.125*grp['Qload3d'][..., -1]
                fsol[:, :, -2] = grp[self.hdf5table3d][clock, ..., -1] + 0.25*grp['Qload3d'][..., -1]
                fsol[:, :, -3] = grp[self.hdf5table3d][clock, ..., -1] + 0.5*grp['Qload3d'][..., -1]
                #fsol[ grp['xxx'][..., None]**2 + grp['yyy'][..., None]**2 < (grp.attrs['Lx'] + grp.attrs['Ly'])**2/4] = grp.attrs['TBC']
                fsol_x[:, :, :] = grp.attrs['TBC']
                fsol_y[:, :, :] = grp.attrs['TBC']
                fsol_new[:, :, :] = grp.attrs['TBC']
                # fsol[0, :, :] = grp.attrs['TBC']
                # fsol[-1, :, :] = grp.attrs['TBC']
                # fsol[:, 0, :] = grp.attrs['TBC']
                # fsol[:, -1, :] = grp.attrs['TBC']
                # fsol_x[1:self.nx+1, 1:self.ny+1, 1:self.nz+1] = grp[self.hdf5table3d][clock, ...]# + grp['Qload3d'][...]
                # fsol_x[0, :, :] = grp.attrs['TBC']
                # fsol_x[-1, :, :] = grp.attrs['TBC']
                # fsol_x[:, 0, :] = grp.attrs['TBC']
                # fsol_x[:, -1, :] = grp.attrs['TBC']
                # fsol_y[1:self.nx+1, 1:self.ny+1, 1:self.nz+1] = grp[self.hdf5table3d][clock, ...]# + grp['Qload3d'][...]
                # fsol_y[0, :, :] = grp.attrs['TBC']
                # fsol_y[-1, :, :] = grp.attrs['TBC']
                # fsol_y[:, 0, :] = grp.attrs['TBC']
                # fsol_y[:, -1, :] = grp.attrs['TBC']
                # fsol_new[1:self.nx+1, 1:self.ny+1, 1:self.nz+1] = grp[self.hdf5table3d][clock, ...]# + grp['Qload3d'][...]
                # fsol_new[0, :, :] = grp.attrs['TBC']
                # fsol_new[-1, :, :] = grp.attrs['TBC']
                # fsol_new[:, 0, :] = grp.attrs['TBC']
                # fsol_new[:, -1, :] = grp.attrs['TBC']
                fdrive[:, :, 3:self.nz+3] = grp['Qload3d'][...] # Option for clock, could be tripling effect of source, check
                #fdrive[1:self.nx+1, 1:self.ny+1, 0] = grp['Qload3d'][..., 0]
                #fdrive[:, :, 2] = grp['Qload3d'][..., 0]
                #fdrive[:, :, 1] = grp['Qload3d'][..., 0]
                #fdrive[:, :, 0] = grp['Qload3d'][..., 0]
                #fdrive[1:self.nx+1, 1:self.ny+1, -1] = grp['Qload3d'][..., -1]
                #fdrive[:, :, -1] = grp['Qload3d'][..., -1]
                #fdrive[:, :, -2] = grp['Qload3d'][..., -1]
                #fdrive[:, :, -3] = grp['Qload3d'][..., -1]
                #fdrive[:, :, 0] = grp['Qload3d'][..., 0]
                #fdrive[:, :, -1] = grp['Qload3d'][..., -1]

                #d_x = np.zeros(self.nx-2) # Might be slightly faster to move here
                for j in range(1, self.ny-1):
                    for k in range(1, self.nz+5):
                        d_x = np.zeros(self.nx-2)
                        #d_x_BC = np.zeros([self.nx-2, 2])
                        for i in range(1, self.nx-1):
                            d_x[i-1] = self.sx*fsol[i-1, j, k] + self.sx*fsol[i+1, j, k] \
                                + 2*self.sy*fsol[i, j-1, k] + 2*self.sy*fsol[i, j+1, k] \
                                + 2*self.sz*fsol[i, j, k-1] + 2*self.sz*fsol[i, j, k+1] \
                                + (1 - 2*self.sx - 4*self.sy - 4*self.sz)*fsol[i, j, k] + fdrive[i, j, k]

                        d_x[0] += 1*self.sx*fsol[0, j, k]
                        d_x[-1] += 1*self.sx*fsol[-1, j, k]
                        fsol_x[1:self.nx-1, j, k] = self.tdma_solver(ax, bx, cx, d_x)[...]

                for i in range(1, self.nx-1):
                    for k in range(1, self.nz+5):
                        d_y = np.zeros(self.ny-2)
                        #d_y_BC = np.zeros([self.ny-2, 2])
                        for j in range(1, self.ny-1):
                            d_y[j-1] = self.sx*fsol[i-1, j, k] + self.sx*fsol[i+1, j, k] \
                                + self.sx*fsol_x[i-1, j, k] - 2*self.sx*fsol_x[i, j, k] + self.sx*fsol_x[i+1, j, k] \
                                + self.sy*fsol[i, j-1, k] + self.sy*fsol[i, j+1, k] \
                                + 2*self.sz*fsol[i, j, k-1] + 2*self.sz*fsol[i, j, k+1] \
                                + (1 - 2*self.sx - 2*self.sy - 4*self.sz)*fsol[i, j, k] + fdrive[i, j, k]

                        d_y[0] += 1*self.sy*fsol[i, 0, k]
                        d_y[-1] += 1*self.sy*fsol[i, -1, k]
                        fsol_y[i, 1:self.ny-1, k] = self.tdma_solver(ay, by, cy, d_y)[...]

                for i in range(1, self.nx-1):
                    for j in range(1, self.ny-1):
                        d_z = np.zeros(self.nz+4)
                        for k in range(1, self.nz+5):#if (k-1 != 0) and (k != self.nz-1):
                            d_z[k-1] = self.sx*fsol[i-1, j, k] + self.sx*fsol[i+1, j, k] \
                                + self.sx*fsol_x[i-1, j, k] - 2*self.sx*fsol_x[i, j, k] + self.sx*fsol_x[i+1, j, k] \
                                + self.sy*fsol[i, j-1, k] + self.sy*fsol[i, j+1, k] \
                                + self.sy*fsol_y[i, j-1, k] - 2*self.sy*fsol_y[i, j, k] + self.sy*fsol_y[i, j+1, k] \
                                + self.sz*fsol[i, j, k-1] + self.sz*fsol[i, j, k+1] \
                                + (1 - 2*self.sx - 2*self.sy - 2*self.sz)*fsol[i, j, k] + fdrive[i, j, k]

                        d_z[0] += 1*self.sz*fsol[i, j, 0]
                        d_z[-1] += 1*self.sz*fsol[i, j, -1]
                        #d_z[0] += 1*self.sz*fsol[i, j, 0] + fdrive[i, j, 0]
                        #d_z[-1] += 1*self.sz*fsol[i, j, -1] + fdrive[i, j, -1]
                        #fsol_new[i, j, 1:self.nz-1] = self.tdma_solver(az, bz, cz, d_z)[...]
                        fsol_new[i, j, 1:self.nz+5] = self.tdma_solver(az, bz, cz, d_z)[...]

                clock+=1

                # TODO: Time it and test if an additional with h5py statement must come here
                # TODO: Verify proper assignment of values i.e. explicit ellipses to copy values and not pointer!
                #grp[self.hdf5table3d][..., clock] = fsol_new[...] + grp['Qload3d'][..., clock-1]
                #grp[self.hdf5table3d][clock, ...] = fsol_new[...] + grp['Qload3d'][...]
                #grp[self.hdf5table3d][clock, ...] = fsol_new[...]
                grp[self.hdf5table3d][clock, ...] = fsol_new[:, :, 3:self.nz+3]
                #print("Convergence Test: {0}".format(np.allclose(fsol_new, fsol, rtol, atol)))
                #if (np.allclose(fsol_new, fsol, rtol, atol)):
                    #convergence = True
                if (convergence) or (clock == self.nt-1):
                    grp.attrs.create('time', clock)

    #@time_it
    #@jit(f8[:] (f8[:],f8[:],f8[:],f8[:] ))
    def tdma_solver(self, a, b, c, d):
        """TDMA solver, a b c d can be NumPy array type or Python list type.

        References:
            * `TheoChristiaanse Numba Accelerated Gist <https://gist.github.com/TheoChristiaanse/d168b7e57dd30342a81aa1dc4eb3e469>`_
            * `Wikipedia TriDiagonal Matrix Algorithm <http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm>`_
            * `Wikipedia TDMA (Thomas Algorithm) <http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)>`_
        Todo:
            * Compare `SO Solve TDM using solve banded <https://stackoverflow.com/questions/23120164/optimize-ax-b-solution-for-a-tridiagonal-coefficient-matrix>`_

        """

        #print("Entering tdma solver: {0}".format(a, b, c, d))
        #print("entering number of equations {0}".format(len(d)))
        nf = len(d) # number of equations
        ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays
        for it in range(1, nf):
            mc = ac[it-1]/bc[it-1]
            bc[it] = bc[it] - mc*cc[it-1]
            dc[it] = dc[it] - mc*dc[it-1]

        xc = bc
        xc[-1] = dc[-1]/bc[-1]

        for il in range(nf-2, -1, -1):
            xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

        return xc

    @time_it
    def explicit_method(self, rtol=1e-01, atol=1e-03, neumann_BC=0.1):
        """Explicit forward-time, central step finite-difference method.

        Implements and writes solution to HDF5 file. Try to minimise hdf5 access reads and writes.
        Treats three-dimensional solution `block` per time-slice.

        Args:
            rtol (float): Relative tolerance for convergence criterion test (Numpy default is 1e-05).
            atol (float): Absolute tolerance for convergence criterion test (Numpy default is 1e-08).
            neumann_BC (float): Neumann Boundary Condition for constant flux set through fictitious BC's.

        Attributes:
            clock (:obj:`~numpy.ndarray`): Temporal evolution for the solution.

        Todo:
            * Test performance of block hdf5 IO vs deep inner for loop hdf5 IO. Need parallel hdf5 accesses.
        """
        clock = 0
        #fsol = np.dstack([self.finit]*self.nz)
        #Fdri = np.dstack([self.Fdrive]*self.nz) * np.exp(-np.abs(self.alpha)*self.zz)
        fsol = np.zeros([self.nx, self.ny, self.nz])
        temp = np.zeros([self.nx, self.ny, self.nz])

        #temp[:] = self.finit3d[:]
        fsol[:] = self.finit3d[:]
        temp[0, :, :] = self.TBC
        temp[self.nx-1, :, :] = self.TBC
        temp[:, 0, :] = self.TBC
        temp[:, self.ny-1, :] = self.TBC
        convergence = False
        while (clock < self.nt-1) and (not convergence):
            for i in range(1, self.nx-2):
                for j in range(1, self.ny-2):
                    for k in range(self.nz):
                        if (k == 0):
                            # Neumann BC for crystal front-face (-sign)
                            temp[i, j, k] = (fsol[i, j, k]
                                         + self.sx*(fsol[i-1, j, k] -2*fsol[i, j, k] + fsol[i+1, j, k]) \
                                         + self.sy*(fsol[i, j-1, k] -2*fsol[i, j, k] + fsol[i, j+1, k]) \
                                         + self.sz*(-2*self.dz*neumann_BC -2*fsol[i, j, k] \
                                         + 2*fsol[i, j, k+1]) + self.FDrive3d[i, j, k])
                        elif (k == self.nz-1):
                            # Neumann BC for crystal end-face (+sign)
                            temp[i, j, k] = (fsol[i, j, k] \
                                         + self.sx*(fsol[i-1, j, k] -2*fsol[i, j, k] + fsol[i+1, j, k]) \
                                         + self.sy*(fsol[i, j-1, k] -2*fsol[i, j, k] + fsol[i, j+1, k]) \
                                         + self.sz*(2*fsol[i, j, k-1] -2*fsol[i, j, k] \
                                         + 2*self.dz*neumann_BC) + self.FDrive3d[i, j, k])
                        else:
                            temp[i, j, k] = (fsol[i, j, k] \
                                         + self.sx*(fsol[i-1, j, k] -2*fsol[i, j, k] + fsol[i+1, j, k]) \
                                         + self.sy*(fsol[i, j-1, k] -2*fsol[i, j, k] + fsol[i, j+1, k]) \
                                         + self.sz*(fsol[i, j, k-1] -2*fsol[i, j, k] + fsol[i, j, k+1]) \
                                         + self.FDrive3d[i, j, k])
            clock+=1
            if (np.allclose(temp, fsol, rtol, atol)):
                convergence = True

            with h5py.File(self.hdf5filename, 'a') as hf:
                grp = hf.require_group(self.hdf5grp)
                grp[self.hdf5table3d][..., clock] = temp[...]
                if (convergence) or (clock == self.nt-1):
                    grp.attrs.create('time', clock)
            fsol = []
            #fsol[...] = temp[...]
            fsol = temp[...]

#  LocalWords:  hdf fsol
