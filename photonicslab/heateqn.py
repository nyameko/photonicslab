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
#import skimage
#import imageio

from photonicslab.config import *
from photonicslab.isotropy import GRINLens

class HeatEqn2D(GRINLens):
    """Two-dimensional heat equation initialisation object.

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
        * Test meshgrid creation with sparse matrices.
        * Make sure to catch things like nd:ylf etc, don't want to pass on nd alone in exceptions.
        * Consider using kwargs for methods with more than6 parameters, eg see numpy.meshgrid docs.
        * Complete add_supergaussian() and replace add_gaussian and add_flattop with calls to add_supergaussian in their respective limits.

    References:
        * `Northrop Gruman YAG Properties <http://www.northropgrumman.com/BusinessVentures/SYNOPTICS/Products/LaserCrystals/Pages/NdYAG.aspx>`_
        * `Scientific Materials Nd:YAG Properties <http://www.scientificmaterials.com/downloads/Nd_YAG.pdf>`_
        * `Scientific Materials Yb:YAG Properties <http://www.scientificmaterials.com/downloads/Yb_YAG.pdf>`_
        * `Scientific Materials YVO4 Properties <http://www.castech.com/manage/upfile/fileload/20170824171530.pdf>`_
        * `H5PY Github Issue #289 <https://github.com/h5py/h5py/issues/289>`_
        * `H5PY Github Issue #624 <https://github.com/h5py/h5py/issues/624>`_
    """
    def __init__(self, nx=301, ny=301, nt=500, Lx=0.4, Ly=0.4, Lz=2.0, Tbc=293.15, Qmax=1000,
                 gain_medium ='yb_yag', hdf5filename='hdf5db/storage.hdf5', hdf5grp='nb1',
                 dt_stability_factor=.985):
        #need to add super here.
        self.nx = int(np.abs(nx))
        self.ny = int(np.abs(ny))
        self.nt = int(np.abs(nt))
        self.Lx = float(np.abs(Lx))
        self.Ly = float(np.abs(Ly))
        self.Lz = float(np.abs(Lz))
        self.dx = self.Lx/(self.nx-1)
        self.dy = self.Ly/(self.ny-1)
        self.Tbc = float(Tbc)
        self.Qmax = float(Qmax)

        self._vmin = 0
        self._vmax = self.Qmax
        self.num_q = 0

        try:
            if gain_medium.lower() in ['nd_ylf', 'nd:ylf', 'nd-ylf', 'ndylf']:
                self.rho = 3.99
                self.kx, self.ky = 0.063, 0.063
                self.cp = 0.79
                self.dndT = 7.86e-6
                self.nbc = 1.4538
                self.gain_medium = str('Nd:YLF')
            elif gain_medium.lower() in ['nd_yag', 'nd:yag', 'nd-yag', 'ndyag']:
                self.rho = 4.55
                self.kx, self.ky = 0.112, 0.112
                self.cp = 0.59
                self.dndT = 7.8e-6
                self.nbc = 1.8197
                self.tau = 230e-3
                #self.tau77k =
                # Need to check absorption and emission numbers / avoiding errors
                self.wl_p = 0.808e-4
                self.wl_l = 1.064e-4
                self.gain_medium = str('Nd:YAG')
            elif gain_medium.lower() in ['yb_yag', 'yb:yag', 'yb-yag', 'ybyag']:
                self.rho = 4.56
                self.kx, self.ky = 0.112, 0.112
                self.cp = 0.59
                self.dndT = 7.8e-6
                self.nbc = 1.8197
                self.tau = 967e-6
                self.tau77k = 738e-6
                self.tau300k = 1660e-6
                #self.sigma =
                self.wl_p = 0.941e-4
                self.wl_l = 1.064e-4
                self.gain_medium = str('Yb:YAG')
            elif gain_medium.lower() in ['tm_yvo4', 'tm:yvo4', 'tm-yvo4', 'tmyvo4']:
                self.rho = 4.22
                self.kx, self.ky = 0.0510, 0.0525
                self.cp = 0.56
                self.dndT = 8.5e-6
                self.nbc = 1.95
                self.tau = 967e-6
                self.tau77k = 738e-6
                self.tau300k = 1660e-6
                #self.sigma =
                self.wl_p = 0.941e-4
                self.wl_l = 1.064e-4
                self.gain_medium = str('Tm:YVO4')
            else:
                raise ValueError("Unknown gain medium.")
        except ValueError as unknown_gain:
            print("ValuerError: {0}".format(unknown_gain))
            raise ValueError from unknown_gain

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


        self.dt=0.01

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

    def add_supergaussian(self, center_x=0.0, center_y=0.0, q=None, sigma_x=250, sigma_y=250, p_x=1, p_y=1):
        pass

    def add_supergaussian_ring(self):
        pass

    def add_supergaussian_block(self):
        pass

    def add_source_from_image(self, savepath='images/Q_image.png', center_x=None, center_y=None,
                              q=None, nt_start=0, nt_end=None, w_rgb=[0.8554, 0.0525, 0.0921]):
        """Adds a heat source to the current instance of the heateqn2D object from an image file.

        Reads in an arbitrary image file, converts this to greyscale with user
        adjustable rgb weightings. Background subtraction is done automatically
        and the weighted center of mass is used to center the image, unless
        center_x and center_y values are provided in which case the source
        image in centered on those user values.

        Args:
            savepath (str): Image filename savepath.
            center_x (float): X co-ordinate of the center of mass of the profile of the heat source [cm].
            center_y (float): Y co-ordinate of the center of mass of the profile of the heat source [cm].
            q (float): Amplitude of Gaussian heat source. Uses class constructor default :obj:`~HeatEqn2D.Qinit`.
            nt_start (float): Starting time-step from which to add source.
            nt_end (float): Ending time-step after which source no longer contributes to solver.
            w_rgb (~numpy.ndarray): Red, Green and Blue weighting factors for grayscale conversion.

        Attributes:
            imageRGB (:obj:`~numpy.ndarray`): Raw RGB image resized to [Nx, Ny].
            image (:obj:`~numpy.ndarray`): Greyscale version of imageRGB converted using w_rgb weightings.
            threshold_value (:obj:`float`): Return threshold value based on Otsu's method,\
                                            for background subtraction.
            labeled_foreground (:obj:`~numpy.ndarray`): Boolean array, for integer values above background\
                                                        threshold.
            properties (:obj:`list`): Measure properties of labeled image regions.
            weighted_center_of_mass: (:obj:`list`): Centroid coordinate tuple [nx, ny] weighted with\
                                                    intensity image.
            shift_x (:obj:`int`): Index value used to adjust x center value of image.
            shift_y (:obj:`int`): Index value used to adjust y center value of image.

        Raises:
            ValueError: If too much power is provided.
            ValueError: If the coordinates of the position of the heat source are out of bounds
                        of crystal face.

        ..seealso::
            * :func:`~skimage.color.rgb2gray`
            * `Wikipedia Otsu's Method <https://en.wikipedia.org/wiki/Otsu%27s_method>`_

        References:
            * `SO Using skimage.measure.regionprops <https://stackoverflow.com/questions/48888239/finding-the-center-of-mass-in-an-image>`_

        Todo:
            * Check for rollover/wraparound/reflect, and make that section zero.
            * Exception handling and testing.

        """

        from skimage.io import imread
        from skimage import filters
        from skimage.color import rgb2gray
        from skimage.measure import regionprops
        from skimage.transform import resize, rotate
        q = self._verify_q(q)
        nt_start, nt_end = self._verify_nt_range(nt_start, nt_end)
        imageRGB = resize(rotate((imread(str(savepath))), -90), (self.nx, self.ny), mode='constant')
        image = np.sum(w_rgb[:]*imageRGB[...,:], axis=2)
        threshold_value = filters.threshold_otsu(image)
        labeled_foreground = (image > threshold_value).astype(int)
        properties = regionprops(labeled_foreground, image)
        center_of_mass = properties[0].centroid
        weighted_center_of_mass = properties[0].weighted_centroid

        if center_x is not None:
            try:
                if not (-self.Lx/2 <= center_x <= self.Lx/2):
                    raise ValueError("Position of heat source out of bounds of crystal face.")
            except ValueError as outofbounds:
                warnings.warn("OutOfBoundsWarning: {0}".format(outofbounds), RuntimeWarning)
            finally:
                shift_x =  np.round((np.abs(self.xx[:,0] - center_x).argmin()) - weighted_center_of_mass[0])
        else:
            shift_x = np.round(self.nx/2 - weighted_center_of_mass[0])
        if center_y is not None:
            try:
                if not (-self.Ly/2 <= center_y <= self.Ly/2):
                    raise ValueError("Position of heat source out of bounds of crystal face.")
            except ValueError as outofbounds:
                warnings.warn("OutOfBoundsWarning: {0}".format(outofbounds), RuntimeWarning)
            finally:
                shift_y =  np.round((np.abs(self.yy[0,:] - center_y).argmin()) - weighted_center_of_mass[1])
        else:
            shift_y = np.round(self.ny/2 - weighted_center_of_mass[1])

        self.Qinit[nt_start:nt_end,]+= q*(np.roll(image, [int(shift_x), int(shift_y)], axis=[0, 1])[None, ...])
        self._set_plot_limits()

    def add_source_from_hdf5(self, hdf5grp='nb1', hdf5table='seed_mod_convol',
                             q=None, nt_start=0, nt_end=None):
        """Adds a heat source to the current instance of the heateqn2D object from an hdf5 file.

        Args:
            hdf5grp (str): Group name of the HDF5 group object.
            hdf5table (str): Dataset name of the HDF5 dataset entry object.
            q (float): Amplitude of Gaussian heat source. Uses class constructor default :obj:`~HeatEqn2D.Qinit`.
            nt_start (float): Starting time-step from which to add source.
            nt_end (float): Ending time-step after which source no longer contributes to solver.

        Raises:
            ValueError: If too much power is provided.
            KeyError: If the coordinates of the position of the heat source are out of bounds
                      of crystal face.

        """
        q = self._verify_q(q)
        nt_start, nt_end = self._verify_nt_range(nt_start, nt_end)
        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            grp = hf.require_group(hdf5grp)
            try:
                if (str(hdf5table)) not in grp:
                    raise KeyError("{0} not found in database: {1}".format(str(hdf5table), str(grp)))
            except KeyError as keynotfound:
                warnings.warn("KeyNotFoundWarning: {0}".format(keynotfound), RuntimeWarning)
            finally:
                self.Qinit[nt_start:nt_end,] += q*grp[str(hdf5table)][nt_start:nt_end][:]
                self._set_plot_limits()

    def add_gaussian(self, center_x=0.0, center_y=0.0, q=None, nt_start=0, nt_end=None, b_x=250, b_y=250):
        """Adds a Gaussian heat source to the current instance of the heateqn2D object.

        Args:
            center_x (float): X co-ordinate of the center of the Gaussian heat source [cm].
            center_y (float): Y co-ordinate of the center of the Gaussian heat source [cm].
            q (float): Amplitude of Gaussian heat source. Uses class constructor default :obj:`~HeatEqn2D.Qinit`.
            nt_start (float): Starting time-step from which to add source.
            nt_end (float): Ending time-step after which source no longer contributes to solver.
            b_x (float): Gaussian shape in x-direction.
            b_y (float): Gaussian shape in the y-direction.

        Raises:
            ValueError: If too much power is provided.
            ValueError: If the coordinates of the position of the heat source are out of bounds
                        of crystal face.

        Example:
            Adding two adjacent and non-equal sources.

            >>> import photonicslab
            >>> myGaussianPair = photonicslab.heateqn.HeatEqn2D()
            >>> myGaussianPair.add_gaussian_source()
            >>> myGaussianPair.add_gaussian_source(q=800)

            .. figure:: ../notebooks/images/nb1_sourceq.png
               :align:  center
               :width:  60%

               Pair of adjacent Gaussian sources.

        """

        warnings.warn("DeprecationWarning: Superseded by add_supergaussian() method.",
                      DeprecationWarning)

        q = self._verify_q(q)
        nt_start, nt_end = self._verify_nt_range(nt_start, nt_end)
        try:
            if not (-self.Lx/2 <= center_x <= self.Lx/2 and -self.Ly/2 <= center_y <= self.Ly/2):
                raise ValueError("Position of heat source out of bounds of crystal face.")
        except ValueError as outofbounds:
            warnings.warn("OutOfBoundsWarning: {0}".format(outofbounds), RuntimeWarning)
        finally:
            self.Qinit[nt_start:nt_end,] += q*np.exp(-b_x*(self.xx-center_x)**2 -b_y*(self.yy-center_y)**2)
            self._set_plot_limits()

    def add_flattop(self, center_x=0.0, center_y=0.0, center_r = 0.05, q=None, nt_start=0, nt_end=None):
        """Adds a step or top-hat like circular heat source to the current instance of the heateqn2D object.

        Args:
            center_x (float): X co-ordinate of the center of the heat source [cm].
            center_y (float): Y co-ordinate of the center of the heat source [cm].
            center_r (float): Radius from the center of the heat source [cm].
            q (float): Amplitude of heat source.
            nt_start (float): Starting time-step from which to add source.
            nt_end (float): Ending time-step after which source no longer contributes to solver.

        Raises:
            ValueError: If too much power is provided.
            ValueError: If the coordinates of the position of the heat source are out of bounds
                        of crystal face.

        Attributes:
            pts (:obj:`~numpy.ndarray`): Cross sectional coordinates for which to add heat source.

        References:
            * `SO Circle Adv Indexing <https://stackoverflow.com/questions/15432294/changing-the-values-in-a-numpy-array-depending-on-the-area-of-a-circle>`_

        """

        warnings.warn("PendingDeprecationWarning: Superseded by add_supergaussian() method.",
                      PendingDeprecationWarning)

        q = self._verify_q(q)
        nt_start, nt_end = self._verify_nt_range(nt_start, nt_end)
        try:
            if center_r <= 0:
                center_r = np.abs(center_r)
                raise ValueError("Source radius provided is negative or zero.")
            if not (-self.Lx/2 + center_r <= center_x <= self.Lx/2 - center_r and
                    -self.Ly/2 + center_r <= center_y <= self.Ly/2 - center_r):
                raise ValueError("Position of flattop heat source outside of bounds of crystal face.")
        except ValueError as outofbounds:
            warnings.warn("OutOfBoundsWarning: {0}".format(outofbounds), RuntimeWarning)
        finally:
            pts = (self.xx-center_x)**2 + (self.yy-center_y)**2 < center_r**2
            self.Qinit[nt_start:nt_end, pts] += q
            self._set_plot_limits()

    def add_flattop_ring(self, center_x=0.0, center_y=0.0, inner_r=0.05, outer_r=0.1,
                         q=None, nt_start=0, nt_end=None):
        """Adds a step or top-hat like ring heat source to the current instance of the heateqn2D object.

        Args:
            center_x (float): X co-ordinate of the center of the heat source [cm].
            center_y (float): Y co-ordinate of the center of the heat source [cm].
            inner_r (float): Radius from the center to inner section of the heat source [cm].
            outer_r (float): Radius from the center to outer section of the heat source [cm].
            q (float): Amplitude of heat source.
            nt_start (float): Starting time-step from which to add source.
            nt_end (float): Ending time-step after which source no longer contributes to solver.

        Raises:
            ValueError: If too much power is provided.
            ValueError: If the coordinates of the position of the heat source are out of bounds
                        of crystal face.

        Attributes:
            inner_pts (:obj:`~numpy.ndarray`): Cross sectional coordinates for which to remove heat source.
            outer_pts (:obj:`~numpy.ndarray`): Cross sectional coordinates for which to add heat source.

        """
        q = self._verify_q(q)
        nt_start, nt_end = self._verify_nt_range(nt_start, nt_end)
        try:

            if (inner_r < 0 or outer_r <= 0):
                inner_r = np.abs(inner_r)
                outer_r = np.abs(outer_r)
                warnings.warn("Corrected one of the ring source radii provided are either negative or zero.", RuntimeWarning)
            if np.abs(inner_r) >= np.abs(outer_r):
                temp_r = outer_r
                outer_r = inner_r
                inner_r = temp_r
                warnings.warn("Inner heat source ring radius >= outer radius", RuntimeWarning)
            if not ((-self.Lx/2 + outer_r < center_x < self.Lx/2 - outer_r) and
                    (-self.Ly/2 + outer_r < center_y < self.Ly/2 - outer_r)):
                raise ValueError("Position of heat source outside of bounds of crystal face.")
        except ValueError as outofbounds:
            warnings.warn("OutOfBoundsWarning: {0}".format(outofbounds), RuntimeWarning)
        finally:
            outer_pts = (self.xx-center_x)**2 + (self.yy-center_y)**2 <= outer_r**2
            inner_pts = (self.xx-center_x)**2 + (self.yy-center_y)**2 < inner_r**2
            self.Qinit[nt_start:nt_end, outer_pts] += q
            self.Qinit[nt_start:nt_end, inner_pts] -= q
            self._set_plot_limits()

    def add_flattop_block(self, lower_left_x=-0.05, lower_left_y=-0.05,
                          upper_right_x=0.05, upper_right_y=0.05, q=None,
                          nt_start=0, nt_end=None):
        """Adds a step or top-hat like ring heat source to the current instance of the heateqn2D object.

        Args:
            lower_left_x (float): X co-ordinate of the lower left corner of the heat bar [cm].
            lower_left_y (float): Y co-ordinate of the lower left corner of the heat bar [cm].
            upper_right_x (float): X co-ordinate of the upper right corner of the heat bar [cm].
            upper_right_y (float): Y co-ordinate of the upper right corner of the heat bar [cm].
            q (float): Amplitude of heat source.
            nt_start (float): Starting time-step from which to add source.
            nt_end (float): Ending time-step after which source no longer contributes to solver.

        Raises:
            ValueError: If too much power is provided.
            ValueError: If the coordinates of the position of the heat source are out of bounds
                        of crystal face.
        """
        q = self._verify_q(q)
        nt_start, nt_end = self._verify_nt_range(nt_start, nt_end)
        try:
            if (lower_left_x > upper_right_x):
                temp_x = lower_left_x
                lower_left_x = upper_right_x
                upper_right_x = temp_x
                raise ValueError("Invalid heat bar x-coordinates.")
            if (lower_left_y > upper_right_y):
                temp_y = lower_left_y
                lower_left_y = upper_right_y
                upper_right_y = temp_y
                raise ValueError("Invalid heat bar y-coordinates.")
            if not ((-self.Lx/2 < lower_left_x < upper_right_x < self.Lx/2) and
                    (-self.Ly/2 < lower_left_y < upper_right_y < self.Ly/2)):
                raise ValueError("Position of heat source outside of bounds of crystal face.")
        except ValueError as outofbounds:
            warnings.warn("OutOfBoundsWarning: {0}".format(outofbounds), RuntimeWarning)
        finally:
            self.Qinit[nt_start:nt_end, (self.xx >= lower_left_x) & (self.yy >= lower_left_y) &
                       (self.xx <= upper_right_x) & (self.yy <= upper_right_y)] += q
            self._set_plot_limits()

    def _verify_q(self, q):
        try:
            if q is None:
                q = self.Qmax
            elif float(q) > self.Qmax:
                raise ValueError("Value of heat source too high.")
        except ValueError:
            q = self.Qmax
            warnings.warn("RuntimeWarning: Heat source to be added to Qinit reduced to Qmax.",
                          RuntimeWarning)
        return q

    def _verify_nt_range(self, nt_start, nt_end):
        try:
            if nt_end is None:
                nt_end = self.nt
            elif int(nt_end) > self.nt:
                raise ValueError("Value of add heat source stop-time too high.")
        except ValueError:
            nt_end = self.nt
            warnings.warn("RuntimeWarning: End-time for heat source to be added to Qinit reduced to nt.",
                          RuntimeWarning)
        try:
            if int(nt_start) > self.nt:
                raise ValueError("Value of add heat source start-time too high.")
        except ValueError:
            nt_start = 0
            warnings.warn("RuntimeWarning: Start-time for heat source to be added to Qinit reduced to 0.",
                          RuntimeWarning)

        try:
            if int(nt_start) > int(nt_end):
                raise ValueError("Values of add source time-intervals inverted.")
        except ValueError:
            nt_temp = nt_start
            nt_start = nt_end
            nt_end = nt_temp
            warnings.warn("RuntimeWarning: Start and end times for heat source addition are inverted",
                          RuntimeWarning)

        return int(nt_start), int(nt_end)

    def remove_sources(self):
        """Removes all of the initial heat sources, and resets plot limits."""
        self.Qinit = np.zeros([self.nt, self.nx, self.ny])
        self._vmin = 0
        self._vmax = self.Qmax
        self.num_q = 0

    def _set_plot_limits(self):
        """Update the values of _vmin and _vmax for profile plotting.

        Attributes:
            _vmin (:obj:`float`): Private attribute used by plotting function.
            _vmax (:obj:`float`): Private attribute used by plotting function.

        Todo:
            * [FIXME] Bug with warning.
        """
        if np.max(self.Qinit) > self._vmax and np.max(self.Qinit) > self.Qmax:
            warnings.warn("RuntimeWarning: Current peak Qinit exceeds peak visualisation point. Start with higher sources first.",
                          RuntimeWarning)
        if self.num_q == 0:
            self._vmin, self._vmax = (np.max([self._vmin, np.min(self.Qinit)]),
                                      np.min([self._vmax, np.max(self.Qinit)]))
        else:
            self._vmin, self._vmax = (np.max([self._vmin, np.min(self.Qinit)]),
                                      np.max([self._vmax, np.max(self.Qinit)]))
        if self._vmin >= self._vmax:
            warnings.warn("RuntimeWarning: Zero-range plot limit - boundaries have the same value.",
                          RuntimeWarning)
        self.num_q+=1

    def plot_source_profile(self, savepath='images/Q_init.png', nt_plot=0):
        """Plot the current source profile from heat load object.

        Args:
            savepath (str): Path to save the image.
            nt_plot (int): Time-step to plot (Default it 0).
        Todo:
            * FIXME: Exception handling and warnings for bad nt_plot values
        """

        warnings.warn("PendingDeprecationWarning: All plotting and imaging will be handled by\
        visualisation module.", PendingDeprecationWarning)

        #LABEL_FONTSIZE=6.2
        #AXIS_TICK_FONTSIZE=5.
        plt.close('all')
        plt.clf()
        # Use gnuplot cmap for gain
        # Use jet for temperature / heat
        # plt.pcolor(self.xx, self.yy, self.Qinit[nt_plot,], vmin=self._vmin,
        #            vmax=self._vmax, cmap='gnuplot2')
        plt.pcolor(self.xx, self.yy, self.Qinit[nt_plot,], vmin=self._vmin,
                   vmax=self._vmax, cmap='gnuplot2')
        #plt.axes().set_aspect('equal')
        # plt.rc('xtick', labelsize=AXIS_TICK_FONTSIZE)
        # plt.rc('ytick', labelsize=AXIS_TICK_FONTSIZE)
        # plt.xlabel("X [cm]", fontsize=LABEL_FONTSIZE)
        # plt.ylabel("Y [cm]", fontsize=LABEL_FONTSIZE)
        # plt.title("Initial heat source at pump-side endface", fontsize=LABEL_FONTSIZE)
        plt.rc('xtick')
        plt.rc('ytick')
        plt.xlabel("X [cm]")
        plt.ylabel("Y [cm]")
        plt.title("Initial heat source at pump-side endface")
        plt.colorbar()
        plt.savefig(savepath, dpi=MAX_DPI)

    def get_explicit_fd_params(self):
        """Return parameters and coefficients to the explicit finite difference approximation:

        .. math::

            T_{i,j}^{n+1} = T_{i,j}^{n} + sx\\left(T_{i-1,j}^{n} - 2T_{i,j}^{n} + T_{i+1,j}^{n}\\right)
            + sy\\left(T _{i,j-1}^{n} - 2T_{i,j}^{n} + T_{i,j+1}^{n}\\right) + ss\\times Q_\\text{init}

        Where :math`T^{n+1mm}` is the solution to the differential equation
        :math:`Q_\\text{load} = ss\\times Q_\\text{init}` is the driving source term i.e. heat or thermal load, and
        :math:`s_{x_{i}} = \\frac{\\Delta t\\kappa_{x_{i}}}{(\\Delta x_{i})^2} \\text{ for } i \\in [1,2,3]`
        Attributes:
            sx (:obj:`float`): Integration coefficient for second-order finite difference along x.
            sy (:obj:`float`): Integration coefficient for second-order finite difference along y.
            Qload (:obj:`~numpy.ndarray`): Normalised 2D (unit-correct) system driving / damping  force.
            clock (:obj:`~numpy.ndarray`): Time evolution.

        TODO:
            * Replace maxshape numbers with class constants.
        """

        sx = self.dt*self.kx/(self.rho*self.cp*(self.dx)**2)
        sy = self.dt*self.ky/(self.rho*self.cp*(self.dy)**2)
        ss = self.dt/(self.rho*self.cp)
        clock = np.linspace(0, self.nt*self.dt, self.nt)
        Qload = ss*self.Qinit
        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            grp = hf.require_group(self.hdf5grp)
            grp.attrs.create('sx', sx)
            grp.attrs.create('sy', sy)
            if 'clock' not in grp:
                grp.create_dataset('clock', shape=(self.nt,), dtype=np.dtype(float).type, data=clock,
                                   maxshape=(MAX_NT,))
            else:
                grp['clock'].resize((self.nt,))
                grp['clock'][...] = clock
            if 'Qload' not in grp:
                grp.create_dataset('Qload', shape=(self.nt, self.nx, self.ny), dtype=np.dtype(float).type,
                                   data=Qload, maxshape=(MAX_NT, MAX_NX, MAX_NY))
            else:
                grp['Qload'].resize((self.nt, self.nx, self.ny))
                grp['Qload'][...] = Qload
        #return [self.xx, self.yy, clock, Tinit, Qload, sx, sy]

    def get_cn_fd_params(self):
        """Return parameters and coefficients to the implicit finite difference approximation,
        implemented within a Crank-Nicolson Scheme.

        .. math::

            T_{i,j}^{n+1} - sx\\left(T_{i-1,j}^{n+1} - 2T_{i,j}^{n+1} + T_{i+1,j}^{n+1}\\right)
            - sy\\left(T _{i,j-1}^{n+1} - 2T_{i,j}^{n+1} + T_{i,j+1}^{n+1}\\right) = T_{i,j}^{n} + sx\\left(T_{i-1,j}^{n} - 2T_{i,j}^{n} + T_{i+1,j}^{n}\\right)
            + sy\\left(T _{i,j-1}^{n} - 2T_{i,j}^{n} + T_{i,j+1}^{n}\\right) + ss\\times Q^{n}

        Where T is the solution to the differential equation
        :math:`Q_\\text{load} = ss\\times Q^{n}` is the driving \ source term or heat\thermal load,
        :math:`s_{x_{i}} = \\frac{\\Delta t}{2}\\frac{\\kappa_{x_{i}}}{(\\Delta x_{i})^2} \\text{ for } i \\in [1,2,3]`, and
        :math:`s_s = \\frac{1}{\\rho c_{p}}`, forming linear algebra system
        :math:`Ax = b`.

        Attributes:
            sx (:obj:`float`): Integration coefficient for second-order finite difference along x.
            sy (:obj:`float`): Integration coefficient for second-order finite difference along y.
            Qload (:obj:`~numpy.ndarray`): Normalised 2D (unit-correct) system driving / damping  force.
            clock (:obj:`~numpy.ndarray`): Time evolution.

        """
        self.get_explicit_fd_params()
        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            grp = hf.require_group(self.hdf5grp)
            sx = grp.attrs['sx']
            sy = grp.attrs['sy']
            grp.attrs.create('sx', 0.5*sx)
            grp.attrs.create('sy', 0.5*sy)

class HeatEqn3D(HeatEqn2D):
    """Three-dimensional heat equation initialisation object.

    Args:
        nz (int): Number of steps along z-direction of axial / cross-sectional profile [cm].
        alpha (float): Exponential decay factor representing heat absorption in gain medium.

    Attributes:
        nz (:obj:`int`): Number of steps along z-direction of axial / cross-sectional profile [cm].
        alphamax (:obj:`float`): Exponential decay factor representing heat absorption in gain medium [units?].
        dz (:obj:`float`): Spacing along z-direction [cm].
        kz (:obj:`float`): Thermal conductivity coefficient along z-direction [W/(cm K)].
        Qinit3d (:obj:`~numpy.ndarray`): Initial heat source distribution for 3D solid.

    Note:
        * Be very careful here with the ordering of 3D meshgrid, :math:` [i,j,k] \\rightarrow [z,y,x]`.
        * The Crank-Nicolson scheme should not require a stability factor.

    Todo:
        * Consider using ``**kwargs`` to reduce the fugly number of params in constructor. (Purely aesthetic / Pythonic)
        * Consider using sparse keyword in meshgrid creation
        * TODO: Go back to old style 3d source creation with 2d, and z decay
        * TODO: Overload and inherit all methods for shape construction in 3d with decay.
        * TODO: Setup self.num_q variables, and make sure each beam is parameterised.
    """
    def __init__(self, nx=100, ny=100, nz=200, nt=50, Lx=0.4, Ly=0.4, Lz=2., Tbc=293.15, Qmax=1000,
                 gain_medium='ybyag', hdf5filename='hdf5db/storage.hdf5', hdf5grp='nb1',
                 dt_stability_factor=0.99, alphamax=10, cavity_mode_waist=191.5e-4):
        super().__init__(nx, ny, nt, Lx, Ly, Lz, Tbc, Qmax, gain_medium, hdf5filename, hdf5grp, dt_stability_factor)
        self.nz = nz
        self.dz = self.Lz/(self.nz-1)
        self.alphamax = alphamax
        self.waist = cavity_mode_waist

        if self.gain_medium.lower() == 'nd:ylf':
            self.kz = 0.063
        elif self.gain_medium.lower() == 'nd:yag':
            self.kz = 0.112
        elif self.gain_medium.lower() == 'yb:yag':
            self.kz = 0.112
        elif self.gain_medium.lower() == 'tm:yvo4':
            self.kz = 0.95

        # Variation in z-step "should" not important.
        # And with implicit methods, stability is all but gaurenteed
        #self.dt = dt_stability_factor*(.5*self.rho*self.cp*np.min([self.dx**2, self.dy**2, self.dz**2])
        #                                   /np.max([self.kx, self.ky, self.kz]))

        try:
            if (self.nz <= 0):
                raise ValueError("Invalid number of steps.")
            self.xxx, self.yyy, self.zzz = np.meshgrid(np.linspace(-1*self.Lx/2, self.Lx/2, self.nx),
                                                       np.linspace(-1*self.Ly/2, self.Ly/2, self.ny),
                                                       np.linspace(0, self.Lz, self.nz), indexing='ij')
        except ValueError as bad_constr:
            print(bad_constr)

        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            grp = hf.require_group(self.hdf5grp)
            grp.attrs.create('thermal_conductivity', [self.kx, self.ky, self.kz])
            grp.attrs.create('nz', self.nz)
            grp.attrs.create('dz', self.dz)
            grp.attrs.create('waist', self.waist)
            if 'zzz' not in grp:
                grp.create_dataset('zzz', shape=(self.nx, self.ny, self.nz), dtype=np.dtype(float).type,
                                   data=self.zzz, maxshape=(MAX_NX, MAX_NY, MAX_NZ))
            else:
                grp['zzz'].resize((self.nx, self.ny, self.nz))
                grp['zzz'][...] = self.zzz
            if 'yyy' not in grp:
                grp.create_dataset('yyy', shape=(self.nx, self.ny, self.nz), dtype=np.dtype(float).type,
                                   data=self.yyy, maxshape=(MAX_NX, MAX_NY, MAX_NZ))
            else:
                grp['yyy'].resize((self.nx, self.ny, self.nz))
                grp['yyy'][...] = self.yyy
            if 'xxx' not in grp:
                grp.create_dataset('xxx', shape=(self.nx, self.ny, self.nz), dtype=np.dtype(float).type,
                                   data=self.xxx, maxshape=(MAX_NX, MAX_NY, MAX_NZ))
            else:
                grp['xxx'].resize((self.nx, self.ny, self.nz))
                grp['xxx'][...] = self.xxx

        # self.Pinit3d = np.zeros([self.nx, self.ny, self.nz, self.nt, self.num_q])
        # self.Qinit3d = np.zeros([self.nx, self.ny, self.nz, self.nt, self.num_q])
        self.Pinit3d = np.zeros([self.nx, self.ny, self.nz])
        self.Qinit3d = np.zeros([self.nx, self.ny, self.nz])
        self.Ptot_in = 0
        self.Ptot_out = 0

    def add_normalized_pump_gaussian(self, center_x=0.0, center_y=0.0,  wp=None, alpha=12, flatness=1, eta=None, q=None, nt_start=0, nt_end=None):
        """End-pumped normalized pump distribution for gain medium treated as a four-level system.

        Args:
            wp (float): Pump beam waist. Set to beam waist of fundamental cavity transverse mode for mode-matched pumping.
            alpha (float): Crystal (gain medium) absorption coefficient.
            eta (float): Crystal (gain medium) absorption efficiency.

        Todo:
            * Add conditionals / separate private member functions to check validity of assignments, i.e. waist less than Lx, Ly etc
            * Normalise Qinit3d
            * Need to keep a tab of the number of sources and allocate them individually.
            * Each pump source will be a list of lists, and structure will be stored i hdf5 db.
        """
        q = self._verify_q(q)
        nt_start, nt_end = self._verify_nt_range(nt_start, nt_end)

        if eta is not None:
            self.eta = eta
        else:
            self.eta = 1 - np.exp(-2*alpha*self.dz)

        if wp is not None:
            self.wp = wp
        else:
            self.wp = self.waist

        if alpha <= self.alphamax:
            self.alpha = alpha
        else:
            self.alpha = self.alphamax

        # self.Pinit3d[:, :, :, nt_start:nt_end]
        # self.Pinit3d[:, :, :, nt_start:nt_end] += (2*alpha/(np.pi*self.wp**2*self.eta) \
        #                                         * ((np.exp(self.alpha*self.zzz - self.alpha*self.dz) \
        #                                         + np.exp(-self.alpha*self.zzz - self.alpha*self.dz)) \
        #                                         * np.exp(-2*(self.xxx**2 + self.yyy**2)/self.wp**2)))[:, :, :, None]

        # self.Qinit3d[:, :, :, nt_start:nt_end] += q*self.Pinit3d[:, :, :, nt_start:nt_end]
        # self.Pinit3d = (2*alpha/(np.pi*self.wp**2*self.eta) \
        #                 * ((np.exp(self.alpha*self.Lz - self.alpha*self.zzz) \
        #                 + np.exp(-self.alpha*self.Lz - self.alpha*self.zzz)) \
        #                 * np.exp(-2*((self.xxx-center_x)**2 + (self.yyy-center_y)**2)/self.wp**2)))
        self.Pinit3d = (2*alpha/(np.pi*self.wp**2*self.eta) \
                        * ((np.exp(self.alpha*self.Lz - self.alpha*self.zzz) \
                        + np.exp(-self.alpha*self.Lz - self.alpha*self.zzz)) \
                        * np.exp(-2*((self.xxx-center_x)**2 + (self.yyy-center_y)**2)**flatness/self.wp**(2*flatness))))

        # FIXME: Weired bug and warning with regards to what I think may be zero division
        # TODO: Check multiplication by 1e-4
        self.Qinit3d += q*np.abs((self.Pinit3d/np.max(self.Pinit3d))**2)
        #self.Ptot_in = q*np.sum(np.abs((self.Pinit3d[:, :, 0]/np.max(self.Pinit3d))**2))*(1e-2)
        self.Ptot_in += q*np.sum(self.Pinit3d[:, :, 0]/np.max(self.Pinit3d))*(1e-4)
        self.Ptot_out += q*np.sum((self.Pinit3d[:, :, -1]/np.max(self.Pinit3d))*(1e-4))
        #self.Qinit[nt_start:nt_end, :, :] = self.Qinit3d[:, :, 0, nt_start:nt_end]
        self._set_plot_limits()

    def remove_sources(self):
        """Removes all of the initial heat sources, and resets plot limits."""
        super().remove_sources
        self.Pinit3d = np.zeros([self.nx, self.ny, self.nz])
        self.Qinit3d = np.zeros([self.nx, self.ny, self.nz])

    def _set_plot_limits(self):
        """Update the values of _vmin and _vmax for profile plotting.

        Attributes:
            _vmin (:obj:`float`): Private attribute used by plotting function.
            _vmax (:obj:`float`): Private attribute used by plotting function.
        """

        if np.max(self.Qinit3d) > self._vmax and np.max(self.Qinit3d) > self.Qmax:
            warnings.warn("RuntimeWarning: Current peak Qinit exceeds peak visualisation point. Start with higher sources first.",
                          RuntimeWarning)

        self._vmin, self._vmax = (np.max([self._vmin, np.min(self.Qinit3d)])/(self.nx*self.ny),\
                                  #np.max([self._vmax, np.max(self.Qinit3d)])*np.pi/2*(self.wp**2*1e-2)\
                                  np.max([self._vmax, np.max(self.Qinit3d)])/(self.nx*self.ny))
        if self._vmin >= self._vmax:
            warnings.warn("RuntimeWarning: Zero-range plot limit - boundaries have the same value.",
                          RuntimeWarning)
        self.num_q+=1

    def plot_source_profile(self, savepath='images/Q_init3d.png', z_ind=-1, time_step=0):
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

        try:
            if int(time_step) > self.nt:
                raise ValueError("Invalid time index provided.")
        except ValueError as t_out_of_bounds:
            time_step = int(self.nt)
            print(t_out_of_bounds)

        print('Pump side total input power: {0:.2f} W'.format(self.Ptot_in))
        print('Output coupler total power: {0:.2f} W'.format(self.Ptot_out))
        # try:
        #     if (z < 0) or (z > self.Lz):
        #         raise ValueError("Invalid attenuation distance provided.")
        # except ValueError as z_out_of_bounds:
        #     z = self.Lz
        #     print(z_out_of_bounds)

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
        plt.title("Initial normalized intensity profile of pump EM field")

        _fig = plt.figure()
        _ax1 = _fig.add_subplot(2, 2, 1)
        _ax2 = _fig.add_subplot(2, 2, 2)
        _ax3 = _fig.add_subplot(2, 2, 3, projection='3d')
        _ax4 = _fig.add_subplot(2, 2, 4, projection='3d')
        _fig.suptitle("Power per pixel [W] at t = {0:.5f}".format(time_step), fontsize=LABEL_FONTSIZE+1.0)
        #_fig.set_tight_layout(True)
        _fig.subplots_adjust(wspace=0.25, hspace=0.4)

        #######################
        _ax1.set_xlabel('X [cm]', fontsize=LABEL_FONTSIZE)
        _ax1.set_ylabel('Y [cm]', fontsize=LABEL_FONTSIZE)
        _ax1.set_title('Pump-side end face', fontsize=LABEL_FONTSIZE)
        _quad1 = _ax1.pcolormesh(self.xx, self.yy, self.Qinit3d[:, :, 0]/(self.nx*self.ny), vmin=self._vmin,
                                 vmax=self._vmax, cmap='jet', shading='gouraud')

        #######################
        _ax2.set_xlabel('X [cm]', fontsize=LABEL_FONTSIZE)
        _ax2.set_ylabel('Y [cm]', fontsize=LABEL_FONTSIZE)
        _ax2.set_title('Output coupler side end-face'.format(z_ind), fontsize=LABEL_FONTSIZE)
        _quad2 = _ax2.pcolormesh(self.xx, self.yy, self.Qinit3d[:, :, z_ind]/(self.nx*self.ny),
                                           vmin=self._vmin, vmax=self._vmax, cmap='jet',
                                           shading='gouraud')

        #######################
        _ax3.set_xlabel('X [cm]', fontsize=LABEL_FONTSIZE)
        _ax3.set_xlim(-1*self.Lx/2, self.Lx/2)
        _ax3.set_ylabel('Z [cm]', fontsize=LABEL_FONTSIZE)
        _ax3.set_ylim(0, 1.1*self.Lz)
#        _ax3.set_zlabel('Q', fontsize=LABEL_FONTSIZE)
#        _ax3.set_zlim(self._vmin, self._vmax)
        _quad3 = _ax3.plot_surface(self.xxx.T[:, round(self.ny/2), :], \
                                   self.zzz.T[:, round(self.ny/2), :], \
                                   self.Qinit3d.T[:, round(self.ny/2), :]/(self.nx*self.ny), \
                                   alpha=0.3, cmap='jet')
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
                               zdir='z', offset=0, cmap='jet')

        #######################
        _ax4.set_xlabel('Y [cm]', fontsize=LABEL_FONTSIZE)
        _ax4.set_xlim(-0.5*self.Ly, 0.5*self.Ly)
        _ax4.set_ylabel('Z [cm]', fontsize=LABEL_FONTSIZE)
        _ax4.set_ylim(0, 1.1*self.Lz)
#        _ax4.set_title('Q', fontsize=LABEL_FONTSIZE)
#        _ax4.set_zlim(self._vmin, self._vmax)
        _quad4 = _ax4.plot_surface(self.yyy[round(self.nx/2), :, :], \
                                   self.zzz[round(self.nx/2), :, :], \
                                   self.Qinit3d[round(self.nx/2), :, :]/(self.nx*self.ny), \
                                   alpha=0.3, cmap='jet')
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
                               zdir='z', offset=0, cmap='jet')

        #cax = _fig.add_axes([0.85, 0.1, 0.03, 0.8])
        cax = _fig.add_axes([0.95, 0.1, 0.01, 0.75])
        _fig.colorbar(_quad1, cax=cax).set_label('Q')

        plt.savefig(savepath, dpi=MAX_DPI)

    def get_explicit_fd_params(self):
        """Return parameters and coefficients to the explicit 3D finite difference approximation.

        .. math::

            T_{i,j,k}^{n+1} = T_{i,j,k}^{n} + sx\\left(T_{i-1,j,k}^{n} - 2T_{i,j,k}^{n}
            + T_{i+1,j,k}^{n}\\right) + sy\\left(T_{i,j-1,k}^{n} - 2T_{i,j,k}^{n}
            + T_{i,j+1,k}^{n}\\right) + sz\\left(T_{i,j,k-1}^{n} - 2T_{i,j,k}^{n}
            + T_{i,j,k+1}^{n}\\right) + ss\\times Q\\text{init}(i,j,k)

        Returns:
            (list): Finite-difference integration variables.

        Attributes:
            sz (:obj:`float`): Integration coefficient for second-order finite difference approx along z.
            Qload3d (:obj:`~numpy.ndarray`): Normalised 3D (unit-correct) system driving / damping force.
            Tinit3d (:obj:`~numpy.ndarray`): Initial 3D (boundary condition based) temperature profile.

        Note:
            * Order of child method's params list does not directly logically correspond to base.

        Todo:
            * Instead of 2D and 3D versions of Qload and Tinit, rather overload *(Solved?)*
            * Need to move generation of meshgrids to hdf5 and only pass nt, nx, ny, nz etc.
            * Watch out for memory overflow with 2D -> 3D casting.
        """
        super().get_explicit_fd_params()
        sz = self.dt*self.kz/(self.rho*self.cp*(self.dz)**2)
        ss = self.dt/(self.rho*self.cp)
        Qload3d = ss*self.Qinit3d
        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            grp = hf.require_group(self.hdf5grp)
            grp.attrs.create('sz', sz)
            grp.attrs.create('waist_pump', self.wp)
            grp.attrs.create('alpha', self.alpha)
            grp.attrs.create('eta', self.eta)

            if 'Qload3d' not in grp:
                grp.create_dataset('Qload3d', shape=(self.nx, self.ny, self.nz), dtype=np.dtype(float).type,
                                   data=Qload3d, maxshape=(MAX_NX, MAX_NY, MAX_NZ))
            else:
                grp['Qload3d'].resize((self.nx, self.ny, self.nz))
                grp['Qload3d'][...] = Qload3d

            if 'Pinit3d' not in grp:
                grp.create_dataset('Pinit3d', shape=(self.nx, self.ny, self.nz), dtype=np.dtype(float).type,
                                   data=self.Pinit3d, maxshape=(MAX_NX, MAX_NY, MAX_NZ))
            else:
                grp['Pinit3d'].resize((self.nx, self.ny, self.nz))
                grp['Pinit3d'][...] = self.Pinit3d

        #Qload = ss*self.Qinit3d
        #Tinit = self.Tbc*np.ones([self.ny,self.nx])
        #return [super().get_explicit_fd_params(), self.zz, sz, self.alphamax, self.dz]

    def get_cn_fd_params(self):
        """Return parameters and coefficients to the implicit finite difference approximation,
        implemented within a Crank-Nicolson Scheme.

        .. math::

            T_{i,j,k}^{n+1} - sz\\left(T_{i-1,j,k}^{n+1} - 2T_{i,j,k}^{n+1} + T_{i+1,j,k}^{n+1}\\right)
            - sy\\left(T _{i,j-1,k}^{n+1} - 2T_{i,j,k}^{n+1} + T_{i,j+1,k}^{n+1}\\right)
            - sx\\left(T _{i,j,k-1}^{n+1} - 2T_{i,j,k}^{n+1} + T_{i,j,k+1}^{n+1}\\right)
            = T_{i,j,k}^{n} + sz\\left(T_{i-1,j,k}^{n} - 2T_{i,j,k}^{n} + T_{i+1,j,k}^{n}\\right)
            + sy\\left(T _{i,j-1,k}^{n} - 2T_{i,j,k}^{n} + T_{i,j+1,k}^{n}\\right)
            + sx\\left(T _{i,j,k-1}^{n} - 2T_{i,j,k}^{n} + T_{i,j,k+1}^{n}\\right) + ss\\times Q^{n}

        Where T is the solution to the differential equation
        :math:`Q_\\text{load} = ss\\times Q^{n}` is the driving \ source term or heat\thermal load,
        :math:`s_{x_{i}} = \\frac{\\Delta t}{2}\\frac{\\kappa_{x_{i}}}{(\\Delta x_{i})^2} \\text{ for } i \\in [1,2,3]`, and
        :math:`s_s = \\frac{1}{\\rho c_{p}}`, forming linear algebra system
        :math:`Ax = b`.

        Attributes:
            sz (:obj:`float`): Integration coefficient for second-order finite difference along z.
        """

        super().get_cn_fd_params()
        with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
            grp = hf.require_group(self.hdf5grp)
            sz = grp.attrs['sz']
            grp.attrs.create('sz', 0.5*sz)

#  LocalWords:  hdf ij const ss Qload
