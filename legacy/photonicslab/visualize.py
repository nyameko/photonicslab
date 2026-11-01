# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""Visualisation module for generating arbitrary contour plot animations.

Attributes:
    LABEL_FONTSIZE (float): Font size for the image titles and labels.
    AXIS_TICK_FONTSIZE (float): Tick size and spacing along axes.

Todo:
    * Refactor: Each cross-sectional plane should be read in and processed slice by slice not as a whole at once.

"""

import h5py
import numpy as np
import matplotlib
import warnings
#matplotlib.use('Agg')
import matplotlib.ticker as tck
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import Axes3D
from photonicslab.beamprop import BeamProp2D
from photonicslab.config import *

LABEL_FONTSIZE = 6.2
AXIS_TICK_FONTSIZE = 5.

LABEL_FONTSIZE_3D = 4
AXIS_TICK_FONTSIZE_3D = 4.5

class Visualize2D(object):
    """Two-dimensional temperature visualisation object.

    Args:
        hdf5filename (str): Filename of the HDF5 file object to be read from.
        hdf5table (str): Dataset name of the HDF5 group/dataset object to read.

    Attributes:
        fevolmap (:obj:`~numpy.ndarray`): Two-dimensional cross-sectional profile array of evolution map.
        xx (:obj:`~numpy.ndarray`): Meshgrid array description along x-direction.
        yy (:obj:`~numpy.ndarray`): Meshgrid array description along y-direction.
        clock (:obj:`~numpy.ndarray`): 1d array describing temporal evolution.
        _fsource (:obj:`~numpy.ndarray`): Meshgrid array description of driving / damping source.
        _vmin (:obj:`float`): Private attribute used by plotting function.
        _vmax (:obj:`float`): Private attribute used by plotting function.

    Todo:
        * Document private internal attributes for quadrant plotting / animation.
        * DRY Methods for quadrant assignments.
        * See `SO arrow on axis <https://stackoverflow.com/questions/23311466/using-matplotlib-and-point-to-specific-value-on-the-x-axis>`_ for highlighting temporal evolution of focal length

    """

    def __init__(self, hdf5filename='hdf5db/storage.hdf5', hdf5grp='nb1', hdf5table='temperature'):
        self.hdf5filename = str(hdf5filename)
        self.hdf5grp = str(hdf5grp)
        self.hdf5table = str(hdf5table)
        self.dsetfsol = str( str(hdf5grp)+'/'+str(hdf5table) )
        with h5py.File(self.hdf5filename, 'r', libver='latest') as hf:
            # self._fevolmap = np.remainder(hf[self.dsetfsol][:], 2*np.pi)
            self._fevolmap = hf[self.dsetfsol][:]
            grp = hf.require_group(self.hdf5grp)
            self.xx = grp['xx'][:]
            self.yy = grp['yy'][:]
            self.clock = grp['clock'][:]
            self._fsource = grp['Qload'][:]
            if 'polyfit' in hf[self.hdf5grp]:
                grp_poly = hf.require_group(str(self.hdf5grp +'/polyfit/'+ self.hdf5table))
                self.x_ind = grp_poly.attrs.get('x_ind')
                self.y_ind = grp_poly.attrs.get('y_ind')
                self.xpolyval = grp_poly.get('xpolyval')[:]
                self.ypolyval = grp_poly.get('ypolyval')[:]

        self._vmin_src = np.min(self._fsource)
        self._vmax_src = np.max(self._fsource)
        self._vmin_evol = np.min(self._fevolmap)
        self._vmax_evol = np.max(self._fevolmap)

    def _clear_canvas(self, labelsize=AXIS_TICK_FONTSIZE):
        """Private method to clear plotting canvas."""
        plt.close("all")
        plt.clf()
        plt.cla()
        plt.figure(dpi=MAX_DPI)
        plt.rc('xtick', labelsize=labelsize)
        plt.rc('ytick', labelsize=labelsize)

    def _custom_div_cmap(self, numcolors=255, name='custom_div_cmap',
                         mincol='black', midcol='white', maxcol='black'):
        """Create a custom diverging colormap with three colors.

        Default is grey to white to grey with 20 colors.  Colors can be specified
        in any way understandable by matplotlib.colors.ColorConverter.to_rgb()

        Reference:
            * PyHOGS Colormaps Example `<http://pyhogs.github.io/colormap-examples.html>`_
            * SO: Numpy Masked Arrays for Splitting CB `<https://stackoverflow.com/questions/22128166/two-different-color-colormaps-in-the-same-imshow-matplotlib>`_
        """

        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list(name=name, colors =[mincol, midcol, maxcol], N=numcolors)
        return cmap

    def _init_canvas(self):
        """Private method to prepare plotting canvas."""
        self._clear_canvas()
        self._fig, ((self._ax1, self._ax2), (self._ax3, self._ax4)) = plt.subplots(nrows=2, ncols=2)
        self._fig.suptitle('T [$^\circ$K] at t = {0:.5f}'.format(self.clock[-1]),
                           fontsize=LABEL_FONTSIZE+1.0)
        #self._fig.set_tight_layout(True)
        self._fig.subplots_adjust(wspace=0.25, hspace=0.4)

        #######################
        self._ax1.set_xlabel('X [cm]', fontsize=LABEL_FONTSIZE)
        self._ax1.set_ylabel('Y [cm]', fontsize=LABEL_FONTSIZE)
        self._ax1.set_title('Heat load [W/cm$^3$]', fontsize=LABEL_FONTSIZE)
        self._quad1 = self._ax1.pcolormesh(self.xx, self.yy, self._fsource[-1], vmin=self._vmin_src,
                                           vmax=self._vmax_src, cmap='jet', shading='gouraud')
        self._fig.colorbar(self._quad1, ax=self._ax1)

        #######################
        self._ax2.set_ylim(self._vmin_evol, self._vmax_evol)
        self._ax2.set_xlabel('X [cm]', fontsize=LABEL_FONTSIZE)
        self._ax2.set_ylabel('T [$^\circ$K]', fontsize=LABEL_FONTSIZE)
        self._ax2.set_title('On-axis temperature cross-section', fontsize=LABEL_FONTSIZE)
        self._quad2, = self._ax2.plot(self.xx.T[round(np.shape(self._fevolmap)[2]/2)],
                                      self._fevolmap[-1].T[round(np.shape(self._fevolmap)[2]/2)],
                                      '-b', label='Evolution')
        self._quad2b, = self._ax2.plot(self.xx[self.x_ind[0]:self.x_ind[1],
                                               round(np.shape(self._fevolmap)[2]/2)], self.xpolyval[-1],
                                       '-r', label='Parabolic Fit', lw=2)
        _legend = self._ax2.legend(loc='upper right', prop={'size':AXIS_TICK_FONTSIZE-1})
        for label in _legend.get_lines():
            label.set_linewidth(0.5)

        #######################
        self._ax3.set_xlabel('X [cm]', fontsize=LABEL_FONTSIZE)
        self._ax3.set_ylabel('Y [cm]', fontsize=LABEL_FONTSIZE)
        self._ax3.set_title('Temperature [$^\circ$K]', fontsize=LABEL_FONTSIZE)
        self._quad3 = self._ax3.pcolormesh(self.xx, self.yy, self._fevolmap[-1], vmin=self._vmin_evol,
                                           vmax=self._vmax_evol, cmap='jet', shading='gouraud')
        self._cb3 = self._fig.colorbar(self._quad3, ax=self._ax3)

        #######################
        self._ax4.set_ylim(self._vmin_evol, self._vmax_evol)
        self._ax4.set_xlabel('Y [cm]', fontsize=LABEL_FONTSIZE)
        self._ax4.set_ylabel('T [$^\circ$K]', fontsize=LABEL_FONTSIZE)
        self._ax4.set_title('On-axis temperature cross-section', fontsize=LABEL_FONTSIZE)
        self._quad4, = self._ax4.plot(self.yy[round(np.shape(self._fevolmap)[1]/2)],
                                      self._fevolmap[-1][round(np.shape(self._fevolmap)[1]/2)],
                                      '-b', label='Evolution')
        self._quad4b, = self._ax4.plot(self.yy[round(np.shape(self._fevolmap)[1]/2),
                                               self.y_ind[0]:self.y_ind[1]],
                                       self.ypolyval[-1], '-r', label='Parabolic Fit', lw=2)
        _legend = self._ax4.legend(loc='upper right', prop={'size':AXIS_TICK_FONTSIZE-1})
        for label in _legend.get_lines():
            label.set_linewidth(0.5)

    def _init_anim(self):
        """Private method to prepare animation canvas."""
        self._quad1.set_array([])
        self._quad3.set_array([])
        self._quad2.set_data([], [])
        self._quad2b.set_data([], [])
        self._quad4.set_data([], [])
        self._quad4b.set_data([], [])
        self._fig.suptitle([])
        return self._quad1, self._quad2, self._quad2b, self._quad3, self._quad4, self._quad4b, self._fig

    def _step_anim(self, t):
        """Private method to step through animation canvases."""
        self._quad1.set_array(self._fsource[t].ravel())
        self._quad3.set_array(self._fevolmap[t].ravel())
        self._quad2.set_data(self.xx.T[round(np.shape(self._fevolmap)[2]/2)],
                             self._fevolmap[t].T[round(np.shape(self._fevolmap)[2]/2)])
        self._quad2b.set_data(self.xx[self.x_ind[0]:self.x_ind[1], round(np.shape(self._fevolmap)[2]/2)],
                              self.xpolyval[t])
        self._quad4.set_data(self.yy[round(np.shape(self._fevolmap)[1]/2)],
                             self._fevolmap[t][round(np.shape(self._fevolmap)[1]/2)])
        self._quad4b.set_data(self.yy[round(np.shape(self._fevolmap)[1]/2), self.y_ind[0]:self.y_ind[1]],
                              self.ypolyval[t])
        self._fig.suptitle('T [$^\circ$K] at t = {0:.5f}'.format(self.clock[t]),
                           fontsize=LABEL_FONTSIZE+1.0)
        return self._quad1, self._quad2, self._quad2b, self._quad3, self._quad4, self._quad4b, self._fig

    @time_it
    def transient_profiles(self, savepath='vidoes/transient_profiles.mp4'):
        """Generate and save transient profile as MP4 file.

        Args:
            savepath (str): Name of MP4 file to save animation to.
        """
        self._init_canvas()
        anim = FuncAnimation(self._fig, self._step_anim, init_func=self._init_anim, \
                             frames=self.clock.size, interval=1000*(self.clock[1]-self.clock[0]),
                             blit=False)
        anim.save(savepath, fps=30, extra_args=['-vcodec', 'libx264'], dpi=MAX_DPI)

    def steadystate_profiles(self, savepath='images/steadystate_profiles.png'):
        """Generate and save steady-state profile as PNG file.

        Args:
            savepath (str): Name of PNG file to save image to.
        """
        self._init_canvas()
        plt.savefig(savepath, dpi=MAX_DPI)

class Visualize3D(Visualize2D):
    """Three-dimensional slice-by-slice visualisations of heat source and temperature profiles.

    The visualisations are based on monitoring the cross-sectional profile
    evolution across xy planes and along the xz and yz axes.

    Args:
        zz (ndarray): Meshgrid array description along axial z-direction.

    Todo:
        * Use rectangles and cylinders from `SO Colorbar for Plot Surface Sphere <https://stackoverflow.com/questions/6600579/colorbar-for-matplotlib-plot-surface-command>`_
        * Test mpl_toolkits.mplot3d Axes3D and Axes3d functions, and link path to mpl_toolkits docs.
        * Test the generation of a single colorbar for all for quadrants.
        * Need to go back and eliminate fsource parameter from 2d initializer
    """

    def __init__(self, z=-1, hdf5filename='hdf5db/storage.hdf5', hdf5grp='nb8',\
                 hdf5table='temperature3d', hdf5tabledrive='qinit'):
        super().__init__(hdf5filename, hdf5grp, hdf5table)

        with h5py.File(str(hdf5filename), 'r', libver='latest') as hf:
            #self._fevolmap = hdf[self.dsetfsol][self.t_ind,]
            #self._fsource = hdf[str( str(hdf5grp)+'/'+str(hdf5tabledrive) )][:]
            grp = hf.require_group(self.hdf5grp)
            self.fevolmap3d = grp[self.hdf5table][-1, ...][:]
            self.nx, self.ny, self.nz = grp.attrs.get('nx'), grp.attrs.get('ny'), grp.attrs.get('nz')
            self.Lx, self.Ly, self.Lz = grp.attrs.get('Lx'), grp.attrs.get('Ly'), grp.attrs.get('Lz')
            self.xxx = grp['xxx'][...]
            self.yyy = grp['yyy'][...]
            self.zzz = grp['zzz'][...]
            self.fsource3d = grp['Qload3d'][...]

        self.set_depth(z)

        self._vmin_src = np.min(self.fsource3d)
        self._vmax_src = np.max(self.fsource3d)
        self._vmin_evol = np.min(self.fevolmap3d)
        self._vmax_evol = np.max(self.fevolmap3d)

        self.cmap = 'jet'
        #self.cmap_proj = 'jet'
        self.cmap_proj = 'gist_gray'

    def set_depth(self, z=-1):
        """Set view depth within crystal / optical element.

        Todo:
            *TODO: Add test to check if still within crystal element.
            *TODO: Add all the try exception handling boiler plate.
        """

        if z < self.Lz and z > 0:
            self.z_ind = (np.abs(self.zzz[0, 0, :] - z).argmin())
            #self.t_ind = (np.abs(clock-t).argmin())
        elif z == -1:
            self.z_ind = -1

    def _init_canvas(self):
        """Private method to prepare plotting canvas."""

        #AXIS_TICK_FONTSIZE = 4 # -2.2
        #LABEL_FONTSIZE = 4.5 # -.5
        self._clear_canvas(AXIS_TICK_FONTSIZE_3D-0.5)

        # FIXME: Initial configuration of setup canvas
        # TODO: Implement one single colorbar for all quadrants
        self._fig = plt.figure()
        self._ax1 = self._fig.add_subplot(2, 2, 1)
        self._ax2 = self._fig.add_subplot(2, 2, 2)
        self._ax3 = self._fig.add_subplot(2, 2, 3, projection='3d')
        self._ax4 = self._fig.add_subplot(2, 2, 4, projection='3d')
        # self._ax5 = self._fig.add_subplot(2, 2, 5, projection='3d')
        # self._ax6 = self._fig.add_subplot(2, 2, 6, projection='3d')
        self._fig.suptitle("T [$^\circ$K] at t = {0:.5f}".format(self.clock[-1]), \
                           fontsize=LABEL_FONTSIZE_3D+1.0)
        #self._fig.set_tight_layout(True)
        self._fig.subplots_adjust(wspace=0.25, hspace=0.4)

        #######################
        self._ax1.set_xlabel('X [cm]', fontsize=LABEL_FONTSIZE_3D)
        self._ax1.set_ylabel('Y [cm]', fontsize=LABEL_FONTSIZE_3D)
        self._ax1.set_title('Pump-side end face temperature', fontsize=LABEL_FONTSIZE_3D)
        self._quad1 = self._ax1.pcolormesh(self.xx, self.yy, self.fevolmap3d[:, :, 0], \
                                           vmin=self._vmin_evol, vmax=self._vmax_evol, \
                                           cmap=self.cmap, shading='gouraud')

        #######################
        self._ax2.set_xlabel('X [cm]', fontsize=LABEL_FONTSIZE_3D)
        self._ax2.set_ylabel('Y [cm]', fontsize=LABEL_FONTSIZE_3D)
        self._ax2.set_title('Temperature at slice z = {0:.0f}'.format(self.z_ind), \
                            fontsize=LABEL_FONTSIZE_3D)
        self._quad2 = self._ax2.pcolormesh(self.xx, self.yy, self.fevolmap3d[:, :, self.z_ind], \
                                           vmin=self._vmin_evol, vmax=self._vmax_evol, \
                                           cmap=self.cmap, shading='gouraud')

        #######################
        self._ax3.set_xlabel('X [cm]', fontsize=LABEL_FONTSIZE_3D)
        self._ax3.set_xlim(-0.6*self.Lx, 0.6*self.Lx)
        self._ax3.set_ylabel('Z [cm]', fontsize=LABEL_FONTSIZE_3D)
        self._ax3.set_ylim(0, 1.1*self.Lz)
        #self._ax3.set_zlabel('Temperature [$^\circ$K]', fontsize=LABEL_FONTSIZE_3D)
        #self._ax3.set_zlim(self._vmin_evol-0.1*self._vmin_evol, self._vmax_evol)
        #print(np.shape(self.fevolmap3d))
        # if self.cmap == 'jet':
        #     self._quad3 = self._ax3.plot_surface(self.xxx.T[:, int(round(self.ny/2)), :], \
        #                                      self.zzz.T[:, int(round(self.ny/2)), :], \
        #                                      self.fevolmap3d.T[:, int(round(self.ny/2)), :], \
        #                                      alpha=0.8, cmap=self.cmap)
        self._cset3 = self._ax3.contourf(self.xxx.T[:, int(round(self.ny/2)), :], \
                                         self.zzz.T[:, int(round(self.ny/2)), :], \
                                         self.fevolmap3d.T[:, int(round(self.ny/2)), :], \
                                         zdir='x', offset=-0.6*self.Lx, cmap=self.cmap_proj, \
                                         vmin=self._vmin_evol, vmax=self._vmax_evol)
        self._cset3 = self._ax3.contourf(self.xxx.T[:, int(round(self.ny/2)), :], \
                                         self.zzz.T[:, int(round(self.ny/2)), :], \
                                         self.fevolmap3d.T[:, int(round(self.ny/2)), :], \
                                         zdir='y', offset=1.1*self.Lz, cmap=self.cmap_proj, \
                                         vmin=self._vmin_evol, vmax=self._vmax_evol)
        self._quad3 = self._ax3.contourf(self.xxx.T[:, int(round(self.ny/2)), :], \
                                         self.zzz.T[:, int(round(self.ny/2)), :], \
                                         self.fevolmap3d.T[:, int(round(self.ny/2)), :], \
                                         zdir='z', offset=self._vmin_evol, cmap=self.cmap, \
                                         vmin=self._vmin_evol, vmax=self._vmax_evol)

        #######################
        self._ax4.set_xlabel('Y [cm]', fontsize=LABEL_FONTSIZE_3D)
        self._ax4.set_xlim(-0.6*self.Ly, 0.6*self.Ly)
        self._ax4.set_ylabel('Z [cm]', fontsize=LABEL_FONTSIZE_3D)
        self._ax4.set_ylim(0, 1.1*self.Lz)
        #self._ax4.set_title('Temperature [$^\circ$K]', fontsize=LABEL_FONTSIZE_3D)
        #self._ax3.set_zlim(self._vmin_evol, self._vmax_evol)
        #self._ax4.set_zlim(self._vmin_evol-0.1*self._vmin_evol, self._vmax_evol)
        #if self.cmap == 'jet':
            # self._quad4 = self._ax4.plot_surface(self.yyy[int(round(self.nx/2)), :, :], \
            #                                  self.zzz[int(round(self.nx/2)), :, :], \
            #                                  self.fevolmap3d[int(round(self.nx/2)), :, :], \
            #                                  alpha=0.8, cmap=self.cmap)
        self._cset4 = self._ax4.contourf(self.yyy[int(round(self.nx/2)), :, :], \
                                         self.zzz[int(round(self.nx/2)), :, :], \
                                         self.fevolmap3d[int(round(self.nx/2)), :, :], \
                                         zdir='x', offset=-0.6*self.Ly, cmap=self.cmap_proj, \
                                         vmin=self._vmin_evol, vmax=self._vmax_evol)
        self._cset4 = self._ax4.contourf(self.yyy[int(round(self.nx/2)), :, :], \
                                         self.zzz[int(round(self.nx/2)), :, :], \
                                         self.fevolmap3d[int(round(self.nx/2)), :, :], \
                                         zdir='y', offset=1.1*self.Lz, cmap=self.cmap_proj, \
                                         vmin=self._vmin_evol, vmax=self._vmax_evol)
        # self._cset4ya, = self._ax4.plot(self.yyy[int(round(self.nx/2)), :, 0], \
        #                                self.fevolmap3d[int(round(self.nx/2)), :, 0], \
        #                                zdir='y', label='Pump-Side')
        # self._cset4yb, = self._ax4.plot(self.yyy[int(round(self.nx/2)), :, -1], \
        #                                self.fevolmap3d[int(round(self.nx/2)), :, -1], \
        #                                zdir='y', label='Output Coupler')
        self._quad4 = self._ax4.contourf(self.yyy[int(round(self.nx/2)), :, :], \
                                         self.zzz[int(round(self.nx/2)), :, :], \
                                         self.fevolmap3d[int(round(self.nx/2)), :, :], \
                                         zdir='z', offset=self._vmin_evol, cmap=self.cmap, \
                                         vmin=self._vmin_evol, vmax=self._vmax_evol)

        #######################
        # self._ax5.set_xlabel('X [cm]', fontsize=LABEL_FONTSIZE_3D)
        # self._ax5.set_xlim(-0.6*self.Lx, 0.6*self.Lx)
        # self._ax5.set_ylabel('Z [cm]', fontsize=LABEL_FONTSIZE_3D)
        # self._ax5.set_ylim(0, 1.1*self.Lz)
        #self._ax4.set_title('Temperature [$^\circ$K]', fontsize=LABEL_FONTSIZE_3D)
        #self._ax3.set_zlim(self._vmin_evol, self._vmax_evol)
        #self._ax4.set_zlim(self._vmin_evol-0.1*self._vmin_evol, self._vmax_evol)
        #if self.cmap == 'jet':
        self._quad5 = self._ax3.plot_surface(self.xxx.T[:, int(round(self.ny/2)), :], \
                                             self.zzz.T[:, int(round(self.ny/2)), :], \
                                             self.fevolmap3d.T[:, int(round(self.ny/2)), :], \
                                             alpha=0.1, cmap=self.cmap, vmin=self._vmin_evol, \
                                             vmax=self._vmax_evol)

        #######################
        # self._ax6.set_xlabel('Y [cm]', fontsize=LABEL_FONTSIZE_3D)
        # self._ax6.set_xlim(-0.6*self.Ly, 0.6*self.Ly)
        # self._ax6.set_ylabel('Z [cm]', fontsize=LABEL_FONTSIZE_3D)
        # self._ax6.set_ylim(0, 1.1*self.Lz)
        #self._ax4.set_title('Temperature [$^\circ$K]', fontsize=LABEL_FONTSIZE_3D)
        #self._ax3.set_zlim(self._vmin_evol, self._vmax_evol)
        #self._ax4.set_zlim(self._vmin_evol-0.1*self._vmin_evol, self._vmax_evol)
        #if self.cmap == 'jet':
        self._quad6 = self._ax4.plot_surface(self.yyy[int(round(self.nx/2)), :, :], \
                                             self.zzz[int(round(self.nx/2)), :, :], \
                                             self.fevolmap3d[int(round(self.nx/2)), :, :], \
                                             alpha=0.1, cmap=self.cmap, vmin=self._vmin_evol, \
                                             vmax=self._vmax_evol)

        #print(self._cset4x)
        # print(self._cset4ya)
        # print(self._cset4yb)
        #print(self._cset4z)

        #cax = _fig.add_axes([0.85, 0.1, 0.03, 0.8])
        cax = self._fig.add_axes([0.95, 0.1, 0.01, 0.75])
        self.figcb = self._fig.colorbar(self._quad1, cax=cax)
        self.figcb.set_label('T')

    def _init_anim(self):
        """Private method to prepare animation canvas."""
        self._quad1.set_array([])
        self._quad2.set_array([])
        # self._cset4ya.set_data([], [])
        # self._cset4yb.set_data([], [])
        #self._quad5.remove()
        #self._quad6.remove()
        for coll in self._quad3.collections:
            #plt.gca().collections.remove(coll)
            #self._fig.gca().collections.remove(coll)
            coll.remove()
        #for coll in self._ax4.collections:
            #plt.gca().collections.remove(coll)
            #self._fig.gca().collections.remove(coll)
            #coll.remove()
        for coll in self._quad4.collections:
            coll.remove()
        # for coll in self._quad5.collections:
        #     coll.remove()
        # for coll in self._quad6.collections:
        #     coll.remove()
        # for coll in self._cset4y.collections:
        #     coll.remove()
        # # for coll in self._cset4yb.collections:
        # #     coll.remove()
        # for coll in self._cset4z.collections:
        #     coll.remove()
        # FIXME:
        # for coll in self._cset3.collections:
        #     #plt.gca().collections.remove(coll)
        #     #self._fig.gca().collections.remove(coll)
        #     coll.remove()
        # for coll in self._cset4.collections:
        #     #plt.gca().collections.remove(coll)
        #     #self._fig.gca().collections.remove(coll)
        #     coll.remove()
        # FIXME:
        #self._ax3.collections = []
        #self._ax4.collections = []

        self._fig.suptitle([])
        #return self._quad1, self._quad2, self._quad3, self._cset3, self._quad4, self._cset4, self._fig
        return self._quad1, self._quad2, self._quad3, self._quad4, self._fig

    def _step_anim(self, t):
        """Private method to step through animation canvases.
        Todo:
            * Check indices for nx and ny and which should be transposed by changing range values of x and y
            * Check the use iter instead of t
            * Will need to generalize sup title for phase as well
        """

        with h5py.File(self.hdf5filename, 'r', libver='latest') as hdf:
            self.fevolmap3d = hdf[self.dsetfsol][t, ...][:]

        self._quad1.set_array(self.fevolmap3d[:, :, 0].ravel())
        self._quad2.set_array(self.fevolmap3d[:, :, self.z_ind].ravel())
        #self._cset4ya.set_data(self.yyy[int(round(self.nx/2)), :, 0], self.fevolmap3d[int(round(self.nx/2)), :, 0])
        #self._cset4yb.set_data(self.yyy[int(round(self.nx/2)), :, -1], self.fevolmap3d[int(round(self.nx/2)), :, -1])

        # if self.cmap == 'jet':
        #     self._quad3 = self._ax3.plot_surface(self.xxx.T[:, int(round(self.ny/2)), :], \
        #                                      self.zzz.T[:, int(round(self.ny/2)), :], \
        #                                      self.fevolmap3d.T[:, int(round(self.ny/2)), :], \
        #                                      alpha=0.8, cmap=self.cmap)
        # self._cset3 = self._ax3.contourf(self.xxx.T[:, int(round(self.ny/2)), :], \
        #                                  self.zzz.T[:, int(round(self.ny/2)), :], \
        #                                  self.fevolmap3d.T[:, int(round(self.ny/2)), :], \
        #                                  zdir='x', offset=-0.6*self.Lx, cmap=self.cmap)
        self._quad3 = self._ax3.contourf(self.xxx.T[:, int(round(self.ny/2)), :], \
                                         self.zzz.T[:, int(round(self.ny/2)), :], \
                                         self.fevolmap3d.T[:, int(round(self.ny/2)), :], \
                                         zdir='z', offset=self._vmin_evol, cmap=self.cmap, \
                                         vmin=self._vmin_evol, vmax=self._vmax_evol)
        # self._cset3 = self._ax3.contourf(self.xxx.T[:, int(round(self.ny/2)), :], \
        #                                  self.zzz.T[:, int(round(self.ny/2)), :], \
        #                                  self.fevolmap3d.T[:, int(round(self.ny/2)), :], \
        #                                  zdir='z', offset=self._vmin_evol, cmap=self.cmap)

        # #if self.cmap == 'jet':
        #     self._quad4 = self._ax4.plot_surface(self.yyy[int(round(self.nx/2)), :, :], \
        #                                      self.zzz[int(round(self.nx/2)), :, :], \
        #                                      self.fevolmap3d[int(round(self.nx/2)), :, :], \
        #                                      alpha=0.8, cmap=self.cmap)
        # self._cset4 = self._ax4.contourf(self.yyy[int(round(self.nx/2)), :, :], \
        #                                  self.zzz[int(round(self.nx/2)), :, :], \
        #                                  self.fevolmap3d[int(round(self.nx/2)), :, :], \
        #                                  zdir='x', offset=-0.6*self.Ly, cmap=self.cmap_proj)
        self._quad4 = self._ax4.contourf(self.yyy[int(round(self.nx/2)), :, :], \
                                         self.zzz[int(round(self.nx/2)), :, :], \
                                         self.fevolmap3d[int(round(self.nx/2)), :, :], \
                                         zdir='z', offset=self._vmin_evol, cmap=self.cmap, \
                                         vmin=self._vmin_evol, vmax=self._vmax_evol)
        # self._cset4ya, = self._ax4.plot(self.yyy[int(round(self.nx/2)), :, 0], \
        #                                self.fevolmap3d[int(round(self.nx/2)), :, -1], \
        #                                zdir='y', label='Pump-Side')
        # self._cset4yb, = self._ax4.plot(self.yyy[int(round(self.nx/2)), :, -1], \
        #                                self.fevolmap3d[int(round(self.nx/2)), :, -1], \
        #                                zdir='y', label='Output Coupler')
        # self._cset4 = self._ax4.contourf(self.yyy[int(round(self.nx/2)), :, :], \
        #                                  self.zzz[int(round(self.nx/2)), :, :], \
        #                                  self.fevolmap3d[int(round(self.nx/2)), :, :], \
        #                                  zdir='z', offset=self._vmin_evol, cmap=self.cmap_proj)
        # self._quad5 = self._ax3.plot_surface(self.xxx.T[:, int(round(self.ny/2)), :], \
        #                                      self.zzz.T[:, int(round(self.ny/2)), :], \
        #                                      self.fevolmap3d.T[:, int(round(self.ny/2)), :], \
        #                                      alpha=0.2, cmap=self.cmap)

        # self._quad6 = self._ax4.plot_surface(self.yyy[int(round(self.nx/2)), :, :], \
        #                                      self.zzz[int(round(self.nx/2)), :, :], \
        #                                      self.fevolmap3d[int(round(self.nx/2)), :, :], \
        #                                      alpha=0.2, cmap=self.cmap)

        self._fig.suptitle('T [$^\circ$K] at t = {0:.5f}'.format(self.clock[t]), fontsize=LABEL_FONTSIZE_3D+1.0)
        return self._quad1, self._quad2, self._quad3, self._quad4, self._fig

    def steadystate_source_profiles(self, savepath='images/steadystate_source_profiles.png'):
        """Generate and save steady-state source profile as PNG file.

        Args:
            savepath (str): Name of PNG file to save image to.
        """
        self._init_canvas()
        # self._init_anim()

        # self._quad1.set_array(self.fsource3d[:, :, 0].ravel())
        # self._quad2.set_array(self.fsource3d[:, :, self.z_ind].ravel())

        # self._quad3 = self._ax3.plot_surface(self.xxx[:, round(self.ny/2), :], self.zzz[:, round(self.ny/2), :],
        #                                      self.fsource3d[:, round(self.ny/2), :], alpha=0.8, cmap='jet')
        # self._cset3 = self._ax3.contourf(self.xxx[:, round(self.ny/2), :], self.zzz[:, round(self.ny/2), :],
        #                                  self.fsource3d[:, round(self.ny/2), :], zdir='x', offset=-1*self.Lx, cmap='jet')
        # self._cset3 = self._ax3.contourf(self.xxx[:, round(self.ny/2), :], self.zzz[:, round(self.ny/2), :],
        #                                  self.fsource3d[:, round(self.ny/2), :], zdir='y', offset=self.Lz, cmap='jet')
        # self._cset3 = self._ax3.contourf(self.xxx[:, round(self.ny/2), :], self.zzz[:, round(self.ny/2), :],
        #                                  self.fsource3d[:, round(self.ny/2), :], zdir='z', offset=self._vmin_evol, cmap='jet')
        # self._quad4 = self._ax4.plot_surface(self.yyy[round(self.nx/2), :, :], self.zzz[round(self.nx/2), :, :],
        #                                      self.fsource3d[round(self.nx/2), :, :], alpha=0.8, cmap='jet')
        # self._cset4 = self._ax4.contourf(self.yyy[round(self.nx/2), :, :], self.zzz[round(self.nx/2), :, :],
        #                                   self.fsource3d[round(self.nx/2), :, :], zdir='x', offset=-1*self.Ly, cmap='jet')
        # self._cset4 = self._ax4.contourf(self.yyy[round(self.nx/2), :, :], self.zzz[round(self.nx/2), :, :],
        #                                   self.fsource3d[round(self.nx/2), :, :], zdir='y', offset=self.Lz, cmap='jet')
        # self._cset4 = self._ax4.contourf(self.yyy[round(self.nx/2), :, :], self.zzz[round(self.nx/2), :, :],
        #                                   self.fsource3d[round(self.nx/2), :, :], zdir='z', offset=self._vmin_evol, cmap='jet')

        # self._ax1.set_title('Pump-side end face source heat load', fontsize=LABEL_FONTSIZE_3D)
        # self._ax2.set_title('Source heat load at slice z = {0:.0f}'.format(self.z_ind), fontsize=LABEL_FONTSIZE_3D)
        # self._ax3.set_zlabel('Q', fontsize=LABEL_FONTSIZE_3D)
        # self._ax4.set_zlabel('Q', fontsize=LABEL_FONTSIZE_3D)
        # self._fig.suptitle('Q at t = {0:.5f}'.format(self.clock[t]), fontsize=LABEL_FONTSIZE_3D+1.0)

        plt.savefig(savepath, dpi=MAX_DPI)

class VisualizePhase2D(Visualize2D):
    """Two-dimensional phase visualisation object.

    Todo:
        * Big decision to make, will this class import or inherit beamprop?
        * Check test to ensure polyfit profiles exist
    """

    def __init__(self, hdf5filename='hdf5db/storage.hdf5', hdf5grp='nb1', hdf5table='phase'):
        super().__init__(hdf5filename, hdf5grp, hdf5table)
        #self._fevolmap = np.remainder(self._fevolmap, 2*np.pi)
        self.xpolyval = np.remainder(self.xpolyval, 2*np.pi)
        self.ypolyval = np.remainder(self.ypolyval, 2*np.pi)
        self._vmin_evol = np.min(self._fevolmap)
        self._vmax_evol = np.max(self._fevolmap)

    @time_it
    def _init_canvas(self):
        """Private method to prepare plotting canvas.

        References:
            * `SO Get XY_Data from Plot<https://stackoverflow.com/questions/20130768/retrieve-xy-data-from-matplotlib-figure>`_
        """
        super()._init_canvas()
        self._quad2.set_data([],[])
        self._quad2b.set_data([],[])
        self._quad4.set_data([],[])
        self._quad4b.set_data([],[])
        self._fig.suptitle('Phase Map at t = {0:.5f}'.format(self.clock[-1]),
                           fontsize=LABEL_FONTSIZE+1.0)

        self._ax2.set_ylabel('Phase [rad]', fontsize=LABEL_FONTSIZE)
        self._ax2.set_title('On-axis phase profiles', fontsize=LABEL_FONTSIZE)
        self._ax2.set_yticklabels([r"$0$", r"$\frac{\pi}{3}$", r"$\frac{2\pi}{3}$", r"$\pi$",
                                   r"$\frac{4\pi}{3}$", r"$\frac{5\pi}{3}$", r"$2\pi$"])

        cmap_phase = self._custom_div_cmap()
        self._quad3.set_cmap(cmap_phase)
        self._ax3.set_title('Phasemap', fontsize=LABEL_FONTSIZE)
        #self._cb3.set_ticks([np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3, 2*np.pi])
        self._cb3.ax.set_yticklabels([r"$\frac{\pi}{3}$", r"$\frac{2\pi}{3}$", r"$\pi$",
                                      r"$\frac{4\pi}{3}$", r"$\frac{5\pi}{3}$", r"$2\pi$"])

        self._ax4.set_ylabel('Phase [rad]', fontsize=LABEL_FONTSIZE)
        self._ax4.set_title('On-axis phase profiles', fontsize=LABEL_FONTSIZE)
        self._ax4.set_yticklabels([r"$0$", r"$\frac{\pi}{3}$", r"$\frac{2\pi}{3}$", r"$\pi$",
                                   r"$\frac{4\pi}{3}$", r"$\frac{5\pi}{3}$", r"$2\pi$"])

    def _step_anim(self, t):
        """Private method to step through animation canvases."""
        self._quad1.set_array(self._fsource[t].ravel())
        self._quad3.set_array(self._fevolmap[t].ravel())

        self._quad2.set_data(self.xx.T[round(np.shape(self._fevolmap)[2]/2)],
                             self._fevolmap[t].T[round(np.shape(self._fevolmap)[2]/2)])

        self._quad2b.set_data(self.xx[self.x_ind[0]:self.x_ind[1], round(np.shape(self._fevolmap)[2]/2)],
                              self.xpolyval[t])

        self._quad4.set_data(self.yy[round(np.shape(self._fevolmap)[1]/2)],
                             self._fevolmap[t][round(np.shape(self._fevolmap)[1]/2)])

        self._quad4b.set_data(self.yy[round(np.shape(self._fevolmap)[1]/2), self.y_ind[0]:self.y_ind[1]],
                              self.ypolyval[t])

        self._fig.suptitle('Phase Profiles at t = {0:.5f}'.format(self.clock[t]),
                           fontsize=LABEL_FONTSIZE+1.0)
        return self._quad1, self._quad2, self._quad2b, self._quad3, self._quad4, self._quad4b, self._fig

    #@time_it
    def _unlink_wrap_around(self, dat, lims=[0, 2*np.pi], thresh=0.95):
        """Remove discontinuous linewrap from on-axis phase profile.

        Iterate over contiguous regions of `dat` (i.e. where it does not
        jump from near one limit to the other).

        This function returns an iterator object that yields slice
        objects, which index the contiguous portions of `dat`.

        This function implicitly assumes that all points in `dat` fall
        within `lims`.

        Reference:
            * `SO Solution from farenorth <https://stackoverflow.com/questions/27138751/preventing-plot-joining-when-values-wrap-in-matplotlib-plots>`_

        Todo:
            * Find place to use this function
        >>> for slc in unlink_wrap(x, lims):
        >>>     ax.plot(x[slc], y[slc], 'b-', linewidth=2)

        """

        jump = np.nonzero(np.abs(np.diff(dat)) > ((lims[1] - lims[0]) * thresh))[0]
        lasti = 0
        for ind in jump:
            yield slice(lasti, ind + 1)
            lasti = ind + 1
            yield slice(lasti, len(dat))

    @time_it
    def steadystate_profiles(self, savepath='images/steadystate_profiles.png', threshold=0.95):
        """Generate and save steady-state profile as PNG file.

        Args:
            savepath (str): Name of PNG file to save image to.
        """
        self._init_canvas()

        for slc in self._unlink_wrap_around(self._fevolmap[-1].T[round(np.shape(self._fevolmap)[2]/2)], thresh=threshold):
            self._ax2.plot(self.xx.T[round(np.shape(self._fevolmap)[2]/2)][slc],
                           self._fevolmap[-1].T[round(np.shape(self._fevolmap)[2]/2)][slc],
                           '-b', label='Evolution')

        for slc in self._unlink_wrap_around(self.xpolyval[-1], thresh=threshold):
            self._ax2.plot(self.xx[self.x_ind[0]:self.x_ind[1], round(np.shape(self._fevolmap)[2]/2)][slc],
                           self.xpolyval[-1][slc], '-r', label='Parabolic Fit', lw=2)

        for slc in self._unlink_wrap_around(self._fevolmap[-1][round(np.shape(self._fevolmap)[1]/2)], thresh=threshold):
            self._ax4.plot(self.yy[round(np.shape(self._fevolmap)[1]/2)][slc],
                           self._fevolmap[-1][round(np.shape(self._fevolmap)[1]/2)][slc],
                           '-b', label='Evolution')

        for slc in self._unlink_wrap_around(self.ypolyval[-1], thresh=threshold):
            self._ax4.plot(self.yy[round(np.shape(self._fevolmap)[1]/2), self.y_ind[0]:self.y_ind[1]][slc],
                           self.ypolyval[-1][slc], '-r', label='Parabolic Fit', lw=2)

        plt.savefig(savepath, dpi=MAX_DPI)

class VisualizePhase3D(Visualize3D):
    """Three-dimensional phase visualisation object."""

    def __init__(self, z=-1, hdf5filename='hdf5db/storage.db',
                 hdf5grp='nb8', hdf5table='phase3d'):
        super().__init__(z, hdf5filename, hdf5grp, hdf5table)
        self.fevolmap3d = np.remainder(self.fevolmap3d, 2*np.pi)
        self._vmin_evol, self._vmax_evol = np.min(self.fevolmap3d), np.max(self.fevolmap3d)
        self.cmap = self._custom_div_cmap()
        self.cmap_proj = self._custom_div_cmap()

    def _init_canvas(self):
        super()._init_canvas()
        #temp = super()._init_anim()
        self._fig.suptitle('Phasemap at t = {0:.5f}'.format(self.clock[-1]),
                           fontsize=LABEL_FONTSIZE_3D+1.0)

        self._ax1.set_title('Pump-side end face phasemap', fontsize=LABEL_FONTSIZE)
        #self._quad1.set_cmap(self.cmap)

        self._ax2.set_title('Phasemap at slice z = {0:.0f}'.format(self.z_ind), fontsize=LABEL_FONTSIZE)
        #self._quad2.set_cmap(self.cmap)

        #self._quad3.set_cmap(self.cmap)
        #self._quad3.set_cmap(self.cmap)
        self._ax3.set_zticklabels([r"$0$", r"$\frac{\pi}{3}$", r"$\frac{2\pi}{3}$", r"$\pi$",
                                   r"$\frac{4\pi}{3}$", r"$\frac{5\pi}{3}$", r"$2\pi$"])
        #self._quad4.set_cmap(self.cmap)
        #self._quad4.set_cmap
        self._ax4.set_zticklabels([r"$0$", r"$\frac{\pi}{3}$", r"$\frac{2\pi}{3}$", r"$\pi$",
                                   r"$\frac{4\pi}{3}$", r"$\frac{5\pi}{3}$", r"$2\pi$"])

        # self._quad5.set_cmap(self.cmap)
        # self._ax5.set_zticklabels([r"$0$", r"$\frac{\pi}{3}$", r"$\frac{2\pi}{3}$", r"$\pi$",
        #                            r"$\frac{4\pi}{3}$", r"$\frac{5\pi}{3}$", r"$2\pi$"])

        # self._quad6.set_cmap
        # self._ax6.set_zticklabels([r"$0$", r"$\frac{\pi}{3}$", r"$\frac{2\pi}{3}$", r"$\pi$",
        #                            r"$\frac{4\pi}{3}$", r"$\frac{5\pi}{3}$", r"$2\pi$"])

        # FIXME: Ticklabels not showing correctly
        self.figcb.set_ticklabels([r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$', r'$\pi$',
                                   r'$\frac{4\pi}{3}$', r'$\frac{5\pi}{3}$', r'$2\pi$'])
        self.figcb.set_label(r'$\theta$')
        self.figcb.update_ticks()

        #self._fig.colorbar(self._quad1, cax=self.cax).set_label(r'\theta')
        #self._fig.colorbar(self._quad1, cax=self.cax).set_ticks([np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3, 2*np .pi])
        # self._fig.colorbar(self._quad1, cax=self.cax).set_ticklabels([r"$\frac{\pi}{3}$", r"$\frac{2\pi}{3}$", r"$\pi$",
        #                                                           r"$\frac{4\pi}{3}$", r"$\frac{5\pi}{3}$", r"$2\pi$"])
        # self.figcb.set_yticklabels([r"$\frac{\pi}{3}$", r"$\frac{2\pi}{3}$", r"$\pi$",
        #                             r"$\frac{4\pi}{3}$", r"$\frac{5\pi}{3}$", r"$2\pi$"])
        # self.figcb.ax.set_label(r"$\theta")

    def _step_anim(self, t):
        temp = super()._step_anim(t)
        self._fig.suptitle(r'Phasemap [$\theta$] at t = {0:.5f}'.format(self.clock[t]), fontsize=LABEL_FONTSIZE_3D+1.0)
        return temp

class VisualizeBeam2D(Visualize2D):
    """Two-dimensional beam visualisation object.

    Uses the :class:`~photonicslab.beamprop.BeamProp2D` class to the generate original
    and propagated seed beam profiles.

    Args:
        p (int, p>=0): Radial mode index denoting number of radial nodes.
        l (int): Azimuthal mode index denoting number of intertwined helices.
        z (float): Propagation distance along optical axis [cm].

    Attributes:
        fseed (:obj:`~numpy.ndarray`): Incident seed beam profile. Default is a Gaussian.
        fseed_mod (:obj:`~numpy.ndarray`): Phase modulated incident seed beam profile.
        fseed_convol (:obj:`~numpy.ndarray`): Propagation kernel convolution of incident seed beam profile.
        fseed_mod_convol (:obj:`~numpy.ndarray`): Propagation kernel convolution of modulated incident seed beam profile.

    """

    def __init__(self, wavelen=1.064e-4, waist=191.5e-4, p=0, l=0, z=0, phi=0, hdf5filename='hdf5db/storage.hdf5', hdf5grp='nb1', hdf5set_phase='phase'):
        super().__init__(hdf5filename, hdf5grp)
        self.wavelen = float(wavelen)
        self.waist = float(waist)
        self.hdf5set_phase = str(hdf5set_phase)
        self.fseed = BeamProp2D(wavelen=self.wavelen, waist=self.waist, hdf5filename=self.hdf5filename,
                                hdf5grp=self.hdf5grp).lg_mode(p, l, z, phi)
        self._fsource = self.fseed[2]

        self._vmin_src = np.min(self._fsource)
        self._vmax_src = np.max(self._fsource)

        # TODO: This returns for all times, like _fevol
        self.fseed_mod = BeamProp2D(wavelen=self.wavelen, waist=self.waist, hdf5filename=self.hdf5filename,
                                    hdf5grp=self.hdf5grp).modulate(self.fseed[0], hdf5table=self.hdf5set_phase)

        self._vmin_evol = np.min(self.fseed[2])
        self._vmax_evol = np.max(self.fseed[2])
        self.p = p
        self.l = l
        self.z = z
        self.phi = phi
        # FIXME: This may be redundant?
        #self.fseed_convol = np.empty(self.xx, self.yy)
        #self.fseed_mod_convol = np.empty(self.xx, self.yy)
        #print("Rayleigh Range: {0}".format(np.round(float(np.pi*(waist)**2/wavelen))))

    def _init_canvas(self):
        """Private method to prepare animation canvas."""
        self._clear_canvas()
        self._fig, ((self._ax1, self._ax2), (self._ax3, self._ax4)) = plt.subplots(nrows=2, ncols=2)
        self._fig.suptitle('Beam Intensities I at t = {0:.5f}s and propagated to z = {1:.1f}cm'\
                           .format(self.clock[-1], self.z), fontsize=LABEL_FONTSIZE+1.0)
        #self._fig.set_tight_layout(True)
        self._fig.subplots_adjust(wspace=0.25, hspace=0.4)

        self._ax1.set_xlabel('X [cm]', fontsize=LABEL_FONTSIZE)
        self._ax1.set_ylabel('Y [cm]', fontsize=LABEL_FONTSIZE)
        self._ax1.set_title('Incident seed profile at z=0', fontsize=LABEL_FONTSIZE)
        self._quad1 = self._ax1.pcolormesh(self.xx, self.yy, self._fsource, vmin=self._vmin_src,
                                           vmax=self._vmax_src, cmap='gnuplot2', shading='gouraud')
        self._fig.colorbar(self._quad1, ax=self._ax1)
        # TODO: Complete _init_canvas, and single colorbar

        #######################
        self._ax2.set_ylim(self._vmin_evol, self._vmax_evol)
        self._ax2.set_xlabel('X [cm]', fontsize=LABEL_FONTSIZE)
        self._ax2.set_ylabel('Normalized Intensity', fontsize=LABEL_FONTSIZE)
        self._ax2.set_title('On-axis intensity cross-section', fontsize=LABEL_FONTSIZE)
        self._quad2, = self._ax2.plot(self.xx.T[round(np.shape(self.fseed[2])[1]/2)],
                                      self.fseed[2].T[round(np.shape(self.fseed[2])[1]/2)],
                                      '-b', label='Seed at z=0')
        self._quad2b, = self._ax2.plot(self.xx.T[round(np.shape(self.fseed_convol)[1]/2)],
                                      self.fseed_convol.T[round(np.shape(self.fseed_convol)[1]/2)],
                                      '-r', label='Un-modulated')
        if self.fseed_mod_convol.ndim == 2:
            self._quad2c, = self._ax2.plot(self.xx.T[round(np.shape(self.fseed_mod_convol)[1]/2)],
                            self.fseed_mod_convol.T[round(np.shape(self.fseed_mod_convol)[1]/2)],
                                           '-g', label='Phase Modulated')
        elif self.fseed_mod_convol.ndim == 3:
            #print(np.shape(self.fseed_mod_convol))
            self._quad2c, = self._ax2.plot(self.xx.T[round(np.shape(self.fseed_mod_convol)[2]/2)],
                            self.fseed_mod_convol[-1].T[round(np.shape(self.fseed_mod_convol)[2]/2)],
                                           '-g', label='Phase Modulated')

        _legend = self._ax2.legend(loc='upper right', prop={'size':AXIS_TICK_FONTSIZE-1})
        for label in _legend.get_lines():
            label.set_linewidth(0.5)

        #######################
        self._ax3.set_xlabel('X [cm]', fontsize=LABEL_FONTSIZE)
        self._ax3.set_ylabel('Y [cm]', fontsize=LABEL_FONTSIZE)
        self._ax3.set_title('Normalized Intensity', fontsize=LABEL_FONTSIZE)
        if self.fseed_mod_convol.ndim == 2:
            self._quad3 = self._ax3.pcolormesh(self.xx, self.yy, self.fseed_mod_convol,
                                               vmin=self._vmin_evol, vmax=self._vmax_evol,
                                               cmap='gnuplot2', shading='gouraud')
        elif self.fseed_mod_convol.ndim == 3:
            self._quad3 = self._ax3.pcolormesh(self.xx, self.yy, self.fseed_mod_convol[-1],
                                               vmin=self._vmin_evol, vmax=self._vmax_evol,
                                               cmap='gnuplot2', shading='gouraud')
        self._fig.colorbar(self._quad3, ax=self._ax3)

        #######################
        self._ax4.set_ylim(self._vmin_evol, self._vmax_evol)
        self._ax4.set_xlabel('Y [cm]', fontsize=LABEL_FONTSIZE)
        self._ax4.set_ylabel('Normalized Intensity', fontsize=LABEL_FONTSIZE)
        self._ax4.set_title('On-axis intensity cross-section', fontsize=LABEL_FONTSIZE)
        self._quad4, = self._ax4.plot(self.yy[round(np.shape(self.fseed[2])[0]/2)],
                                      self.fseed[2][round(np.shape(self.fseed[2])[0]/2)],
                                      '-b', label='Seed at z=0')
        self._quad4b, = self._ax4.plot(self.yy[round(np.shape(self.fseed_convol)[0]/2)],
                                       self.fseed_convol[round(np.shape(self.fseed_convol)[0]/2)],
                                       '-r', label='Un-modulated')
        if self.fseed_mod_convol.ndim == 2:
            self._quad4c, = self._ax4.plot(self.yy[round(np.shape(self.fseed_mod_convol)[0]/2)],
                            self.fseed_mod_convol[round(np.shape(self.fseed_mod_convol)[0]/2)],
                                           '-g', label='Phase Modulated')
        elif self.fseed_mod_convol.ndim == 3:
            self._quad4c, = self._ax4.plot(self.yy[round(np.shape(self.fseed_mod_convol)[1]/2)],
                            self.fseed_mod_convol[-1][round(np.shape(self.fseed_mod_convol)[1]/2)],
                                           '-g', label='Phase Modulated')
        _legend = self._ax4.legend(loc='upper right', prop={'size':AXIS_TICK_FONTSIZE-1})
        for label in _legend.get_lines():
            label.set_linewidth(0.5)

    def add_filter_circular(self, center_x=0.0, center_y=0.0, inner_r=0.0, outer_r=0.05, filter_nd=100.0):
        """Adds a circular obstruction to the incident beam.

        Args:
            center_x (float): X co-ordinate of the center of the heat source [cm].
            center_y (float): Y co-ordinate of the center of the heat source [cm].
            inner_r (float): Radius from the center to inner section of the heat source [cm].
            outer_r (float): Radius from the center to outer section of the heat source [cm].

        Raises:
            ValueError: If the coordinates of the position of the heat source are out of bounds
                        of crystal face.

        Attributes:
            pts (:obj:`~numpy.ndarray`): Cross sectional coordinates for which to add obstruction.

        """
        pts = ((self.xx-center_x)**2 + (self.yy-center_y)**2 <= outer_r**2) & ((self.xx-center_x)**2 + (self.yy-center_y)**2 < inner_r**2)
        self.fseed[0][pts] /= filter_nd
        self.fseed[1][pts] /= filter_nd
        self.fseed[2][pts] /= filter_nd
        self._fsource = self.fseed[2]
        self.fseed_mod = BeamProp2D(wavelen=self.wavelen, waist=self.waist, hdf5filename=self.hdf5filename,
                                    hdf5grp=self.hdf5grp).modulate(self.fseed[0], hdf5table=self.hdf5set_phase)
        # try:

        #     if (inner_r < 0 or outer_r <= 0):
        #         inner_r = np.abs(inner_r)
        #         outer_r = np.abs(outer_r)
        #         warnings.warn("Corrected one of the ring source radii provided are either negative or zero.", RuntimeWarning)
        #     if np.abs(inner_r) >= np.abs(outer_r):
        #         temp_r = outer_r
        #         outer_r = inner_r
        #         inner_r = temp_r
        #         warnings.warn("Inner heat source ring radius >= outer radius", RuntimeWarning)
        #     if not ((-self.Lx/2 + outer_r < center_x < self.Lx/2 - outer_r) and
        #             (-self.Ly/2 + outer_r < center_y < self.Ly/2 - outer_r)):
        #         raise ValueError("Position of heat source outside of bounds of crystal face.")
        # except ValueError as outofbounds:
        #     warnings.warn("OutOfBoundsWarning: {0}".format(outofbounds), RuntimeWarning)
        # finally:
        #     outer_pts = (self.xx-center_x)**2 + (self.yy-center_y)**2 <= outer_r**2
        #     inner_pts = (self.xx-center_x)**2 + (self.yy-center_y)**2 < inner_r**2

    def add_filter_rectangular(self, lower_left_x=-0.05, lower_left_y=-0.05, upper_right_x=0.05, upper_right_y=0.05, filter_nd=0.0):
        """Adds a rectangular obstruction to the incident beam.

        Args:
            center_x (float): X co-ordinate of the center of the heat source [cm].
            center_y (float): Y co-ordinate of the center of the heat source [cm].
            inner_r (float): Radius from the center to inner section of the heat source [cm].
            outer_r (float): Radius from the center to outer section of the heat source [cm].

        Raises:
            ValueError: If the coordinates of the position of the heat source are out of bounds
                        of crystal face.

        Attributes:
            pts (:obj:`~numpy.ndarray`): Cross sectional coordinates for which to add obstruction.

        """
        self.fseed[:][(self.xx >= lower_left_x) & (self.yy >= lower_left_y) & (self.xx <= upper_right_x) & (self.yy <= upper_right_y)] *= filter_nd
        self._fsource = self.fseed[2]
        self.fseed_mod = BeamProp2D(wavelen=self.wavelen, waist=self.waist, hdf5filename=self.hdf5filename,
                                    hdf5grp=self.hdf5grp).modulate(self.fseed[0], hdf5table=self.hdf5set_phase)

    def _convolve_beams(self, temporal=False):
        """Determines convolutions of original and phase modulated incident seed beams.

        Args:
            temporal (bool): Determine transient or steady-state normalized intensity profile of convolution.

        Todo:
            TODO: Fix complex casting in hdf5 file.
        """

        # TODO: Fix convolution
        self.fseed_convol = BeamProp2D(waist=self.waist, hdf5filename=self.hdf5filename,
                                       hdf5grp=self.hdf5grp).convolve(self.fseed[0], self.z)[1]
        #self.fseed_convol = BeamProp2D(wavelen=self.wavelen, waist=self.waist, hdf5filename=self.hdf5filename,
        #                               hdf5grp=self.hdf5grp).lg_mode(self.p, self.l, self.z, self.phi)
        #self.fseed_convol = self.fseed_convol.convolve(self.fseed, self.z)
        if temporal:
            self.fseed_mod_convol = np.zeros(np.shape(self.fseed_mod), dtype=np.dtype(float).type)
            self.fseed_mod_convol = np.asarray([BeamProp2D(waist=self.waist, hdf5filename=self.hdf5filename,
                                                hdf5grp=self.hdf5grp).convolve(self.fseed_mod[t], self.z)[1]
                                                for t in range(np.shape(self.fseed_mod)[0])], dtype=np.dtype(float).type)

            self.seed_propagation = np.zeros(np.shape(self.fseed_mod), dtype=np.dtype(complex).type)
            self.seed_propagation = np.asarray([BeamProp2D(waist=self.waist, hdf5filename=self.hdf5filename,
                                                hdf5grp=self.hdf5grp).convolve(self.fseed_mod[t], self.z)[0]
                                                for t in range(np.shape(self.fseed_mod)[0])], dtype=np.dtype(complex).type)

            with h5py.File(self.hdf5filename, 'a', libver='latest') as hf:
                grp = hf.require_group(self.hdf5grp)
                if str(self.hdf5_seed_intens) not in grp:
                    grp.create_dataset(str(self.hdf5_seed_intens), shape=(np.shape(self.fseed_mod_convol)),
                                       dtype=np.dtype(float).type, data=self.fseed_mod_convol,
                                       maxshape=(MAX_NT, MAX_NX, MAX_NY))
                else:
                    grp[str(self.hdf5_seed_intens)].resize(np.shape(self.fseed_mod_convol))
                    grp[str(self.hdf5_seed_intens)][...] = self.fseed_mod_convol
                if str(self.hdf5_seed_prop) not in grp:
                    grp.create_dataset(str(self.hdf5_seed_prop), shape=(np.shape(self.seed_propagation)),
                                       dtype=np.dtype(complex).type, data=self.seed_propagation,
                                       maxshape=(MAX_NT, MAX_NX, MAX_NY))
                else:
                    grp[str(self.hdf5_seed_prop)].resize(np.shape(self.seed_propagation))
                    grp[str(self.hdf5_seed_prop)][...] = self.seed_propagation
        else:
            self.fseed_mod_convol = np.empty(np.shape(self.fseed_mod[-1]))
            self.fseed_mod_convol = BeamProp2D(waist=self.waist, hdf5filename=self.hdf5filename,
                                               hdf5grp=self.hdf5grp).convolve(self.fseed_mod[-1], self.z)[1]

        self._vmax_src = np.max(self._fsource)
        self._vmax_evol = np.max([np.max(self._fsource), np.max(self.fseed_mod_convol)])

        #self.fseed_mod_convol = self.fseed_mod_convol.convolve(self.fseed_mod, self.z)

    def _init_anim(self):
        #self._quad2, self._quad2b, self._quad3, self._quad4, self._quad4b, self._fig = super()._init_anim()
        self._quad2c.set_data([],[])
        self._quad3.set_array([])
        self._quad4c.set_data([],[])
        self._fig.suptitle([])
        return self._quad2c, self._quad3, self._quad4c, self._fig

    def _step_anim(self, t):
        self._fig.suptitle('Beam Intensities I at t = {0:.5f}s and propagated to z = {1:.1f}cm'\
                           .format(self.clock[t], self.z), fontsize=LABEL_FONTSIZE+1.0)
        self._quad3.set_array(self.fseed_mod_convol[t].ravel())
        self._quad2c.set_data(self.xx.T[round(np.shape(self.fseed_mod_convol)[2]/2)],
                              self.fseed_mod_convol[t].T[round(np.shape(self.fseed_mod_convol)[2]/2)])
        self._quad4c.set_data(self.yy[round(np.shape(self.fseed_mod_convol)[1]/2)],
                              self.fseed_mod_convol[t][round(np.shape(self.fseed_mod_convol)[1]/2)])
        return self._quad2c, self._quad3, self._quad4c, self._fig

    def transient_profiles(self, z_dist, savepath='videos/transient_beam_profiles.mp4',
                           hdf5_seed_intens='seed_intensity', hdf5_seed_prop='seed_propagation'):
        """Generate and save transient profile as MP4 file.

        Profiles are animated at a specific distance z along the optical axis for the beam.

        Args:
            z_dist(float): Propagation distance along optical axis.
            savepath (str): Name of MP4 file to save animation to.
            hdf5_seed_intens (str): Dataset name of the HDF5 normalized seed intensity object.
            hdf5_seed_prop (str): Dataset name of the HDF5 normalized seed propagation object.
        """
        self.z = z_dist
        self.hdf5_seed_intens = str(hdf5_seed_intens)
        self.hdf5_seed_prop = str(hdf5_seed_prop)
        self._convolve_beams(True)
        super().transient_profiles(savepath)

    def steadystate_profiles(self, z_dist, savepath='images/steadystate_beam_profiles.png'):
        """Generate and save steady-state profile as PNG file.

        Profile is recorded at a specific distance z along the optical axis for the beam.

        Args:
            z_dist(float): Propagation distance along optical axis.
            savepath (str): Name of PNG file to save image to.
        """
        self.z = z_dist
        self._convolve_beams()
        super().steadystate_profiles(savepath)

class VisualizeBeam3D(Visualize3D):
    """Three-dimensional intra-crystal (gain medium) beam generation and propagation class.

    """

    def __init__(self, z=-1, hdf5filename='hdf5db/storage.db', hdf5grp='nb8', \
                 hdf5subgrp_beam='beam3d', hdf5table='intensity3d'):
        super().__init__(z, hdf5filename, hdf5grp, hdf5table=str(hdf5subgrp_beam +'/'+ hdf5table))
        self._vmin_evol, self._vmax_evol = np.min(self.fevolmap3d), np.max(self.fevolmap3d)
        self.cmap = 'gnuplot2'
        self.cmap_proj = 'gist_gray'
        #self.cmap_proj = self._custom_div_cmap()

    def _init_canvas(self):
        super()._init_canvas()
        #temp = super()._init_anim()
        self._fig.suptitle('Intensity at t = {0:.5f}'.format(self.clock[-1]),
                           fontsize=LABEL_FONTSIZE_3D+1.0)

        self._ax1.set_title('Pump-side end face beam intensity', fontsize=LABEL_FONTSIZE)
        self._quad1.set_cmap(self.cmap)

        self._ax2.set_title('Intensity at slice z = {0:.0f}'.format(self.z_ind), fontsize=LABEL_FONTSIZE)
        self.figcb.set_label(r'$\theta$')

        #self._fig.colorbar(self._quad1, cax=self.cax).set_label(r'\theta')
        #self._fig.colorbar(self._quad1, cax=self.cax).set_ticks([np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3, 2*np .pi])
        # self._fig.colorbar(self._quad1, cax=self.cax).set_ticklabels([r"$\frac{\pi}{3}$", r"$\frac{2\pi}{3}$", r"$\pi$",
        #                                                           r"$\frac{4\pi}{3}$", r"$\frac{5\pi}{3}$", r"$2\pi$"])
        # self.figcb.set_yticklabels([r"$\frac{\pi}{3}$", r"$\frac{2\pi}{3}$", r"$\pi$",
        #                             r"$\frac{4\pi}{3}$", r"$\frac{5\pi}{3}$", r"$2\pi$"])

    def _step_anim(self, t):
        temp = super()._step_anim(t)
        self._fig.suptitle(r'Beam propagation [$\theta$] at t = {0:.5f}'.format(self.clock[t]), fontsize=LABEL_FONTSIZE_3D+1.0)
        return temp

class VisualizeBeamProp3D(VisualizeBeam3D):
    """Three-dimensional extra-crystal (gain medium) beam propagation class.

    Args:
        prop_dist_zRx = Rayleigh Range Multiplication factor to propagate to.

    """

    def __init__(self, z=-1, z_prop=100, hdf5filename='hdf5db/storage.db', \
                 hdf5grp='nb8', hdf5subgrp_beam='beam3d', hdf5table='intensity3d'):
        super().__init__(z, hdf5filename, hdf5grp, hdf5subgrp_beam, hdf5table)


class VisualizeSelfLensBeam2D(VisualizeBeam2D):
    """Two-dimensional beam visualisation object for sequential self thermal lensing.

    Args:
        hdf5_seed_intens (str): Dataset name of the HDF5 group/dataset normalized seed intensity object to read.
        hdf5_seed_intens (str): Dataset name of the HDF5 group/dataset seed object to read.

    Attributes:
        fseed_t (:obj:`~numpy.ndarray`): Temporal incident seed beam profile.
        _fsource_t (:obj:`~numpy.ndarray`): Temporal incident source profile.

    Todo:
        * Source / seed limits (normalization) when dealing with tight focusing. I.e. peak will be greater than 1.
        * Tidy up and restructure this, convolve and modulate.

    """
    def __init__(self, hdf5filename='hdf5db/storage.hdf5', hdf5grp='nb1', hdf5_seed_intens='nb1/seed_intensity',
                 hdf5_seed_prop='nb1/seed_propagation'):
        super().__init__(hdf5filename=hdf5filename, hdf5grp=hdf5grp)
        with h5py.File(self.hdf5filename, 'r', libver='latest') as hf:
            #grp = hf.require_group(str(self.hdf5grp))
            self.fseed_t = [hf[str(hdf5_seed_prop)][:],
                            #hf[str(hdf5_seed_prop)][:]/np.sqrt(np.sum(np.abs(hf[str(hdf5_seed_prop)][:])**2)),
                            hf[str(hdf5_seed_prop)][:],
                            hf[str(hdf5_seed_intens)][:]]

        self._fsource_t = self.fseed_t[2]
        self._fsource = self._fsource_t[-1]
        self.fseed = [self.fseed_t[i][-1] for i in range(3)]
        self.fseed_mod = BeamProp2D(hdf5filename=self.hdf5filename, hdf5grp=self.hdf5grp).modulate(self.fseed_t[1])

    def _init_canvas(self):
        super()._init_canvas()
        #self._ax1.set_title('Incident seed profile at new z=0', fontsize=LABEL_FONTSIZE)
        self._ax2.legend(['Seed', 'Propagated Seed', 'Self Modulated'], loc='upper right', prop={'size':AXIS_TICK_FONTSIZE-1})

        self._ax4.legend(['Seed', 'Propagated Seed', 'Self Modulated'], loc='upper right', prop={'size':AXIS_TICK_FONTSIZE-1})

    def _init_anim(self):
        self._quad1.set_array([])
        self._quad2.set_data([],[])
        self._quad2b.set_data([],[])
        self._quad4.set_data([],[])
        self._quad4b.set_data([],[])
        return self._quad1, self._quad2, self._quad2b, self._quad4, self._quad4b, super()._init_anim()

    def _step_anim(self, t):
        self.fseed_convol = BeamProp2D(waist=self.waist, hdf5filename=self.hdf5filename,
                                       hdf5grp=self.hdf5grp).convolve(self.fseed_t[0][t], self.z)[1]
        self._quad1.set_array(self.fseed_t[2][t].ravel())
        self._quad2.set_data(self.xx.T[round(np.shape(self.fseed[2])[1]/2)],
                             self.fseed_t[2][t].T[round(np.shape(self.fseed[2])[1]/2)])
        self._quad2b.set_data(self.xx.T[round(np.shape(self.fseed_convol)[1]/2)],
                              self.fseed_convol.T[round(np.shape(self.fseed_convol)[1]/2)])
        self._quad4.set_data(self.yy[round(np.shape(self.fseed[2])[0]/2)],
                             self.fseed_t[2][t][round(np.shape(self.fseed[2])[0]/2)])
        self._quad4b.set_data(self.yy[round(np.shape(self.fseed_convol)[0]/2)],
                              self.fseed_convol[round(np.shape(self.fseed_convol)[0]/2)])
        return self._quad1, self._quad2, self._quad2b, self._quad4, self._quad4b, super()._step_anim(t)



#  LocalWords:  plt fevolmap
