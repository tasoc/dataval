#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plotting utilities.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import logging
import copy
import numpy as np
from bottleneck import allnan, anynan
import matplotlib
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy.visualization as viz

# Change to a non-GUI backend since this
# should be able to run on a cluster:
plt.switch_backend('Agg')

#--------------------------------------------------------------------------------------------------
def plots_interactive(backend=('Qt5Agg', 'MacOSX', 'Qt4Agg', 'GTK3Agg', 'Qt5Cairo', 'GTK3Cairo', 'TkAgg')):
	"""
	Change plotting to using an interactive backend.

	Parameters:
		backend (str or list): Backend to change to. If not provided, will try different
			interactive backends and use the first one that works.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)
	logger.debug("Valid interactive backends: %s", matplotlib.rcsetup.interactive_bk)

	if isinstance(backend, str):
		backend = [backend]

	for bckend in backend:
		if bckend not in matplotlib.rcsetup.interactive_bk:
			logger.debug("Interactive backend '%s' is not found", bckend)
			continue

		# Try to change the backend, and catch errors
		# it it didn't work:
		try:
			plt.switch_backend(bckend)
		except (ModuleNotFoundError, ImportError):
			pass
		else:
			break

#--------------------------------------------------------------------------------------------------
def plots_noninteractive():
	"""
	Change plotting to using a non-interactive backend, which can e.g. be used on a cluster.
	Will set backend to 'Agg'.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""
	plt.switch_backend('Agg')

#--------------------------------------------------------------------------------------------------
def plot_image(image, ax=None, scale='log', cmap=None, origin='lower', xlabel=None,
	ylabel=None, cbar=None, clabel='Flux ($e^{-}s^{-1}$)', cbar_ticks=None,
	cbar_ticklabels=None, cbar_pad=None, cbar_size='5%', title=None,
	percentile=95.0, vmin=None, vmax=None, offset_axes=None, color_bad='k', **kwargs):
	"""
	Utility function to plot a 2D image.

	Parameters:
		image (2d array): Image data.
		ax (matplotlib.pyplot.axes, optional): Axes in which to plot.
			Default (None) is to use current active axes.
		scale (str or :py:class:`astropy.visualization.ImageNormalize` object, optional):
			Normalization used to stretch the colormap.
			Options: ``'linear'``, ``'sqrt'``, ``'log'``, ``'asinh'``, ``'histeq'``, ``'sinh'``
			and ``'squared'``.
			Can also be a :py:class:`astropy.visualization.ImageNormalize` object.
			Default is ``'log'``.
		origin (str, optional): The origin of the coordinate system.
		xlabel (str, optional): Label for the x-axis.
		ylabel (str, optional): Label for the y-axis.
		cbar (string, optional): Location of color bar.
			Choises are ``'right'``, ``'left'``, ``'top'``, ``'bottom'``.
			Default is not to create colorbar.
		clabel (str, optional): Label for the color bar.
		cbar_size (float, optional): Fractional size of colorbar compared to axes. Default=0.03.
		cbar_pad (float, optional): Padding between axes and colorbar.
		title (str or None, optional): Title for the plot.
		percentile (float, optional): The fraction of pixels to keep in color-trim.
			If single float given, the same fraction of pixels is eliminated from both ends.
			If tuple of two floats is given, the two are used as the percentiles.
			Default=95.
		cmap (matplotlib colormap, optional): Colormap to use. Default is the ``Blues`` colormap.
		vmin (float, optional): Lower limit to use for colormap.
		vmax (float, optional): Upper limit to use for colormap.
		color_bad (str, optional): Color to apply to bad pixels (NaN). Default is black.
		kwargs (dict, optional): Keyword arguments to be passed to :py:func:`matplotlib.pyplot.imshow`.

	Returns:
		:py:class:`matplotlib.image.AxesImage`: Image from returned
			by :py:func:`matplotlib.pyplot.imshow`.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)

	# Backward compatible settings:
	make_cbar = kwargs.pop('make_cbar', None)
	if make_cbar:
		raise FutureWarning("'make_cbar' is deprecated. Use 'cbar' instead.")
		if not cbar:
			cbar = make_cbar

	# Special treatment for boolean arrays:
	if isinstance(image, np.ndarray) and image.dtype == 'bool':
		if vmin is None: vmin = 0
		if vmax is None: vmax = 1
		if cbar_ticks is None: cbar_ticks = [0, 1]
		if cbar_ticklabels is None: cbar_ticklabels = ['False', 'True']

	# Calculate limits of color scaling:
	interval = None
	if vmin is None or vmax is None:
		if allnan(image):
			logger.warning("Image is all NaN")
			vmin = 0
			vmax = 1
			if cbar_ticks is None:
				cbar_ticks = []
			if cbar_ticklabels is None:
				cbar_ticklabels = []
		elif isinstance(percentile, (list, tuple, np.ndarray)):
			interval = viz.AsymmetricPercentileInterval(percentile[0], percentile[1])
		else:
			interval = viz.PercentileInterval(percentile)

	# Create ImageNormalize object with extracted limits:
	if scale in ('log', 'linear', 'sqrt', 'asinh', 'histeq', 'sinh', 'squared'):
		if scale == 'log':
			stretch = viz.LogStretch()
		elif scale == 'linear':
			stretch = viz.LinearStretch()
		elif scale == 'sqrt':
			stretch = viz.SqrtStretch()
		elif scale == 'asinh':
			stretch = viz.AsinhStretch()
		elif scale == 'histeq':
			stretch = viz.HistEqStretch(image[np.isfinite(image)])
		elif scale == 'sinh':
			stretch = viz.SinhStretch()
		elif scale == 'squared':
			stretch = viz.SquaredStretch()

		# Create ImageNormalize object. Very important to use clip=False if the image contains
		# NaNs, otherwise NaN points will not be plotted correctly.
		norm = viz.ImageNormalize(
			data=image[np.isfinite(image)],
			interval=interval,
			vmin=vmin,
			vmax=vmax,
			stretch=stretch,
			clip=not anynan(image))

	elif isinstance(scale, (viz.ImageNormalize, matplotlib.colors.Normalize)):
		norm = scale
	else:
		raise ValueError("scale {} is not available.".format(scale))

	if offset_axes:
		extent = (offset_axes[0]-0.5, offset_axes[0] + image.shape[1]-0.5, offset_axes[1]-0.5, offset_axes[1] + image.shape[0]-0.5)
	else:
		extent = (-0.5, image.shape[1]-0.5, -0.5, image.shape[0]-0.5)

	if ax is None:
		ax = plt.gca()

	# Set up the colormap to use. If a bad color is defined,
	# add it to the colormap:
	if cmap is None:
		cmap = copy.copy(plt.get_cmap('Blues'))
	elif isinstance(cmap, str):
		cmap = copy.copy(plt.get_cmap(cmap))

	if color_bad:
		cmap.set_bad(color_bad, 1.0)

	# Plotting the image using all the settings set above:
	im = ax.imshow(
		image,
		cmap=cmap,
		norm=norm,
		origin=origin,
		extent=extent,
		interpolation='nearest',
		**kwargs)

	if xlabel is not None:
		ax.set_xlabel(xlabel)
	if ylabel is not None:
		ax.set_ylabel(ylabel)
	if title is not None:
		ax.set_title(title)
	ax.set_xlim([extent[0], extent[1]])
	ax.set_ylim([extent[2], extent[3]])

	if cbar:
		colorbar(im,
			ax=ax,
			loc=cbar,
			size=cbar_size,
			pad=cbar_pad,
			label=clabel,
			ticks=cbar_ticks,
			ticklabels=cbar_ticklabels)

	# Settings for ticks:
	integer_locator = MaxNLocator(nbins=10, integer=True)
	ax.xaxis.set_major_locator(integer_locator)
	ax.xaxis.set_minor_locator(integer_locator)
	ax.yaxis.set_major_locator(integer_locator)
	ax.yaxis.set_minor_locator(integer_locator)
	ax.tick_params(which='both', direction='out', pad=5)
	ax.xaxis.tick_bottom()
	ax.yaxis.tick_left()

	return im

#--------------------------------------------------------------------------------------------------
def plot_image_fit_residuals(fig, image, fit, residuals=None, percentile=95.0):
	"""
	Make a figure with three subplots showing the image, the fit and the
	residuals. The image and the fit are shown with logarithmic scaling and a
	common colorbar. The residuals are shown with linear scaling and a separate
	colorbar.

	Parameters:
		fig (fig object): Figure object in which to make the subplots.
		image (2D array): Image numpy array.
		fit (2D array): Fitted image numpy array.
		residuals (2D array, optional): Fitted image subtracted from image numpy array.

	Returns:
		list: List with Matplotlib subplot axes objects for each subplot.
	"""

	if residuals is None:
		residuals = image - fit

	# Calculate common normalization for the first two subplots:
	vmin_image, vmax_image = viz.PercentileInterval(percentile).get_limits(image)
	vmin_fit, vmax_fit = viz.PercentileInterval(percentile).get_limits(fit)
	vmin = np.nanmin([vmin_image, vmin_fit])
	vmax = np.nanmax([vmax_image, vmax_fit])
	norm = viz.ImageNormalize(vmin=vmin, vmax=vmax, stretch=viz.LogStretch())

	# Add subplot with the image:
	ax1 = fig.add_subplot(131)
	im1 = plot_image(image, ax=ax1, scale=norm, cbar=None, title='Image')

	# Add subplot with the fit:
	ax2 = fig.add_subplot(132)
	plot_image(fit, ax=ax2, scale=norm, cbar=None, title='PSF fit')

	# Calculate the normalization for the third subplot:
	vmin, vmax = viz.PercentileInterval(percentile).get_limits(residuals)
	v = np.max(np.abs([vmin, vmax]))

	# Add subplot with the residuals:
	ax3 = fig.add_subplot(133)
	im3 = plot_image(residuals, ax=ax3, scale='linear', cmap='seismic', vmin=-v, vmax=v, cbar=None, title='Residuals')

	# Make the common colorbar for image and fit subplots:
	cbar_ax12 = fig.add_axes([0.125, 0.2, 0.494, 0.03])
	fig.colorbar(im1, cax=cbar_ax12, orientation='horizontal')

	# Make the colorbar for the residuals subplot:
	cbar_ax3 = fig.add_axes([0.7, 0.2, 0.205, 0.03])
	fig.colorbar(im3, cax=cbar_ax3, orientation='horizontal')

	# Add more space between subplots:
	plt.subplots_adjust(wspace=0.4, hspace=0.4)

	return [ax1, ax2, ax3]

#--------------------------------------------------------------------------------------------------
def colorbar(im, ax=None, loc='right', pad=None, size='5%', label=None, ticks=None, ticklabels=None):
	"""
	Draw colorbar next to the given axes.

	Returns:
		:class:`matplotlib.colorbar.Colorbar`: Colorbar handle.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	if ax is None:
		ax = plt.gca()
	fig = ax.figure

	# Create new colorbar axes:
	divider = make_axes_locatable(ax)
	if loc == 'top':
		pad = 0.05 if pad is None else pad
		cax = divider.append_axes('top', size=size, pad=pad)
		orientation = 'horizontal'
	elif loc == 'bottom':
		pad = 0.35 if pad is None else pad
		cax = divider.append_axes('bottom', size=size, pad=pad)
		orientation = 'horizontal'
	elif loc == 'left':
		pad = 0.35 if pad is None else pad
		cax = divider.append_axes('left', size=size, pad=pad)
		orientation = 'vertical'
	else:
		pad = 0.05 if pad is None else pad
		cax = divider.append_axes('right', size=size, pad=pad)
		orientation = 'vertical'

	cb = fig.colorbar(im, cax=cax, orientation=orientation)

	if loc == 'top':
		cax.xaxis.set_ticks_position('top')
		cax.xaxis.set_label_position('top')
	elif loc == 'left':
		cax.yaxis.set_ticks_position('left')
		cax.yaxis.set_label_position('left')

	if label is not None:
		cb.set_label(label)
	if ticks is not None:
		cb.set_ticks(ticks)
	if ticklabels is not None:
		cb.set_ticklabels(ticklabels)

	#cax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
	#cax.yaxis.set_minor_locator(matplotlib.ticker.AutoLocator())
	cax.tick_params(which='both', direction='out', pad=5)

	cb.set_alpha(1)
	cb.draw_all()

	return cb
