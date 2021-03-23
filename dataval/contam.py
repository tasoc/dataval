#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os.path
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from .plots import plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MultipleLocator
from .quality import DatavalQualityFlags

#--------------------------------------------------------------------------------------------------
def contam(dval):
	"""
	Function to plot the contamination against the stellar TESS magnitudes

	.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger('dataval')
	logger.info('Plotting Contamination vs. Magnitude...')

	xmax = np.arange(0, 21, 1)
	ymax = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.12, 0.2, 0.3, 0.45, 0.6, 0.7, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
	cont_vs_mag = InterpolatedUnivariateSpline(xmax, ymax, k=1, ext=3)

	for cadence in dval.cadences:

		fig, ax = plt.subplots() # plt.figaspect(2.0)
		fig.subplots_adjust(left=0.145, wspace=0.3, top=0.945, bottom=0.145, right=0.975)

		# Search database for all targets processed with aperture photometry:
		star_vals = dval.search_database(
			select=[
				'todolist.priority',
				'todolist.sector',
				'todolist.tmag',
				'contamination'
			],
			search=["method_used='aperture'", f'cadence={cadence:d}'])

		rgba_color = 'k'
		if dval.color_by_sector:
			sec = np.array([star['sector'] for star in star_vals], dtype='int32')
			sectors = list(set(sec))
			if len(sectors) > 1:
				norm = colors.Normalize(vmin=1, vmax=len(sectors))
				scalarMap = ScalarMappable(norm=norm, cmap=plt.get_cmap('Set1') )
				rgba_color = np.array([scalarMap.to_rgba(s) for s in sec])

		pri = np.array([star['priority'] for star in star_vals], dtype='int64')
		tmags = np.array([star['tmag'] for star in star_vals], dtype='float64')
		contam = np.array([star['contamination'] for star in star_vals], dtype='float64')

		# Indices for plotting
		idx_invalid = np.isnan(contam)
		with np.errstate(invalid='ignore'): # We know that some cont may be NaN
			idx_high = (contam > 1)
			idx_low = (contam <= 1)

		# Remove nan contaminations (should be only from Halo targets)
		contam[idx_high] = 1.1
		contam[idx_invalid] = 1.2

		# Plot individual contamination points
		ax.scatter(tmags[idx_low], contam[idx_low],
			marker='o',
			facecolors=rgba_color,
			color=rgba_color,
			alpha=0.1,
			rasterized=True)

		ax.scatter(tmags[idx_high], contam[idx_high],
			marker='o',
			facecolors='None',
			color=rgba_color,
			alpha=0.9)

		# Plot invalid points:
		ax.scatter(tmags[idx_invalid], contam[idx_invalid],
			marker='o',
			facecolors='None',
			color='r',
			alpha=0.9)

		# Indices for finding validation limit
		#idx_low = (contam <= 1)
		# Compute median-bin curve
		#bin_means, bin_edges, binnumber = binning(tmags[idx_low], contam[idx_low], statistic='median', bins=15, range=(np.nanmin(tmags),np.nanmax(tmags)))
		#bin_width = (bin_edges[1] - bin_edges[0])
		#bin_centers = bin_edges[1:] - bin_width/2

		# Plot median-bin curve (1 and 5 times standadised MAD)
		#ax1.scatter(bin_centers, 1.4826*bin_means, marker='o', color='r')
		#ax1.scatter(bin_centers, 1.4826*5*bin_means, marker='.', color='r')
		#ax.plot(xmax, ymax, marker='.', color='r', ls='-')

		mags = np.linspace(dval.tmag_limits[0], dval.tmag_limits[1], 100)
		ax.plot(mags, cont_vs_mag(mags), 'r-')

		ax.set_xlim(dval.tmag_limits)
		ax.set_ylim([-0.05, 1.3])

		ax.axhline(y=0, ls='--', color='k', zorder=-1)
		ax.axhline(y=1.1, ls=':', color='k', zorder=-1)
		ax.axhline(y=1.2, ls=':', color='r', zorder=-1)
		ax.axhline(y=1, ls=':', color='k', zorder=-1)
		ax.set_xlabel('TESS magnitude')
		ax.set_ylabel('Contamination')

		ax.xaxis.set_major_locator(MultipleLocator(2))
		ax.xaxis.set_minor_locator(MultipleLocator(1))
		ax.yaxis.set_major_locator(MultipleLocator(0.2))
		ax.yaxis.set_minor_locator(MultipleLocator(0.1))

		###########

		fig.savefig(os.path.join(dval.outfolder, f'contam_c{cadence:04d}'))
		if not dval.show:
			plt.close(fig)

		# Assign validation bits
		dv = np.zeros_like(pri, dtype='int32')
		dv[contam >= 1] |= DatavalQualityFlags.InvalidContamination
		dv[(contam > cont_vs_mag(tmags)) & (contam < 1)] |= DatavalQualityFlags.ContaminationHigh
		dval.update_dataval(pri, dv)
