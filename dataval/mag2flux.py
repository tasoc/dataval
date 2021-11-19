#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os.path
import numpy as np
from bottleneck import nansum
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import minimize
from .plots import plt, colorbar
from matplotlib import colors
from matplotlib.ticker import MultipleLocator
from .quality import DatavalQualityFlags

#--------------------------------------------------------------------------------------------------
def mag2flux(dval):
	"""
	Function to plot flux values from apertures against the stellar TESS magnitudes,
	and determine coefficient describing the relation

	Parameters:
		dval (:class:`DataValidation`): Data Validation object.

	.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
	"""

	logger = logging.getLogger('dataval')
	logger.info('Plotting Magnitude to Flux conversion...')

	# Get the maximum median flux to be used to set the plot limits:
	max_mean_flux = dval.search_database(
		select=['MAX(mean_flux) AS maxflux'],
		search=["method_used='aperture'"])
	max_mean_flux = max_mean_flux[0]['maxflux']

	zp_grid = np.linspace(19.0, 22.0, 300)
	mags = np.linspace(dval.tmag_limits[0], dval.tmag_limits[1], 300)

	# The interpolation is linear *in log*
	xmin = np.array([0, 1.5, 9, 12.6, 13, 14, 15, 16, 17, 18, 19])
	ymin = np.array([8e7, 1.8e7, 12500, 250, 59, 5, 1, 1, 1, 1, 1])
	min_bound_log = InterpolatedUnivariateSpline(xmin, np.log10(ymin), k=1, ext=3)
	min_bound = lambda x: np.clip(10**(min_bound_log(x)), 1, None) # noqa: E731

	norm = colors.Normalize(vmin=0, vmax=1)
	fig2, ax2 = plt.subplots()
	ax2.set_xlabel('Zeropoint')
	ax2.set_ylabel(r'$\chi^2$')

	chi2lines = []
	for cadence in dval.cadences:

		fig1, ax1 = plt.subplots()
		fig1.subplots_adjust(left=0.1, wspace=0.2, top=0.94, bottom=0.155, right=0.91)

		star_vals = dval.search_database(
			select=[
				'todolist.priority',
				'todolist.tmag',
				'mean_flux',
				'contamination'
			],
			search=["method_used='aperture'", f'cadence={cadence:d}'])

		if not star_vals:
			logger.debug("No targets with cadence=%d", cadence)
			continue

		pri = np.array([star['priority'] for star in star_vals], dtype='int64')
		tmags = np.array([star['tmag'] for star in star_vals], dtype='float64')
		meanfluxes = np.array([star['mean_flux'] for star in star_vals], dtype='float64')
		contam = np.array([star['contamination'] for star in star_vals], dtype='float64')

		# Plot median fluxes as a function of Tmag, color-coded by the contamination:
		# Do a random permutaion just to avoid points of similar color to "cluster":
		perm = dval.random_state.permutation(len(tmags))
		im = ax1.scatter(tmags[perm], meanfluxes[perm], c=contam[perm],
			marker='o',
			norm=norm,
			cmap=plt.get_cmap('PuOr'),
			alpha=0.1,
			rasterized=True)

		logger.info('Optimising coefficient of relation...')

		# Create chi2 function to be minimized:
		with np.errstate(invalid='ignore'):
			idx = np.isfinite(meanfluxes) & np.isfinite(tmags) & (contam < 0.15)

		def chi2(c):
			return np.log10(nansum(( (meanfluxes[idx] - 10**(-0.4*(tmags[idx] - c))) / (contam[idx]+1) )**2))

		# Calculate chi2 on grid of zeropoints:
		zc = np.array([chi2(c) for c in zp_grid])

		# Minimize around the minimum value from the grid:
		cc = minimize(chi2, np.min(zc), method='Nelder-Mead', options={'disp': False})
		logger.info('Optimisation terminated successfully? %s', cc.success)
		logger.info('Zeropoint (cadence=%ds) is found to be %1.4f', cadence, cc.x)

		# Create plot of chi2 values:
		im2 = ax2.plot(zp_grid, zc, ls='-', label=f'{cadence:d}s cadence')
		ax2.axvline(x=cc.x, c=im2[0].get_color(), ls='--')
		chi2lines.append(im2)

		#fig3 = plt.figure(figsize=(15, 5))
		#fig3.subplots_adjust(left=0.14, wspace=0.3, top=0.94, bottom=0.155, right=0.96)
		#ax31 = fig3.add_subplot(121)
		#ax32 = fig3.add_subplot(122)

		#d1 = meanfluxes[idx1]/(10**(-0.4*(tmags[idx1] - cc.x))) - 1
		#d2 = meanfluxes[idx2]/(10**(-0.4*(tmags[idx2] - cc2.x))) - 1
		#ax31.scatter(tmags[idx1], np.abs(d1), alpha=0.1)
		#ax31.scatter(tmags[idx2], np.abs(d2), color='k', alpha=0.1)
		#ax31.axhline(y=0, ls='--', color='k')

		#bin_means1, bin_edges1, binnumber1 = binning(tmags[idx1], np.abs(d1), statistic='median', bins=15, range=(1.5,15))
		#bin_width1 = (bin_edges1[1] - bin_edges1[0])
		#bin_centers1 = bin_edges1[1:] - bin_width1/2

		#bin_means2, bin_edges2, binnumber2 = binning(tmags[idx2], np.abs(d2), statistic='median', bins=15, range=(1.5,15))
		#bin_width2 = (bin_edges2[1] - bin_edges2[0])
		#bin_centers2 = bin_edges2[1:] - bin_width2/2

		#ax31.scatter(bin_centers1, 1.4826*bin_means1, marker='o', color='r')
		#ax31.scatter(bin_centers1, 1.4826*3*bin_means1, marker='.', color='r')
		#ax31.plot(bin_centers1, 1.4826*3*bin_means1, color='r')

		#ax31.scatter(bin_centers2, 1.4826*bin_means2, marker='o', color='g')
		#ax31.scatter(bin_centers2, 1.4826*3*bin_means2, marker='.', color='g')
		#ax31.plot(bin_centers2, 1.4826*3*bin_means2, color='g')

		# Add line with best fit, and the minimim bound:
		ax1.plot(mags, np.clip(10**(-0.4*(mags - cc.x)), 0, None), color='k', ls='--')
		ax1.plot(mags, min_bound(mags), 'r-')

		ax1.set_yscale('log')
		ax1.set_xlim(dval.tmag_limits[1], dval.tmag_limits[0])
		ax1.set_ylim(0.5, 2*max_mean_flux)
		ax1.set_xlabel('TESS magnitude')
		ax1.set_ylabel('Median flux (e$^-$/s)')
		ax1.xaxis.set_major_locator(MultipleLocator(2))
		ax1.xaxis.set_minor_locator(MultipleLocator(1))
		colorbar(im, ax=ax1, label='Contamination')

		fig1.savefig(os.path.join(dval.outfolder, f'mag2flux_c{cadence:04d}'))
		if not dval.show:
			plt.close(fig1)

		# Assign validation bits, for both FFI and TPF
		dv = np.zeros_like(pri, dtype='int32')
		dv[meanfluxes < min_bound(tmags)] |= DatavalQualityFlags.MagVsFluxLow
		dv[~np.isfinite(meanfluxes) | (meanfluxes <= 0)] |= DatavalQualityFlags.InvalidFlux
		dval.update_dataval(pri, dv)

	if not chi2lines:
		plt.close(fig2)
	else:
		ax2.legend(loc='upper left')
		fig2.savefig(os.path.join(dval.outfolder, 'mag2flux_optimize'))

		if not dval.show:
			plt.close(fig2)
