#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os.path
import numpy as np
from scipy.stats import binned_statistic
from scipy.interpolate import InterpolatedUnivariateSpline
from .plots import plt, colorbar
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator
from .quality import DatavalQualityFlags

#--------------------------------------------------------------------------------------------------
def pixinaperture(dval):
	"""
	Function to plot number of pixels in determined apertures against the stellar
	TESS magnitudes.

	.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
	"""

	logger = logging.getLogger('dataval')
	logger.info('Plotting Pixels in aperture vs. Magnitude...')

	mags = np.linspace(dval.tmag_limits[0], dval.tmag_limits[1], 200)

	norm = Normalize(vmin=0, vmax=1)

	for cadence in dval.cadences:

		fig, ax1 = plt.subplots()
		fig.subplots_adjust(left=0.1, wspace=0.2, top=0.94, bottom=0.155, right=0.91)

		star_vals = dval.search_database(
			select=[
				'todolist.priority',
				'todolist.datasource',
				'todolist.sector',
				'todolist.tmag',
				'diagnostics.mask_size',
				'diagnostics.contamination',
				'diagnostics.errors'
			],
			search=f'cadence={cadence:d}')

		if not star_vals:
			continue

		#if self.color_by_sector:
		#	sec = np.array([star['sector'] for star in star_vals], dtype=int)
		#	sectors = np.array(list(set(sec)))
		#	if len(sectors)>1:
		#		norm = colors.Normalize(vmin=1, vmax=len(sectors))
		#		scalarMap = cmx.ScalarMappable(norm=norm, cmap=plt.get_cmap('Set1') )
		#		rgba_color = np.array([scalarMap.to_rgba(s) for s in sec])
		#	else:
		#		rgba_color = 'k'
		#else:
		#	rgba_color = 'k'

		pri = np.array([star['priority'] for star in star_vals], dtype='int64')
		tmags = np.array([star['tmag'] for star in star_vals], dtype='float64')
		masksizes = np.array([star['mask_size'] for star in star_vals], dtype='float64')
		contam = np.array([star['contamination'] for star in star_vals], dtype='float64')

		minimal_mask_used = np.array([False if star['errors'] is None else ('Using minimum aperture.' in star['errors']) for star in star_vals], dtype='bool')

		#dataval = np.zeros_like(pri, dtype='int32')
		#if self.doval:
		#	dataval = np.array([star['dataval'] for star in star_vals], dtype='int32')

		#cats = {}
		#for cams in list(set(camera)):
		#	cats[cams] = {}
		#	for jj in range(4):
		#		cam_file = os.path.join(self.input_folders[0], 'catalog_camera' + str(int(cams)) +'_ccd' + str(jj+1) + '.sqlite')
		#		with contextlib.closing(sqlite3.connect(cam_file)) as conn:
		#		with sqlite3.connect(cam_file) as conn:
		#			conn.row_factory = sqlite3.Row
		#			cursor = conn.cursor()

		#			cats[cams][str(jj+1)] = cursor

		#cams_cen = {}
		#cams_cen['1','ra'] = 324.566998914166
		#cams_cen['1','decl'] = -33.172999301379
		#cams_cen['2','ra'] = 338.5766
		#cams_cen['2','decl'] = -55.0789
		#cams_cen['3','ra'] = 19.4927
		#cams_cen['3','decl'] = -71.9781
		#cams_cen['4','ra'] = 90.0042
		#cams_cen['4','decl'] = -66.5647

		#dists = np.zeros(len(tics))
		#for ii, tic in enumerate(tics):
		#	query = "SELECT ra,decl FROM catalog where starid=%s" %tic
		#	# Ask the database: status=1
		#	cc = cats[camera[ii]][str(ccds[ii])]
		#	cc.execute(query)
		#	star_vals = cc.fetchone()
		#	dists[ii] = sphere_distance(star_vals['ra'], star_vals['decl'], cams_cen[camera[ii],'ra'], cams_cen[camera[ii],'decl'])

		contam[np.isnan(contam)] = 0

		# Get rid of stupid halo targets
		masksizes[np.isnan(masksizes)] = 0

		perm = dval.random_state.permutation(len(tmags))
		im = ax1.scatter(tmags[perm], masksizes[perm], c=contam[perm],
			marker='o',
			norm=norm,
			cmap=plt.get_cmap('PuOr'),
			alpha=0.5,
			rasterized=True)

		# Compute median-bin curve
		bin_means, bin_edges, binnumber = binned_statistic(tmags, masksizes, statistic='median', bins=15, range=(np.nanmin(tmags),np.nanmax(tmags)))
		bin_width = (bin_edges[1] - bin_edges[0])
		bin_centers = bin_edges[1:] - bin_width/2
		ax1.scatter(bin_centers, bin_means, marker='o', color='r')

		#normed0 = masksizes[idx_lc]/med_vs_mag(tmags[idx_lc])
		#normed1 = masksizes[idx_lc]-pix_vs_mag(tmags[idx_lc])

		#bin_mad, bin_edges, binnumber = binning(tmags[idx_lc], np.abs(normed1), statistic='median', bins=15, range=(np.nanmin(tmags[idx_lc]),np.nanmax(tmags[idx_lc])))
		#bin_width = (bin_edges[1] - bin_edges[0]);		bin_centers = bin_edges[1:] - bin_width/2
		#bin_var = bin_mad*1.4826
		#var_vs_mag = INT.InterpolatedUnivariateSpline(bin_centers, bin_var)

		#plt.figure()
		#plt.scatter(tmags[idx_lc], normed1/var_vs_mag(tmags[idx_lc]), color=rgba_color[idx_lc], alpha=0.5)

		#plt.figure()
		#red = masksizes[idx_lc]/pix_vs_mag(tmags[idx_lc])
		#bin_means, bin_edges, binnumber = binning(contam[idx_lc], red, statistic='median', bins=15, range=(np.nanmin(contam[idx_lc]),1))
		#bin_width = (bin_edges[1] - bin_edges[0]);		bin_centers = bin_edges[1:] - bin_width/2
		#plt.scatter(contam[idx_lc], red, alpha=0.1, color='k')
		#plt.scatter(bin_centers, bin_means, color='r')

		# Plot median-bin curve (1 and 5 times standadised MAD)
		#ax1.scatter(bin_centers, 1.4826*5*bin_means, marker='.', color='r')
		#print(masksizes[(tmags>15) & (source != 'ffi')])
		#print(np.max(tmags[idx_lc]))
		#print(bin_centers, bin_means)

		#print(masksizes[idx_lc])
		#print(any(np.isnan(masksizes[idx_lc])))
		#print(pix_vs_mag(tmags[idx_lc]))
		#diff = np.abs(masksizes[idx_lc] - pix_vs_mag(tmags[idx_lc]))

		#fig00=plt.figure()
		#ax00=fig00.add_subplot(111)
		#d = masksizes[idx_lc] - pix_vs_mag(tmags[idx_lc])
		#ax00.hist(d, bins=500)
		#ax00.axvline(x=np.percentile(d, 99.9), c='k')
		#ax00.axvline(x=np.percentile(d, 0.1), c='k')

		# The interpolations are linear *in log space* which is the reason
		# for the stange lambda wrappers in the following code:
		if cadence == 1800:
			# Minimum bound on FFI data

			xmin = np.array([0, 2, 2.7, 3.55, 4.2, 4.8, 5.5, 6.8, 7.6, 8.4, 9.1, 10, 10.5, 11, 11.5, 11.6, 16])
			ymin = np.array([2600, 846, 526, 319, 238, 159, 118, 62, 44, 32, 23, 15.7, 11.15, 8, 5, 4, 4])
			#min_bound = InterpolatedUnivariateSpline(xmin, ymin, k=1)
			min_bound_log = InterpolatedUnivariateSpline(xmin, np.log10(ymin), k=1, ext=3)
			min_bound = lambda x: 10**(min_bound_log(x))

			xmax = np.array([0, 2, 2.7, 3.55, 4.2, 4.8, 5.5, 6.8, 7.6, 8.4, 9.1, 10, 10.5, 11, 11.5, 12, 12.7, 13.3, 14, 14.5, 15, 16])
			ymax = np.array([10000, 3200, 2400, 1400, 1200, 900, 800, 470, 260, 200, 170, 130, 120, 100, 94, 86, 76, 67, 59, 54, 50, 50])
			#max_bound = InterpolatedUnivariateSpline(xmax, ymax, k=1)
			max_bound_log = InterpolatedUnivariateSpline(xmax, np.log10(ymax), k=1, ext=3)
			max_bound = lambda x: 10**(max_bound_log(x))

		else:
			# Minimum bound on TPF data
			xmin = np.array([0, 2, 2.7, 3.55, 4.2, 4.8, 5.5, 6.8, 7.6, 8.4, 9.1, 10, 10.5, 11, 11.5, 11.6, 16, 19])
			ymin = np.array([220, 200, 130, 70, 55, 43, 36, 30, 27, 22, 16, 10, 8, 6, 5, 4, 4, 4])
			min_bound_log = InterpolatedUnivariateSpline(xmin, np.log10(ymin), k=1, ext=3)
			min_bound = lambda x: 10**(min_bound_log(x))
			max_bound = None

		# Plot limits:
		ax1.plot(mags, min_bound(mags), 'r-')
		if max_bound is not None:
			ax1.plot(mags, max_bound(mags), 'r-')

		#idx_lc2 = (source == 'ffi') & (masksizes>4) & (tmags>8)
		#idx_sort = np.argsort(tmags[idx_lc2])
		#perfilt95 = filt.percentile_filter(masksizes[idx_lc2][idx_sort], 99.8, size=1000)
		#perfilt95 = filt.uniform_filter1d(perfilt95, 5000)
		#perfilt95 = filt.gaussian_filter1d(perfilt95, 10000)
		#perfilt05 = filt.percentile_filter(masksizes[idx_lc2][idx_sort], 0.2, size=1000)
		#perfilt05 = filt.uniform_filter1d(perfilt05, 5000)
		#perfilt05 = filt.gaussian_filter1d(perfilt05, 10000)

		#ax1.plot(tmags[idx_lc2][idx_sort], perfilt95, color='m')
		#ax1.plot(tmags[idx_lc2][idx_sort], perfilt05, color='m')
		#print(tmags[idx_lc2][idx_sort])
		#perh_vs_mag = INT.interp1d(tmags[idx_lc2][idx_sort], perfilt95)
		#perl_vs_mag = INT.interp1d(tmags[idx_lc2][idx_sort], perfilt05)

		#ticsh = tics[idx_lc][(masksizes[idx_lc]>max_bound(tmags[idx_lc]))]
		#ticsh_m = tmags[idx_lc][(masksizes[idx_lc]>max_bound(tmags[idx_lc]))]
		#ticsh_mm = masksizes[idx_lc][(masksizes[idx_lc]>max_bound(tmags[idx_lc]))]
		#ticsl = tics[idx_lc][(masksizes[idx_lc]<min_bound(tmags[idx_lc]))]
		#ticsl_m = tmags[idx_lc][(masksizes[idx_lc]<min_bound(tmags[idx_lc]))]
		#ticsl_mm = masksizes[idx_lc][(masksizes[idx_lc]<min_bound(tmags[idx_lc]))]

		#ticsl = tics[idx_sc][(masksizes[idx_sc]<min_bound_sc(tmags[idx_sc])) & (masksizes[idx_sc]>0)]
		#ticsl_m = tmags[idx_sc][(masksizes[idx_sc]<min_bound_sc(tmags[idx_sc])) & (masksizes[idx_sc]>0)]
		#ticsl_mm = masksizes[idx_sc][(masksizes[idx_sc]<min_bound_sc(tmags[idx_sc])) & (masksizes[idx_sc]>0) ]

		#print('HIGH')

		#for ii, tic in enumerate(ticsh):
		#	print(tic, ticsh_m[ii], ticsh_mm[ii])

		#print('LOW')
		#for ii, tic in enumerate(ticsl):
		#	print(tic, ticsl_m[ii], ticsl_mm[ii])

		#print(len(ticsh))
		#print(len(ticsl))

		#bin_means1, bin_edges1, binnumber1 = binned_statistic(tmags[idx_lc], masksizes[idx_lc]-pix_vs_mag(tmags[idx_lc]), statistic=reduce_percentile1, bins=15, range=(np.nanmin(tmags),np.nanmax(tmags)))
		#bin_means2, bin_edges2, binnumber2 = binned_statistic(tmags[idx_lc], masksizes[idx_lc]-pix_vs_mag(tmags[idx_lc]), statistic=reduce_percentile2, bins=15, range=(np.nanmin(tmags),np.nanmax(tmags)))
		#bin_means1, bin_edges1, binnumber1 = binned_statistic(tmags[idx_lc][d>0], d[d>0], statistic='median', bins=15, range=(np.nanmin(tmags),np.nanmax(tmags)))
		#bin_means2, bin_edges2, binnumber2 = binned_statistic(tmags[idx_lc][d<0], d[d<0], statistic='median', bins=15, range=(np.nanmin(tmags),np.nanmax(tmags)))
		#ax1.scatter(bin_centers, bin_means+bin_means1, marker='o', color='b')
		#ax1.scatter(bin_centers, bin_means+5*bin_means1, marker='o', color='b')
		#ax1.scatter(bin_centers, bin_means+4*bin_means2, marker='o', color='b')
		#ax1.scatter(bin_centers, bin_means+bin_means2, marker='o', color='b')
		#ax1.scatter(tmags[idx_lc][idx_sort], P.values, marker='o', color='m')

		#mags = np.linspace(np.nanmin(tmags)-1, np.nanmax(tmags)+1, 500)
		#pix = np.asarray([Pixinaperture(m) for m in mags], dtype='float64')
		#ax1.plot(mags, pix, color='k', ls='-')
		#ax2.plot(mags, pix, color='k', ls='-')

		ax1.set_xlim(dval.tmag_limits)
		ax1.set_ylim([0.99, np.nanmax(masksizes)+500])

		ax1.set_xlabel('TESS magnitude')
		ax1.set_ylabel('Pixels in aperture')
		xtick_major = np.median(np.diff(ax1.get_xticks()))
		ax1.xaxis.set_minor_locator(MultipleLocator(xtick_major/2))
		ytick_major = np.median(np.diff(ax1.get_yticks()))
		ax1.yaxis.set_minor_locator(MultipleLocator(ytick_major/2))
		ax1.set_yscale('log', nonposy='clip')
		#ax1.yaxis.set_major_formatter(ScalarFormatter())
		#axx.legend(loc='upper right')

		colorbar(im, ax=ax1, label='Contamination')

		fig.savefig(os.path.join(dval.outfolder, f'pixinaperture_c{cadence:04d}'))
		if not dval.show:
			plt.close(fig)

		# Assign validation bits, for both FFI and TPF
		dv = np.zeros_like(pri, dtype='int32')

		# Minimal masks were used:
		dv[minimal_mask_used] |= DatavalQualityFlags.MinimalMask

		# Small and Large masks:
		dv[(masksizes < min_bound(tmags)) & (masksizes > 0)] |= DatavalQualityFlags.SmallMask
		if max_bound is not None:
			dv[masksizes > max_bound(tmags)] |= DatavalQualityFlags.LargeMask

		dval.update_dataval(pri, dv)
