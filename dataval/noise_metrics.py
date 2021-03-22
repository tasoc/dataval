#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os.path
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import binned_statistic
from .plots import plt, colorbar
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import Normalize
from .quality import DatavalQualityFlags
from .noise_model import phot_noise
from .utilities import mad

#--------------------------------------------------------------------------------------------------
def noise_metrics(dval):
	"""
	Function to plot the light curve noise against the stellar TESS magnitudes

	.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger('dataval')
	logger.info('Plotting Noise vs. Magnitude...')

	mags = np.linspace(dval.tmag_limits[0], dval.tmag_limits[1], 200)

	# Colors for theoretical lines:
	# Define the colors directly here to avoid having to import
	# the entire seaborn library, but they come from the seaborn
	# 'colorblind' palette:
	#     cols = seaborn.color_palette("colorblind", 4)
	cols = [
		(0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
		(0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
		(0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
		(0.8352941176470589, 0.3686274509803922, 0.0)
	]
	norm = Normalize(vmin=0, vmax=1)

	for cadence in dval.cadences:

		if dval.corr:
			factor = 1
			star_vals = dval.search_database(
				select=[
					'todolist.priority',
					'todolist.sector',
					'todolist.tmag',
					'diagnostics_corr.rms_hour',
					'diagnostics_corr.ptp',
					'diagnostics.contamination'
				],
				search=f'cadence={cadence:d}')
		else:
			factor = 1e6
			star_vals = dval.search_database(
				select=[
					'todolist.priority',
					'todolist.sector',
					'todolist.tmag',
					'diagnostics.rms_hour',
					'diagnostics.ptp',
					'diagnostics.contamination'
				],
				search=f'cadence={cadence:d}')

		tmags = np.array([star['tmag'] for star in star_vals], dtype='float64')
		pri = np.array([star['priority'] for star in star_vals], dtype='int64')
		rms = np.array([star['rms_hour']*factor for star in star_vals], dtype='float64')
		ptp = np.array([star['ptp']*factor for star in star_vals], dtype='float64')
		contam = np.array([star['contamination'] for star in star_vals], dtype='float64')

		fig1, ax1 = plt.subplots()
		fig1.subplots_adjust(left=0.145, wspace=0.3, top=0.945, bottom=0.145, right=0.975)

		fig2, ax2 = plt.subplots()
		fig2.subplots_adjust(left=0.145, wspace=0.3, top=0.945, bottom=0.145, right=0.975)

		perm = dval.random_state.permutation(len(tmags))
		im1 = ax1.scatter(tmags[perm], rms[perm], c=contam[perm],
			marker='o',
			alpha=0.2,
			cmap=plt.get_cmap('PuOr'),
			norm=norm,
			rasterized=True)

		im2 = ax2.scatter(tmags[perm], ptp[perm], c=contam[perm],
			marker='o',
			alpha=0.2,
			cmap=plt.get_cmap('PuOr'),
			norm=norm,
			rasterized=True)

		# Plot theoretical lines:
		# Expected *1-hour* RMS noise ffi
		# TODO: Update elat+elon based on observing sector?
		tot_noise_rms, vals_rms = phot_noise(mags, timescale=3600, cadpix=cadence, sysnoise=dval.sysnoise)
		ax1.semilogy(mags, vals_rms[:, 0], '-', color=cols[0], label='Shot')
		ax1.semilogy(mags, vals_rms[:, 1], '--', color=cols[1], label='Zodiacal')
		ax1.semilogy(mags, vals_rms[:, 2], '-', color=cols[2], label='Read')
		ax1.semilogy(mags, vals_rms[:, 3], '--', color=cols[3], label='Systematic')
		ax1.semilogy(mags, tot_noise_rms, 'k-', label='Total')

		# Expected ptp for 30-min
		# TODO: Update elat+elon based on observing sector?
		tot_noise_ptp, vals_ptp = phot_noise(mags, timescale=cadence, cadpix=cadence, sysnoise=dval.sysnoise)
		ax2.semilogy(mags, vals_ptp[:, 0], '-', color=cols[0], label='Shot')
		ax2.semilogy(mags, vals_ptp[:, 1], '--', color=cols[1], label='Zodiacal')
		ax2.semilogy(mags, vals_ptp[:, 2], '-', color=cols[2], label='Read')
		ax2.semilogy(mags, vals_ptp[:, 3], '--', color=cols[3], label='Systematic')
		ax2.semilogy(mags, tot_noise_ptp, 'k-', label='Total')

		ptp_vs_mag = InterpolatedUnivariateSpline(mags, tot_noise_ptp, k=1, ext=2)
		rms_vs_mag = InterpolatedUnivariateSpline(mags, tot_noise_rms, k=1, ext=2)

		ax1.set_ylabel(r'$\rm RMS\,\, (ppm\,\, hr^{-1})$')
		ax2.set_ylabel('PTP-MDV (ppm)')

		for axx in [ax1, ax2]:
			axx.set_xlim(dval.tmag_limits)
			axx.set_xlabel('TESS magnitude')
			axx.xaxis.set_major_locator(MultipleLocator(2))
			axx.xaxis.set_minor_locator(MultipleLocator(1))
			axx.set_yscale('log')
			#axx.legend(loc='upper left')

		colorbar(im1, ax=ax1, label='Contamination')
		colorbar(im2, ax=ax2, label='Contamination')

		###########

		fig1.savefig(os.path.join(dval.outfolder, f'rms_c{cadence:04d}'))
		fig2.savefig(os.path.join(dval.outfolder, f'ptp_c{cadence:04d}'))
		if not dval.show:
			plt.close(fig1)
			plt.close(fig2)

		# Assign validation bits
		idx_ptp = (ptp < ptp_vs_mag(tmags)) & (ptp > 0)
		idx_rms = (rms < rms_vs_mag(tmags)) & (rms > 0)
		idx_invalid = (rms <= 0) | ~np.isfinite(rms) | (ptp <= 0) | ~np.isfinite(ptp)

		dv = np.zeros_like(pri, dtype='int32')
		dv[idx_ptp] |= DatavalQualityFlags.LowPTP
		dv[idx_rms] |= DatavalQualityFlags.LowRMS
		dv[idx_invalid] |= DatavalQualityFlags.InvalidNoise

		dval.update_dataval(pri, dv)

#--------------------------------------------------------------------------------------------------
def compare_noise(dval):
	"""
	Compare noise metrics before and after correction

	.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
	"""

	logger = logging.getLogger('dataval')
	logger.info('Plotting Noise Comparison...')

	if not dval.corrections_done:
		logger.info("Can not run compare_noise when corrections have not been run")
		return

	fig1 = plt.figure(figsize=(15, 5))
	fig1.subplots_adjust(left=0.145, wspace=0.3, top=0.945, bottom=0.145, right=0.975)
	ax11 = fig1.add_subplot(121)
	ax12 = fig1.add_subplot(122)

	fig2 = plt.figure(figsize=(15, 5))
	fig2.subplots_adjust(left=0.145, wspace=0.3, top=0.945, bottom=0.145, right=0.975)
	ax21 = fig2.add_subplot(121)
	ax22 = fig2.add_subplot(122)

	fig3 = plt.figure(figsize=(15, 5))
	fig3.subplots_adjust(left=0.145, wspace=0.3, top=0.945, bottom=0.145, right=0.975)
	ax31 = fig3.add_subplot(121)
	ax32 = fig3.add_subplot(122)

	#if self.corr:
	star_vals = dval.search_database(select=[
		'todolist.priority',
		'todolist.starid',
		'todolist.datasource',
		'todolist.sector',
		'todolist.tmag',
		'diagnostics_corr.rms_hour',
		'diagnostics_corr.ptp',
		'diagnostics.contamination',
		'ccd'])
	factor = 1
	star_vals2 = dval.search_database(select=[
		'todolist.priority',
		'todolist.starid',
		'todolist.datasource',
		'todolist.sector',
		'todolist.tmag',
		'diagnostics.rms_hour',
		'diagnostics.ptp',
		'diagnostics.contamination',
		'ccd'])
	factor2 = 1e6

	tmags = np.array([star['tmag'] for star in star_vals], dtype=float)
	pri = np.array([star['priority'] for star in star_vals], dtype=int)
	rms = np.array([star['rms_hour']*factor for star in star_vals], dtype=float)
	ptp = np.array([star['ptp']*factor for star in star_vals], dtype=float)
	source = np.array([star['datasource'] for star in star_vals], dtype=str)
	contam = np.array([star['contamination'] for star in star_vals], dtype=float)

	pri2 = np.array([star['priority'] for star in star_vals2], dtype=int)
	tmags2 = np.array([star['tmag'] for star in star_vals2], dtype=float)
	rms2 = np.array([star['rms_hour']*factor2 for star in star_vals2], dtype=float)
	ptp2 = np.array([star['ptp']*factor2 for star in star_vals2], dtype=float)
	source2 = np.array([star['datasource'] for star in star_vals2], dtype=str)

	def overlap(a, b):
		"""
		return the indices in a that overlap with b, also returns
		the corresponding index in b only works if both a and b are unique!
		This is not very efficient but it works
		"""
		bool_a = np.in1d(a,b)
		ind_a = np.arange(len(a))
		ind_a = ind_a[bool_a]
		ind_b = np.array([np.argwhere(b == a[x]) for x in ind_a]).flatten()
		return ind_a,ind_b

	idx_o, idx2_o = overlap(pri[(source == 'ffi')], pri2[(source2 == 'ffi')])

	print(len(star_vals), len(star_vals2))
	print(len(idx_o), len(idx2_o))
	print(tmags[(source == 'ffi')][idx_o])
	print(tmags2[(source2 == 'ffi')][idx2_o])
	print(np.sum(tmags[(source == 'ffi')][idx_o]-tmags2[(source2 == 'ffi')][idx2_o]))

	#88518 956502

	#tcomp = np.array([tmags[(pri==i)] for i in pri_overlap[5000:10000]])
	#rmscomp = np.array([rms[(pri==i)]/rms2[(pri2==i)] for i in pri_overlap[5000:10000]])
	#ptpcomp = np.array([ptp[(pri==i)]/ptp2[(pri2==i)] for i in pri_overlap[5000:10000]])

	tcomp = tmags[(source == 'ffi')][idx_o]
	rmscomp = rms[(source == 'ffi')][idx_o]/rms2[(source2 == 'ffi')][idx2_o]
	ptpcomp = ptp[(source == 'ffi')][idx_o]/ptp2[(source2 == 'ffi')][idx2_o]

	#ccomp = contam[(source == 'ffi')][idx_o]

	#nbins=300
	#data1 = np.column_stack((tcomp, rmscomp))
	#k = kde.gaussian_kde(data1.T)
	#xi, yi = np.mgrid[tcomp.min():tcomp.max():nbins*1j, rmscomp.min():rmscomp.max():nbins*1j]
	#zi = np.log10(k(np.vstack([xi.flatten(), yi.flatten()])))
	#clevels = ax31.contour(xi, yi, zi.reshape(xi.shape),lw=.9,cmap='winter')#,zorder=90)

	#ax31.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.Greens, shading='gouraud')

	#p = clevels.collections[0].get_paths()
	#inside = np.full_like(tcomp,False,dtype=bool)
	#print(inside)
	#for level in p:
	#	print(inside)
	#	inside |= level.contains_points(list(zip(*(rmscomp,tcomp))))

	#print(inside)
	#print(inside.shape, tcomp.shape, rmscomp.shape)
	#ax31.scatter(tcomp[~inside],rmscomp[~inside],marker='.', color='0.2')

	#data2 = np.column_stack((tcomp, ptpcomp))
	#k = kde.gaussian_kde(data2.T)
	#xi, yi = np.mgrid[tcomp.min():tcomp.max():nbins*1j, ptpcomp.min():ptpcomp.max():nbins*1j]
	#zi = k(np.vstack([xi.flatten(), yi.flatten()]))
	#ax32.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.Greens, shading='gouraud',norm=colors.LogNorm(vmin=zi.min(), vmax=zi.max()))

	ax31.scatter(tcomp, rmscomp, marker='o', c=contam, alpha=0.01, label='30-min cadence', cmap=plt.get_cmap('PuOr'))
	ax32.scatter(tcomp, ptpcomp, marker='o', c=contam, alpha=0.01, label='30-min cadence', cmap=plt.get_cmap('PuOr'))

	bin_rms, bin_edge_rms, _ = binned_statistic(tcomp, rmscomp, statistic='median', bins=15, range=(np.nanmin(tcomp),np.nanmax(tcomp)))
	bin_ptp, bin_edge_ptp, _ = binned_statistic(tcomp, ptpcomp, statistic='median', bins=15, range=(np.nanmin(tcomp),np.nanmax(tcomp)))
	bin_width = (bin_edge_rms[1] - bin_edge_rms[0])
	bin_centers = bin_edge_rms[1:] - bin_width/2

	bin_rmsmad, bin_edges_rmsmad, _ = binned_statistic(tcomp, rmscomp, statistic=mad, bins=15, range=(np.nanmin(tcomp),np.nanmax(tcomp)))
	bin_ptpmad, bin_edges_ptpmad, _ = binned_statistic(tcomp, ptpcomp, statistic=mad, bins=15, range=(np.nanmin(tcomp),np.nanmax(tcomp)))

	ax31.errorbar(bin_centers, bin_rms, yerr=bin_rmsmad, ecolor='r', mec='r', mfc='w', capsize=0, marker='o', ls='')
	ax32.errorbar(bin_centers, bin_ptp, yerr=bin_ptpmad, ecolor='r', mec='r', mfc='w', capsize=0, marker='o', ls='')

	ax31.axhline(y=1, ls='--', color='r')
	ax32.axhline(y=1, ls='--', color='r')

	idx_lc = (source == 'ffi')
	idx_sc = (source != 'ffi')

	#ax11.scatter(tmags[idx_lc], rms[idx_lc], marker='o', facecolors=rgba_color, edgecolor=rgba_color, alpha=0.1, label='30-min cadence')
	#ax12.scatter(tmags[idx_sc], rms[idx_sc], marker='o', facecolors=rgba_color, edgecolor=rgba_color, alpha=0.1, label='2-min cadence')

	ax11.scatter(tmags[idx_lc], rms[idx_lc], marker='o', c=contam[idx_lc], alpha=0.1, label='30-min cadence', cmap=plt.get_cmap('PuOr'))
	ax12.scatter(tmags[idx_sc], rms[idx_sc], marker='o', c=contam[idx_sc], alpha=0.1, label='2-min cadence', cmap=plt.get_cmap('PuOr'))

	#ax21.scatter(tmags[idx_lc], ptp[idx_lc], marker='o', facecolors=rgba_color, edgecolor=rgba_color, alpha=0.1, label='30-min cadence')
	#ax22.scatter(tmags[idx_sc], ptp[idx_sc], marker='o', facecolors=rgba_color, edgecolor=rgba_color, alpha=0.1, label='2-min cadence')

	ax21.scatter(tmags[idx_lc], ptp[idx_lc], marker='o', c=contam[idx_lc], alpha=0.1, label='30-min cadence', cmap=plt.get_cmap('PuOr'))
	ax22.scatter(tmags[idx_sc], ptp[idx_sc], marker='o', c=contam[idx_sc], alpha=0.1, label='2-min cadence', cmap=plt.get_cmap('PuOr'))

	#ax11.scatter(tmags2[(source2=='ffi')], rms2[(source2=='ffi')], marker='o', facecolors='r', edgecolor='r', alpha=0.1, label='30-min cadence')
	#ax21.scatter(tmags2[(source2=='ffi')], ptp2[(source2=='ffi')], marker='o', facecolors='r', edgecolor='r', alpha=0.1, label='30-min cadence')

	# Plot theoretical lines
	mags = np.linspace(dval.tmag_limits[0], dval.tmag_limits[1], 200)

	# Expected *1-hour* RMS noise ffi
	# TODO: Update elat+elon based on observing sector?
	tot_noise_rms_ffi, vals_rms_ffi = phot_noise(mags, timescale=3600, cadpix=1800, sysnoise=dval.sysnoise)
	ax11.semilogy(mags, vals_rms_ffi[:, 0], 'r-', label='Shot')
	ax11.semilogy(mags, vals_rms_ffi[:, 1], 'g--', label='Zodiacal')
	ax11.semilogy(mags, vals_rms_ffi[:, 2], '-', label='Read')
	ax11.semilogy(mags, vals_rms_ffi[:, 3], 'b--', label='Systematic')
	ax11.semilogy(mags, tot_noise_rms_ffi, 'k-', label='Total')

	# Expected *1-hour* RMS noise tpf
	# TODO: Update elat+elon based on observing sector?
	tot_noise_rms_tpf, vals_rms_tpf = phot_noise(mags, timescale=3600, cadpix=120, sysnoise=dval.sysnoise)
	ax12.semilogy(mags, vals_rms_tpf[:, 0], 'r-', label='Shot')
	ax12.semilogy(mags, vals_rms_tpf[:, 1], 'g--', label='Zodiacal')
	ax12.semilogy(mags, vals_rms_tpf[:, 2], '-', label='Read')
	ax12.semilogy(mags, vals_rms_tpf[:, 3], 'b--', label='Systematic')
	ax12.semilogy(mags, tot_noise_rms_tpf, 'k-', label='Total')

	# Expected ptp for 30-min
	# TODO: Update elat+elon based on observing sector?
	tot_noise_ptp_ffi, vals_ptp_ffi = phot_noise(mags, timescale=1800, cadpix=1800, sysnoise=dval.sysnoise)
	ax21.semilogy(mags, vals_ptp_ffi[:, 0], 'r-', label='Shot')
	ax21.semilogy(mags, vals_ptp_ffi[:, 1], 'g--', label='Zodiacal')
	ax21.semilogy(mags, vals_ptp_ffi[:, 2], '-', label='Read')
	ax21.semilogy(mags, vals_ptp_ffi[:, 3], 'b--', label='Systematic')
	ax21.semilogy(mags, tot_noise_ptp_ffi, 'k-', label='Total')

	# Expected ptp for 2-min
	# TODO: Update elat+elon based on observing sector?
	tot_noise_ptp_tpf, vals_ptp_tpf = phot_noise(mags, timescale=120, cadpix=120, sysnoise=dval.sysnoise)
	ax22.semilogy(mags, vals_ptp_tpf[:, 0], 'r-', label='Shot')
	ax22.semilogy(mags, vals_ptp_tpf[:, 1], 'g--', label='Zodiacal')
	ax22.semilogy(mags, vals_ptp_tpf[:, 2], '-', label='Read')
	ax22.semilogy(mags, vals_ptp_tpf[:, 3], 'b--', label='Systematic')
	ax22.semilogy(mags, tot_noise_ptp_tpf, 'k-', label='Total')

	ax11.set_ylabel(r'$\rm RMS\,\, (ppm\,\, hr^{-1})$')
	ax21.set_ylabel('PTP-MDV (ppm)')
	ax31.set_ylabel(r'$\rm RMS_{corr} / RMS_{raw}$')
	ax32.set_ylabel(r'$\rm PTP-MDV_{corr} / PTP-MDV_{raw}$')

	for axx in np.array([ax11, ax12, ax21, ax22, ax31, ax32]):
		axx.set_xlim(dval.tmag_limits)
		axx.set_xlabel('TESS magnitude')
		axx.xaxis.set_major_locator(MultipleLocator(2))
		axx.xaxis.set_minor_locator(MultipleLocator(1))
		axx.set_yscale('log')
		#axx.legend(loc='upper left')

	ax31.set_xlim(dval.tmag_limits)
	ax32.set_xlim(dval.tmag_limits)
	###########

	fig1.savefig(os.path.join(dval.outfolder, 'rms_comp'))
	fig2.savefig(os.path.join(dval.outfolder, 'ptp_comp'))
	fig3.savefig(os.path.join(dval.outfolder, 'comp'))
	if not dval.show:
		plt.close(fig1)
		plt.close(fig2)
		plt.close(fig3)
