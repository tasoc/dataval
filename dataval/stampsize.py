#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os.path
import numpy as np
#from scipy.stats import binned_statistic
from .plots import plt, colorbar
from matplotlib.colors import Normalize

#--------------------------------------------------------------------------------------------------
def stampsize(dval):
	"""
	Function to plot width and height of pixel stamps against the stellar TESS magnitudes

	.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
	"""

	logger = logging.getLogger('dataval')
	logger.info('Plotting Stamp sizes...')

	fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(20, 10), sharex='all', sharey='row')
	fig.subplots_adjust(wspace=0.03, hspace=0.03)

	star_vals = dval.search_database(
		select=[
			'todolist.datasource',
			'todolist.sector',
			'todolist.tmag',
			'diagnostics.stamp_resizes',
			'diagnostics.stamp_width',
			'diagnostics.stamp_height'
		],
		search="method_used='aperture'")

	if not star_vals:
		logger.info("No targets to plot")
		return

	tmags = np.array([star['tmag'] for star in star_vals], dtype='float64')
	width = np.array([star['stamp_width'] for star in star_vals], dtype='int32')
	height = np.array([star['stamp_height'] for star in star_vals], dtype='int32')
	resize = np.array([star['stamp_resizes'] for star in star_vals], dtype='int32')
	ffi = np.array([star['datasource'] == 'ffi' for star in star_vals], dtype='bool')
	tpf = np.array([star['datasource'] == 'tpf' for star in star_vals], dtype='bool')
	secondary = np.array([star['datasource'].startswith('tpf:') for star in star_vals], dtype='bool')

	if dval.color_by_sector:
		# Color by sector number
		sec = np.array([star['sector'] for star in star_vals], dtype='int32')
		sectors = sorted(list(set(sec)))
		col = sec
		norm = Normalize(vmin=-0.5, vmax=len(sectors)-0.5)
		cmap = plt.get_cmap('tab10', len(sectors))
		cbar_ticks = np.arange(len(sectors))
		cbar_ticklabels = sectors
		cbar_label = 'Sector'
	else:
		# Color by number of stamp resizes
		maxresize = int(np.max(resize))
		col = resize
		norm = Normalize(vmin=-0.5, vmax=maxresize+0.5)
		cmap = plt.get_cmap('tab10', maxresize+1)
		cbar_ticks = cbar_ticklabels = None
		cbar_label = 'Resizes'

	# Plot the heights and widths as a function of magnitude, color-coded
	# depending on the choices above:
	im2 = ax1.scatter(tmags[ffi], height[ffi], c=col[ffi], norm=norm, cmap=cmap,
		marker='o', facecolors='None', label='FFI - h', alpha=0.3)

	ax2.scatter(tmags[tpf], height[tpf], c=col[tpf], norm=norm, cmap=cmap,
		marker='o', facecolors='None', label='TPF - h', alpha=0.3)

	ax3.scatter(tmags[ffi], width[ffi], c=col[ffi], norm=norm, cmap=cmap,
		marker='o', facecolors='None', label='FFI - w', alpha=0.3)

	ax4.scatter(tmags[tpf], width[tpf], c=col[tpf], norm=norm, cmap=cmap,
		marker='o', facecolors='None', label='TPF- w', alpha=0.3)

	ax5.scatter(tmags[secondary], height[secondary], c=col[secondary], norm=norm, cmap=cmap,
		marker='o', facecolors='None', label='TPF-sec- w', alpha=0.3)

	ax6.scatter(tmags[secondary], width[secondary], c=col[secondary], norm=norm, cmap=cmap,
		marker='o', facecolors='None', label='TPF-sec- w', alpha=0.3)

	#bin_means, bin_edges, binnumber = binned_statistic(tmags[ds], height[ds], statistic='median', bins=20, range=(1.5,10))
	#bin_width = (bin_edges[1] - bin_edges[0])
	#bin_centers = bin_edges[1:] - bin_width/2

	#bin_means2, bin_edges2, binnumber2 = binned_statistic(tmags[ds], width[ds], statistic='median', bins=20, range=(1.5,10))
	#bin_width2 = (bin_edges2[1] - bin_edges2[0])
	#bin_centers2 = bin_edges2[1:] - bin_width2/2

	#ax12.scatter(bin_centers2, bin_means2, marker='o', color='b', zorder=3)
	#ax11.scatter(bin_centers, bin_means, marker='o', color='b', zorder=3)

	# Decide how many pixels to use based on lookup tables as a function of Tmag:
	#mags = np.array([0., 0.52631579, 1.05263158, 1.57894737, 2.10526316,
	#	2.63157895, 3.15789474, 3.68421053, 4.21052632, 4.73684211,
	#	5.26315789, 5.78947368, 6.31578947, 6.84210526, 7.36842105,
	#	7.89473684, 8.42105263, 8.94736842, 9.47368421, 10.])
	#nhei = np.array([831.98319063, 533.58494422, 344.0840884, 223.73963332,
	#	147.31365728, 98.77856016, 67.95585074, 48.38157414,
	#	35.95072974, 28.05639497, 23.043017, 19.85922009,
	#	17.83731732, 16.5532873, 15.73785092, 15.21999971,
	#	14.89113301, 14.68228285, 14.54965042, 14.46542084])
	#nwid = np.array([157.71602062, 125.1238281, 99.99440209, 80.61896267,
	#	65.6799962, 54.16166547, 45.28073365, 38.4333048,
	#	33.15375951, 29.08309311, 25.94450371, 23.52456986,
	#	21.65873807, 20.22013336, 19.1109318, 18.25570862,
	#	17.59630936, 17.08789543, 16.69589509, 16.39365266])

	#mags2 = np.linspace(np.min(tmags)-0.2, np.max(tmags)+0.2, 500)
	#nwid2 = np.array([2*(np.ceil(np.interp(m, mags, nwid))//2)+1 for m in mags2])
	#nhei2 = np.array([2*(np.ceil(np.interp(m, mags, nhei))//2)+1 for m in mags2])

	#nwid2[(nwid2 < 15)] = 15
	#nhei2[(nhei2 < 15)] = 15

	#ax12.plot(mags2,nwid2, 'b--')
	#ax11.plot(mags2,nhei2, 'b--')

	ax1.set_title('FFI')
	ax2.set_title('TPF')
	ax5.set_title('TPF - Secondary')
	ax1.set_ylabel('Stamp height (pixels)')
	ax3.set_ylabel('Stamp width (pixels)')
	ax3.set_xlabel('TESS magnitude')
	ax4.set_xlabel('TESS magnitude')
	ax6.set_xlabel('TESS magnitude')

	for axx in [ax1, ax2, ax3, ax4, ax5, ax6]:
		axx.set_xlim(dval.tmag_limits)
		axx.set_ylim(bottom=0)

	# Add colorbar:
	cb = colorbar(im2, ax=ax5, label=cbar_label, ticks=cbar_ticks, ticklabels=cbar_ticklabels)
	cb.minorticks_off()

	fig.savefig(os.path.join(dval.outfolder, 'stamp_size'))
	if not dval.show:
		plt.close(fig)
