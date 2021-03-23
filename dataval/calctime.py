#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os.path
import numpy as np
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
from .plots import plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MultipleLocator

#--------------------------------------------------------------------------------------------------
def calctime(dval, maxtime=50.0):

	logger = logging.getLogger('dataval')
	logger.info('Plotting calculation times for photometry...')

	for cadence in dval.cadences:

		star_vals = dval.search_database(
			select=[
				'diagnostics.stamp_resizes',
				'diagnostics.elaptime'
			],
			search=[f'cadence={cadence:d}', f'diagnostics.elaptime <= {maxtime:f}'])

		if not star_vals:
			continue

		et = np.array([star['elaptime'] for star in star_vals], dtype='float64')
		resize = np.array([star['stamp_resizes'] for star in star_vals], dtype='int32')

		maxresize = int(np.max(resize))

		fig, ax = plt.subplots(figsize=plt.figaspect(0.5))
		norm = Normalize(vmin=-0.5, vmax=maxresize+0.5)
		scalarMap = ScalarMappable(norm=norm, cmap=plt.get_cmap('tab10'))

		# Calculate KDE of full dataset:
		kde1 = KDE(et)
		kde1.fit(kernel='gau', gridsize=1024)

		# Calculate KDEs for different number of stamp resizes:
		for jj in range(maxresize+1):
			kde_data = et[resize == jj]
			if len(kde_data):
				kde2 = KDE(kde_data)
				kde2.fit(kernel='gau', gridsize=1024)

				rgba_color = scalarMap.to_rgba(jj)

				ax.fill_between(kde2.support, 0, kde2.density, color=rgba_color, alpha=0.5,
					label=f'{jj:d} resizes')

		ax.plot(kde1.support, kde1.density, color='k', lw=2, label='All')
		ax.set_xlim([0, maxtime])
		ax.set_ylim(bottom=0)

		ax.xaxis.set_major_locator(MultipleLocator(5))
		ax.xaxis.set_minor_locator(MultipleLocator(1))
		ax.set_xlabel('Calculation time (sec)')
		ax.legend(loc='upper right')

		fig.savefig(os.path.join(dval.outfolder, f'calctime_c{cadence:04d}'))

		if not dval.show:
			plt.close(fig)

#--------------------------------------------------------------------------------------------------
def calctime_corrections(dval, maxtime=50.0):

	logger = logging.getLogger('dataval')
	if not dval.corrections_done:
		logger.debug("Skipping since corrections not done")
		return
	logger.info('Plotting calculation times for corrections...')

	for cadence in dval.cadences:

		star_vals = dval.search_database(
			select='diagnostics_corr.elaptime',
			search=[f'cadence={cadence:d}', f'diagnostics_corr.elaptime <= {maxtime:f}'])

		if not star_vals:
			continue

		et = np.array([star['elaptime'] for star in star_vals], dtype='float64')

		kde = KDE(et)
		kde.fit(gridsize=1024)

		fig, ax = plt.subplots(figsize=plt.figaspect(0.5))
		ax.plot(kde.support, kde.density, color='k', lw=2)
		ax.set_xlim([0, maxtime])
		ax.set_ylim(bottom=0)
		ax.xaxis.set_major_locator(MultipleLocator(5))
		ax.xaxis.set_minor_locator(MultipleLocator(1))
		ax.set_xlabel('Calculation time (sec)')

		fig.savefig(os.path.join(dval.outfolder, f'calctime_corr_c{cadence:04d}'))

		if not dval.show:
			plt.close(fig)
