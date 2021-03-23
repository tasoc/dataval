#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data validation of wait-time diagnostics.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import logging
import os.path
import numpy as np
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
from .plots import plt
from matplotlib.ticker import MultipleLocator

#--------------------------------------------------------------------------------------------------
def mag_dist(dval):
	"""
	Function to plot magnitude distribution for targets

	.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger('dataval')
	logger.info('Plotting Magnitude distribution...')

	fig, ax = plt.subplots(figsize=plt.figaspect(0.5))
	fig.subplots_adjust(left=0.14, wspace=0.3, top=0.94, bottom=0.155, right=0.96)

	colors = ['r', 'b', 'g'] # TODO: What if there are more than three?
	for k, cadence in enumerate(dval.cadences):

		star_vals = dval.search_database(
			select='todolist.tmag',
			search=f'cadence={cadence:d}')

		if star_vals:
			tmags = np.array([star['tmag'] for star in star_vals])

			kde = KDE(tmags)
			kde.fit(gridsize=1000)

			ax.fill_between(kde.support, 0, kde.density/np.max(kde.density),
				color=colors[k],
				alpha=0.3,
				label=f'{cadence:d}s cadence')

#		kde_all = KDE(tmags)
#		kde_all.fit(gridsize=1000)
#		ax.plot(kde_all.support, kde_all.density/np.max(kde_all.density), 'k-', lw=1.5, label='All')

	ax.set_ylim(bottom=0)
	ax.set_xlabel('TESS magnitude')
	ax.set_ylabel('Normalised Density')
	ax.xaxis.set_major_locator(MultipleLocator(2))
	ax.xaxis.set_minor_locator(MultipleLocator(1))
	ax.legend(frameon=False, loc='upper left', borderaxespad=0, handlelength=2.5, handletextpad=0.4)

	fig.savefig(os.path.join(dval.outfolder, 'mag_dist'))
	if not dval.show:
		plt.close(fig)
