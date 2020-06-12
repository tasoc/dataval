#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data validation of automatic Halo photometry switching.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import logging
import os.path
from astropy.table import Table
from .plots import plt
from .utilities import mag2flux

#--------------------------------------------------------------------------------------------------
def plot_haloswitch(dval):
	"""
	Visiualize the disgnostics used to automatically switch to Halo photometry.

	Parameters:
		dval (:class:`DataValidation`): Data Validation object.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)
	logger.info("Running halo-switch diagnostics...")

	# Check if the edge_flux column is in the database
	# It was not generated in earlier versions of the pipeline
	dval.cursor.execute("PRAGMA table_info(diagnostics)")
	if 'edge_flux' not in [r['name'] for r in dval.cursor.fetchall()]:
		logger.info("EDGE_FLUX is not stored in database.")
		return

	# TODO: Store these in the database as well
	haloswitch_tmag_limit = 6.0 # Maximal Tmag to apply Halo photometry automatically
	haloswitch_flux_limit = 0.01

	# Create figure figure:
	fig = plt.figure(figsize=(12,8))
	ax = fig.add_subplot(111)

	for datasource in ('ffi', 'tpf'):
		# Get the data from the database:
		constraint_cadence = "datasource='ffi'" if datasource == 'ffi' else "datasource!='ffi'"
		star_vals = dval.search_database(
			select=[
				'todolist.priority',
				'todolist.tmag',
				'diagnostics.edge_flux',
				"(diagnostics.errors IS NOT NULL AND INSTR(diagnostics.errors, 'Automatically switched to Halo photometry') > 0) AS switched"
			],
			search=[
				'edge_flux > 0',
				constraint_cadence,
				"(todolist.method IS NULL OR todolist.method != 'halo')" # Don't include things already processed with Halo
			])
		if not star_vals:
			continue

		tab = Table(rows=star_vals,
			names=('priority', 'tmag', 'edge_flux', 'switched'),
			dtype=('int64', 'float32', 'float64', 'bool'))

		switched = tab['switched']
		expected_flux = mag2flux(tab['tmag'])

		i = ax.scatter(tab['tmag'], tab['edge_flux']/expected_flux,
			alpha=0.3, label=datasource.upper())
		ax.scatter(tab['tmag'][switched], tab['edge_flux'][switched]/expected_flux[switched],
			c=i.get_facecolor(), alpha=1, marker='x', label=datasource.upper() + ' Switched')

	ax.axvline(haloswitch_tmag_limit, c='r', ls='--')
	ax.axhline(haloswitch_flux_limit, c='r', ls='--')
	ax.set_xlabel('TESS magnitude')
	ax.set_ylabel('Edge flux / Expected total flux')
	ax.set_yscale('log')
	ax.set_ylim([1e-5, 1.0])
	ax.set_xlim(dval.tmag_limits)
	ax.legend()

	# Save figure to file and close:
	fig.savefig(os.path.join(dval.outfolder, 'haloswitch'))
	if dval.show:
		plt.show()
	else:
		plt.close(fig)
