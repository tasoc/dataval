#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data validation of wait-time diagnostics.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import logging
import os.path
from scipy.stats import binned_statistic as binning
from astropy.table import Table
from .plots import plt

#--------------------------------------------------------------------------------------------------
def waittime(dval):
	"""
	Visualize the worker wait-time during the processing.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger('dataval')
	logger.info("Running waittime diagnostics...")

	run_tables = ['diagnostics']
	if dval.corrections_done:
		run_tables.append('diagnostics_corr')

	for table in run_tables:
		# Check if the worker_waittime column is in the database
		# It was not generated in earlier versions of the pipeline
		dval.cursor.execute("PRAGMA table_info(" + table + ")")
		if 'worker_wait_time' not in [r['name'] for r in dval.cursor.fetchall()]:
			logger.info("WORKER_WAIT_TIME is not stored in database.")
			return

		# Get the data from the database:
		star_vals = dval.search_database(select=['todolist.priority', table + '.worker_wait_time'])
		tab = Table(rows=star_vals,
			names=('priority', 'worker_wait_time'),
			dtype=('int64', 'float32'))

		# Bin the wait-time to see main systematic:
		bin_means, bin_edges, _ = binning(tab['priority'], tab['worker_wait_time'], statistic='median', bins=25)
		bin_width = (bin_edges[1] - bin_edges[0])
		bin_centers = bin_edges[1:] - bin_width/2

		# Create figure figure:
		fig = plt.figure(figsize=(16,8))
		ax = fig.add_subplot(111)
		ax.scatter(tab['priority'], tab['worker_wait_time'], marker='.', alpha=0.1)
		ax.scatter(bin_centers, bin_means, c='r')
		ax.set_xlabel('Priority')
		ax.set_ylabel('Worker wait-time (s)')
		ax.set_xlim(left=0)
		ax.set_ylim(bottom=0)

		# Save figure to file and close:
		fig.savefig(os.path.join(dval.outfolder, 'worker_waittime_' + table))
		if not dval.show:
			plt.close(fig)
