#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data validation of automatic Halo photometry switching.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import logging
import os
from tqdm import tqdm
from .status import STATUS

#--------------------------------------------------------------------------------------------------
def cleanup(dval):
	"""
	Perform cleanup of the database and lightcurve files.

	It is sometimes needed to do some basic cleanups of left-over FITS files
	that were produced by the photometry, but were later deemed better to discard.

	Parameters:
		dval (:class:`DataValidation`): Data Validation object.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)
	logger.info("Running cleanup...")
	tqdm_settings = {'disable': not logger.isEnabledFor(logging.INFO)}

	# Get a list of all targets that were skipped ut still has an associated lightcurve file.
	# This can happen when a targets is marked as SKIPPED after it has already been processed.
	dval.cursor.execute("SELECT todolist.priority,lightcurve FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE status=? AND lightcurve IS NOT NULL;", [STATUS.SKIPPED.value])
	results = dval.cursor.fetchall()
	num_cleanups = len(results)

	logger.info("%d files can be cleaned up", num_cleanups)

	# Only actually perform the cleanups if we are saving validations:
	if dval.doval and num_cleanups > 0:
		rootdir = dval.input_folders[0]

		for row in tqdm(results, **tqdm_settings):
			try:
				# Path to the file on disk:
				fpath = os.path.abspath(os.path.join(rootdir, row['lightcurve']))

				# Remove the record of the file from the database:
				dval.cursor.execute("UPDATE diagnostics SET lightcurve=NULL WHERE priority=?;", [row['priority']])
				dval.conn.commit()

				# Delete the file on disk:
				if os.path.isfile(fpath):
					os.remove(fpath)
			except:
				dval.conn.rollback()
				raise
