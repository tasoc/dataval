#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check sumimages stored in lightcurve FITS files.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import logging
import os.path
from astropy.io import fits
from tqdm import tqdm

#--------------------------------------------------------------------------------------------------
def check_duplicate_headers(dval):
	"""
	Check headers of lightcurve FITS files for duplicates.

	Parameters:
		dval (:class:`DataValidation`): DataValidation instance.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger('dataval')
	logger.info("Checking duplicate headers...")

	# Settings for tqdm:
	tqdm_settings = {
		'disable': None if logger.isEnabledFor(logging.INFO) else True
	}

	# Get list of lightcurves from TODO-file:
	if dval.corr:
		files = dval.search_database(select=['todolist.priority', 'diagnostics_corr.lightcurve'])
	else:
		files = dval.search_database(select=['todolist.priority', 'diagnostics.lightcurve'])

	missing_files = 0
	for row in tqdm(files, **tqdm_settings):
		fpath = os.path.join(dval.input_folder, row['lightcurve'])
		logger.debug("Checking file: %s", fpath)

		if not os.path.isfile(fpath):
			missing_files += 1
		else:
			with fits.open(fpath, mode='readonly', memmap=True) as hdu:
				# Check that no header keywords are duplicated:
				for k, h in enumerate(hdu):
					keys = list(h.header.keys())
					nonunique_keys = set([r for r in keys if keys.count(r) > 1])
					if nonunique_keys:
						logger.error("Non-unique keys found in header #%d: %s", k, nonunique_keys)

	if missing_files:
		logger.error("%d missing lightcurve files detected.", missing_files)
