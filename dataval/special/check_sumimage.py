#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check sumimages stored in lightcurve FITS files.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import logging
import os.path
import numpy as np
from astropy.io import fits
from tqdm import tqdm

#--------------------------------------------------------------------------------------------------
def check_sumimage(dval, warn_abs=5, warn_rel=0.05):
	"""
	Check sumimages stored in lightcurve FITS files.

	Parameters:
		dval (:py:class:`DataValidation`): DataValidation instance.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)
	logger.info("Checking sumimages...")

	if dval.corr:
		files = dval.search_database(select=['todolist.priority', 'diagnostics_corr.lightcurve'])
	else:
		files = dval.search_database(select=['todolist.priority', 'diagnostics.lightcurve'])

	print(files)

	rootdir = dval.input_folders[0]

	missing_files = False
	for row in tqdm(files):
		fpath = os.path.join(rootdir, row['lightcurve'])
		logger.debug("Checking file: %s", fpath)

		if not os.path.isfile(fpath):
			missing_files = True
		else:
			with fits.open(fpath, mode='readonly', memmap=True) as hdu:
				sumimage = hdu['SUMIMAGE'].data
				aperture = np.asarray(hdu['APERTURE'].data, dtype='int32')

				if sumimage.shape != aperture.shape:
					logger.error("{lightcurve:s}: {priority:d}", row)
					continue

				# Only look at the pixels the aperture says are downloaded:
				good_aperture = (aperture & 1 != 0)

				# Count the number of NaNs in the sumimage, both in
				# absolute and relative sense:
				bad_abs = np.sum(~np.isfinite(sumimage[good_aperture]))
				bad_rel = bad_abs / np.sum(good_aperture)

				if bad_abs >= warn_abs or bad_rel >= warn_rel:
					logger.warning("{lightcurve:s}: {priority:d}", row)

	if missing_files:
		logger.error("Missing lightcurve files detected.")
