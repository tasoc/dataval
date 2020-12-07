#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for data release creation.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import logging
import warnings
from astropy.io import fits

#--------------------------------------------------------------------------------------------------
def check_fits_changes(fname, fname_modified, allow_header_value_changes=None):
	"""
	Check FITS files for changes.

	If no changes between the two files are detected, an error is also raised, as it is assumed
	that a change was needed. The header keywords CHECKSUM and DATASUM are always allowed to change.

	Parameters:
		fname (str, fits.HDUList): Original FITS file to check against.
		fname_modified (str, fits.HDUList): Modified FITS file to check. 
		allow_header_value_changes (list): List of header keywords allowed to change.

	Returns:
		bool: True if file check was okay, False otherwise.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)

	if allow_header_value_changes is None:
		allow_header_value_changes = ['CHECKSUM', 'DATASUM']
	else:
		allow_header_value_changes = ['CHECKSUM', 'DATASUM'] + allow_header_value_changes

	with warnings.catch_warnings():
		warnings.filterwarnings('ignore', category=fits.verify.VerifyWarning)
		diff = fits.FITSDiff(fname, fname_modified)

	fname_str = 'HDU' if isinstance(fname, fits.HDUList) else fname

	if logger.isEnabledFor(logging.DEBUG):
		report = diff.report()
		logger.debug(report)

	if diff.identical:
		logger.error("%s: Nothing has changed?", fname_str)
		return False

	if diff.diff_hdu_count:
		logger.error("%s: Different number of HDUs: %s", fname_str, diff.diff_hdu_count)
		return False

	everything_ok = True
	for k, d in diff.diff_hdus:
		# Headers:
		hdr = d.diff_headers
		if not hdr.identical:
			if hdr.diff_keywords:
				# Keywords only in header A:
				if hdr.diff_keywords[0]:
					logger.error("%s: %s", fname_str, hdr.diff_keywords[0])
					everything_ok = False
				# Keywords only in header B:
				if hdr.diff_keywords[1]:
					if any([key not in allow_header_value_changes for key in hdr.diff_keywords[1]]):
						logger.error("%s: %s", fname_str, hdr.diff_keywords[1])
						everything_ok = False

			for key, val in hdr.diff_keyword_values.items():
				if key not in allow_header_value_changes:
					logger.error("%s: Keyword with different values: %s: %s != %s",
						fname_str, key, val[0][0], val[0][1])
					everything_ok = False

			for key, val in hdr.diff_keyword_comments.items():
				if key not in allow_header_value_changes:
					logger.error("%s: Keyword with different comments: %s: %s != %s",
						fname_str, key, val[0][0], val[0][1])
					everything_ok = False

		# Data:
		dat = d.diff_data
		if dat is not None and not dat.identical:
			if isinstance(dat, fits.diff.TableDataDiff):
				if dat.diff_column_count:
					logger.error("%s: Different number of table columns!", fname_str)
					everything_ok = False

				for something in dat.diff_column_attributes:
					column_change = something[0][1]
					if column_change == 'disp' and any([kw.startswith('TDISP') for kw in allow_header_value_changes]):
						pass
					elif column_change == 'unit' and any([kw.startswith('TUNIT') for kw in allow_header_value_changes]):
						pass
					else:
						column_key = something[0][0]
						column_changes = something[1]
						logger.error("%s: Table header with different values: %s (%s): %s != %s",
							fname_str,
							column_key,
							column_change,
							column_changes[0],
							column_changes[1])
						everything_ok = False

				if dat.diff_values:
					logger.error("%s: Data has been changed!", fname_str)
					everything_ok = False
			else:
				logger.error("%s: Data has been changed!", fname_str)
				everything_ok = False

	# If we have made it this far, things should be okay:
	return everything_ok
