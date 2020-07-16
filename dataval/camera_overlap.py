#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check for complications arising from overlapping TESS cameras.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import logging

#--------------------------------------------------------------------------------------------------
def camera_overlap(dval):
	"""
	Check for complications arising from overlapping TESS cameras.

	A target can in some cases be simultaniously observed from two cameras, which
	can (and has!) cause problems. These checks are to a large extend to capture
	older runs where these problems were not yet identified.

	Parameters:
		dval (:class:`DataValidation`): Data Validation object.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)
	logger.info("Checking for overlapping camera problems...")

	# If something overlaps in sector, starid and datasource,
	# they better have different cameras and ccds:
	dval.cursor.execute("SELECT * FROM todolist GROUP BY sector,starid,datasource,camera,ccd HAVING COUNT(*) > 1;")
	results = dval.cursor.fetchall()
	if len(results) > 0:
		logger.error("  Duplicate targets detected: %d", len(results))
	else:
		logger.info("  No duplicate targets.")

	# Check for duplicate lightcurves:
	# In later versions of the pipeline this is impossible, but check it anyway
	dval.cursor.execute("SELECT * FROM diagnostics GROUP BY lightcurve HAVING COUNT(*) > 1;")
	results = dval.cursor.fetchall()
	if len(results) > 0:
		logger.error("  Duplicate raw lightcurved detected: %d", len(results))
	else:
		logger.info("  No duplicate raw lightcurves.")

	if dval.corr:
		# Check for duplicate corrected lightcurves:
		# In later versions of the pipeline this is impossible, but check it anyway
		dval.cursor.execute("SELECT * FROM diagnostics_corr GROUP BY lightcurve HAVING COUNT(*) > 1;")
		results = dval.cursor.fetchall()
		if len(results) > 0:
			logger.error("  Duplicate corrected lightcurved detected: %d", len(results))
		else:
			logger.info("  No duplicate corrected lightcurves.")
