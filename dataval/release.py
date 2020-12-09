#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for data release creation.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import logging
import warnings
import re
import os
import shutil
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from .utilities import find_tpf_files, get_filehash

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

#--------------------------------------------------------------------------------------------------
regex_filename = re.compile(r'^tess(\d+)-s(\d+)-(\d)-(\d)-c(\d+)-dr(\d+)-v(\d+)-tasoc-(cbv|ens)_lc\.fits\.gz$')
regex_fileend = re.compile(r'\.fits\.gz$')

#--------------------------------------------------------------------------------------------------
def fix_file(row, input_folder=None, check_corrector=None, force_version=None, tpf_rootdir=None):

	logger = logging.getLogger(__name__)
	fname = os.path.join(input_folder, row['lightcurve'])

	fname_original = regex_fileend.sub('.original.fits.gz', fname)
	if os.path.exists(fname_original):
		raise Exception("ORIGINAL exists: %s" % fname_original)

	dataval = int(row['dataval'])
	modification_needed = False

	m = regex_filename.match(os.path.basename(fname))
	if not m:
		raise Exception("RegEx doesn't match!")

	starid = int(m.group(1))
	sector = int(m.group(2))
	camera = int(m.group(3))
	ccd = int(m.group(4))
	cadence = int(m.group(5))
	datarel = int(m.group(6))
	version = int(m.group(7))
	corrector = m.group(8)

	# Basic checks:
	if starid != row['starid']:
		raise Exception("STARID")
	if sector != row['sector']:
		raise Exception("SECTOR")
	if camera != row['camera']:
		raise Exception("CAMERA")
	if ccd != row['ccd']:
		raise Exception("CCD")
	#if cadence != row['cadence']:
	#	raise Exception("CADENCE")
	if force_version is not None and version != force_version:
		#modification_needed = True
		raise Exception("Version mismatch!")
	if corrector != check_corrector:
		raise Exception("CORRECTOR")

	# Do we really need to modify the FITS file?
	modification_needed = True # FORCE modification check!
	fix_wcs = False

	if dataval > 0:
		modification_needed = True

	if cadence == 120 and version <= 5:
		modification_needed = True
		fix_wcs = True

	# Find the starid of the TPF which was used to create this lightcurve:
	if row['datasource'] == 'tpf':
		dependency = row['starid']
	elif row['datasource'].startswith('tpf:'):
		dependency = int(row['datasource'][4:])
	else:
		dependency = None

	# Damn, it looks like a modification is needed:
	allow_change = []
	if modification_needed:
		logger.debug("Opening FITS file: %s", fname)
		modification_needed = False

		if fix_wcs:
			if tpf_rootdir is None:
				raise Exception("You need to provide a TPF_ROOTDIR")
			# Find out what the
			if dependency is None:
				raise Exception("We can't fix WCSs of FFI targets!")
			# Find the original TPF file and extract the WCS from its headers:
			tpf_file = find_tpf_files(tpf_rootdir, starid=dependency, sector=sector, camera=camera, ccd=ccd, cadence=cadence)
			if len(tpf_file) != 1:
				raise Exception("Could not find TPF file: starid=%d, sector=%d" % (dependency, sector))
			# Extract the FITS header with the correct WCS:
			with warnings.catch_warnings():
				warnings.filterwarnings('ignore', category=FITSFixedWarning)
				wcs_header = WCS(header=fits.getheader(tpf_file[0], extname='APERTURE'), relax=True).to_header(relax=True)

		shutil.copy(fname, fname_original)
		with fits.open(fname_original, mode='readonly', memmap=True) as hdu:
			prihdr = hdu[0].header

			# Check if the current DATAVAL matches what it should be:
			current_dataval = prihdr.get('DATAVAL')
			if current_dataval != dataval:
				modification_needed = True
				allow_change += ['DATAVAL']
				if current_dataval is None:
					# Insert DATAVAL keyword just before CHECKSUM:
					prihdr.insert('CHECKSUM', ('DATAVAL', dataval, 'Data validation flags'))
				else:
					prihdr['DATAVAL'] = dataval

			if corrector == 'ens' and version <= 5:
				if hdu['ENSEMBLE'].header.get('TDISP2') == 'E':
					logger.info("%s: Changing ENSEMBLE/TDISP2 header", fname)
					modification_needed = True
					allow_change += ['TDISP2']
					hdu['ENSEMBLE'].header['TDISP2'] = 'E26.17'

			if fix_wcs:
				logger.info("%s: Changing WCS", fname)
				modification_needed = True
				allow_change += ['CRPIX1', 'CRPIX2']
				hdu['APERTURE'].header.update(wcs_header)
				hdu['SUMIMAGE'].header.update(wcs_header)

			if modification_needed:
				hdu.writeto(fname, checksum=True, overwrite=True)

	if modification_needed:
		try:
			if check_fits_changes(fname_original, fname, allow_header_value_changes=allow_change):
				os.remove(fname_original)
			else:
				logger.error("File check failed: %s", fname)
				raise Exception("File check failed: %s" % fname)
		except: # noqa: E722
			logger.exception("Whoops: %s", fname)
			if os.path.isfile(fname_original) and os.path.getsize(fname_original) > 0:
				if os.path.exists(fname):
					os.remove(fname)
				os.rename(fname_original, fname)
			raise

	elif os.path.exists(fname_original):
		os.remove(fname_original)

	# Extract information from final file:
	filesize = os.path.getsize(fname)
	filehash = get_filehash(fname)

	return {
		'priority': row['priority'],
		'starid': row['starid'],
		'sector': row['sector'],
		'camera': row['camera'],
		'ccd': row['ccd'],
		'cadence': cadence,
		'lightcurve': row['lightcurve'],
		'dataval': dataval,
		'datarel': datarel,
		'version': version,
		'filesize': filesize,
		'filehash': filehash,
		'dependency': dependency
	}
