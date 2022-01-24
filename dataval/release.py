#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for data release creation.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import logging
import warnings
import re
import os
import shutil
import tempfile
import contextlib
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from .utilities import find_tpf_files, get_filehash

#--------------------------------------------------------------------------------------------------
@contextlib.contextmanager
def temporary_filename(**kwargs):
	"""
	Context that introduces a temporary file.

	Creates a temporary file, yields its name, and upon context exit, deletes it.
	(In contrast, tempfile.NamedTemporaryFile() provides a 'file' object and
	deletes the file as soon as that file object is closed, so the temporary file
	cannot be safely re-opened by another library or process.)

	Yields:
		The name of the temporary file.
	"""
	if 'delete' in kwargs:
		raise ValueError("DELETE keyword can not be used")
	try:
		f = tempfile.NamedTemporaryFile(delete=False, **kwargs)
		tmp_name = f.name
		f.close()
		yield tmp_name
	finally:
		if os.path.exists(tmp_name):
			os.remove(tmp_name)

#--------------------------------------------------------------------------------------------------
def atomic_copy(src, dst):
	"""
	Copy file (using shutil.copy2), but with higher likelihood of being an atomic operation.

	This is done by first copying to a temp file and then renaming this file to the final name.
	This is only atomic on POSIX systems.
	"""
	if os.path.exists(dst):
		raise FileExistsError(dst)

	with temporary_filename(dir=os.path.dirname(dst), suffix='.tmp') as tmp:
		try:
			shutil.copy2(src, tmp)
			os.rename(tmp, dst)
		except: # noqa: E722
			if os.path.exists(dst):
				os.remove(dst)
			raise

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
	for dh in diff.diff_hdus:
		# Historically this has changed in astropy, from containing two to four entries,
		# which is the reason for unpacking like this:
		d = dh[1]
		# Headers:
		hdr = d.diff_headers
		if not hdr.identical:
			if hdr.diff_keywords:
				# Keywords only in header A:
				if hdr.diff_keywords[0]:
					logger.error("%s: Extra keyword in original: %s", fname_str, hdr.diff_keywords[0])
					everything_ok = False
				# Keywords only in header B:
				if hdr.diff_keywords[1]:
					if any([key not in allow_header_value_changes for key in hdr.diff_keywords[1]]):
						logger.error("%s: Extra keyword in modified: %s", fname_str, hdr.diff_keywords[1])
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
		raise RuntimeError(f"ORIGINAL exists: {fname_original}")

	dataval = int(row['dataval'])
	modification_needed = False

	m = regex_filename.match(os.path.basename(fname))
	if not m:
		raise RuntimeError("RegEx doesn't match!")

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
		raise RuntimeError("STARID")
	if sector != row['sector']:
		raise RuntimeError("SECTOR")
	if camera != row['camera']:
		raise RuntimeError("CAMERA")
	if ccd != row['ccd']:
		raise RuntimeError("CCD")
	#if cadence != row['cadence']:
	#	raise RuntimeError("CADENCE")
	if force_version is not None and version != force_version:
		#modification_needed = True
		raise RuntimeError("Version mismatch!")
	if corrector != check_corrector:
		raise RuntimeError("CORRECTOR")

	# Do we really need to modify the FITS file?
	openfile_needed = True # FORCE modification check!
	fix_wcs = False

	# We need to open if there is a dataval to add to the header
	if dataval > 0:
		openfile_needed = True

	# Fix for bug with WCS being incorrect in TPF lightcurves
	if cadence == 120 and version <= 5:
		openfile_needed = True
		fix_wcs = True

	# Because of the problem with multiple MJD-OBS keywords
	# in FITS headers, we have to check files in these cases.
	# TODO: Modify this when we know the CAUSE of this SYMPTOM
	if version <= 5:
		openfile_needed = True

	# We need to open the ensemble files to find the lightcurve dependencies:
	if corrector == 'ens':
		openfile_needed = True

	# Find the starid of the TPF which was used to create this lightcurve:
	if row['datasource'] == 'tpf':
		dependency_tpf = row['starid']
	elif row['datasource'].startswith('tpf:'):
		dependency_tpf = int(row['datasource'][4:])
	else:
		dependency_tpf = None

	# Placeholder for dependencies between lightcurves:
	dependency_lc = None

	# Damn, it looks like a modification is needed:
	allow_change = []
	if openfile_needed:
		logger.debug("Opening FITS file: %s", fname)
		modification_needed = False

		if fix_wcs:
			if tpf_rootdir is None:
				raise RuntimeError("You need to provide a TPF_ROOTDIR")
			# Find out what the
			if dependency_tpf is None:
				raise RuntimeError("We can't fix WCSs of FFI targets!")
			# Find the original TPF file and extract the WCS from its headers:
			tpf_file = find_tpf_files(tpf_rootdir, starid=dependency_tpf, sector=sector, camera=camera, ccd=ccd, cadence=cadence)
			if len(tpf_file) != 1:
				raise RuntimeError(f"Could not find TPF file: starid={dependency_tpf:d}, sector={sector:d}")
			# Extract the FITS header with the correct WCS:
			with warnings.catch_warnings():
				warnings.filterwarnings('ignore', category=FITSFixedWarning)
				wcs_header = WCS(header=fits.getheader(tpf_file[0], extname='APERTURE'), relax=True).to_header(relax=True)

		atomic_copy(fname, fname_original)
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

			if corrector == 'ens':
				# Pick out the list of TIC-IDs used to build ensemble:
				dependency_lc = list(hdu['ENSEMBLE'].data['TIC'])

			if fix_wcs:
				logger.info("%s: Changing WCS", fname)
				modification_needed = True
				allow_change += ['CRPIX1', 'CRPIX2']
				mjdref_remove = ('MJDREF' not in hdu['APERTURE'].header)
				hdu['APERTURE'].header.update(wcs_header)
				hdu['SUMIMAGE'].header.update(wcs_header)
				if mjdref_remove:
					hdu['APERTURE'].header.remove('MJDREF', ignore_missing=True, remove_all=True)
					hdu['SUMIMAGE'].header.remove('MJDREF', ignore_missing=True, remove_all=True)

			# Fix bug with multiple MJD-OBS keywords in FITS headers:
			if version <= 5: # TODO: Modify this when we know the CAUSE of this SYMPTOM
				for extname in ('APERTURE', 'SUMIMAGE'):
					if list(hdu[extname].header.keys()).count('MJD-OBS') > 1:
						logger.info("%s: Multiple MJD-OBS in %s", fname, extname)
						mjdobs = hdu[extname].header['MJD-OBS']
						indx = hdu[extname].header.index('MJD-OBS')
						hdu[extname].header.remove('MJD-OBS', remove_all=True)
						hdu[extname].header.insert(indx, ('MJD-OBS', mjdobs, '[d] MJD at start of observation'))
						allow_change += ['MJD-OBS']
						modification_needed = True

			if modification_needed:
				hdu.writeto(fname, output_verify='exception', checksum=True, overwrite=True)

	if modification_needed:
		try:
			if check_fits_changes(fname_original, fname, allow_header_value_changes=allow_change):
				os.remove(fname_original)
			else:
				logger.error("File check failed: %s", fname)
				raise RuntimeError(f"File check failed: {fname}")
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

	# Check that filesize is not zero:
	if filesize == 0:
		raise RuntimeError(f"File has zero size: {fname}")

	return {
		'priority': row['priority'],
		'starid': row['starid'],
		'sector': row['sector'],
		'camera': row['camera'],
		'ccd': row['ccd'],
		'cbv_area': row['cbv_area'],
		'cadence': cadence,
		'lightcurve': row['lightcurve'],
		'dataval': dataval,
		'datarel': datarel,
		'version': version,
		'filesize': filesize,
		'filehash': filehash,
		'dependency_tpf': dependency_tpf,
		'dependency_lc': dependency_lc
	}

#--------------------------------------------------------------------------------------------------
def process_cbv(fname, input_folder, force_version=None):

	m = re.match(r'^tess-s(\d{4})-c(\d{4})-a(\d{3})-v(\d+)-tasoc_cbv\.fits\.gz$', os.path.basename(fname))
	if m is None:
		raise RuntimeError("CBV file does not have the correct file name format!")
	fname_sector = int(m.group(1))
	fname_cadence = int(m.group(2))
	fname_cbvarea = int(m.group(3))
	fname_camera = int(m.group(3)[0])
	fname_ccd = int(m.group(3)[1])
	fname_version = int(m.group(4))

	# Open the FITS file and check the headers:
	with fits.open(fname, mode='readonly', memmap=True) as hdu:
		hdr = hdu[0].header
		sector = hdr['SECTOR']
		camera = hdr['CAMERA']
		ccd = hdr['CCD']
		data_rel = hdr['DATA_REL']
		version = hdr['VERSION']
		cbv_area = hdr['CBV_AREA']
		cadence = hdr['CADENCE']

		time = np.asarray(hdu[1].data['TIME'])
		cadence_time = int(np.round(86400*np.median(np.diff(time))))

	# Check that the filename and headers are consistent:
	if sector != fname_sector:
		raise RuntimeError("SECTOR does not match filename.")
	if camera != fname_camera:
		raise RuntimeError("CAMERA does not match filename.")
	if ccd != fname_ccd:
		raise RuntimeError("CCD does not match filename.")
	if cadence != fname_cadence:
		raise RuntimeError("CADENCE does not match filename.")
	if cadence != cadence_time:
		raise RuntimeError("CADENCE does not match TIME.")
	if cbv_area != fname_cbvarea:
		raise RuntimeError("CBV_AREA does not match filename.")
	if version != fname_version:
		raise RuntimeError("VERSION does not match filename.")

	if force_version is not None and version != force_version:
		raise RuntimeError("Version mismatch!")

	path = os.path.relpath(fname, input_folder).replace('\\', '/')

	# Extract information from final file:
	filesize = os.path.getsize(fname)
	filehash = get_filehash(fname)

	# Check that filesize is not zero:
	if filesize == 0:
		raise RuntimeError("File has zero size: %s", fname)

	return {
		'path': path,
		'sector': sector,
		'camera': camera,
		'ccd': ccd,
		'cbv_area': cbv_area,
		'cadence': cadence,
		'datarel': data_rel,
		'version': version,
		'filesize': filesize,
		'filehash': filehash
	}
