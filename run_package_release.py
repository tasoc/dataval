#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The purpose of this script is to extract all relevant information from the TODO-file about
files which are ready to be released.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import argparse
import sys
import logging
import warnings
import re
import os
import shutil
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
import sqlite3
from contextlib import closing
import functools
import multiprocessing
from tqdm import tqdm
from dataval import __version__
from dataval.utilities import find_tpf_files, get_filehash, TqdmLoggingHandler, CounterFilter
from dataval.release import check_fits_changes

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
				if os.path.exists(fname):
					os.remove(fname)
				os.rename(fname_original, fname)
				raise Exception("File check failed: %s" % fname)
		except: # noqa: E722
			logger.exception("Whoops: %s", fname)
			if os.path.exists(fname_original):
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

#--------------------------------------------------------------------------------------------------
def main():
	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Create movies of TESS cameras.')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('-o', '--overwrite', help='Overwrite existing release file.', action='store_true')
	parser.add_argument('-j', '--jobs', type=int, default=0, help="Number of parallel jobs.")
	parser.add_argument('--version', type=int, default=None, help="Check that files are of this version.")
	parser.add_argument('--tpf', type=str, default=None, help="Rootdir to search for TPF files.")
	parser.add_argument('todofile', type=str, help="TODO-file.")
	args = parser.parse_args()

	logging_level = logging.INFO
	if args.quiet:
		logging_level = logging.WARNING
	elif args.debug:
		logging_level = logging.DEBUG

	# Setup logging:
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	logger = logging.getLogger(__name__)
	logger.setLevel(logging_level)
	console = TqdmLoggingHandler()
	console.setFormatter(formatter)
	if not logger.hasHandlers():
		logger.addHandler(console)

	# Add a CounterFilter to the logger, which will count the number of log-records
	# being passed through the logger. Can be used to count the number of errors/warnings:
	_counterfilter = CounterFilter()
	logger.addFilter(_counterfilter)

	tqdm_settings = {
		'disable': not logger.isEnabledFor(logging.INFO)
	}

	# Parse input:
	force_version = args.version
	input_file = args.todofile
	tpf_rootdir = args.tpf
	if not os.path.isfile(input_file):
		logger.error("Input file does not exist: %s", input_file)
		return 2
	if tpf_rootdir is not None and not os.path.isdir(tpf_rootdir):
		logger.error("TPF_ROOTDIR does not exist: %s", tpf_rootdir)
		return 2

	# There was a known bug in the processing of versions <=5, so require that TPF rootdir is
	# provided for those versions:
	if force_version <= 5 and tpf_rootdir is None:
		logger.error("You should provide a TPF_ROOTDIR")
		return 2

	# Decide on the number of parallel jobs to start:
	threads = args.jobs
	if threads <= 0:
		threads = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
	logger.info("Using %d processes.", threads)

	# Open the SQLite file in read-only mode and retrieve the full list of files
	# that are due to be released:
	logger.info("Extracting list of files to release...")
	with closing(sqlite3.connect('file:' + input_file + '?mode=ro', uri=True)) as conn:
		conn.row_factory = sqlite3.Row
		cursor = conn.cursor()
		cursor.execute("SELECT * FROM corr_settings;")
		corr_settings = dict(cursor.fetchone())

		# Check that diagnostics_corr exists:
		cursor.execute("SELECT COUNT(*) AS antal FROM sqlite_master WHERE type='table' AND name='diagnostics_corr';")
		if cursor.fetchone()['antal'] != 1:
			logger.error("DIAGNOSTICS_CORR table does not exist")
			return 2

		# Check that datavalidation_corr exists:
		cursor.execute("SELECT COUNT(*) AS antal FROM sqlite_master WHERE type='table' AND name='datavalidation_corr';")
		if cursor.fetchone()['antal'] != 1:
			logger.error("DATAVALIDATION_CORR table does not exist")
			return 2

		# Check that there is a datavalidation entry for all todolist entries:
		cursor.execute("SELECT COUNT(*) AS antal FROM todolist LEFT JOIN datavalidation_corr ON todolist.priority=datavalidation_corr.priority WHERE datavalidation_corr.priority IS NULL;")
		if cursor.fetchone()['antal'] > 0:
			logger.error("DATAVALIDATION_CORR table seems incomplete")
			return 2

		cursor.execute("""
		SELECT
			todolist.priority,
			todolist.starid,
			todolist.camera,
			todolist.ccd,
			todolist.sector,
			todolist.datasource,
			diagnostics_corr.lightcurve,
			datavalidation_corr.dataval
		FROM todolist
			INNER JOIN diagnostics_corr ON todolist.priority=diagnostics_corr.priority
			INNER JOIN datavalidation_corr ON todolist.priority=datavalidation_corr.priority
		WHERE
			datavalidation_corr.approved=1;""")
		files_to_release = [dict(row) for row in cursor.fetchall()]
		cursor.close()

	# Extract information needed below:
	input_folder = os.path.dirname(input_file)
	corrector = corr_settings['corrector']

	if corrector not in ('cbv', 'ensemble'):
		logger.error("Invalid corrector value: %s", corrector)
		return 2

	# Do a simple check that all the files exists:
	logger.info("Checking file existance...")
	for row in tqdm(files_to_release, **tqdm_settings):
		fname = os.path.join(input_folder, row['lightcurve'])
		if not os.path.isfile(fname):
			logger.error("File not found: %s", fname)
			return 2

	fix_file_wrapper = functools.partial(fix_file,
		input_folder=input_folder,
		check_corrector=corrector[:3], # NOTE: Ensemble is only "ens" in filenames
		force_version=force_version,
		tpf_rootdir=tpf_rootdir)

	release_db = os.path.join(input_folder, 'release-{0:s}.sqlite'.format(corrector))
	logger.info("Release file: %s", release_db)
	if args.overwrite and os.path.isfile(release_db):
		os.remove(release_db)

	with closing(sqlite3.connect(release_db)) as conn:
		conn.row_factory = sqlite3.Row
		cursor = conn.cursor()
		cursor.execute("PRAGMA locking_mode=EXCLUSIVE;")
		cursor.execute("PRAGMA journal_mode=TRUNCATE;")
		cursor.execute("""CREATE TABLE IF NOT EXISTS release (
			priority INTEGER NOT NULL PRIMARY KEY,
			lightcurve TEXT NOT NULL,
			starid INTEGER NOT NULL,
			sector INTEGER NOT NULL,
			camera INTEGER NOT NULL,
			ccd INTEGER NOT NULL,
			cadence INTEGER NOT NULL,
			filesize INTEGER NOT NULL,
			filehash TEXT NOT NULL,
			datarel INTEGER NOT NULL,
			dataval INTEGER NOT NULL,
			dependency INTEGER
		);""")
		# Create the settings table if it doesn't exist:
		cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='settings';")
		if cursor.fetchone()[0] == 0:
			cursor.execute("""CREATE TABLE settings (
				dataval_version TEXT NOT NULL,
				corrector TEXT NOT NULL,
				version INTEGER
			);""")
			cursor.execute("INSERT INTO settings (dataval_version, corrector, version) VALUES (?,?,?);", [
				__version__,
				corrector,
				force_version
			])
		conn.commit()

		# Ensure that we are not running an existing file with new settings:
		cursor.execute("SELECT * FROM settings LIMIT 1;")
		settings = cursor.fetchone()
		if settings['corrector'] != corrector:
			logger.error("Inconsistent CORRECTOR provided")
			return 2
		if force_version is not None and settings['version'] != force_version:
			logger.error("Inconsistent VERSION provided")
			return 2

		cursor.execute("SELECT priority FROM release;")
		already_processed = set([row[0] for row in cursor.fetchall()])
		not_yet_processed = []
		for row in files_to_release:
			if row['priority'] not in already_processed:
				not_yet_processed.append(row)

		numfiles = len(not_yet_processed)
		logger.info("Already processed: %d", len(files_to_release)-numfiles)
		logger.info("To be processed: %d", numfiles)
		if numfiles == 0:
			logger.info("Nothing to process")
			return 0
		else:
			with multiprocessing.Pool(threads) as pool:
				if threads > 1:
					m = pool.imap
				else:
					m = map

				inserted = 0
				for info in tqdm(m(fix_file_wrapper, not_yet_processed), total=numfiles, **tqdm_settings):
					logger.debug(info)
					cursor.execute("INSERT INTO release (priority, lightcurve, starid, sector, camera, ccd, cadence, filesize, filehash, datarel, dataval, dependency) VALUES (?,?,?,?,?,?,?,?,?,?,?,?);", [
						info['priority'],
						info['lightcurve'],
						info['starid'],
						info['sector'],
						info['camera'],
						info['ccd'],
						info['cadence'],
						info['filesize'],
						info['filehash'],
						info['datarel'],
						info['dataval'],
						info['dependency']
					])

					inserted += 1
					if inserted >= 100:
						inserted = 0
						conn.commit()

			conn.commit()

		cursor.execute("PRAGMA journal_mode=DELETE;")
		conn.commit()
		cursor.close()

	# Check the number of errors or warnings issued, and convert these to a return-code:
	logcounts = _counterfilter.counter
	if logcounts.get('ERROR', 0) > 0 or logcounts.get('CRITICAL', 0) > 0:
		return 4
	elif logcounts.get('WARNING', 0) > 0:
		return 3
	return 0

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	multiprocessing.freeze_support() # for Windows support

	returncode = main()
	sys.exit(returncode)
