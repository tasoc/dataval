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
import os
import sqlite3
from contextlib import closing
import functools
import multiprocessing
from tqdm import tqdm
from dataval import __version__
from dataval.utilities import TqdmLoggingHandler, CounterFilter
from dataval.release import fix_file

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
		corrector = corr_settings['corrector']

		if corrector not in ('cbv', 'ensemble'):
			logger.error("Invalid corrector value: %s", corrector)
			return 2

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

		# We are not releasing TPF-files from the Ensemble method:
		additional_constrainits = ''
		if corrector == 'ensemble':
			additional_constrainits = " AND todolist.datasource='ffi'"

		# Gather full list of all files to be released:
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
			datavalidation_corr.approved=1""" + additional_constrainits + ";")
		files_to_release = [dict(row) for row in cursor.fetchall()]
		cursor.close()

	# Extract information needed below:
	input_folder = os.path.dirname(input_file)

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
