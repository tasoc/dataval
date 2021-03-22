#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Validation module for TASOC Pipeline.

.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import os
import logging
import numpy as np
import sqlite3
from tqdm import tqdm
import itertools

# Plotting:
from .plots import matplotlib as mpl, plots_interactive

# Data Validation methods in separate sub-packages:
from .noise_metrics import noise_metrics
from .pixinaperture import pixinaperture
from .stampsize import stampsize
from .contam import contam
from .mag_dist import mag_dist
from .mag2flux import mag2flux
from .cleanup import cleanup
from .camera_overlap import camera_overlap
from .calctime import calctime, calctime_corrections
from .waittime import waittime
from .haloswitch import haloswitch

# Local packages:
from .status import STATUS
from .quality import DatavalQualityFlags
from .utilities import CounterFilter, find_lightcurve_files
from .version import get_version

#--------------------------------------------------------------------------------------------------
class DataValidation(object):

	def __init__(self, todo_file, output_folder=None, corr=False, validate=True,
		colorbysector=False, ext='png', showplots=False, sysnoise=5.0):
		"""
		Initialize DataValidation object.

		Parameters:
			todo_file (str): DESCRIPTION.
			output_folder (str, optional): DESCRIPTION. Defaults to None.
			corr (bool, optional): DESCRIPTION. Defaults to False.
			validate (bool, optional): DESCRIPTION. Defaults to True.
			colorbysector (bool, optional): DESCRIPTION. Defaults to False.
			ext (str, optional): DESCRIPTION. Defaults to 'png'.
			showplots (bool, optional): DESCRIPTION. Defaults to False.
			sysnoise (float, optional): DESCRIPTION. Defaults to 0.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		logger = logging.getLogger('dataval')

		# Store inputs:
		if os.path.isdir(todo_file):
			# If it was just a directory, then append the default todo-file:
			self.input_folder = todo_file
			todo_file = os.path.join(todo_file, 'todo.sqlite')
		else:
			self.input_folder = os.path.dirname(todo_file)
		self.extension = ext
		self.show = showplots
		self.outfolder = output_folder
		self.sysnoise = sysnoise
		self.doval = validate
		self.color_by_sector = colorbysector
		self.corr = corr
		self.corr_method = None

		# Other settings:
		self.random_seed = 2187
		self._random_state = None

		if self.corr:
			self.dataval_table = 'datavalidation_corr'
			subdir = 'corr'
		else:
			self.dataval_table = 'datavalidation_raw'
			subdir = 'raw'
		if not self.doval:
			self.dataval_table += '_temp'
			logfilename = 'dataval.log'
		else:
			logfilename = 'dataval_save.log'

		# Make sure it is an absolute path:
		todo_file = os.path.abspath(todo_file)
		logger.info("Loading input data from '%s'", todo_file)
		if not os.path.isfile(todo_file):
			raise FileNotFoundError(f"TODO file not found: '{todo_file}'")

		# Open the SQLite file:
		self.conn = sqlite3.connect(todo_file)
		self.conn.row_factory = sqlite3.Row
		self.cursor = self.conn.cursor()
		self.cursor.execute("PRAGMA foreign_keys=ON;")

		# Check if corrections have been run:
		self.cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='diagnostics_corr';")
		self.corrections_done = bool(self.cursor.fetchone()[0] == 1)
		if self.corr and not self.corrections_done:
			self.close()
			raise ValueError("Can not run dataval on corr when corrections have not been run")

		# Add method_used to the diagnostics table if it doesn't exist:
		self.cursor.execute("PRAGMA table_info(diagnostics)")
		if 'method_used' not in [r['name'] for r in self.cursor.fetchall()]:
			# Since this one is NOT NULL, we have to do some magic to fill out the
			# new column after creation, by finding keywords in other columns.
			# This can be a pretty slow process, but it only has to be done once.
			logger.debug("Adding method_used column to diagnostics")
			self.cursor.execute("ALTER TABLE diagnostics ADD COLUMN method_used TEXT NOT NULL DEFAULT 'aperture';")
			for m in ('aperture', 'halo', 'psf', 'linpsf'):
				self.cursor.execute("UPDATE diagnostics SET method_used=? WHERE priority IN (SELECT priority FROM todolist WHERE method=?);", [m, m])
			self.cursor.execute("UPDATE diagnostics SET method_used='halo' WHERE method_used='aperture' AND errors LIKE '%Automatically switched to Halo photometry%';")
			self.conn.commit()

		# Add the CADENCE column to todolist, if it doesn't exist:
		# This is only for backwards compatibility.
		self.cursor.execute("PRAGMA table_info(todolist)")
		existing_columns = [r['name'] for r in self.cursor.fetchall()]
		if 'cadence' not in existing_columns:
			logger.debug("Adding CADENCE column to todolist")
			self.cursor.execute("ALTER TABLE todolist ADD COLUMN cadence INTEGER DEFAULT NULL;")
			self.cursor.execute("UPDATE todolist SET cadence=1800 WHERE datasource='ffi' AND sector < 27;")
			self.cursor.execute("UPDATE todolist SET cadence=600 WHERE datasource='ffi' AND sector >= 27 AND sector <= 55;")
			self.cursor.execute("UPDATE todolist SET cadence=120 WHERE datasource!='ffi' AND sector < 27;")
			self.cursor.execute("SELECT COUNT(*) AS antal FROM todolist WHERE cadence IS NULL;")
			if self.cursor.fetchone()['antal'] > 0:
				self.close()
				raise ValueError("TODO-file does not contain CADENCE information and it could not be determined automatically. Please recreate TODO-file.")
			self.conn.commit()

		# Get the corrector that was run on this TODO-file, if the corr_settings table is available:
		if self.corr:
			self.cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='corr_settings';")
			if self.cursor.fetchone()[0] == 1:
				self.cursor.execute("SELECT corrector FROM corr_settings LIMIT 1;")
				row = self.cursor.fetchone()
				if row is not None and row['corrector'] is not None:
					self.corr_method = row['corrector'].strip()
					subdir += '-' + row['corrector'].strip()

		# Create table for data-validation:
		# Depending if we are saving the results or not (self.doval) we are creating
		# it either as a real table, or as a TEMPORARY table, which will only exist in memory
		# as long as the database connection is opened.
		logger.info("Creating datavalidation table...")
		self.cursor.execute('DROP TABLE IF EXISTS ' + self.dataval_table + ';')
		if self.doval:
			self.cursor.execute("CREATE TABLE IF NOT EXISTS " + self.dataval_table + """ (
				priority INTEGER PRIMARY KEY ASC NOT NULL,
				dataval INTEGER NOT NULL DEFAULT 0,
				approved BOOLEAN,
				FOREIGN KEY (priority) REFERENCES todolist(priority) ON DELETE CASCADE ON UPDATE CASCADE
			);""")
		else:
			# Temporary tables can not use foreign keys to real tables, which is why
			# we are not putting in the same foreign key here.
			self.cursor.execute("CREATE TEMPORARY TABLE " + self.dataval_table + """ (
				priority INTEGER PRIMARY KEY ASC NOT NULL,
				dataval INTEGER NOT NULL DEFAULT 0,
				approved BOOLEAN
			);""")

		# Fill out the table with zero dataval and NULL in approved:
		logger.info("Initializing datavalidation table...")
		self.cursor.execute("INSERT INTO " + self.dataval_table + " (priority) SELECT priority FROM todolist;")
		self.conn.commit()

		# Create a couple of indicies on the two columns:
		logger.info("Creating indicies on datavalidation table...")
		self.cursor.execute("CREATE INDEX IF NOT EXISTS " + self.dataval_table + "_approved_idx ON " + self.dataval_table + " (approved);")
		self.cursor.execute("CREATE INDEX IF NOT EXISTS " + self.dataval_table + "_dataval_idx ON " + self.dataval_table + " (dataval);")
		self.conn.commit()

		logger.info("Creating lightcurve indicies...")
		self.cursor.execute("CREATE INDEX IF NOT EXISTS diagnostics_lightcurve_idx ON diagnostics (lightcurve);")
		if self.corrections_done:
			self.cursor.execute("CREATE INDEX IF NOT EXISTS diagnostics_corr_lightcurve_idx ON diagnostics_corr (lightcurve);")
		self.conn.commit()

		# Create output directory:
		if self.outfolder is None:
			self.outfolder = os.path.join(self.input_folder, 'data_validation', subdir)
		os.makedirs(self.outfolder, exist_ok=True)
		logger.info("Putting output data in '%s'", self.outfolder)

		# Also write any logging output to the
		formatter = logging.Formatter('%(asctime)s - %(levelname)-7s - %(funcName)-10.10s - %(message)s')
		self._filehandler = logging.FileHandler(os.path.join(self.outfolder, logfilename), mode='w')
		self._filehandler.setFormatter(formatter)
		self._filehandler.setLevel(logging.INFO)
		logger.addHandler(self._filehandler)

		# Add a CounterFilter to the logger, which will count the number of log-records
		# being passed through the logger. Can be used to count the number of errors/warnings:
		self._counterfilter = CounterFilter()
		logger.addFilter(self._counterfilter)

		# Log the version of the data validation being run:
		logger.info("Data Validation version: %s", get_version())

		# Write to log if we are saving or not:
		if self.doval:
			logger.info("Saving final validations in TODO-file.")
		else:
			logger.info("Not saving final validations in TODO-file.")

		# Get the range of Tmags in the tables:
		tmag_limits = self.search_database(select=['MIN(tmag) AS tmag_min', 'MAX(tmag) AS tmag_max'])[0]
		self.tmag_limits = (tmag_limits['tmag_min']-0.5, tmag_limits['tmag_max']+0.5)

		# Get the distinct list of available cadences:
		self.cadences = self.search_database(select='cadence', distinct=True, order_by='cadence')
		self.cadences = [int(cad['cadence']) for cad in self.cadences]

		# Plot settings:
		if self.show:
			plots_interactive()
		mpl.style.use(os.path.join(os.path.dirname(__file__), 'dataval.mplstyle'))
		mpl.rcParams['savefig.format'] = self.extension

	#----------------------------------------------------------------------------------------------
	def __enter__(self):
		return self

	#----------------------------------------------------------------------------------------------
	def __exit__(self, *args):
		self.close()

	#----------------------------------------------------------------------------------------------
	def __del__(self):
		self.close()

	#----------------------------------------------------------------------------------------------
	def close(self):
		"""Close DataValidation object and all associated objects."""
		mpl.style.use('default')
		if hasattr(self, 'cursor'):
			try:
				self.cursor.close()
			except sqlite3.ProgrammingError:
				pass
		if hasattr(self, 'conn'):
			self.conn.close()

		# Close the logging FileHandler:
		if hasattr(self, '_filehandler'):
			logger = logging.getLogger('dataval')
			self._filehandler.close()
			logger.removeHandler(self._filehandler)

	#----------------------------------------------------------------------------------------------
	@property
	def logcounts(self):
		return self._counterfilter.counter

	#----------------------------------------------------------------------------------------------
	@property
	def random_state(self):
		if self._random_state is None:
			self._random_state = np.random.default_rng(self.random_seed)
		return self._random_state

	#----------------------------------------------------------------------------------------------
	def search_database(self, select=None, search=None, order_by=None, limit=None, distinct=False,
		joins=None):
		"""
		Search list of lightcurves and return a list of tasks/stars matching the given criteria.

		Parameters:
			search (list of strings or None): Conditions to apply to the selection of stars
				from the database.
			order_by (list, string or None): Column to order the database output by.
			limit (int or None): Maximum number of rows to retrieve from the database.
				If limit is None, all the rows are retrieved.
			distinct (boolean): Boolean indicating if the query should return unique elements only.

		Returns:
			list of dicts: Returns all stars retrieved by the call to the database as dicts/tasks
			that can be consumed directly by load_lightcurve

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		logger = logging.getLogger('dataval')

		# Which columns to select from the tables:
		if select is None:
			select = '*'
		elif isinstance(select, (list, tuple)):
			select = ",".join(select)

		# Search constraints:
		# Default is to only pass through targets that made the last step successfully
		default_search = ['status IN (1,3)']
		if self.corr:
			default_search.append('corr_status IN (1,3)')

		if isinstance(search, str): search = [search,]
		search = default_search if search is None else default_search + search
		search = "WHERE " + " AND ".join(search)

		if order_by is None:
			order_by = ''
		elif isinstance(order_by, (list, tuple)):
			order_by = " ORDER BY " + ",".join(order_by)
		elif isinstance(order_by, str):
			order_by = " ORDER BY " + order_by

		limit = '' if limit is None else " LIMIT %d" % limit

		# Which tables to join together:
		default_joins = ['INNER JOIN diagnostics ON todolist.priority=diagnostics.priority']

		if self.corrections_done:
			default_joins.append('LEFT JOIN diagnostics_corr ON todolist.priority=diagnostics_corr.priority')

		joins = default_joins if joins is None else default_joins + joins

		# Create query:
		query = "SELECT {distinct:s}{select:s} FROM todolist {joins:s} {search:s}{order_by:s}{limit:s};".format(
			distinct='DISTINCT ' if distinct else '',
			select=select,
			joins=' '.join(joins),
			search=search,
			order_by=order_by,
			limit=limit
		)

		# Ask the database:
		logger.debug("Running query: %s", query)
		self.cursor.execute(query)

		return [dict(row) for row in self.cursor.fetchall()]

	#----------------------------------------------------------------------------------------------
	def update_dataval(self, priorities, values):
		"""
		Update data validation table in database.

		Parameters:
			priorities (array): Array of priorities.
			values (array): Array of data validation flags to be assigned each corresponding priority.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		logger = logging.getLogger('dataval')

		values = np.asarray(values, dtype='int32')
		v = [(int(val), int(pri)) for pri, val in zip(priorities, values) if val != 0]
		if v:
			self.cursor.executemany("UPDATE " + self.dataval_table + " SET dataval=(dataval | ?) WHERE priority=?;", v)
			self.conn.commit()
			logger.info("Updated %d entries of %d possible.", self.cursor.rowcount, len(v))
		else:
			logger.info("Nothing to update.")

	#----------------------------------------------------------------------------------------------
	def validate(self):
		"""
		Run all validations and write out summary.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		logger = logging.getLogger('dataval')
		logger.info('--------------------------------------------------------')

		# Run the cleanup as the first, since this may actually change things:
		self.cleanup()

		# Run all the validation subroutines:
		self.basic()
		self.mag2flux()
		self.pixinaperture()
		self.stampsize()
		self.contam()
		self.noise_metrics()
		self.mag_dist()
		self.calctime()
		self.calctime_corrections()
		self.waittime()
		self.haloswitch()
		self.camera_overlap()

		# All the data validation flags are now saved in the database table, so let's combine
		# them and mark which targets should be approved:
		logger.info('--------------------------------------------------------')
		logger.info("Setting approved flags...")
		self.cursor.execute("UPDATE " + self.dataval_table + " SET approved=1 WHERE dataval=0;")
		self.cursor.execute("UPDATE " + self.dataval_table + " SET approved=(dataval & %d = 0) WHERE dataval > 0;" % DatavalQualityFlags.DEFAULT_BITMASK)

		self.cursor.execute("UPDATE " + self.dataval_table + " SET approved=0 WHERE priority IN (SELECT priority FROM todolist WHERE status NOT IN ({ok:d},{warning:d}));".format(
			ok=STATUS.OK.value,
			warning=STATUS.WARNING.value,
		))

		if self.corr:
			self.cursor.execute("UPDATE " + self.dataval_table + " SET approved=0 WHERE priority IN (SELECT priority FROM todolist WHERE corr_status NOT IN ({ok:d},{warning:d}));".format(
				ok=STATUS.OK.value,
				warning=STATUS.WARNING.value,
			))
		self.conn.commit()

		# Check that all entries have been set:
		self.cursor.execute("SELECT COUNT(*) AS antal FROM " + self.dataval_table + " WHERE approved IS NULL;")
		if self.cursor.fetchone()['antal'] > 0:
			logger.error("Not all approved were set")

		# Write out summary of validations
		logger.info('--------------------------------------------------------')
		logger.info("Summary of approved and rejected targets:")
		for camera, ccd in itertools.product((1,2,3,4), (1,2,3,4)):
			self.cursor.execute("SELECT COUNT(*) AS antal FROM todolist INNER JOIN " + self.dataval_table + " ON todolist.priority=" + self.dataval_table + ".priority WHERE status!=? AND camera=? AND ccd=? AND approved=1;", (
				STATUS.SKIPPED.value,
				camera,
				ccd
			))
			count_approved = self.cursor.fetchone()['antal']
			self.cursor.execute("SELECT COUNT(*) AS antal FROM todolist INNER JOIN " + self.dataval_table + " ON todolist.priority=" + self.dataval_table + ".priority WHERE status!=? AND camera=? AND ccd=? AND approved=0;", (
				STATUS.SKIPPED.value,
				camera,
				ccd
			))
			count_notapproved = self.cursor.fetchone()['antal']

			percent = 100*count_notapproved/(count_notapproved + count_approved)
			logger.info("  CAMERA=%d, CCD=%d: %.2f%% (%d rejected, %d approved)", camera, ccd, percent, count_notapproved, count_approved)

		self.cursor.execute("SELECT COUNT(*) AS antal FROM todolist INNER JOIN " + self.dataval_table + " ON todolist.priority=" + self.dataval_table + ".priority WHERE status!={skipped:d} AND approved=1;".format(
			skipped=STATUS.SKIPPED.value
		))
		count_total_approved = self.cursor.fetchone()['antal']
		self.cursor.execute("SELECT COUNT(*) AS antal FROM todolist INNER JOIN " + self.dataval_table + " ON todolist.priority=" + self.dataval_table + ".priority WHERE status!={skipped:d} AND approved=0;".format(
			skipped=STATUS.SKIPPED.value
		))
		count_total_notapproved = self.cursor.fetchone()['antal']

		percent = 100*count_total_notapproved/(count_total_notapproved + count_total_approved)
		logger.info("  TOTAL: %.2f%% (%d rejected, %d approved)", percent, count_total_notapproved, count_total_approved)

		logger.info("Reasons for rejections:")
		for b in range(14): # TODO: Loop over DatavalQualityFlags instead - requires it to be a real enum
			flag = 2**b

			# Only show flags that cause rejection:
			if flag & DatavalQualityFlags.DEFAULT_BITMASK == 0:
				continue

			# Count the number of targets where the flag is set:
			self.cursor.execute("SELECT COUNT(*) AS antal FROM todolist INNER JOIN " + self.dataval_table + " ON todolist.priority=" + self.dataval_table + ".priority WHERE status IN (:ok,:warning) AND dataval > 0 AND dataval & :dataval != 0;", {
				'ok': STATUS.OK.value,
				'warning': STATUS.WARNING.value,
				'dataval': flag
			})
			count_flag = self.cursor.fetchone()['antal']

			percent = 100*count_flag/count_total_notapproved
			logger.info("  %s: %d (%.2f%%)", flag, count_flag, percent)

		logger.info('--------------------------------------------------------')

	#----------------------------------------------------------------------------------------------
	def basic(self, warn_errors_ratio=0.05):
		"""
		Perform basic checks of the TODO-file and the lightcurve files.

		Parameters:
			warn_errors_ratio (float, optional): Fraction of photometry ERRORs to OK and WARNINGs
				to warn about. Default=5%.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		logger = logging.getLogger('dataval')
		logger.info('Testing basics...')
		tqdm_settings = {'disable': None if logger.isEnabledFor(logging.INFO) else True}

		# Status that we should check for in the database. They should not be present if the
		# processing was completed correctly:
		bad_status = str(STATUS.UNKNOWN.value) + ',' + str(STATUS.STARTED.value) + ',' + str(STATUS.ABORT.value)

		# Check the status of the photometry:
		self.cursor.execute("SELECT COUNT(*) FROM todolist WHERE status IS NULL OR status IN (" + bad_status + ");")
		rowcount = self.cursor.fetchone()[0]
		if rowcount:
			logger.error("%d entries have not had PHOTOMETRY run", rowcount)
		else:
			logger.info("All PHOTOMETRY has been run.")

		# Summary of all photometry status:
		logger.info("Summary of photometry status:")
		self.cursor.execute("SELECT status,COUNT(*) AS antal FROM todolist WHERE status NOT IN ({ok:d},{warning:d}) GROUP BY status;".format(
			ok=STATUS.OK.value,
			warning=STATUS.WARNING.value
		))
		total_bad_phot_status = 0
		for sta in self.cursor.fetchall():
			total_bad_phot_status += sta['antal']
			logger.info("  %s: %d", STATUS(sta['status']).name, sta['antal'])
		logger.info("  TOTAL: %d", total_bad_phot_status)

		# Warn if it seems that there is a large number of ERROR, compared to OK and WARNING:
		logger.info("Checking number of photometry errors:")
		for camera, ccd in itertools.product((1,2,3,4), (1,2,3,4)):
			self.cursor.execute("SELECT COUNT(*) FROM todolist WHERE status IN ({ok:d},{warning:d}) AND camera={camera:d} AND ccd={ccd:d};".format(
				ok=STATUS.OK.value,
				warning=STATUS.WARNING.value,
				camera=camera,
				ccd=ccd
			))
			count_good = self.cursor.fetchone()[0]
			self.cursor.execute("SELECT COUNT(*) FROM todolist WHERE status={error:d} AND camera={camera:d} AND ccd={ccd:d};".format(
				error=STATUS.ERROR.value,
				camera=camera,
				ccd=ccd
			))
			count_errors = self.cursor.fetchone()[0]
			ratio = count_errors/(count_good + count_errors) if count_good + count_errors > 0 else 0
			if ratio > warn_errors_ratio:
				logger.warning("  CAMERA=%d, CCD=%d: High number of errors detected: %.2f%% (%d errors, %d good)",
					camera, ccd, 100*ratio, count_errors, count_good)
			else:
				logger.info("  CAMERA=%d, CCD=%d: %.2f%% (%d errors, %d good)",
					camera, ccd, 100*ratio, count_errors, count_good)

		# Check that everything that should have, has a diagnostics entry:
		# Ignore status=SKIPPED, since these will not have a diagnostics entry.
		self.cursor.execute("SELECT * FROM todolist LEFT JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE diagnostics.priority IS NULL AND status != {skipped:d};".format(
			skipped=STATUS.SKIPPED
		))
		rowcount = len(self.cursor.fetchall())
		logger.log(logging.ERROR if rowcount else logging.INFO, "%d entries missing in DIAGNOSTICS", rowcount)

		# Check photometry_skipped table. All stars marked as SKIPPED in photometry should
		# have an entry explaining which target that was responsible for it being skipped:
		# NOTE: This will currently fail for most TODO-files due to a bug/feature in the photometry
		#       code, where an entry is not created in all cases.
		self.cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='photometry_skipped';")
		if self.cursor.fetchone()[0] == 1:
			self.cursor.execute("SELECT COUNT(*) FROM todolist LEFT JOIN photometry_skipped ON todolist.priority=photometry_skipped.priority WHERE status={skipped:d} AND photometry_skipped.priority IS NULL;".format(
				skipped=STATUS.SKIPPED
			))
			rowcount = self.cursor.fetchone()[0]
			# TODO: For now, this is just a warning, because there is a known bug in photometry
			#       causing this to often having non-zero number of missing entries.
			logger.log(logging.WARNING if rowcount else logging.INFO, "%d entries missing in PHOTOMETRY_SKIPPED", rowcount)
		else:
			logger.warning("PHOTOMETRY_SKIPPED table not found!")

		# Check the status of corrections:
		if self.corr:
			self.cursor.execute("SELECT COUNT(*) FROM todolist WHERE corr_status IS NULL OR corr_status IN (" + bad_status + ");")
			rowcount = self.cursor.fetchone()[0]
			if rowcount:
				logger.error("%d entries have not had CORRECTIONS run", rowcount)
			else:
				logger.info("All CORRECTIONS have been run.")

			# Warn if it seems that there is a large number of ERROR, compared to OK and WARNING:
			logger.info("Checking number of correction errors:")
			for camera, ccd in itertools.product((1,2,3,4), (1,2,3,4)):
				self.cursor.execute("SELECT COUNT(*) FROM todolist WHERE corr_status IN ({ok:d},{warning:d}) AND camera={camera:d} AND ccd={ccd:d};".format(
					ok=STATUS.OK.value,
					warning=STATUS.WARNING.value,
					camera=camera,
					ccd=ccd
				))
				count_good = self.cursor.fetchone()[0]
				self.cursor.execute("SELECT COUNT(*) FROM todolist WHERE corr_status={error:d} AND camera={camera:d} AND ccd={ccd:d};".format(
					error=STATUS.ERROR.value,
					camera=camera,
					ccd=ccd
				))
				count_errors = self.cursor.fetchone()[0]
				ratio = count_errors/(count_good + count_errors) if count_good + count_errors > 0 else 0
				if ratio > warn_errors_ratio:
					logger.warning("  CAMERA=%d, CCD=%d: High number of errors detected: %.2f%% (%d errors, %d good)",
						camera, ccd, 100*ratio, count_errors, count_good)
				else:
					logger.info("  CAMERA=%d, CCD=%d: %.2f%% (%d errors, %d good)",
						camera, ccd, 100*ratio, count_errors, count_good)

			# Check that everything that should have, has a diagnostics_corr entry:
			# Ignore status=SKIPPED, since these will not have a diagnostics_corr entry.
			self.cursor.execute("SELECT * FROM todolist LEFT JOIN diagnostics_corr ON todolist.priority=diagnostics_corr.priority WHERE diagnostics_corr.priority IS NULL AND corr_status != {skipped:d};".format(
				skipped=STATUS.SKIPPED
			))
			rowcount = len(self.cursor.fetchall())
			logger.log(logging.ERROR if rowcount else logging.INFO, "%d entries missing in DIAGNOSTICS_CORR", rowcount)

		# Check for specific errors that should be flagged:
		# Patterns can contain wildcards (% or _):
		specific_errors = [
			'FileNotFoundError',
			'sqlite3.%',
			'TargetNotFoundError', # custom "error" set in photometry.TaskManager.save_result
			'TypeError'
		]

		logger.info("Checking for specific errors...")
		tbls = ('diagnostics', 'diagnostics_corr') if self.corr else ('diagnostics',)
		for tbl, keyword in itertools.product(tbls, specific_errors):
			self.cursor.execute('SELECT COUNT(*) FROM ' + tbl + ' WHERE errors IS NOT NULL AND errors LIKE "%' + keyword + ': %";')
			count_specificerror = self.cursor.fetchone()[0]
			logger.log(logging.ERROR if count_specificerror else logging.INFO,
				"  %s (%s): %d", keyword, tbl, count_specificerror)

		# Check if any raw lightcurve files are missing:
		logger.info("Checking if any raw lightcurve files are missing...")
		missing_phot_lightcurves = 0
		missing_phot_lightcurves_list = os.path.join(self.outfolder, 'missing_raw.txt')
		with open(missing_phot_lightcurves_list, 'w') as fid:
			self.cursor.execute("SELECT todolist.priority,lightcurve FROM todolist LEFT JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE status IN (1,3);")
			for row in tqdm(self.cursor.fetchall(), **tqdm_settings):
				if row['lightcurve'] is None or \
					not os.path.isfile(os.path.join(self.input_folder, row['lightcurve'])) or \
					os.path.getsize(os.path.join(self.input_folder, row['lightcurve'])) == 0:
					missing_phot_lightcurves += 1
					fid.write("{priority:6d}  {lightcurve:s}\n".format(**row))

		if missing_phot_lightcurves == 0:
			logger.info("All photometry lightcurves avaliable.")
			os.remove(missing_phot_lightcurves_list)
		else:
			logger.error("%d missing photometry lightcurves.", missing_phot_lightcurves)

		# Check of any corrected lightcurve files are missing:
		if self.corr:
			logger.info("Checking if any corrected lightcurve files are missing...")
			missing_corr_lightcurves = 0
			missing_corr_lightcurves_list = os.path.join(self.outfolder, 'missing_corr.txt')
			with open(missing_corr_lightcurves_list, 'w') as fid:
				self.cursor.execute("SELECT todolist.priority,diagnostics_corr.lightcurve FROM todolist LEFT JOIN diagnostics_corr ON todolist.priority=diagnostics_corr.priority WHERE corr_status IN (1,3);")
				for row in tqdm(self.cursor.fetchall(), **tqdm_settings):
					if row['lightcurve'] is None or \
						not os.path.isfile(os.path.join(self.input_folder, row['lightcurve'])) or \
						os.path.getsize(os.path.join(self.input_folder, row['lightcurve'])) == 0:
						missing_corr_lightcurves += 1
						fid.write("{priority:6d}  {lightcurve:s}\n".format(**row))

			if missing_corr_lightcurves == 0:
				logger.info("All corrected lightcurves avaliable.")
				os.remove(missing_corr_lightcurves_list)
			else:
				logger.error("%d missing corrected lightcurves.", missing_corr_lightcurves)

		# Checking for leftover lightcurve files:
		logger.info("Checking for any leftover orphaned lightcurve files...")
		leftover_lightcurves = 0
		leftover_lightcurves_list = os.path.join(self.outfolder, 'orphaned_lightcurves.txt')
		with open(leftover_lightcurves_list, 'w') as fid:
			logger.info("  Checking for orphaned raw lightcurves...")
			for fname in tqdm(find_lightcurve_files(self.input_folder, 'tess*-tasoc_lc.fits.gz'), **tqdm_settings):
				# Find relative path to find in database:
				relpath = os.path.relpath(fname, self.input_folder)
				logger.debug("Checking: %s", relpath)

				self.cursor.execute("SELECT * FROM diagnostics WHERE lightcurve=?;", [relpath])
				if self.cursor.fetchone() is None:
					leftover_lightcurves += 1
					fid.write(relpath + "\n")

			if self.corr:
				logger.info("  Checking for orphaned corrected lightcurves...")
				if self.corr_method is None:
					logger.error("Correction method not given")
				fname_filter = {'ensemble': 'ens', 'cbv': 'cbv', 'kasoc_filter': 'kf', None: '*'}[self.corr_method]
				for fname in tqdm(find_lightcurve_files(self.input_folder, 'tess*-tasoc-%s_lc.fits.gz' % fname_filter), **tqdm_settings):
					# Find relative path to find in database:
					relpath = os.path.relpath(fname, self.input_folder)
					logger.debug("Checking: %s", relpath)

					self.cursor.execute("SELECT * FROM diagnostics_corr WHERE lightcurve=?;", [relpath])
					if self.cursor.fetchone() is None:
						leftover_lightcurves += 1
						fid.write(relpath + "\n")

		if leftover_lightcurves == 0:
			logger.info("No orphaned lightcurves.")
			os.remove(leftover_lightcurves_list)
		else:
			logger.error("%d orphaned lightcurves.", leftover_lightcurves)

	#----------------------------------------------------------------------------------------------
	def pixinaperture(self):
		pixinaperture(self)

	#----------------------------------------------------------------------------------------------
	def stampsize(self):
		stampsize(self)

	#----------------------------------------------------------------------------------------------
	def noise_metrics(self):
		noise_metrics(self)

	#----------------------------------------------------------------------------------------------
	def contam(self):
		contam(self)

	#----------------------------------------------------------------------------------------------
	def mag_dist(self):
		mag_dist(self)

	#----------------------------------------------------------------------------------------------
	def mag2flux(self):
		mag2flux(self)

	#----------------------------------------------------------------------------------------------
	def cleanup(self):
		cleanup(self)

	#----------------------------------------------------------------------------------------------
	def camera_overlap(self):
		camera_overlap(self)

	#----------------------------------------------------------------------------------------------
	def calctime(self):
		calctime(self)

	#----------------------------------------------------------------------------------------------
	def calctime_corrections(self):
		calctime_corrections(self)

	#----------------------------------------------------------------------------------------------
	def waittime(self):
		waittime(self)

	#----------------------------------------------------------------------------------------------
	def haloswitch(self):
		haloswitch(self)
