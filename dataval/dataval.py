#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Validation module for TASOC Pipeline.

.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import os
import logging
import numpy as np
from bottleneck import ss
import sqlite3
import scipy.interpolate as INT
from scipy.optimize import minimize
from scipy.stats import binned_statistic as binning
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
from astropy.table import Table
from tqdm import tqdm
import enum
import itertools

# Plotting:
from .plots import plt, matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator, ScalarFormatter
import seaborn as sns

# Local packages:
from .quality import DatavalQualityFlags
from .utilities import mag2flux, mad, CounterFilter, find_lightcurve_files # rms_timescale, sphere_distance
from .noise_model import phot_noise

#--------------------------------------------------------------------------------------------------
class STATUS(enum.IntEnum):
	"""
	Status indicator of the status of the correction.

	"""
	UNKNOWN = 0 #: The status is unknown. The actual calculation has not started yet.
	STARTED = 6 #: The calculation has started, but not yet finished.
	OK = 1      #: Everything has gone well.
	ERROR = 2   #: Encountered a catastrophic error that I could not recover from.
	WARNING = 3 #: Something is a bit fishy. Maybe we should try again with a different algorithm?
	ABORT = 4   #: The calculation was aborted.
	SKIPPED = 5 #: The target was skipped because the algorithm found that to be the best solution.

#--------------------------------------------------------------------------------------------------
class DataValidation(object):

	#----------------------------------------------------------------------------------------------
	def __init__(self, input_folders, output_folder=None, corr=False, validate=True,
			colorbysector=False, ext='png', showplots=False, sysnoise=0):
		"""
		Initialize DataValidation object.

		Parameters:
			input_folders (list): DESCRIPTION.
			output_folder (str, optional): DESCRIPTION. Defaults to None.
			corr (bool, optional): DESCRIPTION. Defaults to False.
			validate (bool, optional): DESCRIPTION. Defaults to True.
			colorbysector (bool, optional): DESCRIPTION. Defaults to False.
			ext (str, optional): DESCRIPTION. Defaults to 'png'.
			showplots (bool, optional): DESCRIPTION. Defaults to False.
			sysnoise (float, optional): DESCRIPTION. Defaults to 0.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)

		# Store inputs:
		self.input_folders = input_folders
		self.extension = ext
		self.show = showplots
		self.outfolder = output_folder
		self.sysnoise = sysnoise
		self.doval = validate
		self.color_by_sector = colorbysector
		self.corr = corr
		self.corr_method = None

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

		# Load SQLite TODO files:
		# TODO: How do we handle cases with more than one input?
		for k, todo_file in enumerate(self.input_folders):
			# If it was just a directory, then append the default todo-file:
			if os.path.isdir(todo_file):
				todo_file = os.path.join(todo_file, 'todo.sqlite')
			else:
				self.input_folders[k] = os.path.dirname(todo_file)

			logger.info("Loading input data from '%s'", todo_file)
			if not os.path.isfile(todo_file):
				raise FileNotFoundError("TODO file not found: '%s'" % todo_file)

			# Open the SQLite file:
			self.conn = sqlite3.connect(todo_file)
			self.conn.row_factory = sqlite3.Row
			self.cursor = self.conn.cursor()
			self.cursor.execute("PRAGMA foreign_keys=ON;")

			# Check if corrections have been run:
			self.cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='diagnostics_corr';")
			self.corrections_done = bool(self.cursor.fetchone()[0] == 1)
			if self.corr and not self.corrections_done:
				raise ValueError("Can not run dataval on corr when corrections have not been run")

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

			# Create a couple of indicies on the two columns:
			self.cursor.execute("CREATE INDEX IF NOT EXISTS " + self.dataval_table + "_approved_idx ON " + self.dataval_table + " (approved);")
			self.cursor.execute("CREATE INDEX IF NOT EXISTS " + self.dataval_table + "_dataval_idx ON " + self.dataval_table + " (dataval);")
			self.conn.commit()

			logger.info("Creating lightcurve indicies...")
			self.cursor.execute("CREATE INDEX IF NOT EXISTS diagnostics_lightcurve_idx ON diagnostics (lightcurve);")
			if self.corrections_done:
				self.cursor.execute("CREATE INDEX IF NOT EXISTS diagnostics_corr_lightcurve_idx ON diagnostics_corr (lightcurve);")
			self.conn.commit()

			# Fill out the table with zero dataval and NULL in approved:
			self.cursor.execute("INSERT INTO " + self.dataval_table + " (priority) SELECT priority FROM todolist;")
			self.conn.commit()

		# Create output directory:
		if len(self.input_folders) == 1 and self.outfolder is None:
			self.outfolder = os.path.join(self.input_folders[0], 'data_validation', subdir)
		os.makedirs(self.outfolder, exist_ok=True)
		logger.info("Putting output data in '%s'", self.outfolder)

		# Also write any logging output to the
		formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
		self._filehandler = logging.FileHandler(os.path.join(self.outfolder, logfilename), mode='w')
		self._filehandler.setFormatter(formatter)
		self._filehandler.setLevel(logging.INFO)
		logger.addHandler(self._filehandler)

		# Add a CounterFilter to the logger, which will count the number of log-records
		# being passed through the logger. Can be used to count the number of errors/warnings:
		self._counterfilter = CounterFilter()
		logger.addFilter(self._counterfilter)

		# Write to log if we are saving or not:
		if self.doval:
			logger.info("Saving final validations in TODO-file.")
		else:
			logger.info("Not saving final validations in TODO-file.")

		# Get the range of Tmags in the tables:
		tmag_limits = self.search_database(select=['MIN(tmag) AS tmag_min', 'MAX(tmag) AS tmag_max'])[0]
		self.tmag_limits = (tmag_limits['tmag_min']-0.5, tmag_limits['tmag_max']+0.5)

		# Plot settings:
		if self.show:
			plt.switch_backend('TKAgg')
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
			logger = logging.getLogger(__name__)
			self._filehandler.close()
			logger.removeHandler(self._filehandler)

	#----------------------------------------------------------------------------------------------
	@property
	def logcounts(self):
		return self._counterfilter.counter

	#----------------------------------------------------------------------------------------------
	def search_database(self, select=None, search=None, order_by=None, limit=None, distinct=False, joins=None):
		"""
		Search list of lightcurves and return a list of tasks/stars matching the given criteria.

		Parameters:
			search (list of strings or None): Conditions to apply to the selection of stars from the database
			order_by (list, string or None): Column to order the database output by.
			limit (int or None): Maximum number of rows to retrieve from the database. If limit is None, all the rows are retrieved.
			distinct (boolean): Boolean indicating if the query should return unique elements only.

		Returns:
			list of dicts: Returns all stars retrieved by the call to the database as dicts/tasks that can be consumed directly by load_lightcurve

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)

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

		logger = logging.getLogger(__name__)

		values = np.asarray(values, dtype='int32')
		v = [(int(val), int(pri)) for pri, val in zip(priorities, values) if val != 0]
		if v:
			self.cursor.executemany("UPDATE " + self.dataval_table + " SET dataval=(dataval | ?) WHERE priority=?;", v)
			self.conn.commit()
			logger.info("Updated %d entries of %d possible.", self.cursor.rowcount, len(v))
		else:
			logger.info("Nothing to update.")

	#----------------------------------------------------------------------------------------------
	def Validations(self):
		"""
		Run all validations and write out summary.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)
		logger.info('--------------------------------------------------------')

		# Run all the validation subroutines:
		self.basic()
		self.plot_mag2flux()
		self.plot_pixinaperture()
		self.plot_contam()
		self.plot_noise()
		self.plot_stamp()
		self.plot_mag_dist()
		self.plot_waittime()
		self.plot_haloswitch()

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

		logger = logging.getLogger(__name__)
		logger.info('Testing basics...')
		tqdm_settings = {'disable': not logger.isEnabledFor(logging.INFO)}

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

		# Root directory for files assocuated with this TODO-file:
		rootdir = self.input_folders[0]

		# Check if any raw lightcurve files are missing:
		logger.info("Checking if any raw lightcurve files are missing...")
		missing_phot_lightcurves = 0
		missing_phot_lightcurves_list = os.path.join(self.outfolder, 'missing_raw.txt')
		with open(missing_phot_lightcurves_list, 'w') as fid:
			self.cursor.execute("SELECT todolist.priority,lightcurve FROM todolist LEFT JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE status IN (1,3);")
			for row in tqdm(self.cursor.fetchall(), **tqdm_settings):
				if row['lightcurve'] is None or not os.path.isfile(os.path.join(rootdir, row['lightcurve'])):
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
					if row['lightcurve'] is None or not os.path.isfile(os.path.join(rootdir, row['lightcurve'])):
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
		leftover_lightcurves_list = os.path.join(self.outfolder, 'leftover_lightcurves.txt')
		with open(leftover_lightcurves_list, 'w') as fid:
			for fname in tqdm(find_lightcurve_files(rootdir, 'tess*-tasoc_lc.fits.gz'), **tqdm_settings):
				# Find relative path to find in database:
				relpath = os.path.relpath(fname, rootdir)
				logger.debug("Checking: %s", relpath)

				self.cursor.execute("SELECT * FROM diagnostics WHERE lightcurve=?;", [relpath])
				if self.cursor.fetchone() is None:
					leftover_lightcurves += 1
					fid.write(relpath + "\n")

			if self.corr:
				if self.corr_method is None:
					logger.error("Correction method not given")
				fname_filter = {'ensemble': 'ens', 'cbv': 'cbv', 'kasoc_filter': 'kf', None: '*'}[self.corr_method]
				for fname in tqdm(find_lightcurve_files(rootdir, 'tess*-tasoc-%s_lc.fits.gz' % fname_filter), **tqdm_settings):
					# Find relative path to find in database:
					relpath = os.path.relpath(fname, rootdir)
					logger.debug("Checking: %s", relpath)

					self.cursor.execute("SELECT * FROM diagnostics_corr WHERE lightcurve=?;", [relpath])
					if self.cursor.fetchone() is None:
						leftover_lightcurves += 1
						fid.write(relpath + "\n")

		if leftover_lightcurves == 0:
			logger.info("No leftover lightcurves.")
			os.remove(leftover_lightcurves_list)
		else:
			logger.error("%d leftover lightcurves.", leftover_lightcurves)

	#----------------------------------------------------------------------------------------------
	def plot_contam(self):
		"""
		Function to plot the contamination against the stellar TESS magnitudes

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)
		logger.info('Plotting Contamination vs. Magnitude...')

		fig = plt.figure(figsize=(10, 5))
		fig.subplots_adjust(left=0.145, wspace=0.3, top=0.945, bottom=0.145, right=0.975)
		ax1 = fig.add_subplot(211)
		ax2 = fig.add_subplot(212)

		# Search database for all targets processed with aperture photometry:
		star_vals = self.search_database(
			select=['todolist.priority','todolist.sector','todolist.datasource','todolist.tmag','contamination'],
			search="(method IS NULL or method='aperture')")

		rgba_color = 'k'
		if self.color_by_sector:
			sec = np.array([star['sector'] for star in star_vals], dtype=int)
			sectors = np.array(list(set(sec)))
			if len(sectors) > 1:
				norm = colors.Normalize(vmin=1, vmax=len(sectors))
				scalarMap = cmx.ScalarMappable(norm=norm, cmap=plt.get_cmap('Set1') )
				rgba_color = np.array([scalarMap.to_rgba(s) for s in sec])

		tmags = np.array([star['tmag'] for star in star_vals], dtype='float64')
		cont = np.array([star['contamination'] for star in star_vals], dtype='float64')
		pri = np.array([star['priority'] for star in star_vals], dtype='int64')
		source = np.array([star['datasource'] for star in star_vals], dtype=str)

		# Indices for plotting
		with np.errstate(invalid='ignore'): # We know that some cont may be NaN
			idx_high_ffi = (cont > 1) & (source == 'ffi')
			idx_high_tpf = (cont > 1) & (source != 'ffi')
			idx_low_ffi = (cont <= 1) & (source == 'ffi')
			idx_low_tpf = (cont <= 1) & (source != 'ffi')

		# Remove nan contaminations (should be only from Halo targets)
		cont[idx_high_ffi] = 1.1
		cont[idx_high_tpf] = 1.1
		cont[np.isnan(cont)] = 1.2

		# Plot individual contamination points
		ax1.scatter(tmags[idx_low_ffi], cont[idx_low_ffi], marker='o', facecolors=rgba_color, color=rgba_color, alpha=0.1, rasterized=True)
		ax2.scatter(tmags[idx_low_tpf], cont[idx_low_tpf], marker='o', facecolors=rgba_color, color=rgba_color, alpha=0.1, rasterized=True)

		ax1.scatter(tmags[idx_high_ffi], cont[idx_high_ffi], marker='o', facecolors='None', color=rgba_color, alpha=0.9)
		ax1.scatter(tmags[(cont == 1.2) & (source == 'ffi')], cont[(cont == 1.2) & (source == 'ffi')], marker='o', facecolors='None', color='r', alpha=0.9)
		ax2.scatter(tmags[idx_high_tpf], cont[idx_high_tpf], marker='o', facecolors='None', color=rgba_color, alpha=0.9)
		ax2.scatter(tmags[(cont == 1.2) & (source != 'ffi')], cont[(cont == 1.2) & (source != 'ffi')], marker='o', facecolors='None', color='r', alpha=0.9)

		# Indices for finding validation limit
		#idx_low = (cont <= 1)
		# Compute median-bin curve
		#bin_means, bin_edges, binnumber = binning(tmags[idx_low], cont[idx_low], statistic='median', bins=15, range=(np.nanmin(tmags),np.nanmax(tmags)))
		#bin_width = (bin_edges[1] - bin_edges[0])
		#bin_centers = bin_edges[1:] - bin_width/2

		xmax = np.arange(0, 20, 1)
		ymax = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.12, 0.2, 0.3, 0.45, 0.6, 0.7, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])

		# Plot median-bin curve (1 and 5 times standadised MAD)
		#ax1.scatter(bin_centers, 1.4826*bin_means, marker='o', color='r')
		#ax1.scatter(bin_centers, 1.4826*5*bin_means, marker='.', color='r')
		ax1.plot(xmax, ymax, marker='.', color='r', ls='-')
		ax2.plot(xmax, ymax, marker='.', color='r', ls='-')

		cont_vs_mag = INT.InterpolatedUnivariateSpline(xmax, ymax)
		#mags = np.linspace(np.nanmin(tmags),np.nanmax(tmags),100)

		ax1.set_xlim(self.tmag_limits)
		ax2.set_xlim(self.tmag_limits)

		# Plotting stuff
		for axx in np.array([ax1, ax2]):
			#axx.plot(mags, cont_vs_mag(mags))

			axx.set_ylim([-0.05, 1.3])

			axx.axhline(y=0, ls='--', color='k', zorder=-1)

			axx.axhline(y=1.1, ls=':', color='k', zorder=-1)
			axx.axhline(y=1.2, ls=':', color='r', zorder=-1)
			axx.axhline(y=1, ls=':', color='k', zorder=-1)
			axx.set_xlabel('TESS magnitude')
			axx.set_ylabel('Contamination')

			axx.xaxis.set_major_locator(MultipleLocator(2))
			axx.xaxis.set_minor_locator(MultipleLocator(1))
			axx.yaxis.set_major_locator(MultipleLocator(0.2))
			axx.yaxis.set_minor_locator(MultipleLocator(0.1))
			#axx.legend(loc='upper left')

		###########
		fig.savefig(os.path.join(self.outfolder, 'contam'))
		if self.show:
			plt.show()
		else:
			plt.close(fig)

		# Assign validation bits
		dv = np.zeros_like(pri, dtype='int32')
		dv[cont >= 1] |= DatavalQualityFlags.InvalidContamination
		dv[(cont > cont_vs_mag(tmags)) & (cont < 1)] |= DatavalQualityFlags.ContaminationHigh
		self.update_dataval(pri, dv)

	#----------------------------------------------------------------------------------------------
	def compare_noise(self):
		"""
		Compare noise metrics before and after correction

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)
		logger.info('Plotting Noise Comparison...')

		if not self.corrections_done:
			logger.info("Can not run compare_noise when corrections have not been run")
			return

		fig1 = plt.figure(figsize=(15, 5))
		fig1.subplots_adjust(left=0.145, wspace=0.3, top=0.945, bottom=0.145, right=0.975)
		ax11 = fig1.add_subplot(121)
		ax12 = fig1.add_subplot(122)

		fig2 = plt.figure(figsize=(15, 5))
		fig2.subplots_adjust(left=0.145, wspace=0.3, top=0.945, bottom=0.145, right=0.975)
		ax21 = fig2.add_subplot(121)
		ax22 = fig2.add_subplot(122)

		fig3 = plt.figure(figsize=(15, 5))
		fig3.subplots_adjust(left=0.145, wspace=0.3, top=0.945, bottom=0.145, right=0.975)
		ax31 = fig3.add_subplot(121)
		ax32 = fig3.add_subplot(122)

		#if self.corr:
		star_vals = self.search_database(select=['todolist.priority','todolist.starid','todolist.datasource','todolist.sector','todolist.tmag','diagnostics_corr.rms_hour','diagnostics_corr.ptp','diagnostics.contamination','ccd'])
		factor = 1
		star_vals2 = self.search_database(select=['todolist.priority','todolist.starid','todolist.datasource','todolist.sector','todolist.tmag','diagnostics.rms_hour','diagnostics.ptp','diagnostics.contamination','ccd'])
		factor2 = 1e6

		tmags = np.array([star['tmag'] for star in star_vals], dtype=float)
		pri = np.array([star['priority'] for star in star_vals], dtype=int)
		rms = np.array([star['rms_hour']*factor for star in star_vals], dtype=float)
		ptp = np.array([star['ptp']*factor for star in star_vals], dtype=float)
		source = np.array([star['datasource'] for star in star_vals], dtype=str)
		contam = np.array([star['contamination'] for star in star_vals], dtype=float)

		pri2 = np.array([star['priority'] for star in star_vals2], dtype=int)
		tmags2 = np.array([star['tmag'] for star in star_vals2], dtype=float)
		rms2 = np.array([star['rms_hour']*factor2 for star in star_vals2], dtype=float)
		ptp2 = np.array([star['ptp']*factor2 for star in star_vals2], dtype=float)
		source2 = np.array([star['datasource'] for star in star_vals2], dtype=str)

		def overlap(a, b):
			"""
			return the indices in a that overlap with b, also returns
			the corresponding index in b only works if both a and b are unique!
			This is not very efficient but it works
			"""
			bool_a = np.in1d(a,b)
			ind_a = np.arange(len(a))
			ind_a = ind_a[bool_a]
			ind_b = np.array([np.argwhere(b == a[x]) for x in ind_a]).flatten()
			return ind_a,ind_b

		idx_o, idx2_o = overlap(pri[(source == 'ffi')], pri2[(source2 == 'ffi')])

		print(len(star_vals), len(star_vals2))
		print(len(idx_o), len(idx2_o))
		print(tmags[(source == 'ffi')][idx_o])
		print(tmags2[(source2 == 'ffi')][idx2_o])
		print(np.sum(tmags[(source == 'ffi')][idx_o]-tmags2[(source2 == 'ffi')][idx2_o]))

		#88518 956502

		#tcomp = np.array([tmags[(pri==i)] for i in pri_overlap[5000:10000]])
		#rmscomp = np.array([rms[(pri==i)]/rms2[(pri2==i)] for i in pri_overlap[5000:10000]])
		#ptpcomp = np.array([ptp[(pri==i)]/ptp2[(pri2==i)] for i in pri_overlap[5000:10000]])

		tcomp = tmags[(source == 'ffi')][idx_o]
		rmscomp = rms[(source == 'ffi')][idx_o]/rms2[(source2 == 'ffi')][idx2_o]
		ptpcomp = ptp[(source == 'ffi')][idx_o]/ptp2[(source2 == 'ffi')][idx2_o]

		#ccomp = contam[(source == 'ffi')][idx_o]

#		nbins=300
#		data1 = np.column_stack((tcomp, rmscomp))
#		k = kde.gaussian_kde(data1.T)
#		xi, yi = np.mgrid[tcomp.min():tcomp.max():nbins*1j, rmscomp.min():rmscomp.max():nbins*1j]
#		zi = np.log10(k(np.vstack([xi.flatten(), yi.flatten()])))
#		clevels = ax31.contour(xi, yi, zi.reshape(xi.shape),lw=.9,cmap='winter')#,zorder=90)
#
#		ax31.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.Greens, shading='gouraud')
#
#		p = clevels.collections[0].get_paths()
#		inside = np.full_like(tcomp,False,dtype=bool)
#		print(inside)
#		for level in p:
#			print(inside)
#			inside |= level.contains_points(list(zip(*(rmscomp,tcomp))))
#
#		print(inside)
#		print(inside.shape, tcomp.shape, rmscomp.shape)
#		ax31.scatter(tcomp[~inside],rmscomp[~inside],marker='.', color='0.2')
#
#		data2 = np.column_stack((tcomp, ptpcomp))
#		k = kde.gaussian_kde(data2.T)
#		xi, yi = np.mgrid[tcomp.min():tcomp.max():nbins*1j, ptpcomp.min():ptpcomp.max():nbins*1j]
#		zi = k(np.vstack([xi.flatten(), yi.flatten()]))
#		ax32.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.Greens, shading='gouraud',norm=colors.LogNorm(vmin=zi.min(), vmax=zi.max()))

		ax31.scatter(tcomp, rmscomp, marker='o', c=contam, alpha=0.01, label='30-min cadence', cmap=plt.get_cmap('PuOr'))
		ax32.scatter(tcomp, ptpcomp, marker='o', c=contam, alpha=0.01, label='30-min cadence', cmap=plt.get_cmap('PuOr'))

		bin_rms, bin_edge_rms, _ = binning(tcomp, rmscomp, statistic='median', bins=15, range=(np.nanmin(tcomp),np.nanmax(tcomp)))
		bin_ptp, bin_edge_ptp, _ = binning(tcomp, ptpcomp, statistic='median', bins=15, range=(np.nanmin(tcomp),np.nanmax(tcomp)))
		bin_width = (bin_edge_rms[1] - bin_edge_rms[0])
		bin_centers = bin_edge_rms[1:] - bin_width/2

		bin_rmsmad, bin_edges_rmsmad, _ = binning(tcomp, rmscomp, statistic=mad, bins=15, range=(np.nanmin(tcomp),np.nanmax(tcomp)))
		bin_ptpmad, bin_edges_ptpmad, _ = binning(tcomp, ptpcomp, statistic=mad, bins=15, range=(np.nanmin(tcomp),np.nanmax(tcomp)))

		ax31.errorbar(bin_centers, bin_rms, yerr=bin_rmsmad, ecolor='r', mec='r', mfc='w', capsize=0, marker='o', ls='')
		ax32.errorbar(bin_centers, bin_ptp, yerr=bin_ptpmad, ecolor='r', mec='r', mfc='w', capsize=0, marker='o', ls='')

		ax31.axhline(y=1, ls='--', color='r')
		ax32.axhline(y=1, ls='--', color='r')

		idx_lc = (source == 'ffi')
		idx_sc = (source != 'ffi')

#		ax11.scatter(tmags[idx_lc], rms[idx_lc], marker='o', facecolors=rgba_color, edgecolor=rgba_color, alpha=0.1, label='30-min cadence')
#		ax12.scatter(tmags[idx_sc], rms[idx_sc], marker='o', facecolors=rgba_color, edgecolor=rgba_color, alpha=0.1, label='2-min cadence')

		ax11.scatter(tmags[idx_lc], rms[idx_lc], marker='o', c=contam[idx_lc], alpha=0.1, label='30-min cadence', cmap=plt.get_cmap('PuOr'))
		ax12.scatter(tmags[idx_sc], rms[idx_sc], marker='o', c=contam[idx_sc], alpha=0.1, label='2-min cadence', cmap=plt.get_cmap('PuOr'))

#		ax21.scatter(tmags[idx_lc], ptp[idx_lc], marker='o', facecolors=rgba_color, edgecolor=rgba_color, alpha=0.1, label='30-min cadence')
#		ax22.scatter(tmags[idx_sc], ptp[idx_sc], marker='o', facecolors=rgba_color, edgecolor=rgba_color, alpha=0.1, label='2-min cadence')

		ax21.scatter(tmags[idx_lc], ptp[idx_lc], marker='o', c=contam[idx_lc], alpha=0.1, label='30-min cadence', cmap=plt.get_cmap('PuOr'))
		ax22.scatter(tmags[idx_sc], ptp[idx_sc], marker='o', c=contam[idx_sc], alpha=0.1, label='2-min cadence', cmap=plt.get_cmap('PuOr'))

#		ax11.scatter(tmags2[(source2=='ffi')], rms2[(source2=='ffi')], marker='o', facecolors='r', edgecolor='r', alpha=0.1, label='30-min cadence')
#		ax21.scatter(tmags2[(source2=='ffi')], ptp2[(source2=='ffi')], marker='o', facecolors='r', edgecolor='r', alpha=0.1, label='30-min cadence')

		# Plot theoretical lines
		mags = np.linspace(self.tmag_limits[0], self.tmag_limits[1], 200)

		# Expected *1-hour* RMS noise ffi
		# TODO: Update elat+elon based on observing sector?
		tot_noise_rms_ffi, vals_rms_ffi = phot_noise(mags, timescale=3600, cadpix=1800, sysnoise=self.sysnoise)
		ax11.semilogy(mags, vals_rms_ffi[:, 0], 'r-', label='Shot')
		ax11.semilogy(mags, vals_rms_ffi[:, 1], 'g--', label='Zodiacal')
		ax11.semilogy(mags, vals_rms_ffi[:, 2], '-', label='Read')
		ax11.semilogy(mags, vals_rms_ffi[:, 3], 'b--', label='Systematic')
		ax11.semilogy(mags, tot_noise_rms_ffi, 'k-', label='Total')

		# Expected *1-hour* RMS noise tpf
		# TODO: Update elat+elon based on observing sector?
		tot_noise_rms_tpf, vals_rms_tpf = phot_noise(mags, timescale=3600, cadpix=120, sysnoise=self.sysnoise)
		ax12.semilogy(mags, vals_rms_tpf[:, 0], 'r-', label='Shot')
		ax12.semilogy(mags, vals_rms_tpf[:, 1], 'g--', label='Zodiacal')
		ax12.semilogy(mags, vals_rms_tpf[:, 2], '-', label='Read')
		ax12.semilogy(mags, vals_rms_tpf[:, 3], 'b--', label='Systematic')
		ax12.semilogy(mags, tot_noise_rms_tpf, 'k-', label='Total')

		# Expected ptp for 30-min
		# TODO: Update elat+elon based on observing sector?
		tot_noise_ptp_ffi, vals_ptp_ffi = phot_noise(mags, timescale=1800, cadpix=1800, sysnoise=self.sysnoise)
		ax21.semilogy(mags, vals_ptp_ffi[:, 0], 'r-', label='Shot')
		ax21.semilogy(mags, vals_ptp_ffi[:, 1], 'g--', label='Zodiacal')
		ax21.semilogy(mags, vals_ptp_ffi[:, 2], '-', label='Read')
		ax21.semilogy(mags, vals_ptp_ffi[:, 3], 'b--', label='Systematic')
		ax21.semilogy(mags, tot_noise_ptp_ffi, 'k-', label='Total')

		# Expected ptp for 2-min
		# TODO: Update elat+elon based on observing sector?
		tot_noise_ptp_tpf, vals_ptp_tpf = phot_noise(mags, timescale=120, cadpix=120, sysnoise=self.sysnoise)
		ax22.semilogy(mags, vals_ptp_tpf[:, 0], 'r-', label='Shot')
		ax22.semilogy(mags, vals_ptp_tpf[:, 1], 'g--', label='Zodiacal')
		ax22.semilogy(mags, vals_ptp_tpf[:, 2], '-', label='Read')
		ax22.semilogy(mags, vals_ptp_tpf[:, 3], 'b--', label='Systematic')
		ax22.semilogy(mags, tot_noise_ptp_tpf, 'k-', label='Total')

		ax11.set_ylabel(r'$\rm RMS\,\, (ppm\,\, hr^{-1})$')
		ax21.set_ylabel('PTP-MDV (ppm)')
		ax31.set_ylabel(r'$\rm RMS_{corr} / RMS_{raw}$')
		ax32.set_ylabel(r'$\rm PTP-MDV_{corr} / PTP-MDV_{raw}$')

		for axx in np.array([ax11, ax12, ax21, ax22, ax31, ax32]):
			axx.set_xlim(self.tmag_limits)
			axx.set_xlabel('TESS magnitude')
			axx.xaxis.set_major_locator(MultipleLocator(2))
			axx.xaxis.set_minor_locator(MultipleLocator(1))
			axx.set_yscale("log", nonposy='clip')
#			axx.legend(loc='upper left')

		ax31.set_xlim(self.tmag_limits)
		ax32.set_xlim(self.tmag_limits)
		###########

		fig1.savefig(os.path.join(self.outfolder, 'rms_comp'))
		fig2.savefig(os.path.join(self.outfolder, 'ptp_comp'))
		fig3.savefig(os.path.join(self.outfolder, 'comp'))
		if self.show:
			plt.show()
		else:
			plt.close('all')

	#----------------------------------------------------------------------------------------------
	def plot_noise(self):
		"""
		Function to plot the light curve noise against the stellar TESS magnitudes

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)
		logger.info('Plotting Noise vs. Magnitude...')

		fig1 = plt.figure(figsize=(15, 5))
		fig1.subplots_adjust(left=0.145, wspace=0.3, top=0.945, bottom=0.145, right=0.975)
		ax11 = fig1.add_subplot(121)
		ax12 = fig1.add_subplot(122)

		fig2 = plt.figure(figsize=(15, 5))
		fig2.subplots_adjust(left=0.145, wspace=0.3, top=0.945, bottom=0.145, right=0.975)
		ax21 = fig2.add_subplot(121)
		ax22 = fig2.add_subplot(122)

		if self.corr:
			star_vals = self.search_database(select=['todolist.priority','todolist.starid','todolist.datasource','todolist.sector','todolist.tmag','diagnostics_corr.rms_hour','diagnostics_corr.ptp','diagnostics.contamination','ccd'])
			factor = 1
		else:
			star_vals = self.search_database(select=['todolist.priority','todolist.starid','todolist.datasource','todolist.sector','todolist.tmag','diagnostics.rms_hour','diagnostics.ptp','diagnostics.contamination','ccd'])
			factor = 1e6

		tmags = np.array([star['tmag'] for star in star_vals], dtype='float64')
		pri = np.array([star['priority'] for star in star_vals], dtype='int64')
		rms = np.array([star['rms_hour']*factor for star in star_vals], dtype='float64')
		ptp = np.array([star['ptp']*factor for star in star_vals], dtype='float64')
		source = np.array([star['datasource'] for star in star_vals], dtype=str)
		contam = np.array([star['contamination'] for star in star_vals], dtype='float64')

		idx_lc = (source == 'ffi')
		idx_sc = (source != 'ffi')

		im1 = ax11.scatter(tmags[idx_lc], rms[idx_lc], marker='o', c=contam[idx_lc], alpha=0.2, label='30-min cadence', cmap=plt.get_cmap('PuOr'), rasterized=True)
		ax12.scatter(tmags[idx_sc], rms[idx_sc], marker='o', c=contam[idx_sc], alpha=0.2, label='2-min cadence', cmap=plt.get_cmap('PuOr'), rasterized=True)

		im3 = ax21.scatter(tmags[idx_lc], ptp[idx_lc], marker='o', c=contam[idx_lc], alpha=0.2, label='30-min cadence', cmap=plt.get_cmap('PuOr'), rasterized=True)
		ax22.scatter(tmags[idx_sc], ptp[idx_sc], marker='o', c=contam[idx_sc], alpha=0.2, label='2-min cadence', cmap=plt.get_cmap('PuOr'), rasterized=True)

		# Plot theoretical lines
		mags = np.linspace(self.tmag_limits[0], self.tmag_limits[1], 200)
		cols = sns.color_palette("colorblind", 4)

		# Expected *1-hour* RMS noise ffi
		# TODO: Update elat+elon based on observing sector?
		tot_noise_rms_ffi, vals_rms_ffi = phot_noise(mags, timescale=3600, cadpix=1800, sysnoise=self.sysnoise)
		ax11.semilogy(mags, vals_rms_ffi[:, 0], '-', color=cols[0], label='Shot')
		ax11.semilogy(mags, vals_rms_ffi[:, 1], '--', color=cols[1], label='Zodiacal')
		ax11.semilogy(mags, vals_rms_ffi[:, 2], '-', color=cols[2], label='Read')
		ax11.semilogy(mags, vals_rms_ffi[:, 3], '--', color=cols[3], label='Systematic')
		ax11.semilogy(mags, tot_noise_rms_ffi, 'k-', label='Total')

		# Expected *1-hour* RMS noise tpf
		# TODO: Update elat+elon based on observing sector?
		tot_noise_rms_tpf, vals_rms_tpf = phot_noise(mags, timescale=3600, cadpix=120, sysnoise=self.sysnoise)
		ax12.semilogy(mags, vals_rms_tpf[:, 0], '-', color=cols[0], label='Shot')
		ax12.semilogy(mags, vals_rms_tpf[:, 1], '--', color=cols[1], label='Zodiacal')
		ax12.semilogy(mags, vals_rms_tpf[:, 2], '-', color=cols[2], label='Read')
		ax12.semilogy(mags, vals_rms_tpf[:, 3], '--', color=cols[3], label='Systematic')
		ax12.semilogy(mags, tot_noise_rms_tpf, 'k-', label='Total')

		# Expected ptp for 30-min
		# TODO: Update elat+elon based on observing sector?
		tot_noise_ptp_ffi, vals_ptp_ffi = phot_noise(mags, timescale=1800, cadpix=1800, sysnoise=self.sysnoise)
		ax21.semilogy(mags, vals_ptp_ffi[:, 0], '-', color=cols[0], label='Shot')
		ax21.semilogy(mags, vals_ptp_ffi[:, 1], '--', color=cols[1], label='Zodiacal')
		ax21.semilogy(mags, vals_ptp_ffi[:, 2], '-', color=cols[2], label='Read')
		ax21.semilogy(mags, vals_ptp_ffi[:, 3], '--', color=cols[3], label='Systematic')
		ax21.semilogy(mags, tot_noise_ptp_ffi, 'k-', label='Total')

		# Expected ptp for 2-min
		# TODO: Update elat+elon based on observing sector?
		tot_noise_ptp_tpf, vals_ptp_tpf = phot_noise(mags, timescale=120, cadpix=120, sysnoise=self.sysnoise)
		ax22.semilogy(mags, vals_ptp_tpf[:, 0], '-', color=cols[0], label='Shot')
		ax22.semilogy(mags, vals_ptp_tpf[:, 1], '--', color=cols[1], label='Zodiacal')
		ax22.semilogy(mags, vals_ptp_tpf[:, 2], '-', color=cols[2], label='Read')
		ax22.semilogy(mags, vals_ptp_tpf[:, 3], '--', color=cols[3], label='Systematic')
		ax22.semilogy(mags, tot_noise_ptp_tpf, 'k-', label='Total')

		ptp_tpf_vs_mag = INT.UnivariateSpline(mags, tot_noise_ptp_tpf)
		ptp_ffi_vs_mag = INT.UnivariateSpline(mags, tot_noise_ptp_ffi)
		rms_tpf_vs_mag = INT.UnivariateSpline(mags, tot_noise_rms_tpf)
		rms_ffi_vs_mag = INT.UnivariateSpline(mags, tot_noise_rms_ffi)

		ax11.set_ylabel(r'$\rm RMS\,\, (ppm\,\, hr^{-1})$')
		ax21.set_ylabel('PTP-MDV (ppm)')

		for axx in np.array([ax11, ax12, ax21, ax22]):
			axx.set_xlim(self.tmag_limits)
			axx.set_xlabel('TESS magnitude')
			axx.xaxis.set_major_locator(MultipleLocator(2))
			axx.xaxis.set_minor_locator(MultipleLocator(1))
			axx.set_yscale("log", nonposy='clip')
#			axx.legend(loc='upper left')

		divider = make_axes_locatable(ax11)
		cax = divider.append_axes('right', size='5%', pad=0.1)
		im1.set_clim(0, 1)
		cbar = fig1.colorbar(im1, cax=cax, orientation='vertical', label='Contamination')
		cbar.set_alpha(1)
		cbar.draw_all()

		divider = make_axes_locatable(ax21)
		cax = divider.append_axes('right', size='5%', pad=0.1)
		im3.set_clim(0, 1)
		cbar = fig2.colorbar(im3, cax=cax, orientation='vertical', label='Contamination')
		cbar.set_alpha(1)
		cbar.draw_all()

		###########

		fig1.savefig(os.path.join(self.outfolder, 'rms'))
		fig2.savefig(os.path.join(self.outfolder, 'ptp'))
		if self.show:
			plt.show()
		else:
			plt.close('all')

		# Assign validation bits
		dv = np.zeros_like(pri, dtype='int32')

		idx_tpf_ptp = (ptp < ptp_tpf_vs_mag(tmags)) & (ptp > 0)
		idx_ffi_ptp = (ptp < ptp_ffi_vs_mag(tmags)) & (ptp > 0)
		idx_tpf_rms = (rms < rms_tpf_vs_mag(tmags)) & (rms > 0)
		idx_ffi_rms = (rms < rms_ffi_vs_mag(tmags)) & (rms > 0)
		idx_invalid = (rms <= 0) | ~np.isfinite(rms) | (ptp <= 0) | ~np.isfinite(ptp)

		dv[idx_sc & idx_tpf_ptp] |= DatavalQualityFlags.LowPTP
		dv[idx_lc & idx_ffi_ptp] |= DatavalQualityFlags.LowPTP
		dv[idx_sc & idx_tpf_rms] |= DatavalQualityFlags.LowRMS
		dv[idx_lc & idx_ffi_rms] |= DatavalQualityFlags.LowRMS
		dv[idx_lc & idx_ffi_rms] |= DatavalQualityFlags.LowRMS
		dv[idx_invalid] |= DatavalQualityFlags.InvalidNoise

		self.update_dataval(pri, dv)

	#----------------------------------------------------------------------------------------------
	def plot_pixinaperture(self):
		"""
		Function to plot number of pixels in determined apertures against the stellar TESS magnitudes

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)
		logger.info('Plotting Pixels in aperture vs. Magnitude...')

		fig = plt.figure(figsize=(15, 5))
		ax1 = fig.add_subplot(121)
		ax2 = fig.add_subplot(122)
		fig.subplots_adjust(left=0.1, wspace=0.2, top=0.94, bottom=0.155, right=0.91)

#		if not self.doval:
		star_vals = self.search_database(select=['todolist.priority','todolist.ccd','todolist.starid','todolist.datasource','todolist.sector','todolist.tmag','diagnostics.mask_size','diagnostics.contamination','todolist.camera','diagnostics.errors'])
#		else:
#			star_vals = self.search_database(select=['todolist.priority','todolist.ccd','todolist.starid','todolist.datasource','todolist.sector','todolist.tmag','diagnostics.mask_size','diagnostics.contamination','todolist.camera','diagnostics.errors','datavalidation_raw.dataval'])

		#if self.color_by_sector:
		#	sec = np.array([star['sector'] for star in star_vals], dtype=int)
		#	sectors = np.array(list(set(sec)))
		#	if len(sectors)>1:
		#		norm = colors.Normalize(vmin=1, vmax=len(sectors))
		#		scalarMap = cmx.ScalarMappable(norm=norm, cmap=plt.get_cmap('Set1') )
		#		rgba_color = np.array([scalarMap.to_rgba(s) for s in sec])
		#	else:
		#		rgba_color = 'k'
		#else:
		#	rgba_color = 'k'

		#tic = np.array([star['starid'] for star in star_vals], dtype=int)
		tmags = np.array([star['tmag'] for star in star_vals], dtype='float64')
		masksizes = np.array([star['mask_size'] for star in star_vals], dtype='float64')
		contam = np.array([star['contamination'] for star in star_vals], dtype='float64')
		source = np.array([star['datasource'] for star in star_vals], dtype=str)
		pri = np.array([star['priority'] for star in star_vals], dtype='int64')
		minimal_mask_used = np.array([False if star['errors'] is None else ('Using minimum aperture.' in star['errors']) for star in star_vals], dtype='bool')

#		dataval = np.zeros_like(pri, dtype='int32')
#		if self.doval:
#			dataval = np.array([star['dataval'] for star in star_vals], dtype='int32')

#		cats = {}
#		for cams in list(set(camera)):
#			cats[cams] = {}
#			for jj in range(4):
#				cam_file = os.path.join(self.input_folders[0], 'catalog_camera' + str(int(cams)) +'_ccd' + str(jj+1) + '.sqlite')
#				with contextlib.closing(sqlite3.connect(cam_file)) as conn:
#				with sqlite3.connect(cam_file) as conn:
#					conn.row_factory = sqlite3.Row
#					cursor = conn.cursor()
#
#					cats[cams][str(jj+1)] = cursor

#		cams_cen = {}
#		cams_cen['1','ra'] = 324.566998914166
#		cams_cen['1','decl'] = -33.172999301379
#		cams_cen['2','ra'] = 338.5766
#		cams_cen['2','decl'] = -55.0789
#		cams_cen['3','ra'] = 19.4927
#		cams_cen['3','decl'] = -71.9781
#		cams_cen['4','ra'] = 90.0042
#		cams_cen['4','decl'] = -66.5647
#
#		dists = np.zeros(len(tics))
#		for ii, tic in enumerate(tics):
#			query = "SELECT ra,decl FROM catalog where starid=%s" %tic
#			# Ask the database: status=1
#			cc = cats[camera[ii]][str(ccds[ii])]
#			cc.execute(query)
#			star_vals = cc.fetchone()
#			dists[ii] = sphere_distance(star_vals['ra'], star_vals['decl'], cams_cen[camera[ii],'ra'], cams_cen[camera[ii],'decl'])

		contam[np.isnan(contam)] = 0
		norm = colors.Normalize(vmin=0, vmax=1)

		# Get rid of stupid halo targets
		masksizes[np.isnan(masksizes)] = 0

		idx_lc = (source == 'ffi')
		idx_sc = (source != 'ffi')

		idx_lc0 = idx_lc # & (dataval&(32+64) == 0)
		idx_sc0 = idx_sc # & (dataval&(32+64) == 0)

		#idx_lc1 = idx_lc & (dataval&(32+64) != 0)
		#idx_sc1 = idx_sc & (dataval&(32+64) != 0)

		perm_lc = np.random.permutation(sum(idx_lc0))
		perm_sc = np.random.permutation(sum(idx_sc0))

		ax1.scatter(tmags[idx_lc0][perm_lc], masksizes[idx_lc0][perm_lc], marker='o', c=contam[idx_lc0][perm_lc], alpha=0.5, norm=norm, cmap=plt.get_cmap('PuOr'), rasterized=True)
#		ax1.scatter(tmags[idx_lc1], masksizes[idx_lc1], marker='.', c=contam[idx_lc1], alpha=0.2, norm=norm, cmap=plt.get_cmap('PuOr'))

		ax2.scatter(tmags[idx_sc0][perm_sc], masksizes[idx_sc0][perm_sc], marker='o', c=contam[idx_sc0][perm_sc], alpha=0.5, norm=norm, cmap=plt.get_cmap('PuOr'), rasterized=True)
#		ax2.scatter(tmags[idx_sc1], masksizes[idx_sc1], marker='.', c=contam[idx_sc1], alpha=0.2, norm=norm, cmap=plt.get_cmap('PuOr'))

		# Compute median-bin curve
		bin_means, bin_edges, binnumber = binning(tmags[idx_lc], masksizes[idx_lc], statistic='median', bins=15, range=(np.nanmin(tmags[idx_lc]),np.nanmax(tmags[idx_lc])))
		bin_width = (bin_edges[1] - bin_edges[0])
		bin_centers = bin_edges[1:] - bin_width/2
		ax1.scatter(bin_centers, bin_means, marker='o', color='r')

		if np.any(idx_sc):
			bin_means, bin_edges, binnumber = binning(tmags[idx_sc], masksizes[idx_sc], statistic='median', bins=15, range=(np.nanmin(tmags[idx_sc]),np.nanmax(tmags[idx_sc])))
			bin_width = (bin_edges[1] - bin_edges[0])
			bin_centers = bin_edges[1:] - bin_width/2
			ax2.scatter(bin_centers, bin_means, marker='o', color='r')

#		normed0 = masksizes[idx_lc]/med_vs_mag(tmags[idx_lc])
#		normed1 = masksizes[idx_lc]-pix_vs_mag(tmags[idx_lc])

#		bin_mad, bin_edges, binnumber = binning(tmags[idx_lc], np.abs(normed1), statistic='median', bins=15, range=(np.nanmin(tmags[idx_lc]),np.nanmax(tmags[idx_lc])))
#		bin_width = (bin_edges[1] - bin_edges[0]);		bin_centers = bin_edges[1:] - bin_width/2
#		bin_var = bin_mad*1.4826
#		var_vs_mag = INT.InterpolatedUnivariateSpline(bin_centers, bin_var)

#		plt.figure()
#		plt.scatter(tmags[idx_lc], normed1/var_vs_mag(tmags[idx_lc]), color=rgba_color[idx_lc], alpha=0.5)
#
#		plt.figure()
#		red = masksizes[idx_lc]/pix_vs_mag(tmags[idx_lc])
#		bin_means, bin_edges, binnumber = binning(contam[idx_lc], red, statistic='median', bins=15, range=(np.nanmin(contam[idx_lc]),1))
#		bin_width = (bin_edges[1] - bin_edges[0]);		bin_centers = bin_edges[1:] - bin_width/2
#		plt.scatter(contam[idx_lc], red, alpha=0.1, color='k')
#		plt.scatter(bin_centers, bin_means, color='r')

		# Plot median-bin curve (1 and 5 times standadised MAD)
#		ax1.scatter(bin_centers, 1.4826*5*bin_means, marker='.', color='r')
#		print(masksizes[(tmags>15) & (source != 'ffi')])
#		print(np.max(tmags[idx_lc]))
#		print(bin_centers, bin_means)

#		print(masksizes[idx_lc])
#		print(any(np.isnan(masksizes[idx_lc])))
#		print(pix_vs_mag(tmags[idx_lc]))
#		diff = np.abs(masksizes[idx_lc] - pix_vs_mag(tmags[idx_lc]))

#		fig00=plt.figure()
#		ax00=fig00.add_subplot(111)
#		d = masksizes[idx_lc] - pix_vs_mag(tmags[idx_lc])
#		ax00.hist(d, bins=500)
#		ax00.axvline(x=np.percentile(d, 99.9), c='k')
#		ax00.axvline(x=np.percentile(d, 0.1), c='k')

		# Minimum bound on FFI data
		xmin = np.array([0, 2, 2.7, 3.55, 4.2, 4.8, 5.5, 6.8, 7.6, 8.4, 9.1, 10, 10.5, 11, 11.5, 11.6, 16])
		ymin = np.array([2600, 846, 526, 319, 238, 159, 118, 62, 44, 32, 23, 15.7, 11.15, 8, 5, 4, 4])
		min_bound = INT.InterpolatedUnivariateSpline(xmin, ymin, k=1)
		xmax = np.array([0, 2, 2.7, 3.55, 4.2, 4.8, 5.5, 6.8, 7.6, 8.4, 9.1, 10, 10.5, 11, 11.5, 12, 12.7, 13.3, 14, 14.5, 15, 16])
		ymax = np.array([10000, 3200, 2400, 1400, 1200, 900, 800, 470, 260, 200, 170, 130, 120, 100, 94, 86, 76, 67, 59, 54, 50, 50])
		max_bound = INT.InterpolatedUnivariateSpline(xmax, ymax, k=1)

		ax1.plot(xmin, ymin, ls='-', color='r')
		ax1.plot(xmax, ymax, ls='-', color='r')

		# Minimum bound on TPF data
		xmin_sc = np.array([0, 2, 2.7, 3.55, 4.2, 4.8, 5.5, 6.8, 7.6, 8.4, 9.1, 10, 10.5, 11, 11.5, 11.6, 16, 19])
		ymin_sc = np.array([220, 200, 130, 70, 55, 43, 36, 30, 27, 22, 16, 10, 8, 6, 5, 4, 4, 4])
		min_bound_sc = INT.InterpolatedUnivariateSpline(xmin_sc, ymin_sc, k=1)

		ax2.plot(xmin_sc, ymin_sc, ls='-', color='r')

#		idx_lc2 = (source == 'ffi') & (masksizes>4) & (tmags>8)
#		idx_sort = np.argsort(tmags[idx_lc2])
#		perfilt95 = filt.percentile_filter(masksizes[idx_lc2][idx_sort], 99.8, size=1000)
##		perfilt95 = filt.uniform_filter1d(perfilt95, 5000)
#		perfilt95 = filt.gaussian_filter1d(perfilt95, 10000)
#		perfilt05 = filt.percentile_filter(masksizes[idx_lc2][idx_sort], 0.2, size=1000)
##		perfilt05 = filt.uniform_filter1d(perfilt05, 5000)
#		perfilt05 = filt.gaussian_filter1d(perfilt05, 10000)
#
#		ax1.plot(tmags[idx_lc2][idx_sort], perfilt95, color='m')
#		ax1.plot(tmags[idx_lc2][idx_sort], perfilt05, color='m')
#		print(tmags[idx_lc2][idx_sort])
#		perh_vs_mag = INT.interp1d(tmags[idx_lc2][idx_sort], perfilt95)
#		perl_vs_mag = INT.interp1d(tmags[idx_lc2][idx_sort], perfilt05)

#		ticsh = tics[idx_lc][(masksizes[idx_lc]>max_bound(tmags[idx_lc]))]
#		ticsh_m = tmags[idx_lc][(masksizes[idx_lc]>max_bound(tmags[idx_lc]))]
#		ticsh_mm = masksizes[idx_lc][(masksizes[idx_lc]>max_bound(tmags[idx_lc]))]
#		ticsl = tics[idx_lc][(masksizes[idx_lc]<min_bound(tmags[idx_lc]))]
#		ticsl_m = tmags[idx_lc][(masksizes[idx_lc]<min_bound(tmags[idx_lc]))]
#		ticsl_mm = masksizes[idx_lc][(masksizes[idx_lc]<min_bound(tmags[idx_lc]))]

#		ticsl = tics[idx_sc][(masksizes[idx_sc]<min_bound_sc(tmags[idx_sc])) & (masksizes[idx_sc]>0)]
#		ticsl_m = tmags[idx_sc][(masksizes[idx_sc]<min_bound_sc(tmags[idx_sc])) & (masksizes[idx_sc]>0)]
#		ticsl_mm = masksizes[idx_sc][(masksizes[idx_sc]<min_bound_sc(tmags[idx_sc])) & (masksizes[idx_sc]>0) ]

#		print('HIGH')
#
#		for ii, tic in enumerate(ticsh):
#			print(tic, ticsh_m[ii], ticsh_mm[ii])

#		print('LOW')
#		for ii, tic in enumerate(ticsl):
#			print(tic, ticsl_m[ii], ticsl_mm[ii])
#
##		print(len(ticsh))
#		print(len(ticsl))

#		bin_means1, bin_edges1, binnumber1 = binning(tmags[idx_lc], masksizes[idx_lc]-pix_vs_mag(tmags[idx_lc]), statistic=reduce_percentile1, bins=15, range=(np.nanmin(tmags),np.nanmax(tmags)))
#		bin_means2, bin_edges2, binnumber2 = binning(tmags[idx_lc], masksizes[idx_lc]-pix_vs_mag(tmags[idx_lc]), statistic=reduce_percentile2, bins=15, range=(np.nanmin(tmags),np.nanmax(tmags)))
#		bin_means1, bin_edges1, binnumber1 = binning(tmags[idx_lc][d>0], d[d>0], statistic='median', bins=15, range=(np.nanmin(tmags),np.nanmax(tmags)))
#		bin_means2, bin_edges2, binnumber2 = binning(tmags[idx_lc][d<0], d[d<0], statistic='median', bins=15, range=(np.nanmin(tmags),np.nanmax(tmags)))
#		ax1.scatter(bin_centers, bin_means+bin_means1, marker='o', color='b')
#		ax1.scatter(bin_centers, bin_means+5*bin_means1, marker='o', color='b')
#		ax1.scatter(bin_centers, bin_means+4*bin_means2, marker='o', color='b')
#		ax1.scatter(bin_centers, bin_means+bin_means2, marker='o', color='b')
#		ax1.scatter(tmags[idx_lc][idx_sort], P.values, marker='o', color='m')

#		mags = np.linspace(np.nanmin(tmags)-1, np.nanmax(tmags)+1, 500)
#		pix = np.asarray([Pixinaperture(m) for m in mags], dtype='float64')
#		ax1.plot(mags, pix, color='k', ls='-')
#		ax2.plot(mags, pix, color='k', ls='-')

		ax1.set_xlim(self.tmag_limits)
		ax2.set_xlim(self.tmag_limits)

		ax1.set_ylim([0.99, np.nanmax(masksizes)+500])
		ax2.set_ylim([0.99, np.nanmax(masksizes)+500])

		for axx in np.array([ax1, ax2]):
			axx.set_xlabel('TESS magnitude')
			axx.set_ylabel('Pixels in aperture')
			xtick_major = np.median(np.diff(axx.get_xticks()))
			axx.xaxis.set_minor_locator(MultipleLocator(xtick_major/2))
			ytick_major = np.median(np.diff(axx.get_yticks()))
			axx.yaxis.set_minor_locator(MultipleLocator(ytick_major/2))
			axx.set_yscale("log", nonposy='clip')
			axx.yaxis.set_major_formatter(ScalarFormatter())
			#axx.legend(loc='upper right')

		pos = ax2.get_position()
		axc = fig.add_axes([pos.x0 + pos.width+0.01, pos.y0, 0.01, pos.height], zorder=-1)
		cb = mpl.colorbar.ColorbarBase(axc, cmap=plt.get_cmap('PuOr'), norm=norm, orientation='vertical')
		cb.set_label('Contamination', fontsize=12, labelpad=6)
		cb.ax.tick_params(axis='y', direction='out')

		fig.savefig(os.path.join(self.outfolder, 'pix_in_aper'))
		if self.show:
			plt.show()
		else:
			plt.close(fig)

		# Assign validation bits, for both FFI and TPF
		dv = np.zeros_like(pri, dtype='int32')

		# Minimal masks were used:
		dv[minimal_mask_used] |= DatavalQualityFlags.MinimalMask

		# Small and Large masks:
		dv[idx_lc & (masksizes < min_bound(tmags)) & (masksizes > 0)] |= DatavalQualityFlags.SmallMask
		dv[idx_lc & (masksizes > max_bound(tmags))] |= DatavalQualityFlags.LargeMask
		dv[idx_sc & (masksizes < min_bound_sc(tmags)) & (masksizes > 0)] |= DatavalQualityFlags.SmallMask

		self.update_dataval(pri, dv)

	#----------------------------------------------------------------------------------------------
	def plot_mag2flux(self):
		"""
		Function to plot flux values from apertures against the stellar TESS magnitudes,
		and determine coefficient describing the relation

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)
		logger.info('Plotting Magnitude to Flux conversion...')

		fig = plt.figure(figsize=(15, 5))
		fig.subplots_adjust(left=0.1, wspace=0.2, top=0.94, bottom=0.155, right=0.91)
		ax1 = fig.add_subplot(121)
		ax2 = fig.add_subplot(122)

		#fig3 = plt.figure(figsize=(15, 5))
		#fig3.subplots_adjust(left=0.14, wspace=0.3, top=0.94, bottom=0.155, right=0.96)
		#ax31 = fig3.add_subplot(121)
		#ax32 = fig3.add_subplot(122)

		star_vals = self.search_database(select=['todolist.priority','todolist.datasource','todolist.sector','todolist.tmag','mean_flux','contamination'])

		tmags = np.array([star['tmag'] for star in star_vals], dtype='float64')
		meanfluxes = np.array([star['mean_flux'] for star in star_vals], dtype='float64')
		contam = np.array([star['contamination'] for star in star_vals], dtype='float64')
		source = np.array([star['datasource'] for star in star_vals], dtype=str)
		pri = np.array([star['priority'] for star in star_vals], dtype='int64')

		idx_lc = (source == 'ffi')
		idx_sc = (source != 'ffi')

		norm = colors.Normalize(vmin=0, vmax=1)

		perm_lc = np.random.permutation(sum(idx_lc))
		perm_sc = np.random.permutation(sum(idx_sc))

		ax1.scatter(tmags[idx_lc][perm_lc], meanfluxes[idx_lc][perm_lc], marker='o', c=contam[idx_lc][perm_lc], norm=norm, cmap=plt.get_cmap('PuOr'), alpha=0.1, rasterized=True) # , label='30-min cadence')
		ax2.scatter(tmags[idx_sc][perm_sc], meanfluxes[idx_sc][perm_sc], marker='o', c=contam[idx_sc][perm_sc], norm=norm, cmap=plt.get_cmap('PuOr'), alpha=0.1, rasterized=True) # , label='2-min cadence')

		xmin = np.array([0, 1.5, 9, 12.6, 13, 14, 15, 16, 17, 18, 19])
		ymin = np.array([8e7, 1.8e7, 12500, 250, 59, 5, 1, 1, 1, 1, 1])

		min_bound = INT.InterpolatedUnivariateSpline(xmin, ymin, k=1)

		ax1.plot(xmin, ymin, ls='-', color='r')
		ax2.plot(xmin, ymin, ls='-', color='r')

		with np.errstate(invalid='ignore'):
			idx1 = np.isfinite(meanfluxes) & np.isfinite(tmags) & (source == 'ffi') & (contam < 0.15)
			idx2 = np.isfinite(meanfluxes) & np.isfinite(tmags) & (source != 'ffi') & (contam < 0.15)

		logger.info('Optimising coefficient of relation')

		def chi2(c, idx):
			return np.log10(ss( (meanfluxes[idx] - 10**(-0.4*(tmags[idx] - c))) / (contam[idx]+1) ))

		cc = minimize(chi2, 20.5, args=(idx1,), method='Nelder-Mead', options={'disp':False})
		cc2 = minimize(chi2, 20.5, args=(idx2,), method='Nelder-Mead', options={'disp':False})

		logger.info('Optimisation terminated successfully? %s', cc.success)
		logger.info('Coefficient is found to be %1.4f', cc.x)
		logger.info('Coefficient is found to be %1.4f', cc2.x)

		C = np.linspace(19, 22, 100)
		zc = [chi2(c, idx1) for c in C]
		z2c = [chi2(c, idx2) for c in C]

		# Create plot of
		fig2 = plt.figure()
		ax21 = fig2.add_subplot(111)
		ax21.plot(C, zc, 'k-')
		ax21.plot(C, z2c, 'b-')
		ax21.axvline(x=cc.x, color='k', label='30-min')
		ax21.axvline(x=cc2.x, color='b', ls='--', label='2-min')
		ax21.set_xlabel('Coefficient')
		ax21.set_ylabel(r'$\chi^2$')
		ax21.legend(loc='upper left')
		fig2.savefig(os.path.join(self.outfolder, 'mag_to_flux_optimize'))

#		d1 = meanfluxes[idx1]/(10**(-0.4*(tmags[idx1] - cc.x))) - 1
#		d2 = meanfluxes[idx2]/(10**(-0.4*(tmags[idx2] - cc2.x))) - 1
#		ax31.scatter(tmags[idx1], np.abs(d1), alpha=0.1)
#		ax31.scatter(tmags[idx2], np.abs(d2), color='k', alpha=0.1)
#		ax31.axhline(y=0, ls='--', color='k')
#
#		bin_means1, bin_edges1, binnumber1 = binning(tmags[idx1], np.abs(d1), statistic='median', bins=15, range=(1.5,15))
#		bin_width1 = (bin_edges1[1] - bin_edges1[0])
#		bin_centers1 = bin_edges1[1:] - bin_width1/2
#
#		bin_means2, bin_edges2, binnumber2 = binning(tmags[idx2], np.abs(d2), statistic='median', bins=15, range=(1.5,15))
#		bin_width2 = (bin_edges2[1] - bin_edges2[0])
#		bin_centers2 = bin_edges2[1:] - bin_width2/2
#
#		ax31.scatter(bin_centers1, 1.4826*bin_means1, marker='o', color='r')
#		ax31.scatter(bin_centers1, 1.4826*3*bin_means1, marker='.', color='r')
#		ax31.plot(bin_centers1, 1.4826*3*bin_means1, color='r')
#
#		ax31.scatter(bin_centers2, 1.4826*bin_means2, marker='o', color='g')
#		ax31.scatter(bin_centers2, 1.4826*3*bin_means2, marker='.', color='g')
#		ax31.plot(bin_centers2, 1.4826*3*bin_means2, color='g')

		mag = np.linspace(np.nanmin(tmags)-1, np.nanmax(tmags)+1,100)
		ax1.plot(mag, 10**(-0.4*(mag - cc.x)), color='k', ls='--')
		ax2.plot(mag, 10**(-0.4*(mag - cc2.x)), color='k', ls='--')

		for axx in np.array([ax1, ax2]):
			axx.set_yscale("log", nonposy='clip')
			axx.set_xlim(self.tmag_limits)
			axx.set_xlabel('TESS magnitude')

			axx.xaxis.set_major_locator(MultipleLocator(2))
			axx.xaxis.set_minor_locator(MultipleLocator(1))

			axx.set_xlim(axx.get_xlim()[::-1])

#		ax1.text(10, 1e7, r'$\rm Flux = 10^{-0.4\,(T_{mag} - %1.2f)}$' %cc.x, fontsize=14)
		ax1.set_ylabel('Mean flux')

		pos = ax2.get_position()
		axc = fig.add_axes([pos.x0 + pos.width+0.01, pos.y0, 0.01, pos.height], zorder=-1)
		cb = mpl.colorbar.ColorbarBase(axc, cmap=plt.get_cmap('PuOr'), norm=norm, orientation='vertical')
		cb.set_label('Contamination', fontsize=12, labelpad=6)
		cb.ax.tick_params(axis='y', direction='out')

		fig.savefig(os.path.join(self.outfolder, 'mag_to_flux'))
		#fig3.savefig(os.path.join(self.outfolder, 'mag_to_flux_dev'))
		if self.show:
			plt.show()
		else:
			plt.close('all')

		# Assign validation bits, for both FFI and TPF
		dv = np.zeros_like(pri, dtype='int32')
		dv[meanfluxes < min_bound(tmags)] |= DatavalQualityFlags.MagVsFluxLow
		dv[(~np.isfinite(meanfluxes)) | (meanfluxes <= 0)] |= DatavalQualityFlags.InvalidFlux
		self.update_dataval(pri, dv)

	#----------------------------------------------------------------------------------------------
	def plot_stamp(self):
		"""
		Function to plot width and height of pixel stamps against the stellar TESS magnitudes

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)
		logger.info('Plotting Stamp sizes...')

		fig1 = plt.figure(figsize=(15, 10))
		ax11 = fig1.add_subplot(221)
		ax12 = fig1.add_subplot(222)
		ax13 = fig1.add_subplot(223)
		ax14 = fig1.add_subplot(224)

		fig2 = plt.figure(figsize=(15, 7))
		ax21 = fig2.add_subplot(121)
		ax22 = fig2.add_subplot(122)

		star_vals = self.search_database(select=['todolist.datasource','todolist.sector','todolist.tmag','stamp_resizes','stamp_width','stamp_height', 'diagnostics.elaptime'])

		rgba_color = 'r'
		if self.color_by_sector:
			sec = np.array([star['sector'] for star in star_vals], dtype=int)
			sectors = np.array(list(set(sec)))
			if len(sectors) > 1:
				norm = colors.Normalize(vmin=1, vmax=len(sectors))
				scalarMap = cmx.ScalarMappable(norm=norm, cmap=plt.get_cmap('Set1') )
				rgba_color = np.array([scalarMap.to_rgba(s) for s in sec])

		tmags = np.array([star['tmag'] for star in star_vals], dtype='float64')
		et = np.array([star['elaptime'] for star in star_vals], dtype='float64')
		width = np.array([star['stamp_width'] for star in star_vals], dtype='float64')
		height = np.array([star['stamp_height'] for star in star_vals], dtype='float64')
		resize = np.array([star['stamp_resizes'] for star in star_vals], dtype='float64')
		ds = np.array([star['datasource'] == 'ffi' for star in star_vals], dtype='bool')

		idx1 = (resize < 1) & (ds)
		idx2 = (resize < 1) & (~ds)
		idx3 = (resize > 0) & (ds)
		idx4 = (resize > 0) & (~ds)

		ax12.scatter(tmags[idx1], width[idx1], marker='o', facecolors='None', color=rgba_color, label='30-min cadence, no resize', alpha=0.5, zorder=2)
		ax14.scatter(tmags[idx2], width[idx2], marker='o', facecolors='None', color=rgba_color, label='2-min cadence, no resize', alpha=0.5, zorder=2)

		ax12.scatter(tmags[idx3], width[idx3], marker='o', facecolors='None', color='k', label='30-min cadence, resized', alpha=0.5)
		ax14.scatter(tmags[idx4], width[idx4], marker='o', facecolors='None', color='k', label='2-min cadence, resized', alpha=0.5)

		ax11.scatter(tmags[idx1], height[idx1], marker='o', facecolors='None', color=rgba_color, label='30-min cadence, no resize', alpha=0.5, zorder=2)
		ax13.scatter(tmags[idx2], height[idx2], marker='o', facecolors='None', color=rgba_color, label='2-min cadence, no resize', alpha=0.5, zorder=2)

		ax11.scatter(tmags[idx3], height[idx3], marker='o', facecolors='None', color='k', label='30-min cadence, resized', alpha=0.5)
		ax13.scatter(tmags[idx4], height[idx4], marker='o', facecolors='None', color='k', label='2-min cadence, resized', alpha=0.5)

		bin_means, bin_edges, binnumber = binning(tmags[ds], height[ds], statistic='median', bins=20, range=(1.5,10))
		#bin_width = (bin_edges[1] - bin_edges[0])
		#bin_centers = bin_edges[1:] - bin_width/2

		bin_means2, bin_edges2, binnumber2 = binning(tmags[ds], width[ds], statistic='median', bins=20, range=(1.5,10))
		#bin_width2 = (bin_edges2[1] - bin_edges2[0])
		#bin_centers2 = bin_edges2[1:] - bin_width2/2

		#ax12.scatter(bin_centers2, bin_means2, marker='o', color='b', zorder=3)
		#ax11.scatter(bin_centers, bin_means, marker='o', color='b', zorder=3)

		normalize2 = colors.Normalize(vmin=0, vmax=np.max(resize))
		scalarMap = cmx.ScalarMappable(norm=normalize2, cmap=plt.get_cmap('Set1'))
		for jj in range(0, int(np.max(resize))):
			rgba_color = scalarMap.to_rgba(jj)
			try:
				kde1 = KDE(et[ds][(resize[ds] == jj) & (et[ds] < 50)])
				kde1.fit(gridsize=1000)
				ax21.plot(kde1.support, kde1.density, color=rgba_color)
			except:
				pass

		kde1 = KDE(et[ds & (et < 50)])
		kde1.fit(gridsize=1000)
		ax21.plot(kde1.support, kde1.density, color='k', lw=2, label='30-min cadence')
		ax21.set_xlim([0, 50])

		try:
			kde2 = KDE(et[(~ds) & (et < 50)])
			kde2.fit(gridsize=1000)
			ax22.plot(kde2.support, kde2.density, color='k', lw=2, label='2-min candence')
		except ZeroDivisionError:
			pass

		# Decide how many pixels to use based on lookup tables as a function of Tmag:
		mags = np.array([0., 0.52631579, 1.05263158, 1.57894737, 2.10526316,
			2.63157895, 3.15789474, 3.68421053, 4.21052632, 4.73684211,
			5.26315789, 5.78947368, 6.31578947, 6.84210526, 7.36842105,
			7.89473684, 8.42105263, 8.94736842, 9.47368421, 10.])
		nhei = np.array([831.98319063, 533.58494422, 344.0840884, 223.73963332,
			147.31365728, 98.77856016, 67.95585074, 48.38157414,
			35.95072974, 28.05639497, 23.043017, 19.85922009,
			17.83731732, 16.5532873, 15.73785092, 15.21999971,
			14.89113301, 14.68228285, 14.54965042, 14.46542084])
		nwid = np.array([157.71602062, 125.1238281, 99.99440209, 80.61896267,
			65.6799962, 54.16166547, 45.28073365, 38.4333048,
			33.15375951, 29.08309311, 25.94450371, 23.52456986,
			21.65873807, 20.22013336, 19.1109318, 18.25570862,
			17.59630936, 17.08789543, 16.69589509, 16.39365266])

		mags2 = np.linspace(np.min(tmags)-0.2, np.max(tmags)+0.2, 500)
		nwid2 = np.array([2*(np.ceil(np.interp(m, mags, nwid))//2)+1 for m in mags2])
		nhei2 = np.array([2*(np.ceil(np.interp(m, mags, nhei))//2)+1 for m in mags2])

		nwid2[(nwid2 < 15)] = 15
		nhei2[(nhei2 < 15)] = 15

		#ax12.plot(mags2,nwid2, 'b--')
		#ax11.plot(mags2,nhei2, 'b--')

		ax12.set_ylabel('Stamp width (pixels)')
		ax14.set_ylabel('Stamp width (pixels)')
		ax11.set_ylabel('Stamp height (pixels)')
		ax13.set_ylabel('Stamp height (pixels)')

		ax12.yaxis.set_major_locator(MultipleLocator(20))
		ax12.yaxis.set_minor_locator(MultipleLocator(10))

		ax11.yaxis.set_major_locator(MultipleLocator(50))
		ax11.yaxis.set_minor_locator(MultipleLocator(25))

		for axx in np.array([ax11, ax12, ax13, ax14]):
			axx.xaxis.set_major_locator(MultipleLocator(2))
			axx.xaxis.set_minor_locator(MultipleLocator(1))
			axx.set_xlabel('TESS magnitude')
			axx.legend(loc='upper right')

		for axx in np.array([ax21, ax22]):
			axx.xaxis.set_major_locator(MultipleLocator(5))
			axx.xaxis.set_minor_locator(MultipleLocator(2.5))
			axx.set_xlabel('Calculation time (sec)')
			axx.legend(loc='upper right')

		fig1.savefig(os.path.join(self.outfolder, 'stamp_size'))
		fig2.savefig(os.path.join(self.outfolder, 'calc_time'))
		if self.show:
			plt.show()
		else:
			plt.close('all')

	#----------------------------------------------------------------------------------------------
	def plot_mag_dist(self):
		"""
		Function to plot magnitude distribution for targets

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)
		logger.info('Plotting Magnitude distribution...')

		fig = plt.figure(figsize=(10,5))
		ax = fig.add_subplot(111)
		fig.subplots_adjust(left=0.14, wspace=0.3, top=0.94, bottom=0.155, right=0.96)

		star_vals = self.search_database(select=['todolist.datasource','todolist.tmag'])

		tmags = np.array([star['tmag'] for star in star_vals])
		source = np.array([star['datasource'] for star in star_vals])

		idx_lc = (source == 'ffi')
		idx_sc = (source != 'ffi')

		if sum(idx_lc) > 0:
			kde_lc = KDE(tmags[idx_lc])
			kde_lc.fit(gridsize=1000)
#			ax.fill_between(kde_lc.support, 0, kde_lc.density*sum(idx_lc), color='r', alpha=0.3, label='30-min cadence')
			ax.fill_between(kde_lc.support, 0, kde_lc.density/np.max(kde_lc.density), color='r', alpha=0.3, label='30-min cadence')

		if sum(idx_sc) > 0:
			kde_sc = KDE(tmags[idx_sc])
			kde_sc.fit(gridsize=1000)
#			ax.fill_between(kde_sc.support, 0, kde_sc.density*sum(idx_sc), color='b', alpha=0.3, label='2-min cadence')
			ax.fill_between(kde_sc.support, 0, kde_sc.density/np.max(kde_sc.density), color='b', alpha=0.3, label='2-min cadence')

#		kde_all = KDE(tmags)
#		kde_all.fit(gridsize=1000)
#		ax.plot(kde_all.support, kde_all.density/, 'k-', lw=1.5, label='All')

		ax.set_ylim(bottom=0)
		ax.set_xlabel('TESS magnitude')
		ax.set_ylabel('Normalised Density')
		ax.xaxis.set_major_locator(MultipleLocator(2))
		ax.xaxis.set_minor_locator(MultipleLocator(1))
		ax.legend(frameon=False, loc='upper left', borderaxespad=0, handlelength=2.5, handletextpad=0.4)

		fig.savefig(os.path.join(self.outfolder, 'mag_dist'))
		if self.show:
			plt.show()
		else:
			plt.close(fig)

	#----------------------------------------------------------------------------------------------
	def plot_mag_dist_overlap(self):
		"""
		Function to plot magnitude distribution overlap between sectors

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)
		logger.info('Plotting Magnitude distribution...')

		fig = plt.figure(figsize=(10,5))
		ax = fig.add_subplot(111)
		fig.subplots_adjust(left=0.14, wspace=0.3, top=0.94, bottom=0.155, right=0.96)

		sets_lc = []
		sets_sc = []
		tmags_lc = []
		tmags_sc = []
		for i, f in enumerate(self.input_folders):
			todo_file = os.path.join(f, 'todo.sqlite')
			logger.debug("TODO file: %s", todo_file)
			if not os.path.exists(todo_file):
				raise ValueError("TODO file not found")

			# Open the SQLite file:
			conn = sqlite3.connect(todo_file)
			conn.row_factory = sqlite3.Row
			cursor = conn.cursor()

			select = ['todolist.starid','todolist.datasource','todolist.tmag']
			select = ",".join(select)
			#search=["status in (1,3)"]
			search = ["approved=1"]
			search = "WHERE " + " AND ".join(search)

			query = "SELECT {select:s} FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority LEFT JOIN datavalidation_raw ON todolist.priority=datavalidation_raw.priority {search:s};".format(
				select=select,
				search=search)

			# Ask the database: status=1
			cursor.execute(query)
			star_vals = [dict(row) for row in cursor.fetchall()]

			tmags = np.array([star['tmag'] for star in star_vals])
			source = np.array([star['datasource'] for star in star_vals])
			starid = np.array([star['starid'] for star in star_vals], dtype="int32")

			idx_lc = (source == 'ffi')
			idx_sc = (source != 'ffi')

			ind_dict_lc = dict((k,i) for i,k in enumerate(starid[idx_lc]))
			ind_dict_sc = dict((k,i) for i,k in enumerate(starid[idx_sc]))

			print(len(tmags[idx_lc]), len(tmags[idx_sc]))

			tmags_lc.append(tmags[idx_lc])
			tmags_sc.append(tmags[idx_sc])
			sets_lc.append(ind_dict_lc)
			sets_sc.append(ind_dict_sc)

		inter_lc = set(sets_lc[0].keys()).intersection(sets_lc[1].keys())
		inter_sc = set(sets_sc[0].keys()).intersection(sets_sc[1].keys())
		indices_lc = [ sets_lc[1][x] for x in inter_lc ]
		indices_lc2 = [ sets_lc[0][x] for x in inter_lc ]
		indices_sc = [ sets_sc[1][x] for x in inter_sc ]
		indices_sc2 = [ sets_sc[0][x] for x in inter_sc ]

		print(tmags_lc[0][indices_lc2])
		print(tmags_lc[1][indices_lc])
		print(len(indices_lc), len(indices_sc))

#		if sum(idx_lc) > 0:
		kde_lc = KDE(tmags_lc[0][indices_lc2])
		kde_lc.fit(gridsize=1000)
		ax.fill_between(kde_lc.support, 0, kde_lc.density/np.max(kde_lc.density), color='r', alpha=0.3, label='30-min cadence')
#
#		if sum(idx_sc) > 0:
		kde_sc = KDE(tmags_sc[0][indices_sc2])
		kde_sc.fit(gridsize=1000)
#			ax.fill_between(kde_sc.support, 0, kde_sc.density*sum(idx_sc), color='b', alpha=0.3, label='2-min cadence')
		ax.fill_between(kde_sc.support, 0, kde_sc.density/np.max(kde_sc.density), color='b', alpha=0.3, label='2-min cadence')

#		kde_all = KDE(tmags)
#		kde_all.fit(gridsize=1000)
#		ax.plot(kde_all.support, kde_all.density/, 'k-', lw=1.5, label='All')

		ax.set_ylim(ymin=0)
		ax.set_xlabel('TESS magnitude')
		ax.set_ylabel('Normalised Density')
		ax.xaxis.set_major_locator(MultipleLocator(2))
		ax.xaxis.set_minor_locator(MultipleLocator(1))
		ax.legend(frameon=False, loc='upper left', borderaxespad=0,handlelength=2.5, handletextpad=0.4)

#		filename = 'mag_dist.%s' % self.extension
#		fig.savefig(os.path.join(self.outfolder, filename))
		if self.show:
			plt.show()
		else:
			plt.close(fig)

	#----------------------------------------------------------------------------------------------
	def plot_waittime(self):
		"""
		Visiualize the worker wait-time during the processing.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)
		logger.info("Running waittime diagnostics...")

		run_tables = ['diagnostics']
		if self.corrections_done:
			run_tables.append('diagnostics_corr')

		for table in run_tables:
			# Check if the worker_waittime column is in the database
			# It was not generated in earlier versions of the pipeline
			self.cursor.execute("PRAGMA table_info(" + table + ")")
			if 'worker_wait_time' not in [r['name'] for r in self.cursor.fetchall()]:
				logger.info("WORKER_WAIT_TIME is not stored in database.")
				return

			# Get the data from the database:
			star_vals = self.search_database(select=['todolist.priority', table + '.worker_wait_time'])
			tab = Table(rows=star_vals,
				names=('priority', 'worker_wait_time'),
				dtype=('int64', 'float32'))

			# Create figure figure:
			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.scatter(tab['priority'], tab['worker_wait_time'], alpha=0.3)
			ax.set_xlabel('Priority')
			ax.set_ylabel('Worker wait-time (s)')

			# Save figure to file and close:
			fig.savefig(os.path.join(self.outfolder, 'worker_waittime_' + table))
			if self.show:
				plt.show()
			else:
				plt.close(fig)

	#----------------------------------------------------------------------------------------------
	def plot_haloswitch(self):
		"""
		Visiualize the disgnostics used to automatically switch to Halo photometry.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)
		logger.info("Running halo-switch diagnostics...")

		# Check if the edge_flux column is in the database
		# It was not generated in earlier versions of the pipeline
		self.cursor.execute("PRAGMA table_info(diagnostics)")
		if 'edge_flux' not in [r['name'] for r in self.cursor.fetchall()]:
			logger.info("EDGE_FLUX is not stored in database.")
			return

		# TODO: Store these in the database as well
		haloswitch_tmag_limit = 6.0 # Maximal Tmag to apply Halo photometry automatically
		haloswitch_flux_limit = 0.01

		# Create figure figure:
		fig = plt.figure(figsize=(12,8))
		ax = fig.add_subplot(111)

		for datasource in ('ffi', 'tpf'):
			# Get the data from the database:
			constraint_cadence = "datasource='ffi'" if datasource == 'ffi' else "datasource!='ffi'"
			star_vals = self.search_database(
				select=['todolist.priority', 'todolist.tmag', 'diagnostics.edge_flux', "(diagnostics.errors IS NOT NULL AND INSTR(diagnostics.errors, 'Automatically switched to Halo photometry')>0) AS switched"],
				search=['edge_flux > 0', constraint_cadence])
			if not star_vals:
				continue

			tab = Table(rows=star_vals,
				names=('priority', 'tmag', 'edge_flux', 'switched'),
				dtype=('int64', 'float32', 'float64', 'bool'))

			switched = tab['switched']
			expected_flux = mag2flux(tab['tmag'])

			ax.scatter(tab['tmag'], tab['edge_flux']/expected_flux, alpha=0.3, label=datasource)
			ax.scatter(tab['tmag'][switched], tab['edge_flux'][switched]/expected_flux[switched], c='r', marker='x', label='Switched')

		ax.axvline(haloswitch_tmag_limit, c='r', ls='--')
		ax.axhline(haloswitch_flux_limit, c='r', ls='--')
		ax.set_xlabel('TESS magnitude')
		ax.set_ylabel('Edge flux / Expected total flux')
		ax.set_yscale('log')
		ax.set_ylim([1e-5, 1.0])
		ax.set_xlim(self.tmag_limits)
		ax.legend()

		# Save figure to file and close:
		fig.savefig(os.path.join(self.outfolder, 'haloswitch'))
		if self.show:
			plt.show()
		else:
			plt.close(fig)
