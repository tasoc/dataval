# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 20:51:06 2017

@author: Dr. Mikkel N. Lund
"""
#===============================================================================
# Packages
#===============================================================================

import os
import argparse
import logging
import six
import numpy as np
import matplotlib.pyplot as plt

#from dataval.plots import plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.ticker import MultipleLocator, FormatStrFormatter,ScalarFormatter
y_formatter = ScalarFormatter(useOffset=False)
mpl.rcParams['font.family'] = 'serif'
from matplotlib import rc
rc('text', usetex=True)
import scipy.interpolate as INT
import scipy.optimize as OP
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
import sqlite3
from scipy.stats import binned_statistic as binning

from dataval.quality import DatavalQualityFlags
#from dataval.utilities import rms_timescale #, sphere_distance
from dataval.noise_model import phot_noise

plt.ioff()


#def reduce_percentile1(x):
#	return np.nanpercentile(x, 99.9, interpolation='lower')
#def reduce_percentile2(x):
#	return np.nanpercentile(x, 0.5, interpolation='lower')


def combine_flag_dicts(a, b):
	return {key: a.get(key, 0) | b.get(key, 0) for key in set().union(a.keys(), b.keys())}

#------------------------------------------------------------------------------
class DataValidation(object):


	def __init__(self, input_folders, output_folder=None, validate=True, method='all', colorbysector=False, ext='png', showplots=False, sysnoise=0):

		# Store inputs:
		self.input_folders = input_folders
		self.method = method
		self.extension = ext
		self.show = showplots
		self.outfolders = output_folder
		self.sysnoise = sysnoise
		self.doval = validate
		self.color_by_sector = colorbysector

		#load sqlite to-do files
		if len(self.input_folders)==1:
			if self.outfolders is None:
				path = os.path.join(self.input_folders[0], 'data_validation')
				self.outfolders = path
				if not os.path.exists(self.outfolders):
					os.makedirs(self.outfolders)

		for i, f in enumerate(self.input_folders):
			todo_file = os.path.join(f, 'todo.sqlite')
			logger.debug("TODO file: %s", todo_file)
			if not os.path.exists(todo_file):
				raise ValueError("TODO file not found")

			# Open the SQLite file:
			self.conn = sqlite3.connect(todo_file)
			self.conn.row_factory = sqlite3.Row
			self.cursor = self.conn.cursor()

			if self.method == 'all':
				# Create table for data-validation:
				if self.doval:
					self.cursor.execute('DROP TABLE IF EXISTS datavalidation_raw')
				self.cursor.execute("""CREATE TABLE IF NOT EXISTS datavalidation_raw (
					priority INT PRIMARY KEY NOT NULL,
					dataval INT NOT NULL,
					approved BOOLEAN NOT NULL,
					FOREIGN KEY (priority) REFERENCES todolist(priority) ON DELETE CASCADE ON UPDATE CASCADE
				);""")

				self.conn.commit()

	def close(self):
		"""Close DataValidation object and all associated objects."""
		self.cursor.close()
		self.conn.close()

	def __exit__(self, *args):
		self.close()

	def __enter__(self):
		return self

	def search_database(self, select=None, search=None, order_by=None, limit=None, distinct=False):
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

		if select is None:
			select = '*'
		elif isinstance(select, (list, tuple)):
			select = ",".join(select)

		if search is None:
			search = ''
		elif isinstance(search, (list, tuple)):
			search = "WHERE " + " AND ".join(search)
		else:
			search = 'WHERE ' + search

		if order_by is None:
			order_by = ''
		elif isinstance(order_by, (list, tuple)):
			order_by = " ORDER BY " + ",".join(order_by)
		elif isinstance(order_by, six.string_types):
			order_by = " ORDER BY " + order_by

		limit = '' if limit is None else " LIMIT %d" % limit

		query = "SELECT {distinct:s}{select:s} FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority LEFT JOIN datavalidation_raw ON todolist.priority=datavalidation_raw.priority {search:s}{order_by:s}{limit:s};".format(
			distinct='DISTINCT ' if distinct else '',
			select=select,
			search=search,
			order_by=order_by,
			limit=limit
		)
		logger.debug("Running query: %s", query)

		# Ask the database: status=1
		self.cursor.execute(query)
		return [dict(row) for row in self.cursor.fetchall()]



	def Validations(self):

		if self.method == 'all':
			val1 = self.plot_magtoflux(return_val=True)
			val2 = self.plot_pixinaperture(return_val=True)
			val3 = self.plot_contam(return_val=True)
			val4 = self.plot_onehour_noise(return_val=True)
			self.plot_stamp()
			self.plot_mag_dist()

			if self.doval:
				val = combine_flag_dicts(val1, val2)
				val = combine_flag_dicts(val, val3)
				val = combine_flag_dicts(val, val4)

				dv = np.array(list(val.values()), dtype="int32")

				#Reject: Small/High apertures; Contamination>1;
				app = np.ones_like(dv, dtype='bool')
				qf = DatavalQualityFlags.filter(dv)
				app[~qf] = False

				[self.cursor.execute("INSERT INTO datavalidation_raw (priority, dataval, approved) VALUES (?,?,?);", (int(v1), int(v2), bool(v3))) for v1,v2,v3 in
						zip(np.array(list(val.keys()), dtype="int32"),dv,app)]

				self.cursor.execute("INSERT INTO datavalidation_raw (priority, dataval, approved) select todolist.priority, 0, 0 FROM todolist WHERE todolist.status not in (1,3);")
				self.conn.commit()


		elif self.method == 'mag2flux':
			self.plot_magtoflux()
		elif self.method == 'pixvsmag':
			self.plot_pixinaperture()
		elif self.method == 'stamp':
			self.plot_stamp()
		elif self.method == 'magdist':
			self.plot_mag_dist()
		elif self.method == 'noise':
			self.plot_onehour_noise()
		elif self.method == 'magdist':
			self.plot_mag_dist()
		elif self.method == 'contam':
			self.plot_contam()
		elif self.method == 'magdistoverlap':
			self.plot_mag_dist_overlap()



	# =============================================================================
	#
	# =============================================================================

	def plot_contam(self, return_val=False):


		"""
		Function to plot the contamination against the stellar TESS magnitudes

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""

		logger=logging.getLogger(__name__)

		logger.info('------------------------------------------')
		logger.info('Plotting Contamination vs. Magnitude')




		fig = plt.figure(figsize=(10, 5))
		fig.subplots_adjust(left=0.145, wspace=0.3, top=0.945, bottom=0.145, right=0.975)
		ax1 = fig.add_subplot(211)
		ax2 = fig.add_subplot(212)

		star_vals = self.search_database(search=['status in (1,3)'], select=['todolist.priority','todolist.sector','todolist.starid','method','todolist.datasource','todolist.tmag','contamination'])

		if self.color_by_sector:
			sec = np.array([star['sector'] for star in star_vals], dtype=int)
			sectors = np.array(list(set(sec)))
			if len(sectors)>1:
				norm = colors.Normalize(vmin=1, vmax=len(sectors))
				scalarMap = cmx.ScalarMappable(norm=norm, cmap=plt.get_cmap('Set1') )
				rgba_color = np.array([scalarMap.to_rgba(s) for s in sec])
			else:
				rgba_color = 'k'
		else:
			rgba_color = 'k'


		tmags = np.array([star['tmag'] for star in star_vals], dtype=float)
		cont = np.array([star['contamination'] for star in star_vals], dtype=float)
		pri = np.array([star['priority'] for star in star_vals], dtype=int)
		source = np.array([star['datasource'] for star in star_vals], dtype=str)

		# Indices for plotting
		idx_high_ffi = (cont>1) & (source=='ffi')
		idx_high_tpf = (cont>1) & (source=='tpf')
		idx_low_ffi = (cont<=1) & (source=='ffi')
		idx_low_tpf = (cont<=1) & (source=='tpf')
		cont[idx_high_ffi] = 1.1
		cont[idx_high_tpf] = 1.1

		# Remove nan contaminations (should be only from Halo targets)
		cont[np.isnan(cont)] = 1.2

		# Plot individual contamination points
		ax1.scatter(tmags[idx_low_ffi], cont[idx_low_ffi], marker='o', facecolors=rgba_color, color=rgba_color, alpha=0.1)
		ax2.scatter(tmags[idx_low_tpf], cont[idx_low_tpf], marker='o', facecolors=rgba_color, color=rgba_color, alpha=0.1)

		if self.doval:
			ax1.scatter(tmags[idx_high_ffi], cont[idx_high_ffi], marker='o', facecolors='None', color=rgba_color, alpha=0.9)
			ax1.scatter(tmags[(cont==1.2) & (source=='ffi')], cont[(cont==1.2) & (source=='ffi')], marker='o', facecolors='None', color='r', alpha=0.9)
			ax2.scatter(tmags[idx_high_tpf], cont[idx_high_tpf], marker='o', facecolors='None', color=rgba_color, alpha=0.9)
			ax2.scatter(tmags[(cont==1.2) & (source=='tpf')], cont[(cont==1.2) & (source=='tpf')], marker='o', facecolors='None', color='r', alpha=0.9)

		# Indices for finding validation limit
#		idx_low = (cont<=1)
		# Compute median-bin curve
#		bin_means, bin_edges, binnumber = binning(tmags[idx_low], cont[idx_low], statistic='median', bins=15, range=(np.nanmin(tmags),np.nanmax(tmags)))
#		bin_width = (bin_edges[1] - bin_edges[0])
#		bin_centers = bin_edges[1:] - bin_width/2

		xmax = np.arange(0, 20, 1)
		ymax = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.12, 0.2, 0.3, 0.45, 0.6, 0.7, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])


		# Plot median-bin curve (1 and 5 times standadised MAD)
#		ax1.scatter(bin_centers, 1.4826*bin_means, marker='o', color='r')
#		ax1.scatter(bin_centers, 1.4826*5*bin_means, marker='.', color='r')
		ax1.plot(xmax, ymax, marker='.', color='r', ls='-')
		ax2.plot(xmax, ymax, marker='.', color='r', ls='-')

		cont_vs_mag = INT.InterpolatedUnivariateSpline(xmax, ymax)
#		mags = np.linspace(np.nanmin(tmags),np.nanmax(tmags),100)

		ax1.set_xlim([np.min(tmags[(source=='ffi')])-0.5, np.max(tmags[(source=='ffi')])+0.5])
		ax2.set_xlim([np.min(tmags[(source=='tpf')])-0.5, np.max(tmags[(source=='tpf')])+0.5])

		# Plotting stuff
		for axx in np.array([ax1, ax2]):
#			axx.plot(mags, cont_vs_mag(mags))

			if self.doval:
				axx.set_ylim([-0.05, 1.3])
			else:
				axx.set_ylim([-0.05, 1.1])


			axx.axhline(y=0, ls='--', color='k', zorder=-1)

			if self.doval:
				axx.axhline(y=1.1, ls=':', color='k', zorder=-1)
				axx.axhline(y=1.2, ls=':', color='r', zorder=-1)
			axx.axhline(y=1, ls=':', color='k', zorder=-1)
			axx.set_xlabel('TESS magnitude', fontsize=16, labelpad=10)
			axx.set_ylabel('Contamination', fontsize=16, labelpad=10)

			axx.xaxis.set_major_locator(MultipleLocator(2))
			axx.xaxis.set_minor_locator(MultipleLocator(1))
			axx.yaxis.set_major_locator(MultipleLocator(0.2))
			axx.yaxis.set_minor_locator(MultipleLocator(0.1))
			axx.tick_params(direction='out', which='both', pad=5, length=3)
			axx.tick_params(which='major', pad=6, length=5,labelsize='15')
			axx.yaxis.set_ticks_position('both')
#			axx.legend(loc='upper left', prop={'size': 12})


		# Assign validation bits
		if return_val:
			val0 = {}
			val0['dv'] = np.zeros_like(pri, dtype="int32")
			val0['dv'][cont>=1] |= DatavalQualityFlags.ContaminationOne
			val0['dv'][(cont>cont_vs_mag(tmags)) & (cont<1)] |= DatavalQualityFlags.ContaminationHigh

			val = dict(zip(pri, val0['dv']))


		###########
		filename = 'contam.%s' %self.extension
		fig.savefig(os.path.join(self.outfolders, filename))

		if self.show:
			plt.show()
		else:
			plt.close('all')

		if return_val:
			return val


	# =============================================================================
	#
	# =============================================================================

	def plot_onehour_noise(self, return_val=False):

		"""
		Function to plot the light curve noise against the stellar TESS magnitudes

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""

		logger=logging.getLogger(__name__)

		logger.info('------------------------------------------')
		logger.info('Plotting Noise vs. Magnitude')


		fig1 = plt.figure(figsize=(15, 5))
		fig1.subplots_adjust(left=0.145, wspace=0.3, top=0.945, bottom=0.145, right=0.975)
		ax11 = fig1.add_subplot(121)
		ax12 = fig1.add_subplot(122)

		fig2 = plt.figure(figsize=(15, 5))
		fig2.subplots_adjust(left=0.145, wspace=0.3, top=0.945, bottom=0.145, right=0.975)
		ax21 = fig2.add_subplot(121)
		ax22 = fig2.add_subplot(122)

		PARAM = {}

		star_vals = self.search_database(search=['status in (1,3)'], select=['todolist.priority','todolist.starid','todolist.datasource','todolist.sector','todolist.tmag','rms_hour','ptp','ccd'])

		if self.color_by_sector:
			sec = np.array([star['sector'] for star in star_vals], dtype=int)
			sectors = np.array(list(set(sec)))
			if len(sectors)>1:
				norm = colors.Normalize(vmin=1, vmax=len(sectors))
				scalarMap = cmx.ScalarMappable(norm=norm, cmap=plt.get_cmap('Set1') )
				rgba_color = np.array([scalarMap.to_rgba(s) for s in sec])
			else:
				rgba_color = 'k'
		else:
			rgba_color = 'k'


		tmags = np.array([star['tmag'] for star in star_vals], dtype=float)
		pri = np.array([star['priority'] for star in star_vals], dtype=int)
		rms = np.array([star['rms_hour']*1e6 for star in star_vals], dtype=float)
		ptp = np.array([star['ptp']*1e6 for star in star_vals], dtype=float)
		source = np.array([star['datasource'] for star in star_vals], dtype=str)

		# TODO: Update elat+elon based on observing sector?
		PARAM['RA'] = 0
		PARAM['DEC'] = 0

		idx_lc = (source=='ffi')
		idx_sc = (source=='tpf')

		ax11.scatter(tmags[idx_lc], rms[idx_lc], marker='o', facecolors=rgba_color, edgecolor=rgba_color, alpha=0.1, label='30-min cadence')
		ax12.scatter(tmags[idx_sc], rms[idx_sc], marker='o', facecolors=rgba_color, edgecolor=rgba_color, alpha=0.1, label='2-min cadence')

		ax21.scatter(tmags[idx_lc], ptp[idx_lc], marker='o', facecolors=rgba_color, edgecolor=rgba_color, alpha=0.1, label='30-min cadence')
		ax22.scatter(tmags[idx_sc], ptp[idx_sc], marker='o', facecolors=rgba_color, edgecolor=rgba_color, alpha=0.1, label='2-min cadence')

		# Plot theoretical lines
		mags = np.linspace(np.min(tmags)-0.5, np.max(tmags)+0.5, 50)
		vals_rms_tpf = np.zeros([len(mags), 4])
		vals_rms_ffi = np.zeros([len(mags), 4])
		vals_ptp_ffi = np.zeros([len(mags), 4])
		vals_ptp_tpf = np.zeros([len(mags), 4])

		# Expected *1-hour* RMS noise fii
		for i in range(len(mags)):
			vals_rms_ffi[i,:], _ = phot_noise(mags[i], 5775, 3600, PARAM, cadpix='1800', sysnoise=self.sysnoise, verbose=False)
		tot_noise_rms_ffi = np.sqrt(np.sum(vals_rms_ffi**2, axis=1))

		ax11.semilogy(mags, vals_rms_ffi[:, 0], 'r-')
		ax11.semilogy(mags, vals_rms_ffi[:, 1], 'g--')
		ax11.semilogy(mags, vals_rms_ffi[:, 2], '-')
		ax11.semilogy(mags, tot_noise_rms_ffi, 'k-')
		ax11.axhline(y=self.sysnoise, color='b', ls='--')

		# Expected *1-hour* RMS noise tpf
		for i in range(len(mags)):
			vals_rms_tpf[i,:], _ = phot_noise(mags[i], 5775, 3600, PARAM, cadpix='120', sysnoise=self.sysnoise, verbose=False)
		tot_noise_rms_tpf = np.sqrt(np.sum(vals_rms_tpf**2, axis=1))

		ax12.semilogy(mags, vals_rms_tpf[:, 0], 'r-')
		ax12.semilogy(mags, vals_rms_tpf[:, 1], 'g--')
		ax12.semilogy(mags, vals_rms_tpf[:, 2], '-')
		ax12.semilogy(mags, tot_noise_rms_tpf, 'k-')
		ax12.axhline(y=self.sysnoise, color='b', ls='--')

		# Expected ptp for 30-min
		for i in range(len(mags)):
			vals_ptp_ffi[i,:], _ = phot_noise(mags[i], 5775, 1800, PARAM, cadpix='1800', sysnoise=self.sysnoise, verbose=False)
		tot_noise_ptp_ffi = np.sqrt(np.sum(vals_ptp_ffi**2, axis=1))

		ax21.semilogy(mags, vals_ptp_ffi[:, 0], 'r-')
		ax21.semilogy(mags, vals_ptp_ffi[:, 1], 'g--')
		ax21.semilogy(mags, vals_ptp_ffi[:, 2], '-')
		ax21.semilogy(mags, tot_noise_ptp_ffi, 'k-')
		ax21.axhline(y=self.sysnoise, color='b', ls='--')

		# Expected ptp for 2-min
		for i in range(len(mags)):
			vals_ptp_tpf[i,:], _ = phot_noise(mags[i], 5775, 120, PARAM, cadpix='120', sysnoise=self.sysnoise, verbose=False)
		tot_noise_ptp_tpf = np.sqrt(np.sum(vals_ptp_tpf**2, axis=1))


		ax22.semilogy(mags, vals_ptp_tpf[:, 0], 'r-')
		ax22.semilogy(mags, vals_ptp_tpf[:, 1], 'g--')
		ax22.semilogy(mags, vals_ptp_tpf[:, 2], '-')
		ax22.semilogy(mags, tot_noise_ptp_tpf, 'k-')
		ax22.axhline(y=self.sysnoise, color='b', ls='--')

		ptp_tpf_vs_mag = INT.UnivariateSpline(mags, tot_noise_ptp_tpf)
		ptp_ffi_vs_mag = INT.UnivariateSpline(mags, tot_noise_ptp_ffi)
		rms_tpf_vs_mag = INT.UnivariateSpline(mags, tot_noise_rms_tpf)
		rms_ffi_vs_mag = INT.UnivariateSpline(mags, tot_noise_rms_ffi)

#		print(tics[idx_sc][(ptp[idx_sc] < 0.5*ptp_tpf_vs_mag(tmags[idx_sc]))])
#		print(ptp[idx_sc][(ptp[idx_sc] < 0.5*ptp_tpf_vs_mag(tmags[idx_sc]))])
#		print(tmags[idx_sc][(ptp[idx_sc] < 0.5*ptp_tpf_vs_mag(tmags[idx_sc]))])

		ax11.set_ylabel(r'$\rm RMS\,\, (ppm\,\, hr^{-1})$', fontsize=16, labelpad=10)
		ax21.set_ylabel('point-to-point MDV (ppm)', fontsize=16, labelpad=10)

		for axx in np.array([ax11, ax12, ax21, ax22]):
			axx.set_xlim([np.min(tmags)-0.5, np.max(tmags)+0.5])
			axx.set_xlabel('TESS magnitude', fontsize=16, labelpad=10)
			axx.xaxis.set_major_locator(MultipleLocator(2))
			axx.xaxis.set_minor_locator(MultipleLocator(1))
			axx.tick_params(direction='out', which='both', pad=5, length=3)
			axx.tick_params(which='major', pad=6, length=5,labelsize='15')
			axx.yaxis.set_ticks_position('both')
			axx.set_yscale("log", nonposy='clip')
			axx.legend(loc='upper left', prop={'size': 12})

		###########

		filename = 'rms.%s' %self.extension
		filename2 = 'ptp.%s' %self.extension

		fig1.savefig(os.path.join(self.outfolders, filename))
		fig2.savefig(os.path.join(self.outfolders, filename2))


		# Assign validation bits, for both FFI and TPF
		if return_val:

			dv = np.zeros_like(pri, dtype="int32")

			idx_tpf_ptp = (ptp < ptp_tpf_vs_mag(tmags)) & (ptp>0)
			idx_ffi_ptp = (ptp < ptp_ffi_vs_mag(tmags)) & (ptp>0)
			idx_tpf_rms = (rms < rms_tpf_vs_mag(tmags)) & (rms>0)
			idx_ffi_rms = (rms < rms_ffi_vs_mag(tmags)) & (rms>0)
			idx_invalid = (rms<=0) | ~np.isfinite(rms) | (ptp<=0) | ~np.isfinite(ptp)

			dv[idx_sc & idx_tpf_ptp] |= DatavalQualityFlags.LowPTP
			dv[idx_lc & idx_ffi_ptp] |= DatavalQualityFlags.LowPTP
			dv[idx_sc & idx_tpf_rms] |= DatavalQualityFlags.LowRMS
			dv[idx_lc & idx_ffi_rms] |= DatavalQualityFlags.LowRMS
			dv[idx_lc & idx_ffi_rms] |= DatavalQualityFlags.LowRMS
			dv[idx_invalid] |= DatavalQualityFlags.InvalidNoise

			val = dict(zip(list(pri), list(dv)))

		if self.show:
			plt.show()
		else:
			plt.close('all')

		if return_val:
			return val

	# =============================================================================
	#
	# =============================================================================

	def plot_pixinaperture(self, return_val=False):

		"""
		Function to plot number of pixels in determined apertures against the stellar TESS magnitudes

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""

		logger=logging.getLogger(__name__)

		logger.info('------------------------------------------')
		logger.info('Plotting Pixels in aperture vs. Magnitude')

		fig = plt.figure(figsize=(15, 5))
		ax1 = fig.add_subplot(121)
		ax2 = fig.add_subplot(122)
		fig.subplots_adjust(left=0.1, wspace=0.2, top=0.94, bottom=0.155, right=0.91)


		if self.doval:
			star_vals = self.search_database(search=['status in (1,3)'], select=['todolist.priority','todolist.ccd','todolist.starid','todolist.datasource','todolist.sector','todolist.tmag','diagnostics.mask_size','diagnostics.contamination','todolist.camera','diagnostics.errors'])
		else:
			star_vals = self.search_database(search=['status in (1,3)'], select=['todolist.priority','todolist.ccd','todolist.starid','todolist.datasource','todolist.sector','todolist.tmag','diagnostics.mask_size','diagnostics.contamination','todolist.camera','diagnostics.errors','datavalidation_raw.dataval'])



		if self.color_by_sector:
			sec = np.array([star['sector'] for star in star_vals], dtype=int)
			sectors = np.array(list(set(sec)))
			if len(sectors)>1:
				norm = colors.Normalize(vmin=1, vmax=len(sectors))
				scalarMap = cmx.ScalarMappable(norm=norm, cmap=plt.get_cmap('Set1') )
				rgba_color = np.array([scalarMap.to_rgba(s) for s in sec])
			else:
				rgba_color = 'k'
		else:
			rgba_color = 'k'



		tmags = np.array([star['tmag'] for star in star_vals], dtype=float)
		masksizes = np.array([star['mask_size'] for star in star_vals], dtype=float)
		contam = np.array([star['contamination'] for star in star_vals], dtype=float)
		source = np.array([star['datasource'] for star in star_vals], dtype=str)
		pri = np.array([star['priority'] for star in star_vals], dtype=int)
		minimal_mask_used = np.array([False if star['errors'] is None else ('Using minimum aperture.' in star['errors']) for star in star_vals], dtype='bool')

		dataval = np.zeros_like(pri, dtype='int32')
		if self.doval:
			dataval = np.array([star['dataval'] for star in star_vals], dtype='int32')


#		cats = {}
#		for cams in list(set(camera)):
#			cats[cams] = {}
#			for jj in range(4):
#				cam_file = os.path.join(self.input_folders[0], 'catalog_camera' + str(int(cams)) +'_ccd' + str(jj+1) + '.sqlite')
##				with contextlib.closing(sqlite3.connect(cam_file)) as conn:
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

		idx_lc = (source=='ffi')
		idx_sc = (source=='tpf')

		idx_lc0 = (source=='ffi') & (dataval&(32+64) == 0)
		idx_sc0 = (source=='tpf') & (dataval&(32+64) == 0)

		idx_lc1 = (source=='ffi') & (dataval&(32+64) != 0)
		idx_sc1 = (source=='tpf') & (dataval&(32+64) != 0)

		perm_lc = np.random.permutation(sum(idx_lc0))
		perm_sc = np.random.permutation(sum(idx_sc0))

		ax1.scatter(tmags[idx_lc0][perm_lc], masksizes[idx_lc0][perm_lc], marker='o', c=contam[idx_lc0][perm_lc], alpha=0.5, norm=norm, cmap=plt.get_cmap('PuOr'))
		ax1.scatter(tmags[idx_lc1], masksizes[idx_lc1], marker='.', c=contam[idx_lc1], alpha=0.2, norm=norm, cmap=plt.get_cmap('PuOr'))

		ax2.scatter(tmags[idx_sc0][perm_sc], masksizes[idx_sc0][perm_sc], marker='o', c=contam[idx_sc0][perm_sc], alpha=0.5, norm=norm, cmap=plt.get_cmap('PuOr'))
		ax2.scatter(tmags[idx_sc1], masksizes[idx_sc1], marker='.', c=contam[idx_sc1], alpha=0.2, norm=norm, cmap=plt.get_cmap('PuOr'))

		# Compute median-bin curve
		bin_means, bin_edges, binnumber = binning(tmags[idx_lc], masksizes[idx_lc], statistic='median', bins=15, range=(np.nanmin(tmags[idx_lc]),np.nanmax(tmags[idx_lc])))
		bin_width = (bin_edges[1] - bin_edges[0]);		bin_centers = bin_edges[1:] - bin_width/2
		ax1.scatter(bin_centers, bin_means, marker='o', color='r')

		bin_means, bin_edges, binnumber = binning(tmags[idx_sc], masksizes[idx_sc], statistic='median', bins=15, range=(np.nanmin(tmags[idx_sc]),np.nanmax(tmags[idx_sc])))
		bin_width = (bin_edges[1] - bin_edges[0]);		bin_centers = bin_edges[1:] - bin_width/2
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
#		print(masksizes[(tmags>15) & (source=='tpf')])
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

#		idx_lc2 = (source=='ffi') & (masksizes>4) & (tmags>8)
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

		ax1.set_xlim([np.nanmin(tmags[idx_lc])-0.5, np.nanmax(tmags[idx_lc])+0.5])
		ax2.set_xlim([np.nanmin(tmags[idx_sc])-0.5, np.nanmax(tmags[idx_sc])+0.5])
		ax1.set_ylim([0.99, np.nanmax(masksizes)+500])
		ax2.set_ylim([0.99, np.nanmax(masksizes)+500])

		for axx in np.array([ax1, ax2]):
			axx.set_xlabel('TESS magnitude', fontsize=16, labelpad=10)
			axx.set_ylabel('Pixels in aperture', fontsize=16, labelpad=10)
			xtick_major = np.median(np.diff(axx.get_xticks()))
			axx.xaxis.set_minor_locator(MultipleLocator(xtick_major/2))
			ytick_major = np.median(np.diff(axx.get_yticks()))
			axx.yaxis.set_minor_locator(MultipleLocator(ytick_major/2))
			axx.tick_params(direction='out', which='both', pad=5, length=3)
			axx.tick_params(which='major', pad=6, length=5,labelsize='15')
			axx.yaxis.set_ticks_position('both')
			axx.xaxis.set_ticks_position('both')
			axx.set_yscale("log", nonposy='clip')
			axx.yaxis.set_major_formatter(ScalarFormatter())
#			axx.legend(loc='upper right', prop={'size': 12})

		pos = ax2.get_position()
		axc = fig.add_axes([pos.x0 + pos.width+0.01, pos.y0, 0.01, pos.height], zorder=-1)
		cb = mpl.colorbar.ColorbarBase(axc, cmap=plt.get_cmap('PuOr'), norm=norm, orientation='vertical')
		cb.set_label('Contamination', fontsize=12, labelpad=6)
		cb.ax.tick_params(axis='y', direction='out')

		filename = 'pix_in_aper.%s' %self.extension
		fig.savefig(os.path.join(self.outfolders, filename))

		# Assign validation bits, for both FFI and TPF
		if return_val:
			# Create validation dict:
			val0 = {}
			val0['dv'] = np.zeros_like(pri, dtype="int32")

			# Minimal masks were used:
			val0['dv'][minimal_mask_used] |= DatavalQualityFlags.MinimalMask

			# Small and Large masks:
			val0['dv'][idx_lc & (masksizes<min_bound(tmags)) & (masksizes>0)] |= DatavalQualityFlags.SmallMask
			val0['dv'][idx_lc & (masksizes>max_bound(tmags))] |= DatavalQualityFlags.LargeMask
			val0['dv'][idx_sc & (masksizes<min_bound_sc(tmags)) & (masksizes>0)] |= DatavalQualityFlags.SmallMask

			val = dict(zip(pri, val0['dv']))


		if self.show:
			plt.show()
		else:
			plt.close(fig)

		if return_val:
			return val

	# =============================================================================
	#
	# =============================================================================

	def plot_magtoflux(self, return_val=False):
		"""
		Function to plot flux values from apertures against the stellar TESS magnitudes,
		and determine coefficient describing the relation

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""

		logger=logging.getLogger(__name__)

		logger.info('--------------------------------------')
		logger.info('Plotting Magnitude to Flux conversion')

		fig = plt.figure(figsize=(15, 5))
		fig.subplots_adjust(left=0.1, wspace=0.2, top=0.94, bottom=0.155, right=0.91)
		ax1 = fig.add_subplot(121)
		ax2 = fig.add_subplot(122)

		fig2 = plt.figure()
		fig2.subplots_adjust(left=0.1, wspace=0.2, top=0.94, bottom=0.155, right=0.91)
		ax21 = fig2.add_subplot(111)

#		fig3 = plt.figure(figsize=(15, 5))
#		fig3.subplots_adjust(left=0.14, wspace=0.3, top=0.94, bottom=0.155, right=0.96)
#		ax31 = fig3.add_subplot(121)
#		ax32 = fig3.add_subplot(122)

		star_vals = self.search_database(search=["status in (1,3)"], select=['todolist.priority','todolist.datasource','todolist.sector','todolist.starid','todolist.tmag','ccd','mean_flux', 'contamination'])

		if self.color_by_sector:
			sec = np.array([star['sector'] for star in star_vals], dtype=int)
			sectors = np.array(list(set(sec)))
			if len(sectors)>1:
				norm = colors.Normalize(vmin=1, vmax=len(sectors))
				scalarMap = cmx.ScalarMappable(norm=norm, cmap=plt.get_cmap('Set1') )
				rgba_color = np.array([scalarMap.to_rgba(s) for s in sec])
			else:
				rgba_color = 'k'
		else:
			rgba_color = 'k'


		tmags = np.array([star['tmag'] for star in star_vals], dtype=float)
		meanfluxes = np.array([star['mean_flux'] for star in star_vals], dtype=float)
		contam = np.array([star['contamination'] for star in star_vals], dtype=float)
		source = np.array([star['datasource'] for star in star_vals], dtype=str)
		pri = np.array([star['priority'] for star in star_vals], dtype=int)


		idx_lc = (source=='ffi')
		idx_sc = (source=='tpf')

		norm = colors.Normalize(vmin=0, vmax=1)

		perm_lc = np.random.permutation(sum(idx_lc))
		perm_sc = np.random.permutation(sum(idx_sc))

		ax1.scatter(tmags[idx_lc][perm_lc], meanfluxes[idx_lc][perm_lc], marker='o', c=contam[idx_lc][perm_lc], norm = norm, cmap=plt.get_cmap('PuOr'), alpha=0.1)#, label='30-min cadence')
		ax2.scatter(tmags[idx_sc][perm_sc], meanfluxes[idx_sc][perm_sc], marker='o', c=contam[idx_sc][perm_sc], norm = norm, cmap=plt.get_cmap('PuOr'), alpha=0.1)#, label='2-min cadence')

		xmin = np.array([0, 1.5, 9, 12.6, 13, 14, 15, 16, 17, 18, 19])
		ymin = np.array([8e7, 1.8e7, 12500, 250, 59, 5, 1, 1, 1, 1, 1])

		min_bound = INT.InterpolatedUnivariateSpline(xmin, ymin, k=1)

		ax1.plot(xmin, ymin, ls='-', color='r')
		ax2.plot(xmin, ymin, ls='-', color='r')

		idx1 = np.isfinite(meanfluxes) & np.isfinite(tmags) & (source=='ffi') & (contam<0.15)
		idx2 = np.isfinite(meanfluxes) & np.isfinite(tmags) & (source=='tpf') & (contam<0.15)

		logger.info('Optimising coefficient of relation')
		z = lambda c: np.log10(np.nansum(( (meanfluxes[idx1] -  10**(-0.4*(tmags[idx1] - c))) / (contam[idx1]+1) )**2))
		z2 = lambda c: np.log10(np.nansum(( (meanfluxes[idx2] -  10**(-0.4*(tmags[idx2] - c))) / (contam[idx2]+1) )**2))
		cc = OP.minimize(z, 20.5, method='Nelder-Mead', options={'disp':False})
		cc2 = OP.minimize(z2, 20.5, method='Nelder-Mead', options={'disp':False})

		logger.info('Optimisation terminated successfully? %s' %cc.success)
		logger.info('Coefficient is found to be %1.4f' %cc.x)
		logger.info('Coefficient is found to be %1.4f' %cc2.x)

		C=np.linspace(19, 22, 100)
		[ax21.scatter(c, z(c), marker='o', color='k') for c in C]
		[ax21.scatter(c, z2(c), marker='o', color='b') for c in C]
		ax21.axvline(x=cc.x, color='k', label='30-min')
		ax21.axvline(x=cc2.x, color='b', ls='--', label='2-min')
		ax21.set_xlabel('Coefficient')
		ax21.set_ylabel(r'$\chi^2$')
		ax21.legend(loc='upper left', prop={'size': 12})

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

		ax1.set_xlim([np.nanmin(tmags[source=='ffi'])-1, np.nanmax(tmags[source=='ffi'])+1])
		ax2.set_xlim([np.nanmin(tmags[source=='tpf'])-1, np.nanmax(tmags[source=='tpf'])+1])

		for axx in np.array([ax1, ax2]):
			axx.set_yscale("log", nonposy='clip')

		for axx in np.array([ax1, ax2]):
			axx.set_xlabel('TESS magnitude', fontsize=16, labelpad=10)

			axx.xaxis.set_major_locator(MultipleLocator(2))
			axx.xaxis.set_minor_locator(MultipleLocator(1))

			axx.set_xlim(axx.get_xlim()[::-1])
			axx.tick_params(direction='out', which='both', pad=5, length=3)
			axx.tick_params(which='major', pad=6, length=5,labelsize='15')
			axx.yaxis.set_ticks_position('both')
			axx.xaxis.set_ticks_position('both')

#		ax1.text(10, 1e7, r'$\rm Flux = 10^{-0.4\,(T_{mag} - %1.2f)}$' %cc.x, fontsize=14)
		ax1.set_ylabel('Mean flux', fontsize=16, labelpad=10)

		pos = ax2.get_position()
		axc = fig.add_axes([pos.x0 + pos.width+0.01, pos.y0, 0.01, pos.height], zorder=-1)
		cb = mpl.colorbar.ColorbarBase(axc, cmap=plt.get_cmap('PuOr'), norm=norm, orientation='vertical')
		cb.set_label('Contamination', fontsize=12, labelpad=6)
		cb.ax.tick_params(axis='y', direction='out')


		filename = 'mag_to_flux.%s' %self.extension
		filename2 = 'mag_to_flux_optimize.%s' %self.extension
#		filename3 = 'mag_to_flux_dev.%s' %self.extension

		fig.savefig(os.path.join(self.outfolders, filename))
		fig2.savefig(os.path.join(self.outfolders, filename2))
#		fig3.savefig(os.path.join(self.outfolders, filename3))

#		# Assign validation bits, for both FFI and TPF
		if return_val:
			val0 = {}
			val0['dv'] = np.zeros_like(pri, dtype="int32")
			val0['dv'][meanfluxes<min_bound(tmags)] |= DatavalQualityFlags.MagVsFluxLow
			val0['dv'][(~np.isfinite(meanfluxes)) | (meanfluxes<=0)] |= DatavalQualityFlags.InvalidFlux

			val = dict(zip(pri, val0['dv']))

		if self.show:
			plt.show(block=True)
		else:
			plt.close('all')

		if return_val:
			return val


	# =========================================================================
	#
	# =========================================================================

	def plot_stamp(self, return_val=False):

		"""
		Function to plot width and height of pixel stamps against the stellar TESS magnitudes

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""

		logger=logging.getLogger(__name__)

		logger.info('--------------------------------------')
		logger.info('Plotting Stamp sizes')


		fig1 = plt.figure(figsize=(15, 10))
		ax11 = fig1.add_subplot(221)
		ax12 = fig1.add_subplot(222)
		ax13 = fig1.add_subplot(223)
		ax14 = fig1.add_subplot(224)

		fig2 = plt.figure(figsize=(15, 7))
		ax21 = fig2.add_subplot(121)
		ax22 = fig2.add_subplot(122)

		star_vals = self.search_database(search=["status in (1,3)"], select=['todolist.datasource','todolist.sector','todolist.tmag','stamp_resizes','stamp_width','stamp_height','elaptime'])

		if self.color_by_sector:
			sec = np.array([star['sector'] for star in star_vals], dtype=int)
			sectors = np.array(list(set(sec)))
			if len(sectors)>1:
				norm = colors.Normalize(vmin=1, vmax=len(sectors))
				scalarMap = cmx.ScalarMappable(norm=norm, cmap=plt.get_cmap('Set1') )
				rgba_color = np.array([scalarMap.to_rgba(s) for s in sec])
			else:
				rgba_color = 'r'
		else:
			rgba_color = 'r'

		tmags = np.array([star['tmag'] for star in star_vals], dtype=float)
		et = np.array([star['elaptime'] for star in star_vals], dtype=float)
		width = np.array([star['stamp_width'] for star in star_vals], dtype=float)
		height = np.array([star['stamp_height'] for star in star_vals], dtype=float)
		resize = np.array([star['stamp_resizes'] for star in star_vals], dtype=float)
		ds = np.array([star['datasource']=='ffi' for star in star_vals], dtype=bool)

		idx1 = (resize<1) & (ds==True)
		idx2 = (resize<1) & (ds==False)
		idx3 = (resize>0) & (ds==True)
		idx4 = (resize>0) & (ds==False)

		ax12.scatter(tmags[idx1], width[idx1], marker='o', facecolors='None', color=rgba_color, label='30-min cadence, no resize', alpha=0.5, zorder=2)
		ax14.scatter(tmags[idx2], width[idx2], marker='o', facecolors='None', color=rgba_color, label='2-min cadence, no resize', alpha=0.5, zorder=2)

		ax12.scatter(tmags[idx3], width[idx3], marker='o', facecolors='None', color='k', label='30-min cadence, resized', alpha=0.5)
		ax14.scatter(tmags[idx4], width[idx4], marker='o', facecolors='None', color='k', label='2-min cadence, resized', alpha=0.5)

		ax11.scatter(tmags[idx1], height[idx1], marker='o', facecolors='None', color=rgba_color, label='30-min cadence, no resize', alpha=0.5, zorder=2)
		ax13.scatter(tmags[idx2], height[idx2], marker='o', facecolors='None', color=rgba_color, label='2-min cadence, no resize', alpha=0.5, zorder=2)

		ax11.scatter(tmags[idx3], height[idx3], marker='o', facecolors='None', color='k', label='30-min cadence, resized', alpha=0.5)
		ax13.scatter(tmags[idx4], height[idx4], marker='o', facecolors='None', color='k', label='2-min cadence, resized', alpha=0.5)

		bin_means, bin_edges, binnumber = binning(tmags[(ds==True)], height[(ds==True)], statistic='median', bins=20, range=(1.5,10))
		bin_width = (bin_edges[1] - bin_edges[0])
		bin_centers = bin_edges[1:] - bin_width/2

		bin_means2, bin_edges2, binnumber2 = binning(tmags[(ds==True)], width[(ds==True)], statistic='median', bins=20, range=(1.5,10))
		bin_width2 = (bin_edges2[1] - bin_edges2[0])
		bin_centers2 = bin_edges2[1:] - bin_width2/2

#		ax12.scatter(bin_centers2, bin_means2, marker='o', color='b', zorder=3)
#		ax11.scatter(bin_centers, bin_means, marker='o', color='b', zorder=3)

		normalize2 = colors.Normalize(vmin=0, vmax=np.max(resize))
		scalarMap = cmx.ScalarMappable(norm=normalize2, cmap=plt.get_cmap('Set1') )
		for jj in range(0, int(np.max(resize))):
			rgba_color = scalarMap.to_rgba(jj)
			try:
				kde1 = KDE(et[(ds==True)][(resize[(ds==True)]==jj) & (et[(ds==True)]<50)])
				kde1.fit(gridsize=1000)
				ax21.plot(kde1.support, kde1.density, color=rgba_color)
			except:
				pass

		kde1 = KDE(et[(ds==True) & (et<50)])
		kde2 = KDE(et[(ds==False) & (et<50)])
		kde1.fit(gridsize=1000)
		kde2.fit(gridsize=1000)
		ax21.plot(kde1.support, kde1.density, color='k', lw=2, label='30-min cadence')
		ax22.plot(kde2.support, kde2.density, color='k', lw=2, label='2-min candence')
		ax21.set_xlim([0, 50])


		# Decide how many pixels to use based on lookup tables as a function of Tmag:
		mags = np.array([ 0.        ,  0.52631579,  1.05263158,  1.57894737,  2.10526316,
       2.63157895,  3.15789474,  3.68421053,  4.21052632,  4.73684211,
       5.26315789,  5.78947368,  6.31578947,  6.84210526,  7.36842105,
       7.89473684,  8.42105263,  8.94736842,  9.47368421, 10.        ])
		nhei = np.array([831.98319063, 533.58494422, 344.0840884 , 223.73963332,
      147.31365728,  98.77856016,  67.95585074,  48.38157414,
       35.95072974,  28.05639497,  23.043017  ,  19.85922009,
       17.83731732,  16.5532873 ,  15.73785092,  15.21999971,
       14.89113301,  14.68228285,  14.54965042,  14.46542084])
		nwid = np.array([157.71602062, 125.1238281 ,  99.99440209,  80.61896267,
       65.6799962 ,  54.16166547,  45.28073365,  38.4333048 ,
       33.15375951,  29.08309311,  25.94450371,  23.52456986,
       21.65873807,  20.22013336,  19.1109318 ,  18.25570862,
       17.59630936,  17.08789543,  16.69589509,  16.39365266])

		mags2 = np.linspace(np.min(tmags)-0.2, np.max(tmags)+0.2, 500)
		nwid2 = np.array([2*(np.ceil(np.interp(m, mags, nwid))//2)+1 for m in mags2])
		nhei2 = np.array([2*(np.ceil(np.interp(m, mags, nhei))//2)+1 for m in mags2])

		nwid2[(nwid2<15)] = 15
		nhei2[(nhei2<15)] = 15

#		ax12.plot(mags2,nwid2, 'b--')
#		ax11.plot(mags2,nhei2, 'b--')

		ax12.set_ylabel('Stamp width (pixels)', fontsize=16, labelpad=10)
		ax14.set_ylabel('Stamp width (pixels)', fontsize=16, labelpad=10)
		ax11.set_ylabel('Stamp height (pixels)', fontsize=16, labelpad=10)
		ax13.set_ylabel('Stamp height (pixels)', fontsize=16, labelpad=10)

		ax12.yaxis.set_major_locator(MultipleLocator(20))
		ax12.yaxis.set_minor_locator(MultipleLocator(10))

		ax11.yaxis.set_major_locator(MultipleLocator(50))
		ax11.yaxis.set_minor_locator(MultipleLocator(25))

		for axx in np.array([ax11, ax12, ax13, ax14]):
			axx.xaxis.set_major_locator(MultipleLocator(2))
			axx.xaxis.set_minor_locator(MultipleLocator(1))
			axx.tick_params(direction='out', which='both', pad=5, length=3)
			axx.tick_params(which='major', pad=6, length=5,labelsize='15')
			axx.yaxis.set_ticks_position('both')
			axx.xaxis.set_ticks_position('both')
			axx.set_xlabel('TESS magnitude', fontsize=16, labelpad=10)
			axx.legend(loc='upper right', prop={'size': 12})

		for axx in np.array([ax21, ax22]):
			axx.xaxis.set_major_locator(MultipleLocator(5))
			axx.xaxis.set_minor_locator(MultipleLocator(2.5))
			axx.tick_params(direction='out', which='both', pad=5, length=3)
			axx.tick_params(which='major', pad=6, length=5,labelsize='15')
			axx.yaxis.set_ticks_position('both')
			axx.set_xlabel('Calculation time (sec)', fontsize=16, labelpad=10)
			axx.legend(loc='upper right', prop={'size': 12})


		filename = 'stamp_size.%s' %self.extension
		filename2 = 'calc_time.%s' %self.extension

		fig1.savefig(os.path.join(self.outfolders, filename))
		fig2.savefig(os.path.join(self.outfolders, filename2))

		if self.show:
			plt.show()
		else:
			plt.close('all')


	# =============================================================================
	#
	# =============================================================================

	def plot_mag_dist(self):

		"""
		Function to plot magnitude distribution for targets

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""

		logger=logging.getLogger(__name__)

		logger.info('--------------------------------------')
		logger.info('Plotting Magnitude distribution')

		fig = plt.figure(figsize=(10,5))
		ax = fig.add_subplot(111)
		fig.subplots_adjust(left=0.14, wspace=0.3, top=0.94, bottom=0.155, right=0.96)

		star_vals = self.search_database(search=["status in (1,3)"], select=['todolist.datasource','todolist.tmag'])

		tmags = np.array([star['tmag'] for star in star_vals])
		source = np.array([star['datasource'] for star in star_vals])

		idx_lc = (source=='ffi')
		idx_sc = (source=='tpf')

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

		ax.set_ylim(ymin=0)
		ax.set_xlabel('TESS magnitude', fontsize=16, labelpad=10)
		ax.set_ylabel('Normalised Density', fontsize=16, labelpad=10)
		ax.xaxis.set_major_locator(MultipleLocator(2))
		ax.xaxis.set_minor_locator(MultipleLocator(1))
		ax.tick_params(direction='out', which='both', pad=5, length=3)
		ax.tick_params(which='major', pad=6, length=5,labelsize='15')
		ax.yaxis.set_ticks_position('both')
		ax.xaxis.set_ticks_position('both')
		ax.legend(frameon=False, prop={'size':12} ,loc='upper left', borderaxespad=0,handlelength=2.5, handletextpad=0.4)

		filename = 'mag_dist.%s' %self.extension
		fig.savefig(os.path.join(self.outfolders, filename))

		if self.show:
			plt.show()
		else:
			plt.close('all')

	# =============================================================================
	#
	# =============================================================================

	def plot_mag_dist_overlap(self):

		"""
		Function to plot magnitude distribution overlap between sectors

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""

		logger=logging.getLogger(__name__)

		logger.info('--------------------------------------')
		logger.info('Plotting Magnitude distribution')

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

			select=['todolist.starid','todolist.datasource','todolist.tmag']
			select = ",".join(select)
#			search=["status in (1,3)"]
			search=["approved=1"]
			search = "WHERE " + " AND ".join(search)

			query = "SELECT {select:s} FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority LEFT JOIN datavalidation_raw ON todolist.priority=datavalidation_raw.priority  {search:s};".format(
			select=select,
			search=search)

			# Ask the database: status=1
			cursor.execute(query)
			star_vals = [dict(row) for row in cursor.fetchall()]

			tmags = np.array([star['tmag'] for star in star_vals])
			source = np.array([star['datasource'] for star in star_vals])
			starid = np.array([star['starid'] for star in star_vals], dtype="int32")

			idx_lc = (source=='ffi')
			idx_sc = (source=='tpf')

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
		ax.set_xlabel('TESS magnitude', fontsize=16, labelpad=10)
		ax.set_ylabel('Normalised Density', fontsize=16, labelpad=10)
		ax.xaxis.set_major_locator(MultipleLocator(2))
		ax.xaxis.set_minor_locator(MultipleLocator(1))
		ax.tick_params(direction='out', which='both', pad=5, length=3)
		ax.tick_params(which='major', pad=6, length=5,labelsize='15')
		ax.yaxis.set_ticks_position('both')
		ax.xaxis.set_ticks_position('both')
		ax.legend(frameon=False, prop={'size':12} ,loc='upper left', borderaxespad=0,handlelength=2.5, handletextpad=0.4)

#		filename = 'mag_dist.%s' %self.extension
#		fig.savefig(os.path.join(self.outfolders, filename))

		if self.show:
			plt.show()
		else:
			plt.close('all')


	# =============================================================================
	#
	# =============================================================================

	def plot_dist_to_edge(self, return_val=False):

		pass



#------------------------------------------------------------------------------
if __name__ == '__main__':

	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Run Data Validation pipeline.')
	parser.add_argument('-m', '--method', help='Corrector method to use.', default='all', choices=('pixvsmag', 'contam', 'mag2flux', 'stamp', 'noise', 'magdist'))
	parser.add_argument('-e', '--ext', help='Extension of plots.', default='png', choices=('png', 'eps'))
	parser.add_argument('-s', '--show', help='Show plots.', action='store_true')
	parser.add_argument('-v', '--validate', help='Compute validation (only run is method is "all").', action='store_true')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('-cbs', '--colorbysector', help='Color by sector', action='store_true')
	parser.add_argument('-sn', '--sysnoise', type=float, help='systematic noise level for noise plot.', nargs='?', default=5.0)
	parser.add_argument('input_folders', type=str, help='Directory to create catalog files in.', nargs='+')
	parser.add_argument('output_folder', type=str, help='Directory in which to place output if several input folders are given.', nargs='?', default=None)
	args = parser.parse_args()

	if args.output_folder is None and len(args.input_folders) > 1:
		parser.error("Please specify an output directory!")

	# Set logging level:
	logging_level = logging.INFO
	if args.quiet:
		logging_level = logging.WARNING
	elif args.debug:
		logging_level = logging.DEBUG

	# Setup logging:
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	console = logging.StreamHandler()
	console.setFormatter(formatter)
	logger = logging.getLogger(__name__)
	logger.addHandler(console)
	logger.setLevel(logging_level)
	logger_parent = logging.getLogger('dataval')
	logger_parent.addHandler(console)
	logger_parent.setLevel(logging_level)


	logger.info("Loading input data from '%s'", args.input_folders)
	logger.info("Putting output data in '%s'", args.output_folder)

	# Create DataValidation object:
	with DataValidation(args.input_folders, output_folder=args.output_folder,
		validate=args.validate, method=args.method, colorbysector=args.colorbysector,
		showplots=args.show, ext=args.ext, sysnoise=args.sysnoise) as dataval:

		# Run validation
		dataval.Validations()
