# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 20:51:06 2017

@author: Dr. Mikkel N. Lund
"""
#===============================================================================
# Packages
#===============================================================================

from __future__ import with_statement, print_function, division
import os
import argparse
import logging
import six
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.ticker import MultipleLocator, FormatStrFormatter,ScalarFormatter
y_formatter = ScalarFormatter(useOffset=False)
mpl.rcParams['font.family'] = 'serif'
from matplotlib import rc
rc('text', usetex=True)
from astropy import units as u
from astropy.coordinates import SkyCoord
import scipy.interpolate as INT
import scipy.optimize as OP
from bottleneck import move_std
from astropy.io import fits
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
#from pywcsgrid2.allsky_axes import make_allsky_axes_from_header, allsky_header
#import matplotlib.patheffects
import sqlite3
#import pywcsgrid2.healpix_helper as healpix_helper
from scipy.stats import binned_statistic as binning
plt.ioff()


#==============================================================================
# Noise model as a function of magnitude and position
#==============================================================================

def ZLnoise(gal_lat):
    # RMS noise from Zodiacal background
    rms = (16-10)*(gal_lat/90 -1)**2 + 10 # e-1 / pix in 2sec integration
    return rms

def Pixinaperture(Tmag):
    # Approximate relation for pixels in aperture (based on plot in Sullivan et al.)
    pixels = (30 + (((3-30)/(14-7)) * (Tmag-7)))*(Tmag<14) + 3*(Tmag>=14) 
    return int(np.max([pixels, 3]))

def mean_flux_level(Tmag):#, Teff):
    # Magnitude system based on Sullivan et al.
#    collecting_area = np.pi*(10.5/2)**2 # square cm
#    Teff_list = np.array([2450, 3000, 3200, 3400, 3700, 4100, 4500, 5000, 5777, 6500, 7200, 9700]) # Based on Sullivan
#    Flux_list = np.array([2.38, 1.43, 1.40, 1.38, 1.39, 1.41, 1.43, 1.45, 1.45, 1.48, 1.48, 1.56])*1e6 # photons per sec; Based on Sullivan
#    Magn_list = np.array([306, -191, -202, -201, -174, -132, -101, -80, -69.5, -40, -34.1, 35])*1e-3 #Ic-Tmag (mmag)
#
#
#    Flux_int = INT.UnivariateSpline(Teff_list, Flux_list, k=1, s=0)
#    Magn_int = INT.UnivariateSpline(Teff_list, Magn_list, k=1, s=0)
#
#    Imag = Magn_int(Teff)+Tmag
#    Flux = 10**(-0.4*Imag) * Flux_int(Teff) * collecting_area
	
	
    Flux = 10**(-0.4*(Tmag - 20.54))
	

    return Flux


def phot_noise(Tmag, Teff, cad, PARAM, verbose=False, sysnoise=60):

	# Calculate galactic latitude for Zodiacal noise
	gc= SkyCoord(PARAM['RA']*u.degree, PARAM['DEC']*u.degree, frame='icrs')
#    gc = SkyCoord(lon=PARAM['ELON']*u.degree, lat=PARAM['ELAT']*u.degree, frame='barycentrictrueecliptic')
	gc_gal = gc.transform_to('galactic')
	gal_lat0 = gc_gal.b.deg

	gal_lat = np.arcsin(np.abs(np.sin(gal_lat0*np.pi/180)))*180/np.pi

	# Number of 2 sec integrations in cadence
	integrations = cad/2

	# Number of pixels in aperture given Tmag
	pixels = int(Pixinaperture(Tmag))

	# noise values are in rms, so square-root should be used when factoring up
	Flux_factor = np.sqrt(integrations * pixels)

	# Mean flux level in electrons per cadence
	mean_level_ppm = mean_flux_level(Tmag) * cad # electrons (based on measurement) #, Teff

	# Shot noise
	shot_noise = 1e6/np.sqrt(mean_level_ppm)

	# Read noise
	read_noise = 10 * Flux_factor *1e6/mean_level_ppm # ppm

	# Zodiacal noise
	zodiacal_noise = ZLnoise(gal_lat) * Flux_factor *1e6/mean_level_ppm # ppm

	# Systematic noise in ppm
	systematic_noise_ppm = sysnoise / np.sqrt(cad/(60*60)) # ppm / sqrt(hr)


	if verbose:
		print('Galactic latitude', gal_lat)
		print('Systematic noise in ppm', systematic_noise_ppm)
		print('Integrations', integrations)
		print('Pixels', pixels)
		print('Flux factor', Flux_factor)
		print('Mean level ppm', mean_level_ppm)
		print('Shot noise', shot_noise)
		print('Read noise', read_noise)
		print('Zodiacal noise', zodiacal_noise)


	PARAM['Galactic_lat'] = gal_lat
	PARAM['Pixels_in_aper'] = pixels

	noise_vals = np.array([shot_noise, zodiacal_noise, read_noise, systematic_noise_ppm])
	return noise_vals, PARAM # ppm per cadence

def reduce_percentile(x):
	return np.percentile(x[np.isfinite(x)], 95)

# =============================================================================
#
# =============================================================================

def compute_onehour_rms(flux, cad):

	if cad==120:
		N=30
	elif cad==1800:
		N=2
	else:
		N=1

	bins = int(np.ceil(len(flux)/N)) + 1

	idx_finite = np.isfinite(flux)


	flux_finite = flux[idx_finite]
	bin_means = np.array([])
	ii = 0;

	for ii in range(bins):
		try:
			m = np.nanmean(flux_finite[ii*N:(ii+1)*N])
			bin_means = np.append(bin_means, m)
		except:
			continue


	# Compute robust RMS value (MAD scaled to RMS)
	RMS = 1.4826*np.nanmedian(np.abs((bin_means - np.nanmedian(bin_means))))
	PTP = np.nanmedian(np.abs(np.diff(flux_finite)))

	return RMS, PTP



def search_database(cursor, select=None, search=None, order_by=None, limit=None, distinct=False):
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

	query = "SELECT {distinct:s}{select:s} FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority {search:s}{order_by:s}{limit:s};".format(
		distinct='DISTINCT ' if distinct else '',
		select=select,
		search=search,
		order_by=order_by,
		limit=limit
	)
	logger.debug("Running query: %s", query)

	# Ask the database: status=1 
	cursor.execute(query)
	return [dict(row) for row in cursor.fetchall()]


def set_val_flag(D, valtype=None):
	
	dv = np.zeros_like(D)
	
	magvsflux_high = 1
	magvsflux_low = 2
	lc_over_sc = 4
	sc_over_lc = 8
	minmask = 16
	smallmask = 32
	largemask = 64
	smallstamp = 128
	lowptp = 256
	lowrms = 512
	contamone = 1024
	contamhigh = 2048
	negflux = 4096
	
	
	if valtype=='hf': #high flux
		dv[D] = magvsflux_high
	elif valtype=='lf':
		dv[D] = magvsflux_low
	elif valtype=='lc_over_sc':
		dv[D] = lc_over_sc
	elif valtype=='sc_over_lc':
		dv[D] = sc_over_lc
	elif valtype=='min_mask':	
		dv[D] = minmask
	elif valtype=='small_mask':
		dv[D] = smallmask
	elif valtype=='large_mask':
		dv[D] = largemask	
	elif valtype=='small_stamp':
		dv[D] = smallstamp
	elif valtype=='low_ptp':	
		dv[D] = lowptp
	elif valtype=='low_rms':	
		dv[D] = lowrms	
	elif valtype=='contam_one':	
		dv[D] = contamone
	elif valtype=='contam_high':	
		dv[D] = contamhigh	
	elif valtype=='negflux':
		dv[D] = negflux
	
	# Pretty string descriptions for each flag
	STRINGS = {
		1: "Star has higher flux than given by magnitude relation",
		2: "Star has lower flux than given by magnitude relation",
		4: "Star has higher measured flux in 30-min than 2-min",
		5: "Star has higher measured flux in 2-min than 30-min",
		16: "Star has minimum 4x4 mask",
		32: "Star has smaller mask than general relation",
		64: "Star has larger mask than general relation",
		128: "Smaller stamp than default",
		256: "PTP lower than theoretical",
		512: "RMS lower than theoretical",
		1024: "Contamination over 1",
		2048: "Contamination high",
		4096: "Negative mean flux"
	}
	
	return dv

# =============================================================================
#
# =============================================================================
	
class DataValidation(object):
	
	
	def __init__(self, args):
		
#		if not np.asarray(args.input_folders) is args.input_folders:
#			self.input_folders = np.array([args.input_folders,])
#		else:
		self.input_folders = args.input_folders.split(';')
		
		print(self.input_folders)
		
		self.method = args.method
		self.extension = args.ext
		self.show = args.show

		self.cursors = np.array([])
		self.conns = np.array([])
		self.outfolders = args.output_folder
		self.sysnoise = args.sysnoise

		
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
			conn = sqlite3.connect(todo_file)
			conn.row_factory = sqlite3.Row
			
			cursor = conn.cursor()
			
			if self.method == 'all' and self.doval == True:
				# Create table for diagnostics:
				cursor.execute("""CREATE TABLE IF NOT EXISTS dataval (
					priority INT PRIMARY KEY NOT NULL,
					starid BIGINT NOT NULL,
					source TEXT,
					dataval INT,
					errors TEXT,
					FOREIGN KEY (priority) REFERENCES todolist(priority) ON DELETE CASCADE ON UPDATE CASCADE
				);""")
		
				conn.commit()
			
			self.cursors = np.append(self.cursors, cursor)
			self.conns = np.append(self.conns, conn)

		print(self.cursors)
		
		# Run validation
		self.Validations()	
		
		
		
		
		
		
	def Validations(self):
		
		if self.method == 'all':
			val1 = self.plot_magtoflux(return_val=True)
			val2 = self.plot_pixinaperture(return_val=True)
			val3 = self.plot_stamp(return_val=True)
			val4 = self.plot_mag_dist(return_val=True)
			val5 = self.plot_onehour_noise(return_val=True)
			val6 = self.plot_contam(return_val=True)	
			
			dv = val1['dv']+val2['dv']+val3['dv']+val4['dv']+val5['dv']+val6['dv']
			
			for j, cursor in enumerate(self.cursors):
				cursor.execute("INSERT INTO diagnostics (priority, starid, dv) VALUES (?,?,?);", (
				val1['priority'],
				val1['starid'],
				dv))
				
#				details.get('filepath_lightcurve', None),
#				result['time'],
#				details.get('pos_centroid', (None, None))[0],
#				details.get('pos_centroid', (None, None))[1],
#				details.get('mean_flux', None),
#				details.get('variance', None),
#				details.get('variability', None),
#				details.get('rms_hour', None),
#				details.get('ptp', None),
#				details.get('mask_size', None),
#				details.get('contamination', None),
#				stamp_width,
#				stamp_height,
#				details.get('stamp_resizes', 0),
#				error_msg
#			))
				self.conns[j].commit()
#
#		# Write summary file:
#		if self.summary_file and self.summary['tasks_run'] % self.summary_interval == 0:
#			self.write_summary()
#
#	def start_task(self, taskid):
#		"""
#		Mark a task as STARTED in the TODO-list.
#		"""
#		self.cursor.execute("UPDATE todolist SET status=? WHERE priority=?;", (STATUS.STARTED.value, taskid))
#		self.conn.commit()
#		self.summary['STARTED'] += 1
			
			
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
			val = self.plot_contam()	
			print(val['dv'])
		


	# =============================================================================
	# 
	# =============================================================================
	
	def plot_contam(self, return_vals=False):
	
		
		"""
		Function to plot the contamination against the stellar TESS magnitudes

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""
	
		logger=logging.getLogger(__name__)
		
		logger.info('------------------------------------------')
		logger.info('Plotting Contamination vs. Magnitude')
		
		norm = colors.Normalize(vmin=1, vmax=len(self.cursors)+6)
		scalarMap = cmx.ScalarMappable(norm=norm, cmap=plt.get_cmap('Set1') )
	
	
		fig = plt.figure(figsize=(15, 5))
		fig.subplots_adjust(left=0.145, wspace=0.3, top=0.945, bottom=0.145, right=0.975)
		ax = fig.add_subplot(111)
		
		
		for i, c in enumerate(self.cursors):
			star_vals = search_database(c, search=['status in (1,3)'], select=['todolist.starid','method','todolist.datasource','todolist.tmag','contamination'])
			
			rgba_color = scalarMap.to_rgba(i+1)
			
			tmags = np.array([star['tmag'] for star in star_vals], dtype=float)
			cont = np.array([star['contamination'] for star in star_vals], dtype=float)
			tics = np.array([star['starid'] for star in star_vals])
			met = np.array([star['method'] for star in star_vals])
			source = np.array([star['datasource'] for star in star_vals], dtype=str)

			
			
#			idx_finite = np.isfinite(cont)
			idx_low = (cont<=1) & np.isfinite(cont)
			idx_high_ffi = (cont>1) & np.isfinite(cont) & (source=='ffi')
			idx_high_tpf = (cont>1) & np.isfinite(cont) & (source=='tpf')
			idx_low_ffi = (cont<=1) & np.isfinite(cont) & (source=='ffi')
			idx_low_tpf = (cont<=1) & np.isfinite(cont) & (source=='tpf')
			cont[idx_high_ffi] = 1.1
			cont[idx_high_tpf] = 1.1
			
			ax.scatter(tmags[idx_low_ffi], cont[idx_low_ffi], marker='o', facecolors=rgba_color, color=rgba_color, alpha=0.1)
			ax.scatter(tmags[idx_low_tpf], cont[idx_low_tpf], marker='o', facecolors='g', color='g', alpha=0.1)
			ax.scatter(tmags[idx_high_ffi], cont[idx_high_ffi], marker='o', facecolors='None', color=rgba_color, alpha=0.9, label='30-min')
			ax.scatter(tmags[idx_high_tpf], cont[idx_high_tpf], marker='o', facecolors='None', color='g', alpha=0.9, label='2-min')

			bin_means, bin_edges, binnumber = binning(tmags[idx_low], cont[idx_low], statistic='median', bins=15, range=(np.nanmin(tmags),np.nanmax(tmags)))
			bin_width = (bin_edges[1] - bin_edges[0])
			bin_centers = bin_edges[1:] - bin_width/2
			
			
			ax.scatter(bin_centers, 1.4826*bin_means, marker='o', color='k')
			ax.scatter(bin_centers, 1.4826*5*bin_means, marker='.', color='k')
			
			cont_vs_mag = INT.InterpolatedUnivariateSpline(bin_centers, 1.4826*5*bin_means)
			
			mags = np.linspace(np.nanmin(tmags),np.nanmax(tmags),100)
			ax.plot(mags, cont_vs_mag(mags))
			
			if return_vals:
				val = {}
				dv = set_val_flag((cont>1), valtype='contam_one')
				val['dv'] = dv
	
		
		print(sum(np.isfinite(cont)==False))
		print(tics[np.isfinite(cont)==False])
		print(met[np.isfinite(cont)==False])
		ax.set_xlim([np.min(tmags)-0.5, np.max(tmags)+0.5])
		ax.set_ylim([-0.05, 1.15])

		ax.axhline(y=0, ls='--', color='k', zorder=-1)
		ax.axhline(y=1.1, ls=':', color='k', zorder=-1)
		ax.axhline(y=1, ls=':', color='k', zorder=-1)
		ax.set_xlabel('TESS magnitude', fontsize=16, labelpad=10)
		ax.set_ylabel('Contamination', fontsize=16, labelpad=10)

		ax.xaxis.set_major_locator(MultipleLocator(2))
		ax.xaxis.set_minor_locator(MultipleLocator(1))
		ax.yaxis.set_major_locator(MultipleLocator(0.2))
		ax.yaxis.set_minor_locator(MultipleLocator(0.1))
		ax.tick_params(direction='out', which='both', pad=5, length=3)
		ax.tick_params(which='major', pad=6, length=5,labelsize='15')
		ax.yaxis.set_ticks_position('both')
		ax.legend(loc='upper left', prop={'size': 12})
		
		###########

		if len(self.cursors)>1:
			filename = 'contam_joint.%s' %self.extension
		else:
			filename = 'contam.%s' %self.extension
			
			
		fig.savefig(os.path.join(self.outfolders, filename))
			
		if self.show:
			plt.show()
		else:
			plt.close('all')
			
		if return_vals:
			return vals
		
	
	# =============================================================================
	# 	
	# =============================================================================
	
	def plot_onehour_noise(self):
		#, data_paths, sector, cad=1800, version=1, savetex=False, labels=None):
		
		"""
		Function to plot the light curve noise against the stellar TESS magnitudes

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""
	
		logger=logging.getLogger(__name__)
		
		logger.info('------------------------------------------')
		logger.info('Plotting Noise vs. Magnitude')
		
		norm = colors.Normalize(vmin=1, vmax=len(self.cursors)+6)
		scalarMap = cmx.ScalarMappable(norm=norm, cmap=plt.get_cmap('Set1') )
	
	
		fig1 = plt.figure(figsize=(15, 5))
		fig1.subplots_adjust(left=0.145, wspace=0.3, top=0.945, bottom=0.145, right=0.975)
		ax11 = fig1.add_subplot(121)
		ax12 = fig1.add_subplot(122)
	
		fig2 = plt.figure(figsize=(15, 5))
		fig2.subplots_adjust(left=0.145, wspace=0.3, top=0.945, bottom=0.145, right=0.975)
		ax21 = fig2.add_subplot(121)
		ax22 = fig2.add_subplot(122)
	
		PARAM = {}
		
		for i, c in enumerate(self.cursors):
			star_vals = search_database(c, search=['status in (1,3)'], select=['todolist.datasource', 'todolist.tmag','rms_hour', 'ptp', 'ccd'])
			
#			sector = get_sector(c)
#			logger.info('Including data from Sector %i' %int(sector[0]['sector']))
			
			rgba_color = scalarMap.to_rgba(i+1)
			
			tmags = np.array([star['tmag'] for star in star_vals], dtype=float)
			rms = np.array([star['rms_hour']*1e6 for star in star_vals], dtype=float)
			ptp = np.array([star['ptp']*1e6 for star in star_vals], dtype=float)
			source = np.array([star['datasource'] for star in star_vals], dtype=str)


			# TODO: Update elat+elon based on observing sector?
			PARAM['RA'] = 0#hdu[0].header['RA_OBJ']
			PARAM['DEC'] = 0#hdu[0].header['DEC_OBJ']
				
				
			idx_lc = (source=='ffi') & (rms!=0)
			idx_sc = (source=='tpf') & (rms!=0)

			ax11.scatter(tmags[idx_lc], rms[idx_lc], marker='o', facecolors=rgba_color, edgecolor=rgba_color, alpha=0.1, label='30-min cadence')
			ax12.scatter(tmags[idx_sc], rms[idx_sc], marker='o', facecolors=rgba_color, edgecolor=rgba_color, alpha=0.1, label='2-min cadence')

			ax21.scatter(tmags[idx_lc], ptp[idx_lc], marker='o', facecolors=rgba_color, edgecolor=rgba_color, alpha=0.1, label='30-min cadence')
			ax22.scatter(tmags[idx_sc], ptp[idx_sc], marker='o', facecolors=rgba_color, edgecolor=rgba_color, alpha=0.1, label='2-min cadence')
	
	
		# Plot theoretical lines
		mags = np.linspace(np.min(tmags)-0.5, np.max(tmags)+0.5, 50)
		vals = np.zeros([len(mags), 4])
		vals2 = np.zeros([len(mags), 4])
		
		# Expected *1-hour* RMS noise
		for i in range(len(mags)):
			vals[i,:], _ = phot_noise(mags[i], 5775, 3600, PARAM, sysnoise=self.sysnoise, verbose=False)
	
		ax11.semilogy(mags, vals[:, 0], 'r-')
		ax11.semilogy(mags, vals[:, 1], 'g--')
		ax11.semilogy(mags, vals[:, 2], '-')
		ax11.semilogy(mags, np.sqrt(np.sum(vals**2, axis=1)), 'k-')
		ax11.axhline(y=self.sysnoise, color='b', ls='--')
		
		ax12.semilogy(mags, vals[:, 0], 'r-')
		ax12.semilogy(mags, vals[:, 1], 'g--')
		ax12.semilogy(mags, vals[:, 2], '-')
		ax12.semilogy(mags, np.sqrt(np.sum(vals**2, axis=1)), 'k-')
		ax12.axhline(y=self.sysnoise, color='b', ls='--')
	
		# Expected ptp for 30-min
		for i in range(len(mags)):
			vals[i,:], _ = phot_noise(mags[i], 5775, 1800, PARAM, sysnoise=self.sysnoise, verbose=False)
		
		ax21.semilogy(mags, vals[:, 0], 'r-')
		ax21.semilogy(mags, vals[:, 1], 'g--')
		ax21.semilogy(mags, vals[:, 2], '-')
		ax21.semilogy(mags, np.sqrt(np.sum(vals**2, axis=1)), 'k-')
		ax21.axhline(y=self.sysnoise, color='b', ls='--')
	
		# Expected ptp for 2-min
		for i in range(len(mags)):
			vals2[i,:], _ = phot_noise(mags[i], 5775, 120, PARAM, sysnoise=self.sysnoise, verbose=False)
	
		ax22.semilogy(mags, vals2[:, 0], 'r-')
		ax22.semilogy(mags, vals2[:, 1], 'g--')
		ax22.semilogy(mags, vals2[:, 2], '-')
		ax22.semilogy(mags, np.sqrt(np.sum(vals2**2, axis=1)), 'k-')
		ax22.axhline(y=self.sysnoise, color='b', ls='--')

	
#		noi_vs_mag = INT.UnivariateSpline(mags, tot_noise)
#		idx = (rms_tmag_vals[:, 1]/noi_vs_mag(rms_tmag_vals[:, 0]) < 1)
#		print([int(x) for x in rms_tmag_vals[idx, -1]])
#		print([x for x in rms_tmag_vals[idx, 0]])	
	
		
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

		if len(self.cursors)>1:
			filename = 'rms_joint.%s' %self.extension
			filename2 = 'ptp_joint.%s' %self.extension
		else:
			filename = 'rms.%s' %self.extension
			filename2 = 'ptp.%s' %self.extension
			
			
		fig1.savefig(os.path.join(self.outfolders, filename))
		fig2.savefig(os.path.join(self.outfolders, filename2))
			
		if self.show:
			plt.show()
		else:
			plt.close('all')
	
	
	# =============================================================================
	#
	# =============================================================================
	
	def plot_pixinaperture(self):
		
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
		fig.subplots_adjust(left=0.14, wspace=0.3, top=0.94, bottom=0.155, right=0.96)
		norm = colors.Normalize(vmin=0, vmax=len(self.cursors)+5)
		scalarMap = cmx.ScalarMappable(norm=norm, cmap=plt.get_cmap('Set1') )
	
		vx = np.array([])
		vy = np.array([])
	
		for i, c in enumerate(self.cursors):
			star_vals = search_database(c, search=['status in (1,3)'], select=['todolist.datasource', 'todolist.tmag','ccd','diagnostics.mask_size'])
			
#			sector = get_sector(c)
#			logger.info('Including data from Sector %i' %int(sector[0]['sector']))
#			lab = 'Sector %i' %int(sector[0]['sector'])

			rgba_color = scalarMap.to_rgba(i)
			
			tmags = np.array([star['tmag'] for star in star_vals], dtype=float)
			masksizes = np.array([star['mask_size'] for star in star_vals], dtype=float)
			source = np.array([star['datasource'] for star in star_vals], dtype=str)

			vx = np.append(vx, tmags)
			vy = np.append(vy, masksizes)

			idx_lc = (source=='ffi')
			idx_sc = (source=='tpf')
			ax1.scatter(tmags[idx_lc][::10], masksizes[idx_lc][::10], marker='o', color=rgba_color, alpha=0.1, label='30-min cadence') #facecolors='None', 
			ax2.scatter(tmags[idx_sc][::10], masksizes[idx_sc][::10], marker='o', color=rgba_color, alpha=0.1, label='2-min cadence') #facecolors='None', 
	
		print(masksizes[idx_lc])
		mags = np.linspace(np.nanmin(vx)-1, np.nanmax(vx)+1, 500)
		pix = np.asarray([Pixinaperture(m) for m in mags], dtype='float64')
		ax1.plot(mags, pix, color='k', ls='-')
		ax2.plot(mags, pix, color='k', ls='-')
	
		ax1.set_xlim([np.nanmin(vx)-1, np.nanmax(vx)+1])
		ax1.set_ylim([3, np.nanmax(vy)+100])
	
		
	
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
			axx.legend(loc='upper right', prop={'size': 12})
	
		if len(self.cursors)>1:
			filename = 'pix_in_aper_joint.%s' %self.extension
		else:
			filename = 'pix_in_aper.%s' %self.extension
			
			
		fig.savefig(os.path.join(self.outfolders, filename))
			
		if self.show:
			plt.show()
		else:
			plt.close(fig)
	
	
	# =============================================================================
	#
	# =============================================================================
	
	def plot_magtoflux(self):
		"""
		Function to plot flux values from apertures against the stellar TESS magnitudes, 
		and determine coefficient describing the relation

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""
	
		logger=logging.getLogger(__name__)
		
		logger.info('--------------------------------------')
		logger.info('Plotting Magnitude to Flux conversion')
		
		fig = plt.figure(figsize=(15, 5))
		ax1 = fig.add_subplot(121)
		ax2 = fig.add_subplot(122)
	
		norm = colors.Normalize(vmin=0, vmax=len(self.cursors)+5)
		scalarMap = cmx.ScalarMappable(norm=norm, cmap=plt.get_cmap('Set1') )
		fig.subplots_adjust(left=0.14, wspace=0.3, top=0.94, bottom=0.155, right=0.96)
		
		vx = np.array([], dtype=float)
		vy = np.array([], dtype=float)
	
		for i, c in enumerate(self.cursors):
			star_vals = search_database(c, search=["status in (1,3)"], select=['todolist.datasource','todolist.starid','todolist.tmag','ccd','mean_flux'])
#			sector = get_sector(c)
#			logger.info('Including data from Sector %i' %int(sector[0]['sector']))
#			lab = 'Sector %i' %int(sector[0]['sector'])			
			
			rgba_color = scalarMap.to_rgba(i)
			
			tmags = np.array([star['tmag'] for star in star_vals], dtype=float)
			meanfluxes = np.array([star['mean_flux'] for star in star_vals], dtype=float)
			source = np.array([star['datasource'] for star in star_vals], dtype=str)
			tics = np.array([str(star['starid']) for star in star_vals], dtype=str)	

			vx = np.append(vx, tmags)
			vy = np.append(vy, meanfluxes)
				
			idx_lc = (source=='ffi')
			idx_sc = (source=='tpf')
			ax1.scatter(tmags[idx_lc], meanfluxes[idx_lc], marker='o', facecolors='None', color=rgba_color, alpha=0.1, label='30-min cadence')
			ax2.scatter(tmags[idx_sc], meanfluxes[idx_sc], marker='o', facecolors='None', color=rgba_color, alpha=0.1, label='2-min cadence')
	
		
		tics_sc = tics[idx_sc]
		tics_lc = tics[idx_lc]
		tt=np.array([t for t in tics_sc if t in tics_lc])
		
		vy_lc = np.array([vy[(tics==t) & (source=='ffi')][0] for t in tt])
		vy_sc = np.array([vy[(tics==t) & (source=='tpf')][0] for t in tt])
		vx_lc = np.array([vx[(tics==t) & (source=='ffi')][0] for t in tt])


		plt.figure()
		
		bin_means, bin_edges, binnumber = binning(vx_lc, np.abs(vy_lc/vy_sc - 1), statistic='median', bins=15, range=(1.5,10))
		bin_width = (bin_edges[1] - bin_edges[0])
		bin_centers = bin_edges[1:] - bin_width/2
		
		plt.scatter(vx_lc, np.abs(vy_lc/vy_sc - 1), alpha=0.1)
		plt.scatter(bin_centers, 1.4826*bin_means, marker='o', color='r')
		plt.scatter(bin_centers, 1.4826*3*bin_means, marker='.', color='r')
	
		idx0 = np.isfinite(vy) & np.isfinite(vx)
		idx1 = np.isfinite(vy) & np.isfinite(vx) & (source=='ffi')
		idx2 = np.isfinite(vy) & np.isfinite(vx) & (source=='tpf')

		logger.info('Optimising coefficient of relation')
		z = lambda c: np.log10(np.sum((vy[idx1] -  10**(-0.4*(vx[idx1] - c)))**2))
		z2 = lambda c: np.log10(np.sum((vy[idx2] -  10**(-0.4*(vx[idx2] - c)))**2))
		cc = OP.minimize(z, 20.5, method='Nelder-Mead', options={'disp':False})
		cc2 = OP.minimize(z2, 20.5, method='Nelder-Mead', options={'disp':False})
		
		logger.info('Optimisation terminated successfully? %s' %cc.success)
		logger.info('Coefficient is found to be %1.4f' %cc.x)
		
		C=np.linspace(19, 22, 100)
		fig2 = plt.figure()
		ax21=fig2.add_subplot(111)
		[ax21.scatter(c, z(c), marker='o', color='k') for c in C] 
		[ax21.scatter(c, z2(c), marker='o', color='b') for c in C] 
		ax21.axvline(x=cc.x, color='k', label='30-min')
		ax21.axvline(x=cc2.x, color='b', ls='--', label='2-min')
		ax21.set_xlabel('Coefficient')
		ax21.set_ylabel(r'$\chi^2$')
		ax21.legend(loc='upper left', prop={'size': 12})
		
		
		plt.figure()
		d0 = vy[idx0]/(10**(-0.4*(vx[idx0] - cc.x))) - 1
		d1 = vy[idx1]/(10**(-0.4*(vx[idx1] - cc.x))) - 1
		d2 = vy[idx2]/(10**(-0.4*(vx[idx2] - cc.x))) - 1
		plt.scatter(vx[idx1], np.abs(d1), alpha=0.1)
		plt.scatter(vx[idx2], np.abs(d2), color='k', alpha=0.1)
		plt.axhline(y=0, ls='--', color='k')
		
		bin_means1, bin_edges1, binnumber1 = binning(vx[idx1], np.abs(d1), statistic='median', bins=15, range=(1.5,10))
		bin_width1 = (bin_edges1[1] - bin_edges1[0])
		bin_centers1 = bin_edges1[1:] - bin_width1/2
		
		bin_means2, bin_edges2, binnumber2 = binning(vx[idx2], np.abs(d2), statistic='median', bins=15, range=(1.5,10))
		bin_width2 = (bin_edges2[1] - bin_edges2[0])
		bin_centers2 = bin_edges2[1:] - bin_width2/2
		
		plt.scatter(bin_centers1, 1.4826*bin_means1, marker='o', color='r')
		plt.scatter(bin_centers1, 1.4826*3*bin_means1, marker='.', color='r')
		plt.plot(bin_centers1, 1.4826*3*bin_means1, color='r')
		
		plt.scatter(bin_centers2, 1.4826*bin_means2, marker='o', color='g')
		plt.scatter(bin_centers2, 1.4826*3*bin_means2, marker='.', color='g')
		plt.plot(bin_centers2, 1.4826*3*bin_means2, color='g')
#		plt.yscale("log", nonposy='clip')
		
		
		
		mag = np.linspace(np.nanmin(vx)-1, np.nanmax(vx)+1,100)
		for axx in np.array([ax1, ax2]):
			axx.plot(mag, 10**(-0.4*(mag - cc.x)), color='k', ls='--')
			axx.set_yscale("log", nonposy='clip')
		
			axx.set_xlim([np.nanmin(vx)-1, np.nanmax(vx)+1])
			axx.set_xlim(axx.get_xlim()[::-1])
		
			
			axx.set_xlabel('TESS magnitude', fontsize=16, labelpad=10)
			
		
			axx.xaxis.set_major_locator(MultipleLocator(2))
			axx.xaxis.set_minor_locator(MultipleLocator(1))
			axx.tick_params(direction='out', which='both', pad=5, length=3)
			axx.tick_params(which='major', pad=6, length=5,labelsize='15')
			axx.yaxis.set_ticks_position('both')
			axx.xaxis.set_ticks_position('both')
			axx.legend(loc='upper left', prop={'size': 12})
		
		ax1.text(10, 1e7, r'$\rm Flux = 10^{-0.4\,(T_{mag} - %1.2f)}$' %cc.x, fontsize=14)
		ax1.set_ylabel('Mean flux', fontsize=16, labelpad=10)
		
		if len(self.cursors)>1:
			filename = 'mag_to_flux_joint.%s' %self.extension
			filename2 = 'mag_to_flux_optimize_joint.%s' %self.extension
		else:
			filename = 'mag_to_flux.%s' %self.extension
			filename2 = 'mag_to_flux_optimize.%s' %self.extension
			
		fig.savefig(os.path.join(self.outfolders, filename))
		fig2.savefig(os.path.join(self.outfolders, filename2))
			
		if self.show:
			plt.show(block=True)
		else:	
			plt.close('all')
		
		
	# =========================================================================
	# 		
	# =========================================================================
	
	def plot_stamp(self):
		
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
		
	
		norm = colors.Normalize(vmin=0, vmax=len(self.cursors)+5)
		scalarMap = cmx.ScalarMappable(norm=norm, cmap=plt.get_cmap('Set1') )
		
		
		for i, c in enumerate(self.cursors):
			star_vals = search_database(c, search=["status in (1,3)"], select=['todolist.datasource','todolist.tmag','stamp_resizes','stamp_width','stamp_height','elaptime'])
			
#			sector = get_sector(c)
#			logger.info('Including data from Sector %i' %int(sector[0]['sector']))
#			tics = np.array([star['starid'] for star in star_vals])			
			
			rgba_color = scalarMap.to_rgba(i)
			
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
			
			ax12.scatter(bin_centers2, bin_means2, marker='o', color='b', zorder=3)
			ax11.scatter(bin_centers, bin_means, marker='o', color='b', zorder=3)
				
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
		
					
		ax12.plot(mags2,nwid2, 'b--')	
		ax11.plot(mags2,nhei2, 'b--')	
	
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

	
		if len(self.cursors)>1:
			filename = 'stamp_size_joint.%s' %self.extension
			filename2 = 'calc_time_joint.%s' %self.extension
		else:
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
	
		norm = colors.Normalize(vmin=0, vmax=len(self.cursors)*2+5)
		scalarMap = cmx.ScalarMappable(norm=norm, cmap=plt.get_cmap('Set1') )
		fig.subplots_adjust(left=0.14, wspace=0.3, top=0.94, bottom=0.155, right=0.96)
		
		tmag_all = np.array([])
	
		for i, c in enumerate(self.cursors):
			star_vals = search_database(c, search=["status in (1,3)"], select=['todolist.datasource','todolist.tmag'])
			
#			sector = get_sector(c)
#			logger.info('Including data from Sector %i' %int(sector[0]['sector']))
#			lab = 'Sector %i' %int(sector[0]['sector'])			
			
			rgba_color1 = scalarMap.to_rgba(i*2)
			rgba_color2 = scalarMap.to_rgba(i*2+1)
			
			tmags = np.array([star_vals[j]['tmag'] for j in range(len(star_vals))])
			source = np.array([star_vals[j]['datasource'] for j in range(len(star_vals))])
		
			idx_lc = (source=='ffi')
			idx_sc = (source=='tpf')
			
			if sum(idx_lc) > 0:
				kde_lc = KDE(tmags[idx_lc])
				kde_lc.fit(gridsize=1000)
				ax.fill_between(kde_lc.support, 0, kde_lc.density*sum(idx_lc), color=rgba_color1, alpha=0.3, label='30-min cadence')
#				ax.scatter(tmags[idx_lc], -300*np.ones_like(tmags[idx_lc]), lw=1, marker='|', c=rgba_color1, s=30, alpha=0.1)
	
			if sum(idx_sc) > 0:
				kde_sc = KDE(tmags[idx_sc])
				kde_sc.fit(gridsize=1000)
				ax.fill_between(kde_sc.support, 0, kde_sc.density*sum(idx_sc), color=rgba_color2, alpha=0.3, label='2-min cadence')
#				ax.scatter(tmags[idx_sc], -300*np.ones_like(tmags[idx_sc]), lw=1, marker='|', c=rgba_color2, s=30, alpha=0.1)
		
			tmag_all = np.append(tmag_all, tmags)
			kde_all = KDE(tmag_all)
			kde_all.fit(gridsize=1000)
			ax.plot(kde_all.support, kde_all.density*len(tmag_all), 'k-', lw=1.5, label='All')
			
			
			
		ax.set_ylim(ymin=0)
		ax.set_xlabel('TESS magnitude', fontsize=16, labelpad=10)
		ax.set_ylabel('Number of stars', fontsize=16, labelpad=10)
		ax.xaxis.set_major_locator(MultipleLocator(2))
		ax.xaxis.set_minor_locator(MultipleLocator(1))
		ax.tick_params(direction='out', which='both', pad=5, length=3)
		ax.tick_params(which='major', pad=6, length=5,labelsize='15')
		ax.yaxis.set_ticks_position('both')
		ax.xaxis.set_ticks_position('both')
		ax.legend(frameon=False, prop={'size':12} ,loc='upper left', borderaxespad=0,handlelength=2.5, handletextpad=0.4)
	

		if len(self.cursors)>1:
			filename = 'mag_dist_joint.%s' %self.extension
		else:
			filename = 'mag_dist.%s' %self.extension
			
		fig.savefig(os.path.join(self.outfolders, filename))
			
		if self.show:
			plt.show()
		else:
			plt.close('all')
		


# =============================================================================
#
# =============================================================================

if __name__ == '__main__':

	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Run TESS Corrector pipeline on single star.')
	parser.add_argument('-m', '--method', help='Corrector method to use.', default='all', choices=('pixvsmag', 'contam', 'mag2flux', 'stamp', 'noise', 'magdist'))
	parser.add_argument('-e', '--ext', help='Extension of plots.', default='png', choices=('png', 'eps'))
	parser.add_argument('-s', '--show', help='Show plots.', default=False, choices=('True', 'False'))
	parser.add_argument('-v', '--validate', help='Compute validation (only run is method is "all").', default=True, choices=('True', 'False'))
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('input_folders', type=str, help='Directory to create catalog files in.', nargs='?', default=None)
	parser.add_argument('output_folder', type=str, help='Directory in which to place output if several input folders are given.', nargs='?', default=None)
	parser.add_argument('-sn', '--sysnoise', type=float, help='systematic noise level for noise plot.', nargs='?', default=0)
	args = parser.parse_args()


	args.show = 'True'
	args.method = 'contam'
	args.sysnoise = 5
	args.input_folders = '/media/mikkelnl/Elements/TESS/S01_tests/lightcurves-2127753/'
	
	if args.output_folder is None and len(args.input_folders.split(';'))>1:
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
	logger_parent = logging.getLogger('corrections')
	logger_parent.addHandler(console)
	logger_parent.setLevel(logging_level)


	logger.info("Loading input data from '%s'", args.input_folders)
	logger.info("Putting output data in '%s'", args.output_folder)

	# Get the class for the selected method:
	DataValidation(args)

	# Initialize the corrector class:
#	with CorrClass(input_folder, plot=args.plot) as corr:

#		# Start the TaskManager:
#		with corrections.TaskManager(input_folder) as tm:
#			while True:
#				if args.all:
#					task = tm.get_task()
#					if task is None: break
#				elif args.starid is not None:
#					task = tm.get_task(starid=args.starid)
#				elif args.random:
#					task = tm.get_random_task()
#
#				# Run the correction:
#				result = corr.correct(task)
#
#				# Construct results to return to TaskManager:
#				tm.save_results(result)
#
#				if not args.all:
#					break

#	plt.close('all')
