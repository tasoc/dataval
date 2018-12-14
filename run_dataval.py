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

def mean_flux_level(Tmag, Teff):
    # Magnitude system based on Sullivan et al.
    collecting_area = np.pi*(10.5/2)**2 # square cm
    Teff_list = np.array([2450, 3000, 3200, 3400, 3700, 4100, 4500, 5000, 5777, 6500, 7200, 9700]) # Based on Sullivan
    Flux_list = np.array([2.38, 1.43, 1.40, 1.38, 1.39, 1.41, 1.43, 1.45, 1.45, 1.48, 1.48, 1.56])*1e6 # photons per sec; Based on Sullivan
    Magn_list = np.array([306, -191, -202, -201, -174, -132, -101, -80, -69.5, -40, -34.1, 35])*1e-3 #Ic-Tmag (mmag)


    Flux_int = INT.UnivariateSpline(Teff_list, Flux_list, k=1, s=0)
    Magn_int = INT.UnivariateSpline(Teff_list, Magn_list, k=1, s=0)

    Imag = Magn_int(Teff)+Tmag
    Flux = 10**(-0.4*Imag) * Flux_int(Teff) * collecting_area

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
	mean_level_ppm = mean_flux_level(Tmag, Teff) * cad # electrons

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


def get_sector(cursor):
	logger = logging.getLogger(__name__)

	query = "SELECT sector FROM settings;"
	logger.debug("Running query: %s", query)

	# Ask the database:
	cursor.execute(query)
	return [dict(row) for row in cursor.fetchall()]


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
		search = "AND " + " AND ".join(search)
	else:
		search = 'AND ' + search

	if order_by is None:
		order_by = ''
	elif isinstance(order_by, (list, tuple)):
		order_by = " ORDER BY " + ",".join(order_by)
	elif isinstance(order_by, six.string_types):
		order_by = " ORDER BY " + order_by

	limit = '' if limit is None else " LIMIT %d" % limit

	query = "SELECT {distinct:s}{select:s} FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE status=1 {search:s}{order_by:s}{limit:s};".format(
		distinct='DISTINCT ' if distinct else '',
		select=select,
		search=search,
		order_by=order_by,
		limit=limit
	)
	logger.debug("Running query: %s", query)

	# Ask the database:
	cursor.execute(query)
	return [dict(row) for row in cursor.fetchall()]

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
		self.outfolders = []
		
		#load sqlite to-do files
		for i, f in enumerate(self.input_folders):
			path = os.path.join(f, 'data_validation')
			if not os.path.exists(path):
				os.makedirs(path)
			self.outfolders = np.append(self.outfolders, path)
			
			todo_file = os.path.join(f, 'todo.sqlite')
			logger.debug("TODO file: %s", todo_file)
			if not os.path.exists(todo_file):
				raise ValueError("TODO file not found")

			# Open the SQLite file:
			conn = sqlite3.connect(todo_file)
			conn.row_factory = sqlite3.Row
			self.cursors = np.append(self.cursors, conn.cursor())

		print(self.cursors)
		# Run validation
		self.Validations()	
		
	def Validations(self):
		
		if self.method == 'all':
			self.plot_magtoflux()
			self.plot_pixinaperture()
		elif self.method == 'mag2flux':
			self.plot_magtoflux()
		elif self.method == 'pixvsmag':
			self.plot_pixinaperture()
			
			
		

		
#	def plot_folder(self, todo):
#		"""
#		Return folder path where plots for a given lightcurve should be saved.
#
#		Parameters:
#			lc (``lightkurve.TessLightCurve``): Lightcurve to return plot path for.
#
#		Returns:
#			string: Path to directory where plots should be saved.
#
#		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
#		"""
#		lcfile = os.path.join(self.input_folder, lc.meta['task']['lightcurve'])
#		plot_folder = os.path.join(os.path.dirname(lcfile), 'plots', '%011d' % lc.targetid)
#		return plot_folder	



	def plot_bg(data_paths, sector, cad=1800, sysnoise=0, version=1, savetex=False):
		norm = colors.Normalize(vmin=1, vmax=4)
		scalarMap = cmx.ScalarMappable(norm=norm, cmap=plt.get_cmap('Set1') )
	
	
		fig = plt.figure(figsize=(15, 6))
		ax1 = fig.add_subplot(221)
		ax2 = fig.add_subplot(222)
		ax3 = fig.add_subplot(223)
		ax4 = fig.add_subplot(224)
		ax1.set_rasterization_zorder(0)
		ax2.set_rasterization_zorder(0)
		ax3.set_rasterization_zorder(0)
		ax4.set_rasterization_zorder(0)
		
		fig.subplots_adjust(left=0.08, wspace=0.3, top=0.945, bottom=0.145, right=0.98)
		
		for k, d in enumerate(data_paths):
			
			files = np.array([])
			for root, dirs, fil in os.walk(d):
				for file in fil:
					file_path = root + os.sep + file
					if ('corr' in file_path) and ('.fits' in file_path):
						files = np.append(files, file_path)
						
			for i, f in enumerate(files):
				with fits.open(f) as hdu:
					time = hdu[1].data['TIME']
					bg = hdu[1].data['FLUX_BKG']
	
					cam = hdu[0].header['CAMERA']
					
					print(cam)
					rgba_color = scalarMap.to_rgba(cam)
					if cam==1:
						ax1.scatter(time, bg, marker='o', s=1, facecolors='None', edgecolor=rgba_color, zorder=-1)
					if cam==2:
						ax2.scatter(time, bg, marker='o', s=1, facecolors='None', edgecolor=rgba_color, zorder=-1)
					if cam==3:
						ax3.scatter(time, bg, marker='o', s=1, facecolors='None', edgecolor=rgba_color, zorder=-1)
					if cam==4:
						ax4.scatter(time, bg, marker='o', s=1, facecolors='None', edgecolor=rgba_color, zorder=-1)	
	#				ax.scatter(rms_tmag_vals[idx_lc, 0], rms_tmag_vals[idx_lc, 2], marker='s', facecolors='None', edgecolor=rgba_color)
	
	#	ax4.set_xlim([np.min(times)-500, np.max(times)+500])
	#	ax4.set_ylim([-0.15, 0.15])
						
		ax1.text(0.05, 0.9, 'Camera 1', transform=ax1.transAxes, fontsize=14)				
		ax2.text(0.05, 0.9, 'Camera 2', transform=ax2.transAxes, fontsize=14)				
		ax3.text(0.05, 0.9, 'Camera 3', transform=ax3.transAxes, fontsize=14)				
		ax4.text(0.05, 0.9, 'Camera 4', transform=ax4.transAxes, fontsize=14)
					
		ax3.set_xlabel('Time TBJD (days)', fontsize=16, labelpad=10)
		ax4.set_xlabel('Time TBJD (days)', fontsize=16, labelpad=10)
		ax1.set_ylabel(r'$\rm Counts\,\, (e^{-}/s)$', fontsize=16, labelpad=10)
		ax3.set_ylabel(r'$\rm Counts\,\, (e^{-}/s)$', fontsize=16, labelpad=10)
		
	#	ax2.set_ylim([2200, 8000])
		for ax in np.array([ax1, ax2, ax3, ax4]):
			ax.xaxis.set_major_locator(MultipleLocator(5))
			ax.xaxis.set_minor_locator(MultipleLocator(2.5))
			ax.tick_params(direction='out', which='both', pad=5, length=3)
			ax.tick_params(which='major', pad=6, length=5,labelsize='15')
	
		
		if version!=1:
			save_path = 'plots/sector%02d/v%1d/' %(sector,version)
		else:
			save_path = 'plots/sector%02d/' %sector
	
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		fig.savefig(os.path.join(save_path, 'BG.pdf'), bb_inches='tight')		
		fig.savefig(os.path.join(save_path, 'BG.png'), bb_inches='tight')		
			
		if savetex:
			save_path2 = '../releasenote_tex/Release_note%1d/' %sector
			fig.savefig(os.path.join(save_path2, 'BG.pdf'), bb_inches='tight')
	
	
		plt.show()
		
	# =============================================================================
	# 
	# =============================================================================
	
#	def plot_noice_lc(data_paths, sector, cad=1800, sysnoise=0, version=1, savetex=False):
#	
#		norm = colors.Normalize(vmin=0, vmax=len(data_paths)-1)
#	
#		fig2 = plt.figure()
#		fig2.subplots_adjust(left=0.18, wspace=0.3, top=0.97, bottom=0.145, right=0.97)
#		ax2 = fig2.add_subplot(111)
#		
#		fig3 = plt.figure()
#		fig3.subplots_adjust(left=0.18, wspace=0.3, top=0.97, bottom=0.145, right=0.97)
#		ax3 = fig3.add_subplot(111)
#		
#		fig4 = plt.figure()
#		fig4.subplots_adjust(left=0.18, wspace=0.3, top=0.97, bottom=0.145, right=0.97)
#		ax4 = fig4.add_subplot(111)
#	
#	
#		times = np.array([]); c1s = np.array([]); c2s = np.array([])
#		for k, d in enumerate(data_paths):
#			
#			files = np.array([])
#			for root, dirs, fil in os.walk(d):
#				for file in fil:
#					file_path = root + os.sep + file
#					if ('corr' in file_path) and ('.fits' in file_path):
#						print(file_path)
#						files = np.append(files, file_path)
#						
#	
#			for i, f in enumerate(files):
#				with fits.open(f) as hdu:
#					
#					c1 = hdu[1].data['MOM_CENTR1']
#					c2 = hdu[1].data['MOM_CENTR2']
#					time = hdu[1].data['CADENCENO']
#					
#					if '00279741379' in f:
#						ax4.plot(time-time[0], c2-np.nanmedian(c2), ls='-',color='k', label='row')
#						ax4.plot(time-time[0], c1-np.nanmedian(c1), ls='-',color='r', label='column')
#	
#					times = np.append(times, time)
#					c1s = np.append(c1s, c1-np.nanmedian(c1))
#					c2s = np.append(c2s, c2-np.nanmedian(c2))
#				
#			
#		idx = np.argsort(times)	
#		c1s = c1s[idx]
#		c2s = c2s[idx]
#		times = times[idx]
#		times -= times[0]
#		
#		ax2.plot(times, c2s, ls='-',color='k', label='row')
#		ax2.plot(times, c1s, ls='-',color='r', label='column')
#		
#	#	ax2.scatter(times, c2s, marker='o',color='k', alpha=0.1, label='row')
#	#	ax2.scatter(times, c1s, marker='o',color='r', alpha=0.1, label='column')
#	
#		idx_zoom = (times<15870) & (times>15840)
#		ax3.scatter((times[idx_zoom]-15855), c2s[idx_zoom], marker='o', edgecolors='k', facecolors='none',label='row')
#		ax3.scatter((times[idx_zoom]-15855), c1s[idx_zoom], marker='o' ,edgecolors='r', facecolors='none',label='column')
#		ax3.set_xlabel('Cadence number - 15855', fontsize=16, labelpad=10)
#		ax3.set_ylabel(r'$\rm Relative\,\, position\,\, (pixels)$', fontsize=16, labelpad=10)
#		ax3.xaxis.set_major_locator(MultipleLocator(5))
#		ax3.xaxis.set_minor_locator(MultipleLocator(1))
#		ax3.tick_params(direction='out', which='both', pad=5, length=3)
#		ax3.tick_params(which='major', pad=6, length=5,labelsize='15')
#		ax3.yaxis.set_ticks_position('both')
#		ax3.legend(loc='upper right')
#		
#		
#		ax4.set_xlim([np.min(times)-500, np.max(times)+500])
#		ax4.set_ylim([-0.15, 0.15])
#		ax4.set_xlabel('Cadence number', fontsize=16, labelpad=10)
#		ax4.set_ylabel(r'$\rm Relative\,\, position\,\, (pixels)$', fontsize=16, labelpad=10)
#		ax4.xaxis.set_major_locator(MultipleLocator(5000))
#		ax4.xaxis.set_minor_locator(MultipleLocator(2500))
#		ax4.tick_params(direction='out', which='both', pad=5, length=3)
#		ax4.tick_params(which='major', pad=6, length=5,labelsize='15')
#		ax4.yaxis.set_ticks_position('both')
#		ax4.legend(loc='upper right')
#	
#	#	###########
#		ax2.set_xlim([np.min(times)-500, np.max(times)+500])
#		ax2.set_ylim([-0.3,0.3])
#		ax2.set_xlabel('Cadence number', fontsize=16, labelpad=10)
#		ax2.set_ylabel(r'$\rm Relative\,\, position\,\, (pixels)$', fontsize=16, labelpad=10)
#		ax2.xaxis.set_major_locator(MultipleLocator(5000))
#		ax2.xaxis.set_minor_locator(MultipleLocator(2500))
#		ax2.tick_params(direction='out', which='both', pad=5, length=3)
#		ax2.tick_params(which='major', pad=6, length=5,labelsize='15')
#		ax2.yaxis.set_ticks_position('both')
#		ax2.legend(loc='upper right')
#	
#		if version!=1:
#			save_path = 'plots/sector%02d/v%1d/' %(sector,version)
#		else:
#			save_path = 'plots/sector%02d/' %sector
#	
#		if not os.path.exists(save_path):
#			os.makedirs(save_path)
#			
#		fig2.savefig(os.path.join(save_path, 'pixelpos.pdf'), bb_inches='tight')
#		fig3.savefig(os.path.join(save_path, 'pixelpos_zoom.pdf'), bb_inches='tight')
#		fig4.savefig(os.path.join(save_path, 'pixel_specific.pdf'), bb_inches='tight')
#		fig2.savefig(os.path.join(save_path, 'pixelpos.png'), bb_inches='tight')
#		fig3.savefig(os.path.join(save_path, 'pixelpos_zoom.png'), bb_inches='tight')
#		fig4.savefig(os.path.join(save_path, 'pixel_specific.png'), bb_inches='tight')
#		
#		
#		if savetex:
#			save_path2 = '../releasenote_tex/Release_note%1d/' %sector
#			fig2.savefig(os.path.join(save_path2, 'pixelpos.pdf'), bb_inches='tight')
#			fig3.savefig(os.path.join(save_path2, 'pixelpos_zoom.pdf'), bb_inches='tight')
#			fig4.savefig(os.path.join(save_path2, 'pixel_specific.pdf'), bb_inches='tight')
#		
#	
#		plt.show()
	
	# =============================================================================
	# 
	# =============================================================================
	
	def plot_onehour_noise(self, data_paths, sector, cad=1800, sysnoise=0, version=1, savetex=False, labels=None):
	
		norm = colors.Normalize(vmin=0, vmax=len(data_paths)+5)
		scalarMap = cmx.ScalarMappable(norm=norm, cmap=plt.get_cmap('Set1') )
	
	
		fig = plt.figure()
		fig.subplots_adjust(left=0.145, wspace=0.3, top=0.945, bottom=0.145, right=0.975)
		ax = fig.add_subplot(111)
	
		fig2 = plt.figure()
		fig2.subplots_adjust(left=0.145, wspace=0.3, top=0.945, bottom=0.145, right=0.975)
		ax2 = fig2.add_subplot(111)
	
		PARAM = {}
	
		# Add data values
	#	files = np.array([])
	#	files = files.flatten()
	
	#	cols = np.array(['r', 'b', 'c', 'g', 'm'])
	
		# Add data values
		
		
		for k, d in enumerate(data_paths):
			
			files = np.array([])
			for root, dirs, fil in os.walk(d):
				for file in fil:
					file_path = root + os.sep + file
					if ('corr' in file_path) and ('.fits' in file_path):
						print(file_path)
						files = np.append(files, file_path)
						
	#		files = np.append(files, np.array([os.path.join(d, f) for f in os.listdir(d) if f.endswith('.fits')]))
	#		files = np.array([os.path.join(d, f) for f in os.listdir(d) if f.endswith('.fits.gz')])
	
			print(k, d)
	#		if k==0:
	#			tot_rms_tmag_vals = np.zeros([len(files), 6])
	
			rms_tmag_vals = np.zeros([len(files), 5])
			for i, f in enumerate(files):
				with fits.open(f) as hdu:
					tmag = hdu[0].header['TESSMAG']
					flux = hdu[1].data['FLUX_CORR']
	
					rms_tmag_vals[i, 0] = tmag
	
		#			if k==0:
		#				tot_rms_tmag_vals[i, 0] = tmag
	
					if hdu[1].header.get('NUM_FRM', 60) == 60:
						rms, ptp = compute_onehour_rms(flux, 120)
						rms_tmag_vals[i, 1] = rms
						rms_tmag_vals[i, 3] = ptp
	
		#				tot_rms_tmag_vals[i,k+1] = rms
		#				tot_rms_tmag_vals[i,k+1] = np.nanmedian(np.diff(flux))
					else:
						rms, ptp = compute_onehour_rms(flux, 1800)
						rms_tmag_vals[i, 2] = rms
						rms_tmag_vals[i, 4] = ptp
	
					# TODO: Update elat+elon based on observing sector?
					PARAM['RA'] = hdu[0].header['RA_OBJ']
					PARAM['DEC'] = hdu[0].header['DEC_OBJ']
	
	
	
			idx_sc = np.nonzero(rms_tmag_vals[:, 1])
			idx_lc = np.nonzero(rms_tmag_vals[:, 2])
	
			rgba_color = scalarMap.to_rgba(k)
	
			if not labels is None:
				lab = labels[k]
			else:
				lab=''
	
			ax.scatter(rms_tmag_vals[idx_sc, 0], rms_tmag_vals[idx_sc, 1], marker='o', facecolors='None', edgecolor=rgba_color, label=lab)
			ax.scatter(rms_tmag_vals[idx_lc, 0], rms_tmag_vals[idx_lc, 2], marker='s', facecolors='None', edgecolor=rgba_color)
	
			ax2.scatter(rms_tmag_vals[idx_sc, 0], rms_tmag_vals[idx_sc, 3], marker='o', facecolors='None', edgecolor=rgba_color, label=lab)
			ax2.scatter(rms_tmag_vals[idx_lc, 0], rms_tmag_vals[idx_lc, 4], marker='s', facecolors='None', edgecolor=rgba_color)
	
		# Plot theoretical lines
		mags = np.linspace(3.5, 16.5, 50)
		vals = np.zeros([len(mags), 4])
		vals2 = np.zeros([len(mags), 4])
	
	#	print(tot_rms_tmag_vals)
	
	#	plt.figure()
	#	plt.scatter(tot_rms_tmag_vals[:, 0], tot_rms_tmag_vals[:, 1] - tot_rms_tmag_vals[:, 3], facecolors='r', marker='+', color='r')
	#	plt.scatter(tot_rms_tmag_vals[:, 0], tot_rms_tmag_vals[:, 2] - tot_rms_tmag_vals[:, 3], facecolors='b', marker='+', color='b')
	#	plt.scatter(tot_rms_tmag_vals[:, 0], tot_rms_tmag_vals[:, 4] - tot_rms_tmag_vals[:, 3], facecolors='g', marker='+', color='g')
	#	plt.scatter(tot_rms_tmag_vals[:, 0], tot_rms_tmag_vals[:, 5] - tot_rms_tmag_vals[:, 3], facecolors='m', marker='+', color='m')
	
		for i in range(len(mags)):
			vals[i,:], _ = phot_noise(mags[i], 5775, cad, PARAM, sysnoise=sysnoise, verbose=False)
	
		ax.semilogy(mags, vals[:, 0], 'r-')
		ax.semilogy(mags, vals[:, 1], 'g--')
		ax.semilogy(mags, vals[:, 2], '-')
		ax.semilogy(mags, np.sqrt(np.sum(vals**2, axis=1)), 'k-')
		ax.axhline(y=sysnoise, color='b', ls='--')
	
		for i in range(len(mags)):
			vals2[i,:], _ = phot_noise(mags[i], 5775, 120, PARAM, sysnoise=sysnoise, verbose=False)
	
		ax2.semilogy(mags, vals2[:, 0], 'r-')
		ax2.semilogy(mags, vals2[:, 1], 'g--')
		ax2.semilogy(mags, vals2[:, 2], '-')
		ax2.semilogy(mags, np.sqrt(np.sum(vals2**2, axis=1)), 'k-')
	
		for i in range(len(mags)):
			vals[i,:], _ = phot_noise(mags[i], 5775, cad, PARAM, sysnoise=sysnoise, verbose=False)
		tot_noise = np.sqrt(np.sum(vals**2, axis=1))
	
		noi_vs_mag = INT.UnivariateSpline(mags, tot_noise)
		idx = (rms_tmag_vals[:, 1]/noi_vs_mag(rms_tmag_vals[:, 0]) < 1)
		print([int(x) for x in rms_tmag_vals[idx, -1]])
		print([x for x in rms_tmag_vals[idx, 0]])
	
		ax.semilogy(mags, vals[:, 0], 'r-')
		ax.semilogy(mags, vals[:, 1], 'g--')
		ax.semilogy(mags, vals[:, 2], '-')
		ax.semilogy(mags, tot_noise, 'k-')
		ax.axhline(y=sysnoise, color='b', ls='--')
	
		for i in range(len(mags)):
			vals2[i,:], _ = phot_noise(mags[i], 5775, 120, PARAM, sysnoise=sysnoise, verbose=False)
		tot_noise2 = np.sqrt(np.sum(vals2**2, axis=1))
	
		ax2.semilogy(mags, vals2[:, 0], 'r-')
		ax2.semilogy(mags, vals2[:, 1], 'g--')
		ax2.semilogy(mags, vals2[:, 2], '-')
		ax2.semilogy(mags, tot_noise2, 'k-')
	
		ax.set_xlim([3.5, 16.5])
		ax.set_ylim([10, 1e5])
		ax.set_xlabel('TESS magnitude', fontsize=16, labelpad=10)
		ax.set_ylabel(r'$\rm RMS\,\, (ppm\,\, hr^{-1})$', fontsize=16, labelpad=10)
		ax.xaxis.set_major_locator(MultipleLocator(2))
		ax.xaxis.set_minor_locator(MultipleLocator(1))
		ax.tick_params(direction='out', which='both', pad=5, length=3)
		ax.tick_params(which='major', pad=6, length=5,labelsize='15')
		ax.yaxis.set_ticks_position('both')
	
		###########
		ax2.set_xlim([3.5, 16.5])
		ax2.set_xlabel('TESS magnitude', fontsize=16, labelpad=10)
		ax2.set_ylabel('point-to-point MDV (ppm)', fontsize=16, labelpad=10)
		ax2.set_yscale("log", nonposy='clip')
		ax2.xaxis.set_major_locator(MultipleLocator(2))
		ax2.xaxis.set_minor_locator(MultipleLocator(1))
		ax2.tick_params(direction='out', which='both', pad=5, length=3)
		ax2.tick_params(which='major', pad=6, length=5,labelsize='15')
		ax2.yaxis.set_ticks_position('both')
	
	
		ax.legend(loc='upper left', prop={'size': 12})
		ax2.legend(loc='upper left', prop={'size': 12})
	#	plt.tight_layout()
		
		if version!=1:
			save_path = 'plots/sector%02d/v%1d/' %(sector,version)
		else:
			save_path = 'plots/sector%02d/' %sector
	
		if not os.path.exists(save_path):
			os.makedirs(save_path)
			
		fig.savefig(os.path.join(save_path, 'rms_noise.pdf'), bb_inches='tight')
		fig.savefig(os.path.join(save_path, 'rms_noise.png'), bb_inches='tight')
		fig2.savefig(os.path.join(save_path, 'mvd_noise.pdf'), bb_inches='tight')
		fig2.savefig(os.path.join(save_path, 'mvd_noise.png'), bb_inches='tight')
		
		if savetex:
			save_path2 = '../releasenote_tex/Release_note%1d/' %sector
	
			fig.savefig(os.path.join(save_path2, 'rms_noise.pdf'), bb_inches='tight')
			fig2.savefig(os.path.join(save_path2, 'mvd_noise.pdf'), bb_inches='tight')
	
		plt.show()
	
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
		
		fig = plt.figure()
		ax = fig.add_subplot(111)
		fig.subplots_adjust(left=0.14, wspace=0.3, top=0.94, bottom=0.155, right=0.96)
		norm = colors.Normalize(vmin=0, vmax=len(self.cursors)+5)
		scalarMap = cmx.ScalarMappable(norm=norm, cmap=plt.get_cmap('Set1') )
	
		vx = np.array([])
		vy = np.array([])
	
		for i, c in enumerate(self.cursors):
			star_vals = search_database(c, search=['corr_status=1'], select=['todolist.datasource', 'todolist.tmag','ccd','diagnostics.mask_size'])
			sector = get_sector(c)
			
			
			logger.info('Including data from Sector %i' %int(sector[0]['sector']))
			
			rgba_color = scalarMap.to_rgba(i)
			
			tmags = np.array([star_vals[j]['tmag'] for j in range(len(star_vals))])
			masksizes = np.array([star_vals[j]['mask_size'] for j in range(len(star_vals))])
			source = np.array([star_vals[j]['datasource'] for j in range(len(star_vals))])

			vx = np.append(vx, tmags)
			vy = np.append(vy, masksizes)

			lab = 'Sector %i' %int(sector[0]['sector'])

			idx_lc = (source=='ffi')
			idx_sc = (source=='tpf')
			ax.scatter(tmags[idx_lc], masksizes[idx_lc], marker='s', facecolors='None', color=rgba_color)
			ax.scatter(tmags[idx_sc], masksizes[idx_sc], marker='o', facecolors='None', color=rgba_color, label=lab)
	
		mags = np.linspace(np.nanmin(vx)-1, np.nanmax(vx)+1, 500)
		pix = np.asarray([Pixinaperture(m) for m in mags], dtype='float64')
		ax.plot(mags, pix, color='k', ls='-')
	
		ax.set_xlim([np.nanmin(vx)-1, np.nanmax(vx)+1])
		ax.set_ylim([3, np.nanmax(vy)+100])
	
		ax.set_xlabel('TESS magnitude', fontsize=16, labelpad=10)
		ax.set_ylabel('Pixels in aperture', fontsize=16, labelpad=10)
	
		xtick_major = np.median(np.diff(ax.get_xticks()))
		ax.xaxis.set_minor_locator(MultipleLocator(xtick_major/2))
		ytick_major = np.median(np.diff(ax.get_yticks()))
		ax.yaxis.set_minor_locator(MultipleLocator(ytick_major/2))
		ax.tick_params(direction='out', which='both', pad=5, length=3)
		ax.tick_params(which='major', pad=6, length=5,labelsize='15')
		ax.yaxis.set_ticks_position('both')
		ax.xaxis.set_ticks_position('both')
		ax.set_yscale("log", nonposy='clip')
		ax.yaxis.set_major_formatter(ScalarFormatter())
		ax.legend(loc='upper right', prop={'size': 12})
	
		if len(self.cursors)>1:
			filename = 'pix_in_aper_joint.%s' %self.extension
		else:
			filename = 'pix_in_aper.%s' %self.extension
			
			
		for i, fol in enumerate(self.outfolders):
			fig.savefig(os.path.join(fol, filename))
			
		if self.show:
			plt.show()
			
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
		
		fig = plt.figure()
		ax = fig.add_subplot(111)
	
		norm = colors.Normalize(vmin=0, vmax=len(self.cursors)+5)
		scalarMap = cmx.ScalarMappable(norm=norm, cmap=plt.get_cmap('Set1') )
		fig.subplots_adjust(left=0.14, wspace=0.3, top=0.94, bottom=0.155, right=0.96)
		
		vx = np.array([])
		vy = np.array([])
	
		for i, c in enumerate(self.cursors):
			star_vals = search_database(c, search=['corr_status=1'], select=['todolist.datasource','todolist.tmag','ccd','mean_flux'])
			sector = get_sector(c)
			
			logger.info('Including data from Sector %i' %int(sector[0]['sector']))
			
			rgba_color = scalarMap.to_rgba(i)
			
			tmags = np.array([star_vals[j]['tmag'] for j in range(len(star_vals))])
			meanfluxes = np.array([star_vals[j]['mean_flux'] for j in range(len(star_vals))])
			source = np.array([star_vals[j]['datasource'] for j in range(len(star_vals))])
			
			vx = np.append(vx, tmags)
			vy = np.append(vy, meanfluxes)
				
			lab = 'Sector %i' %int(sector[0]['sector'])
			idx_lc = (source=='ffi')
			idx_sc = (source=='tpf')
			ax.scatter(tmags[idx_lc], meanfluxes[idx_lc], marker='s', facecolors='None', color=rgba_color)
			ax.scatter(tmags[idx_sc], meanfluxes[idx_sc], marker='o', facecolors='None', color=rgba_color, label=lab)
	
	
		logger.info('Optimising coefficient of relation')
		z = lambda c: np.sum((np.log10(vy) + 0.4*(vx - c))**2)
		cc = OP.minimize(z, 20.5, method='Nelder-Mead', options={'disp':False})
		
		logger.info('Optimisation terminated successfully? %s' %cc.success)
		logger.info('Coefficient is found to be %1.4f' %cc.x)
		
		C=np.linspace(19, 22, 100)
		fig2 = plt.figure()
		ax2=fig2.add_subplot(111)
		[ax2.scatter(c, z(c), marker='o', color='k') for c in C] 
		ax2.axvline(x=cc.x, color='r')
		ax2.set_xlabel('Coefficient')
		ax2.set_ylabel(r'$\chi^2$')
		
		
		mag = np.linspace(np.nanmin(vx)-1, np.nanmax(vx)+1,100)
		ax.plot(mag, 10**(-0.4*(mag - cc.x)), color='k', ls='--')
		ax.set_yscale("log", nonposy='clip')
	
		ax.set_xlim([np.nanmin(vx)-1, np.nanmax(vx)+1])
		ax.set_xlim(ax.get_xlim()[::-1])
	
		ax.text(8, 1000, r'$\rm Flux = 10^{-0.4\,(T_{mag} - %1.2f)}$' %cc.x, fontsize=14)
		ax.set_xlabel('TESS magnitude', fontsize=16, labelpad=10)
		ax.set_ylabel('Mean flux', fontsize=16, labelpad=10)
	
		ax.xaxis.set_major_locator(MultipleLocator(2))
		ax.xaxis.set_minor_locator(MultipleLocator(1))
		ax.tick_params(direction='out', which='both', pad=5, length=3)
		ax.tick_params(which='major', pad=6, length=5,labelsize='15')
		ax.yaxis.set_ticks_position('both')
		ax.xaxis.set_ticks_position('both')
		ax.legend(loc='upper left', prop={'size': 12})
		
		
		if len(self.cursors)>1:
			filename = 'mag_to_flux_joint.%s' %self.extension
			filename2 = 'mag_to_flux_optimize_joint.%s' %self.extension
		else:
			filename = 'mag_to_flux.%s' %self.extension
			filename2 = 'mag_to_flux_optimize.%s' %self.extension
			
		for i, fol in enumerate(self.outfolders):
			fig.savefig(os.path.join(fol, filename))
			fig2.savefig(os.path.join(fol, filename2))
			
		if self.show:
			plt.show()
			
		plt.close(fig)
		
			
	# =============================================================================
	#
	# =============================================================================
	
	def plot_mag_dist(self):
		
		"""
		Function to plot flux values from apertures against the stellar TESS magnitudes, 
		and determine coefficient describing the relation

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""
	
		logger=logging.getLogger(__name__)
		
		logger.info('--------------------------------------')
		logger.info('Plotting Magnitude to Flux conversion')
		
		fig = plt.figure()
		ax = fig.add_subplot(111)
	
		norm = colors.Normalize(vmin=0, vmax=len(self.cursors)+5)
		scalarMap = cmx.ScalarMappable(norm=norm, cmap=plt.get_cmap('Set1') )
		fig.subplots_adjust(left=0.14, wspace=0.3, top=0.94, bottom=0.155, right=0.96)
		

	
		for i, c in enumerate(self.cursors):
			star_vals = search_database(c, search=['corr_status=1'], select=['todolist.datasource','todolist.tmag'])
			sector = get_sector(c)
			
			logger.info('Including data from Sector %i' %int(sector[0]['sector']))
			
			rgba_color = scalarMap.to_rgba(i)
			
			tmags = np.array([star_vals[j]['tmag'] for j in range(len(star_vals))])
			meanfluxes = np.array([star_vals[j]['mean_flux'] for j in range(len(star_vals))])
			source = np.array([star_vals[j]['datasource'] for j in range(len(star_vals))])
			
#			vx = np.append(vx, tmags)
#			vy = np.append(vy, meanfluxes)
				
			lab = 'Sector %i' %int(sector[0]['sector'])
			idx_lc = (source=='ffi')
			idx_sc = (source=='tpf')
			
			if len(tmag_vals_lc) > 0:
				kde_lc = KDE(tmag_vals_lc)
				kde_lc.fit(gridsize=1000)
				ax.fill_between(kde_lc.support, 0, kde_lc.density*len(tmag_vals_lc), color='b', alpha=0.3, label='1800s')
				ax.scatter(tmag_vals_lc, np.zeros_like(tmag_vals_lc), lw=1, marker='|', c='k', s=80)
	
			if len(tmag_vals_sc) > 0:
				kde_sc = KDE(tmag_vals_sc)
				kde_sc.fit(gridsize=1000)
				ax.fill_between(kde_sc.support, 0, kde_sc.density*len(tmag_vals_sc), color='r', alpha=0.3, label='120s')
				ax.scatter(tmag_vals_sc, np.zeros_like(tmag_vals_sc), lw=1, marker='|', c='k', s=80)
		
			tmag_all = np.append(tmag_vals_lc, tmag_vals_sc)
			kde_all = KDE(tmag_all)
			kde_all.fit(gridsize=1000)
			ax.plot(kde_all.support, kde_all.density*len(tmag_all), 'k-', lw=1.5, label='All')
			
			
			ax.scatter(tmags[idx_lc], meanfluxes[idx_lc], marker='s', facecolors='None', color=rgba_color)
			ax.scatter(tmags[idx_sc], meanfluxes[idx_sc], marker='o', facecolors='None', color=rgba_color, label=lab)
			
			
	
		# Add data values
		files = np.array([])
		for root, dirs, fil in os.walk(data_path):
			for file in fil:
				file_path = root + os.sep + file
				if ('corr' in file_path) and ('.fits' in file_path):
					print(file_path)
					files = np.append(files, file_path)
			
			
		tmag_vals_sc = np.array([])
		tmag_vals_lc = np.array([])
		for f in files:
			with fits.open(f) as hdu:
				tmag = hdu[0].header['TESSMAG']
				dt = hdu[1].header['TIMEDEL'] * 86400
				
				print(tmag, dt)
	
				if dt < 1000:
					tmag_vals_sc = np.append(tmag_vals_sc, tmag)
				else:
					tmag_vals_lc = np.append(tmag_vals_lc, tmag)
	
		fig = plt.figure()
		ax = fig.add_subplot(111)
	
		if len(tmag_vals_lc) > 0:
			kde_lc = KDE(tmag_vals_lc)
			kde_lc.fit(gridsize=1000)
			ax.fill_between(kde_lc.support, 0, kde_lc.density*len(tmag_vals_lc), color='b', alpha=0.3, label='1800s')
			ax.scatter(tmag_vals_lc, np.zeros_like(tmag_vals_lc), lw=1, marker='|', c='k', s=80)
	
		if len(tmag_vals_sc) > 0:
			kde_sc = KDE(tmag_vals_sc)
			kde_sc.fit(gridsize=1000)
			ax.fill_between(kde_sc.support, 0, kde_sc.density*len(tmag_vals_sc), color='r', alpha=0.3, label='120s')
			ax.scatter(tmag_vals_sc, np.zeros_like(tmag_vals_sc), lw=1, marker='|', c='k', s=80)
	
		tmag_all = np.append(tmag_vals_lc, tmag_vals_sc)
		kde_all = KDE(tmag_all)
		kde_all.fit(gridsize=1000)
		ax.plot(kde_all.support, kde_all.density*len(tmag_all), 'k-', lw=1.5, label='All')
	
		
	#	try:
	#		kde_sc = KDE(tmag_vals_sc)
	#		kde_sc.fit(gridsize=1000)
	#		ax.plot(kde_sc.support, kde_sc.density*len(tmag_vals_sc), label='SC')
	#		ax.scatter(tmag_vals_sc, np.zeros_like(tmag_vals_sc), lw=1, marker='+', s=80)
	#	except:
	#		pass		
		
	#	ax.set_xlim([3.5, 16.5])
		ax.set_ylim(ymin=0)
		ax.set_xlabel('TESS magnitude', fontsize=16, labelpad=10)
		ax.set_ylabel('Number of stars', fontsize=16, labelpad=10)
		ax.xaxis.set_major_locator(MultipleLocator(2))
		ax.xaxis.set_minor_locator(MultipleLocator(1))
		ax.tick_params(direction='out', which='both', pad=5, length=3)
		ax.tick_params(which='major', pad=6, length=5,labelsize='15')
		ax.yaxis.set_ticks_position('both')
		plt.tight_layout()
		ax.legend(frameon=False, prop={'size':12} ,loc='upper right', borderaxespad=0,handlelength=2.5, handletextpad=0.4)
	
		ax.yaxis.set_ticks_position('both')
	
		if version!=1:
			save_path = 'plots/sector%02d/v%1d/' %(sector,version)
		else:
			save_path = 'plots/sector%02d/' %sector
	
		if not os.path.exists(save_path):
			os.makedirs(save_path)
	
		fig.savefig(os.path.join(save_path, 'magnitudes.pdf'), bb_inches='tight')
		fig.savefig(os.path.join(save_path, 'magnitudes.png'), bb_inches='tight')
		
		if savetex:
			save_path2 = '../releasenote_tex/Release_note%1d/' %sector
				
			fig.savefig(os.path.join(save_path2, 'magnitudes.pdf'), bb_inches='tight')
		
		
		plt.show()


# =============================================================================
#
# =============================================================================

if __name__ == '__main__':

	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Run TESS Corrector pipeline on single star.')
	parser.add_argument('-m', '--method', help='Corrector method to use.', default='all', choices=('pixvsmag', 'mag2flux'))
	parser.add_argument('-e', '--ext', help='Extension of plots.', default='png', choices=('png', 'eps'))
	parser.add_argument('-s', '--show', help='Show plots.', default=False, choices=('True', 'False'))
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('input_folders', type=str, help='Directory to create catalog files in.', nargs='?', default=None)
	args = parser.parse_args()

	# Make sure at least one setting is given:
#	if not args.all and args.starid is None and not args.random:
#		parser.error("Please select either a specific STARID or RANDOM.")

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

	# Get input and output folder from environment variables:
#	input_folder = args.input_folder
#	if input_folder is None:
#		test_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), 'tests', 'input'))
#		if args.test:
#			input_folder = test_folder
#		else:
#			input_folder = os.environ.get('TESSCORR_INPUT', test_folder)
#	output_folder = os.environ.get('TESSCORR_OUTPUT', os.path.abspath('.'))
#	logger.info("Loading input data from '%s'", input_folder)
#	logger.info("Putting output data in '%s'", output_folder)

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
#
#
#	path0 = '/media/mikkelnl/Elements/TASOC/TASOC_S01_fasttrack_thres08_newWS/'
#	path1 = '/media/mikkelnl/Elements/TASOC/TASOC_S02_fasttrack_thres08_newWS/'
#	
##	path0 = r'C:\Users\au195407\Downloads\Ny mappe'
##	data_paths = np.array([path0 + '08', path0 + '09', path0 + '10', path0 + '11', path0 + '12'])
#	#data_paths = np.array([path0 + '10',]
#
#	magtoflux([path0, path1], sector=1, version=5, savetex=False, labels=np.array(['Sector 1','Sector 2']))
##	plot_bg([path0, path1], cad=120, sector=1, version=3, sysnoise=0, savetex=False)
##	plot_noice_lc([path0, path1], cad=120, sector=1, version=4, sysnoise=0, savetex=False)
#	plot_onehour_noise([path0, path1], cad=120, sector=1, version=5, sysnoise=0, savetex=False, labels=np.array(['Sector 1','Sector 2']))
#	plot_pixinaperture([path0, path1], sector=1, version=5, savetex=False, labels=np.array(['Sector 1','Sector 2']))
##	plot_mag_dist([path0, path1], sector=1, version=3, savetex=False)

	
