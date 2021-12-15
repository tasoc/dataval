#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of utility functions that can be used throughout
the corrections package.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import pickle
import gzip
import os
import fnmatch
import hashlib
import re
from bottleneck import nanmedian, nanmean, allnan
from scipy.stats import binned_statistic
from astropy.io import fits
import logging
import tqdm
from collections import defaultdict

# Constants:
mad_to_sigma = 1.482602218505602 #: Conversion constant from MAD to Sigma. Constant is 1/norm.ppf(3/4)

PICKLE_DEFAULT_PROTOCOL = 4 #: Default protocol to use for saving pickle files.

#------------------------------------------------------------------------------
def savePickle(fname, obj):
	"""
	Save an object to file using pickle.

	Parameters:
		fname (string): File name to save to. If the name ends in '.gz' the file
			will be automatically gzipped.
		obj (object): Any pickalble object to be saved to file.
	"""

	if fname.endswith('.gz'):
		o = gzip.open
	else:
		o = open

	with o(fname, 'wb') as fid:
		pickle.dump(obj, fid, protocol=PICKLE_DEFAULT_PROTOCOL)

#------------------------------------------------------------------------------
def loadPickle(fname):
	"""
	Load an object from file using pickle.

	Parameters:
		fname (string): File name to save to. If the name ends in '.gz' the file
			will be automatically unzipped.
		obj (object): Any pickalble object to be saved to file.

	Returns:
		object: The unpickled object from the file.
	"""

	if fname.endswith('.gz'):
		o = gzip.open
	else:
		o = open

	with o(fname, 'rb') as fid:
		return pickle.load(fid)

#--------------------------------------------------------------------------------------------------
def find_tpf_files(rootdir, starid=None, sector=None, camera=None, ccd=None, cadence=None,
	findmax=None):
	"""
	Search directory recursively for TESS Target Pixel Files.

	Parameters:
		rootdir (str): Directory to search recursively for TESS TPF files.
		starid (int, optional): Only return files from the given TIC number.
			If ``None``, files from all TIC numbers are returned.
		sector (int, optional): Only return files from the given sector.
			If ``None``, files from all sectors are returned.
		camera (int or None, optional): Only return files from the given camera number (1-4).
			If ``None``, files from all cameras are returned.
		ccd (int, optional): Only return files from the given CCD number (1-4).
			If ``None``, files from all CCDs are returned.
		cadence (int, optional): Only return files from the given cadence (20 or 120).
			If ``None``, files from all cadences are returned.
		findmax (int, optional): Maximum number of files to return.
			If ``None``, return all files.

	Note:
		Filtering on camera and/or ccd will cause the program to read the headers
		of the files in order to determine the camera and ccd from which they came.
		This can significantly slow down the query.

	Returns:
		list: List of full paths to TPF FITS files found in directory. The list will
			be sorted according to the filename of the files, e.g. primarily by time.
	"""

	logger = logging.getLogger(__name__)

	# Create the filename pattern to search for:
	sector_str = r'\d{4}' if sector is None else '{0:04d}'.format(sector)
	starid_str = r'\d+' if starid is None else '{0:016d}'.format(starid)
	suffix = {None: 'tp(-fast)?', 120: 'tp', 20: 'tp-fast'}[cadence]
	re_pattern = r'^tess\d+-s' + sector_str + '-' + starid_str + r'-\d{4}-[xsab]_' + suffix + r'\.fits(\.gz)?$'
	regex = re.compile(re_pattern)

	# Pattern used for TESS Alert data:
	sector_str = '??' if sector is None else '{0:02d}'.format(sector)
	starid_str = '*' if starid is None else '{0:011d}'.format(starid)
	filename_pattern2 = 'hlsp_tess-data-alerts_tess_phot_{starid:s}-s{sector:s}_tess_v?_tp.fits*'.format(
		sector=sector_str,
		starid=starid_str
	)

	logger.debug("Searching for TPFs in '%s' using pattern '%s'", rootdir, re_pattern)
	logger.debug("Searching for TPFs in '%s' using pattern '%s'", rootdir, filename_pattern2)

	# Do a recursive search in the directory, finding all files that match the pattern:
	breakout = False
	matches = []
	for root, dirnames, filenames in os.walk(rootdir, followlinks=True):
		for filename in filenames:
			if regex.match(filename) or fnmatch.fnmatch(filename, filename_pattern2):
				fpath = os.path.join(root, filename)
				if camera is not None and fits.getval(fpath, 'CAMERA', ext=0) != camera:
					continue

				if ccd is not None and fits.getval(fpath, 'CCD', ext=0) != ccd:
					continue

				matches.append(fpath)
				if findmax is not None and len(matches) >= findmax:
					breakout = True
					break
		if breakout:
			break

	# Sort the list of files by thir filename:
	matches.sort(key=lambda x: os.path.basename(x))

	return matches

#------------------------------------------------------------------------------
def sphere_distance(ra1, dec1, ra2, dec2):
	"""
	Calculate the great circle distance between two points using the Vincenty formulae.

	Parameters:
		ra1 (float or ndarray): Longitude of first point in degrees.
		dec1 (float or ndarray): Lattitude of first point in degrees.
		ra2 (float or ndarray): Longitude of second point in degrees.
		dec2 (float or ndarray): Lattitude of second point in degrees.

	Returns:
		ndarray: Distance between points in degrees.

	Note:
		https://en.wikipedia.org/wiki/Great-circle_distance
	"""

	# Convert angles to radians:
	ra1 = np.deg2rad(ra1)
	ra2 = np.deg2rad(ra2)
	dec1 = np.deg2rad(dec1)
	dec2 = np.deg2rad(dec2)

	# Calculate distance using Vincenty formulae:
	return np.rad2deg(np.arctan2(
		np.sqrt( (np.cos(dec2)*np.sin(ra2-ra1))**2 + (np.cos(dec1)*np.sin(dec2) - np.sin(dec1)*np.cos(dec2)*np.cos(ra2-ra1))**2 ),
		np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra2-ra1)
	))

#------------------------------------------------------------------------------
def rms_timescale(lc, timescale=3600/86400):
	"""
	Compute robust RMS on specified timescale. Using MAD scaled to RMS.

	Parameters:
		lc (``lightkurve.TessLightCurve`` object): Timeseries to calculate RMS for.
		timescale (float, optional): Timescale to bin timeseries before calculating RMS. Default=1 hour.

	Returns:
		float: Robust RMS on specified timescale.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	if len(lc.flux) == 0 or allnan(lc.flux):
		return np.nan

	time_min = np.nanmin(lc.time)
	time_max = np.nanmax(lc.time)
	if not np.isfinite(time_min) or not np.isfinite(time_max) or time_max - time_min <= 0:
		raise ValueError("Invalid time-vector specified")

	# Construct the bin edges seperated by the timescale:
	bins = np.arange(time_min, time_max, timescale)
	bins = np.append(bins, time_max)

	# Bin the timeseries to one hour:
	indx = np.isfinite(lc.flux)
	flux_bin, _, _ = binned_statistic(lc.time[indx], lc.flux[indx], nanmean, bins=bins)

	# Compute robust RMS value (MAD scaled to RMS)
	return mad_to_sigma * nanmedian(np.abs(flux_bin - nanmedian(flux_bin)))

#------------------------------------------------------------------------------
def mad(x):
	"""
	Median absolute deviation scaled to standard deviation.

	Parameters:
		x (ndarray): Array to calculate robust standard deviation for.

	Returns:
		float: Median absolute deviation scaled to standard deviation.
	"""
	return mad_to_sigma * nanmedian(np.abs(x - nanmedian(x)))

#--------------------------------------------------------------------------------------------------
def mag2flux(mag, zp=20.451):
	"""
	Convert from magnitude to flux using scaling relation from
	aperture photometry. This is an estimate.

	The default scaling is based on TASOC Data Release 5 from sectors 1-5.

	Parameters:
		mag (ndarray): Magnitude in TESS band.
		zp (float): Zero-point to use in scaling. Default is estimated from
			TASOC Data Release 5 from TESS sectors 1-5.

	Returns:
		ndarray: Corresponding flux value
	"""
	return np.clip(10**(-0.4*(mag - zp)), 0, None)

#--------------------------------------------------------------------------------------------------
def find_lightcurve_files(rootdir, pattern='tess*-tasoc_lc.fits.gz'):
	"""
	Find lightcurve files matching filename pattern.

	Parameters:
		rootdir (str): Root directory to search for lightcurve files.
		pattern (str): Pattern to match filenames. Default is for raw TASOC lightcurves.

	Returns:
		iterator: Iterator of paths to found lightcurve files.
	"""
	for root, dirnames, filenames in os.walk(rootdir, followlinks=True):
		for filename in fnmatch.filter(filenames, pattern):
			yield os.path.join(root, filename)

#--------------------------------------------------------------------------------------------------
def get_filehash(fname):
	"""Calculate SHA1-hash of file."""
	buf = 65536
	s = hashlib.sha1()
	with open(fname, 'rb') as fid:
		while True:
			data = fid.read(buf)
			if not data:
				break
			s.update(data)

	sha1sum = s.hexdigest().lower()
	if len(sha1sum) != 40:
		raise Exception("Invalid file hash")
	return sha1sum

#--------------------------------------------------------------------------------------------------
class CounterFilter(logging.Filter):
	"""
	A logging filter which counts the number of log records in each level.
	"""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.counter = defaultdict(int)

	def filter(self, record): # noqa: A003
		self.counter[record.levelname] += 1
		return True

#--------------------------------------------------------------------------------------------------
class TqdmLoggingHandler(logging.Handler):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def emit(self, record):
		try:
			msg = self.format(record)
			tqdm.tqdm.write(msg)
			self.flush()
		except (KeyboardInterrupt, SystemExit):
			raise
		except: # noqa: E722
			self.handleError(record)
