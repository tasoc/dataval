#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Noise model as a function of magnitude and position

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.interpolate import InterpolatedUnivariateSpline
from .utilities import mag2flux

#--------------------------------------------------------------------------------------------------
def ZLnoise(gal_lat):
	"""
	RMS noise from Zodiacal background.

	.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
	"""
	rms = (16 - 10) * (gal_lat/90 - 1)**2 + 10 # e-1 / pix in 2sec integration
	return rms

#--------------------------------------------------------------------------------------------------
def Pixinaperture(Tmag, cad=1800):
	"""
	Number of pixels in aperture as a function of Tmag

	.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
	"""

	# Approximate relation for pixels in aperture (based on plot in Sullivan et al.)
	#pixels = (30 + (((3-30)/(14-7)) * (Tmag-7)))*(Tmag<14) + 3*(Tmag>=14)

	if cad in (1800, 600, 200):
		masksize = np.array([
			[2.05920002, 1484.5],
			[2.95159999, 715],
			[3.84399996, 447],
			[4.73639993, 282.5],
			[5.62879990, 185],
			[6.52119987, 126],
			[7.41359984, 98],
			[8.30599982, 76],
			[9.19839979, 61],
			[10.09079976, 49],
			[10.98319973, 38],
			[11.8755997, 28],
			[12.76799967, 20],
			[13.66039964, 14],
			[14.55279961, 8]
		])

	elif cad in (120, 20):
		masksize = np.array([
			[2.48170001, 473],
			[3.56310005, 210],
			[4, 174],
			[5.72590014, 85],
			[6.80730019, 69],
			[7.88870023, 61],
			[8.97010028, 50],
			[10.05150032, 38],
			[11.13290037, 26],
			[12.5, 13],
			[15.0, 4]
		])

	else:
		raise NotImplementedError()

	# Interpolate linearly in log-space:
	pix_log = InterpolatedUnivariateSpline(masksize[:,0], np.log10(masksize[:,1]), k=1, ext=0)
	pix = lambda x: np.round(10**(pix_log(x)), decimals=13) # noqa: E731
	pixels = pix(Tmag)

	# Ensure lower limit of 4 pixels:
	pixels = np.clip(pixels, 4, None)

	return np.asarray(pixels, dtype='int32')

#--------------------------------------------------------------------------------------------------
def phot_noise(Tmag, timescale=3600, coord=None, sysnoise=60, Teff=5775, cadpix=1800):
	"""
	Photometric noise model in ppm/timescale.

	Parameters:
		Tmag (ndarray or float): TESS magnitudes to calculate noise model for.
		timescale (float): Timescale of noise model in seconds. Defaults to 3600 (1 hour).
		coord (SkyCoord or dict, optional): Sky coordinate used for zodiacal noise. Defaults to RA=0, DEC=0.
		sysnoise (float, optional): Systematic noise contribution in ppm/hr. Defaults to 60.
		Teff (float, optional): Effective temperture. Defaults to 5775.
		cadpix (TYPE, optional): Cadence where images are taken. Choices are 1800 and 120. Defaults to 1800.

	Returns:
		tuple:
			- ndarray: Total noise model as a function of TESS magnitide.
			- ndarray: Individual noise components in 4 columns: Shot, Zodiacal, Read and Systematic.

	Raises:
		ValueError: On invalid coord input.

	.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	# Make sure Tmag is a numpy array:
	Tmag = np.asarray(Tmag)

	# Create SkyCoord object from the coordinates supplied:
	if coord is None:
		# TODO: Nothing is given, so use complete dummy coordinates
		gc = SkyCoord(0*u.degree, 0*u.degree, frame='icrs')

	elif isinstance(coord, dict):
		if 'RA' in coord and 'DEC' in coord:
			gc = SkyCoord(coord['RA']*u.degree, coord['DEC']*u.degree, frame='icrs')
		elif 'ELON' in coord and 'ELAT' in coord:
			gc = SkyCoord(lon=coord['ELON']*u.degree, lat=coord['ELAT']*u.degree, frame='barycentrictrueecliptic')
		else:
			raise ValueError("Invalid coord in dict format")

	elif not isinstance(coord, SkyCoord):
		raise ValueError("Invalid coord")

	# Calculate galactic latitude for Zodiacal noise
	gc_gal = gc.transform_to('galactic')
	gal_lat0 = gc_gal.b.deg
	gal_lat = np.arcsin(np.abs(np.sin(gal_lat0*np.pi/180)))*180/np.pi

	# Number of 2 sec integrations in cadence
	integrations = timescale/2

	# Number of pixels in aperture given Tmag
	pixels = Pixinaperture(Tmag, cadpix)

	# noise values are in rms, so square-root should be used when factoring up
	Flux_factor = np.sqrt(integrations * pixels)

	# Mean flux level in electrons per cadence
	mean_level_ppm = mag2flux(Tmag) * timescale # electrons (based on measurement) #, Teff

	# Shot noise
	shot_noise = 1e6/np.sqrt(mean_level_ppm)

	# Read noise
	read_noise = 10 * Flux_factor * 1e6/mean_level_ppm # ppm

	# Zodiacal noise
	zodiacal_noise = ZLnoise(gal_lat) * Flux_factor * 1e6/mean_level_ppm # ppm

	# Systematic noise in ppm
	systematic_noise = np.full_like(Tmag, sysnoise / np.sqrt(timescale/3600)) # ppm / sqrt(hr)

	# Put individual components together in single table:
	noise_vals = np.column_stack((shot_noise, zodiacal_noise, read_noise, systematic_noise))
	noise_vals = np.clip(noise_vals, 0, None)

	# Calculate the total noise model by adding up the individual contributions:
	total_noise = np.sqrt(np.sum(noise_vals**2, axis=1))
	total_noise = np.clip(total_noise, 0, None)

	return total_noise, noise_vals # ppm per cadence
