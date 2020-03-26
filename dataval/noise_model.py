#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Noise model as a function of magnitude and position

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
import scipy.interpolate as INT
import logging

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

	if cad == 1800:
		ffi_tmag = np.array([2.05920002, 2.95159999, 3.84399996, 4.73639993, 5.6287999,
			6.52119987, 7.41359984, 8.30599982, 9.19839979, 10.09079976,
			10.98319973, 11.8755997, 12.76799967, 13.66039964, 14.55279961])
		ffi_mask = np.array([1484.5, 715., 447., 282.5, 185., 126., 98., 76.,
			61., 49., 38., 28., 20., 14., 8.])
		ffi_pix = INT.InterpolatedUnivariateSpline(ffi_tmag, ffi_mask, k=1)
		pixels = ffi_pix(Tmag)

	elif cad == 120:
		tpf_tmag = np.array([2.48170001, 3.56310005, 4.6445001, 5.72590014, 6.80730019,
			7.88870023, 8.97010028, 10.05150032, 11.13290037, 12.21430041,
			13.29570045, 14.3771005, 15.45850054, 16.53990059, 17.62130063])
		tpf_mask = np.array([473., 188., 82., 61., 55., 49., 43., 38., 31., 23., 18.,
			13., 10., 12., 11.])
		tpf_pix = INT.InterpolatedUnivariateSpline(tpf_tmag, tpf_mask, k=1)
		pixels = tpf_pix(Tmag)

	else:
		raise NotImplementedError()

	# Approximate relation for pixels in aperture (based on plot in Sullivan et al.)
	#pixels = (30 + (((3-30)/(14-7)) * (Tmag-7)))*(Tmag<14) + 3*(Tmag>=14)

	return np.asarray(np.maximum(pixels, 3), dtype='int32')

#--------------------------------------------------------------------------------------------------
def mean_flux_level(Tmag):
	"""
	Mean flux from TESS magnitude

	.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
	"""
	# Magnitude system based on Sullivan et al.
	#collecting_area = np.pi*(10.5/2)**2 # square cm
	#Teff_list = np.array([2450, 3000, 3200, 3400, 3700, 4100, 4500, 5000, 5777, 6500, 7200, 9700]) # Based on Sullivan
	#Flux_list = np.array([2.38, 1.43, 1.40, 1.38, 1.39, 1.41, 1.43, 1.45, 1.45, 1.48, 1.48, 1.56])*1e6 # photons per sec; Based on Sullivan
	#Magn_list = np.array([306, -191, -202, -201, -174, -132, -101, -80, -69.5, -40, -34.1, 35])*1e-3 #Ic-Tmag (mmag)

	#Flux_int = INT.UnivariateSpline(Teff_list, Flux_list, k=1, s=0)
	#Magn_int = INT.UnivariateSpline(Teff_list, Magn_list, k=1, s=0)

	#Imag = Magn_int(Teff)+Tmag
	#Flux = 10**(-0.4*Imag) * Flux_int(Teff) * collecting_area

	Flux = 10**(-0.4*(Tmag - 20.54))
	return Flux

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
	mean_level_ppm = mean_flux_level(Tmag) * timescale # electrons (based on measurement) #, Teff

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

	# Calculate the total noise model by adding up the individual contributions:
	total_noise = np.sqrt(np.sum(noise_vals**2, axis=1))

	return total_noise, noise_vals # ppm per cadence
