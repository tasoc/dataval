#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of utility functions.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
import numpy as np
import tempfile
#from lightkurve import LightCurve
import conftest # noqa: F401
from dataval import utilities

#--------------------------------------------------------------------------------------------------
def test_pickle():

	# Pretty random object to save and restore:
	test_object = {'test': 42, 'whatever': (1,2,3,4,5432)}

	with tempfile.TemporaryDirectory() as tmpdir:
		fname = os.path.join(tmpdir, 'test.pickle')

		# Save object in file:
		utilities.savePickle(fname, test_object)
		assert os.path.exists(fname), "File does not exist"

		# Recover the object again:
		recovered_object = utilities.loadPickle(fname)
		print(recovered_object)
		assert test_object == recovered_object, "The object was not recovered"

		# Save object in file:
		utilities.savePickle(fname + ".gz", test_object)
		assert os.path.exists(fname + ".gz"), "File does not exist"

		# Recover the object again:
		recovered_object = utilities.loadPickle(fname + ".gz")
		print(recovered_object)
		assert test_object == recovered_object, "The object was not recovered"

#--------------------------------------------------------------------------------------------------
def test_sphere_distance():
	np.testing.assert_allclose(utilities.sphere_distance(0, 0, 90, 0), 90)
	np.testing.assert_allclose(utilities.sphere_distance(90, 0, 0, 0), 90)
	np.testing.assert_allclose(utilities.sphere_distance(0, -90, 0, 90), 180)
	np.testing.assert_allclose(utilities.sphere_distance(45, 45, 45, 45), 0)
	np.testing.assert_allclose(utilities.sphere_distance(33.2, 45, 33.2, -45), 90)
	np.testing.assert_allclose(utilities.sphere_distance(337.5, 0, 22.5, 0), 45)
	np.testing.assert_allclose(utilities.sphere_distance(22.5, 0, 337.5, 0), 45)
	np.testing.assert_allclose(utilities.sphere_distance(0, 0, np.array([0, 90]), np.array([90, 90])), np.array([90, 90]))

#--------------------------------------------------------------------------------------------------
"""
def test_rms_timescale():

	time = np.linspace(0, 27, 100)
	flux = np.zeros(len(time))

	rms = utilities.rms_timescale(LightCurve(time=time, flux=flux))
	print(rms)
	np.testing.assert_allclose(rms, 0)

	rms = utilities.rms_timescale(LightCurve(time=time, flux=flux*np.nan))
	print(rms)
	assert np.isnan(rms), "Should return nan on pure nan input"

	rms = utilities.rms_timescale(LightCurve(time=[], flux=[]))
	print(rms)
	assert np.isnan(rms), "Should return nan on empty input"

	# Pure nan in the time-column should raise ValueError:
	with np.testing.assert_raises(ValueError):
		rms = utilities.rms_timescale(LightCurve(time=time*np.nan, flux=flux))

	# Test with timescale longer than timespan should return zero:
	flux = np.random.randn(1000)
	time = np.linspace(0, 27, len(flux))
	rms = utilities.rms_timescale(LightCurve(time=time, flux=flux), timescale=30.0)
	print(rms)
	np.testing.assert_allclose(rms, 0)
"""

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
