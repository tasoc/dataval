#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of DataValidation.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dataval.noise_model import phot_noise

#--------------------------------------------------------------------------------------------------
def test_noise_model():
	"""
	Try initializing DataValidation with a wrong input path
	"""

	# Call with a single float:
	total_noise, noise_components = phot_noise(1)
	assert total_noise.shape == (1,)
	assert noise_components.shape == (1, 4)

	# Call with an ndarray:
	Tmag = np.linspace(0, 20, 500)
	total_noise, noise_components = phot_noise(Tmag)
	assert total_noise.shape == (500,)
	assert noise_components.shape == (500, 4)

	# Individual noise components should be lower than the total noise:
	assert np.all(noise_components[:, 0] <= total_noise)
	assert np.all(noise_components[:, 1] <= total_noise)
	assert np.all(noise_components[:, 2] <= total_noise)
	assert np.all(noise_components[:, 3] <= total_noise)

#--------------------------------------------------------------------------------------------------
def test_noise_model_invalid_input():

	Tmag = np.linspace(0, 20, 100)

	with pytest.raises(ValueError):
		phot_noise(Tmag, coord='not-a-coord')

	with pytest.raises(ValueError):
		phot_noise(Tmag, coord={'not': 'a-coord'})

#--------------------------------------------------------------------------------------------------
def test_noise_model_coord():

	Tmag = np.linspace(0, 20, 100)

	# Default is to use RA=0, DEC=0:
	n_default, n_default_comp = phot_noise(Tmag)

	# These two should give exactly the same results:
	n_coordin, n_coordin_comp = phot_noise(Tmag, coord={'RA': 0, 'DEC': 0})
	np.testing.assert_allclose(n_default, n_coordin)
	np.testing.assert_allclose(n_default_comp, n_coordin_comp)

	# Ecliptic (0,0) is the same as RA-DEC (0,0):
	n_coordin2, n_coordin2_comp = phot_noise(Tmag, coord={'ELAT': 0, 'ELON': 0})
	np.testing.assert_allclose(n_default, n_coordin2)
	np.testing.assert_allclose(n_default_comp, n_coordin2_comp)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
