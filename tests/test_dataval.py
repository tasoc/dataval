#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of DataValidation.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dataval import DataValidation

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')

#----------------------------------------------------------------------
def test_dataval_notodo():
	"""
	Try initializing DataValidation with a wrong input path
	"""

	INPUT_DIR_NOTODO = os.path.join(os.path.dirname(__file__), 'input', 'does-not-exist')

	# Create DataValidation object:
	with pytest.raises(FileNotFoundError):
		with DataValidation([INPUT_DIR_NOTODO]):
			pass


#----------------------------------------------------------------------
def test_dataval_raw():
	"""
	Try to run DataValidation on ONLY_RAW input
	"""

	INPUT = os.path.join(INPUT_DIR, 'only_raw')

	# We should throw an exception when trying to run corr=True on TODO-file where
	# corrections have not been run:
	with pytest.raises(Exception):
		with DataValidation([INPUT], corr=True):
			pass

	# Create DataValidation object:
	with DataValidation([INPUT], corr=False) as dataval:
		dataval.Validations()


#----------------------------------------------------------------------
@pytest.mark.parametrize("corr", [False, True])
def test_dataval_corr(corr):
	"""
	Try to run DataValidation on CORRECTED input
	"""

	INPUT = os.path.join(INPUT_DIR, 'with_corr')

	# On this input file it should be possible to run with both
	with DataValidation([INPUT], corr=corr) as dataval:
		dataval.Validations()


#----------------------------------------------------------------------
if __name__ == '__main__':
	test_dataval_notodo()
	test_dataval_raw()
	test_dataval_corr(False)
	test_dataval_corr(True)