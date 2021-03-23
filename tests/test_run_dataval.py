#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of DataValidation command-line interface.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
from conftest import capture_run_cli

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("inp,corr,save", [
	('only_raw', False, False),
	('with_corr', False, False),
	('with_corr', True, False),
	('with_corr', True, True),
])
def test_run_dataval(PRIVATE_INPUT_DIR, inp, corr, save):
	"""
	Try to run DataValidation on different input
	"""

	test_dir = os.path.join(PRIVATE_INPUT_DIR, inp, 'todo.sqlite')

	params = ['--quiet']
	if corr:
		params.append('--corrected')
	if save:
		params.append('--validate')
	params.append(test_dir)
	print(params)

	out, err, exitcode = capture_run_cli('run_dataval.py', params)
	assert exitcode == 4 # Since the files are missing, this should result in error-state
	#assert False

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('method,expected_retcode', [
	['pixvsmag', 0],
	['stampsize', 0],
	['contam', 0],
	['mag2flux', 0],
	['noise', 0],
	['magdist', 0],
	['calctime', 0],
	['waittime', 0],
	['haloswitch', 0],
	['sumimage', 4],
	['camera_overlap', 4]
])
def test_run_dataval_methods(PRIVATE_INPUT_DIR, method, expected_retcode):
	"""
	Try to run DataValidation on different individual methods
	"""
	test_dir = os.path.join(PRIVATE_INPUT_DIR, 'with_corr', 'todo.sqlite')
	out, err, exitcode = capture_run_cli('run_dataval.py', [
		'--corrected',
		f'--method={method:s}',
		test_dir])
	assert exitcode == expected_retcode

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
