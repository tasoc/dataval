#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of DataValidation.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
import sys
import shlex
import subprocess
import conftest # noqa: F401

#--------------------------------------------------------------------------------------------------
def capture_run_dataval(params):

	command = '"%s" run_dataval.py %s' % (sys.executable, params.strip())
	print(command)

	cmd = shlex.split(command)
	proc = subprocess.Popen(cmd,
		cwd=os.path.join(os.path.dirname(__file__), '..'),
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		universal_newlines=True
	)
	out, err = proc.communicate()
	exitcode = proc.returncode
	proc.kill()

	print("ExitCode: %d" % exitcode)
	print("StdOut:\n%s" % out)
	print("StdErr:\n%s" % err)
	return out, err, exitcode

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

	params = '--quiet {corr:s} {validate:s} "{input_dir:s}"'.format(
		corr='--corr' if corr else '',
		validate='--validate' if save else '',
		input_dir=test_dir
	)
	out, err, exitcode = capture_run_dataval(params)

	#assert exitcode == 0
	#assert False

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
