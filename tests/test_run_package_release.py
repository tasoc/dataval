#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of DataValidation command-line interface.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
import sys
import shlex
import subprocess
import sqlite3
from contextlib import closing
from astropy.io import fits
import conftest # noqa: F401
from dataval.utilities import get_filehash

#--------------------------------------------------------------------------------------------------
def capture_run_release(params):

	command = '"%s" run_package_release.py %s' % (sys.executable, params.strip())
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
@pytest.mark.parametrize("corrector", ['cbv', ]) # 'ensemble'
def test_run_release(PRIVATE_INPUT_DIR, corrector):
	"""
	Try to run package release on different input.

	Todo-files for the tests can be produced by taking the final TASOC_DR05/S06 todo-XX.sqlite
	files and trimming them using the following SQL commands:

	.. code-block:: SQL

		DROP TABLE datavalidation_raw;
		DROP TABLE diagnostics;
		DROP TABLE photometry_skipped;
		DELETE FROM todolist WHERE starid >= 5000000;
		DELETE FROM todolist WHERE priority NOT IN (SELECT priority FROM todolist ORDER BY priority LIMIT 20);
		VACUUM;
		ANALYZE;
		VACUUM;

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	input_file = os.path.join(PRIVATE_INPUT_DIR, 'ready_for_release', 'todo-{0:s}.sqlite'.format(corrector))
	print(input_file)

	params = '--quiet --jobs=1 "{input_file:s}"'.format(
		input_file=input_file
	)
	out, err, exitcode = capture_run_release(params)
	assert exitcode == 0

	# It should have created a release file:
	release_file = os.path.join(PRIVATE_INPUT_DIR, 'ready_for_release', 'release-{0:s}.sqlite'.format(corrector))
	print(release_file)
	assert os.path.isfile(release_file), "Release file does not exist"

	with closing(sqlite3.connect(release_file)) as conn:
		conn.row_factory = sqlite3.Row
		cursor = conn.cursor()

		cursor.execute("SELECT COUNT(*) FROM release;")
		antal = cursor.fetchone()[0]
		assert antal == 19

		cursor.execute("SELECT * FROM release;")
		for row in cursor.fetchall():
			fpath = os.path.join(PRIVATE_INPUT_DIR, 'ready_for_release', row['lightcurve'])
			print(fpath)

			assert os.path.isfile(fpath), "File does not exist"
			assert get_filehash(fpath) == row['filehash']
			assert os.path.getsize(fpath) == row['filesize']

			with fits.open(fpath, mode='readonly', memmap=True) as hdu:
				hdr = hdu[0].header
				assert hdr['DATAVAL'] == row['dataval']
				assert hdr['DATA_REL'] == row['datarel']
				assert hdr['TICID'] == row['starid']
				assert hdr['CAMERA'] == row['camera']
				assert hdr['CCD'] == row['ccd']

		cursor.close()

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
