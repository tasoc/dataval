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

	params = '--jobs=1 --version=5 "{input_file:s}"'.format(
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
				assert hdr['SECTOR'] == row['sector']

		cursor.close()

	# Re-running should not process anything:
	out, err, exitcode = capture_run_release(params)
	assert exitcode == 0
	assert 'Nothing to process' in out

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("changes,expect_returncode,expect_msg", [
	["UPDATE corr_settings SET corrector='nonsense';", 2, 'Invalid corrector value'],
	["UPDATE diagnostics_corr SET lightcurve='does-not-exists.fits.gz';", 2, 'File not found'],
	["UPDATE todolist SET starid=-1 WHERE priority=1220;", 1, 'STARID'],
	["UPDATE todolist SET sector=-1 WHERE priority=1220;", 1, 'SECTOR'],
	["UPDATE todolist SET camera=-1 WHERE priority=1220;", 1, 'CAMERA'],
	["UPDATE todolist SET ccd=-1 WHERE priority=1220;", 1, 'CCD'],
])
def test_run_release_wrong_db(PRIVATE_INPUT_DIR, changes, expect_returncode, expect_msg):
	"""
	Try to run package release on different input.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	input_file = os.path.join(PRIVATE_INPUT_DIR, 'ready_for_release', 'todo-cbv.sqlite')
	print(input_file)

	with closing(sqlite3.connect(input_file)) as conn:
		conn.row_factory = sqlite3.Row
		cursor = conn.cursor()
		cursor.execute(changes)
		conn.commit()

	params = '--quiet --version=5 --jobs=1 "{input_file:s}"'.format(
		input_file=input_file
	)
	out, err, exitcode = capture_run_release(params)
	assert exitcode == expect_returncode
	assert expect_msg in out or expect_msg in err

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
