#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of PACKAGE command-line interface.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
import sys
import subprocess
import sqlite3
import warnings
import logging
from contextlib import closing
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
import conftest # noqa: F401
from dataval import __version__
from dataval.utilities import get_filehash, find_tpf_files
from dataval.release import check_fits_changes

#--------------------------------------------------------------------------------------------------
def capture_run_release(params):

	cmd = [sys.executable, 'run_package_release.py'] + params
	print(cmd)
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
def test_run_release_wrong_file(SHARED_INPUT_DIR):
	"""
	Try to run package release on different input.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	input_file = os.path.join(SHARED_INPUT_DIR, 'ready_for_release', 'todo-does-not-exist.sqlite')
	out, err, exitcode = capture_run_release(['--debug', input_file])
	assert exitcode == 2
	assert 'Input file does not exist' in out

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("jobs", [1, 0])
@pytest.mark.parametrize("corrector", ['cbv', 'ensemble'])
def test_run_release(PRIVATE_INPUT_DIR, jobs, corrector):
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
	tpf_rootdir = os.path.dirname(input_file)

	params = ['--jobs={0:d}'.format(jobs), '--version=5', '--tpf=' + tpf_rootdir, input_file]
	out, err, exitcode = capture_run_release(params)
	assert exitcode == 0

	# It should have created a release file:
	release_file = os.path.join(PRIVATE_INPUT_DIR, 'ready_for_release', 'release-{0:s}.sqlite'.format(corrector))
	assert os.path.isfile(release_file), "Release file does not exist"

	with closing(sqlite3.connect(release_file)) as conn:
		conn.row_factory = sqlite3.Row
		cursor = conn.cursor()

		cursor.execute("SELECT * FROM settings;")
		row = cursor.fetchone()
		assert row['dataval_version'] == __version__
		assert row['corrector'] == corrector
		assert row['version'] == 5

		cursor.execute("SELECT COUNT(*) FROM release;")
		antal = cursor.fetchone()[0]
		if corrector == 'cbv':
			assert antal == 19
		else:
			assert antal == 12

		if corrector == 'cbv':
			cursor.execute("SELECT * FROM release_cbv ORDER BY cadence DESC;")
			cbvs = cursor.fetchall()
			assert len(cbvs) == 2
			row = dict(cbvs[0])
			assert row['path'] == 'cbv-prepare/tess-s0006-c1800-a114-v5-tasoc_cbv.fits.gz'
			assert row['sector'] == 6
			assert row['camera'] == 1
			assert row['ccd'] == 1
			assert row['cadence'] == 1800
			assert row['cbv_area'] == 114
			assert row['datarel'] == 8

			row = dict(cbvs[1])
			assert row['path'] == 'cbv-prepare/tess-s0006-c0120-a114-v5-tasoc_cbv.fits.gz'
			assert row['sector'] == 6
			assert row['camera'] == 1
			assert row['ccd'] == 1
			assert row['cadence'] == 120
			assert row['cbv_area'] == 114
			assert row['datarel'] == 8

		cursor.execute("SELECT * FROM release;")
		for row in cursor.fetchall():
			fpath = os.path.join(PRIVATE_INPUT_DIR, 'ready_for_release', row['lightcurve'])
			print("-" * 30)
			print(fpath)

			assert os.path.isfile(fpath), "File does not exist"
			assert get_filehash(fpath) == row['filehash']
			assert os.path.getsize(fpath) == row['filesize']

			# Test the dependency:
			if row['cadence'] > 200:
				assert row['dependency_tpf'] is None
			else:
				assert row['dependency_tpf'] is not None
				if row['starid'] == 4256961: # This is a secondary target
					assert row['dependency_tpf'] == 4255638
				else: # These are "main" targets:
					assert row['dependency_tpf'] == row['starid']

			with fits.open(fpath, mode='readonly', memmap=True) as hdu:
				hdr = hdu[0].header
				assert hdr['DATAVAL'] == row['dataval']
				assert hdr['DATA_REL'] == row['datarel']
				assert hdr['TICID'] == row['starid']
				assert hdr['CAMERA'] == row['camera']
				assert hdr['CCD'] == row['ccd']
				assert hdr['SECTOR'] == row['sector']

				assert row['cadence'] == int(np.round(hdu[1].header['TIMEDEL']*86400))

				# Check the fix of invalid header in ENSEMBLE extension:
				if corrector == 'ensemble':
					# Check the fix of invalid header in ENSEMBLE extension:
					assert hdu['ENSEMBLE'].header['TDISP2'] != 'E'

					# Check that the stars used to build ensemble were stored:
					dependency_lc = set([int(t) for t in row['dependency_lc'].split(',')])
					assert set(hdu['ENSEMBLE'].data['TIC']) == dependency_lc

				elif corrector == 'cbv':
					assert hdu[1].header['CBV_AREA'] == row['cbv_area']

				# Check the modification of the WCS solution in 120s data:
				if row['cadence'] == 120:
					tpf_file = find_tpf_files(tpf_rootdir,
						starid=row['dependency_tpf'],
						sector=row['sector'],
						camera=row['camera'],
						ccd=row['ccd'],
						cadence=row['cadence'])
					print( tpf_file )

					with warnings.catch_warnings():
						warnings.filterwarnings('ignore', category=FITSFixedWarning)

						# World Coordinate System from the original Target Pixel File:
						wcs_tpf = WCS(header=fits.getheader(tpf_file[0], extname='APERTURE'), relax=True)

						# World coordinate systems from the final FITS lightcurve files:
						wcs_aperture = WCS(header=hdu['APERTURE'].header, relax=True)
						wcs_sumimage = WCS(header=hdu['SUMIMAGE'].header, relax=True)
						#wcs_tpf.printwcs()
						#wcs_aperture.printwcs()
						#wcs_sumimage.printwcs()

					# Try calculating the pixel-coordinate of the target star in the three WCS:
					radec = [[hdr['RA_OBJ'], hdr['DEC_OBJ']]]
					pix_tpf = wcs_tpf.all_world2pix(radec, 0)
					pix_aperture = wcs_aperture.all_world2pix(radec, 0)
					pix_sumimage = wcs_sumimage.all_world2pix(radec, 0)

					# They should give exactly the same results:
					np.testing.assert_allclose(pix_aperture, pix_tpf)
					np.testing.assert_allclose(pix_sumimage, pix_tpf)

		cursor.close()

	# Re-running should not process anything:
	out, err, exitcode = capture_run_release(params)
	assert exitcode == 0
	assert 'Nothing to process' in out

	# Re-running with different VERSION should trigger error:
	params = ['--jobs={0:d}'.format(jobs), '--version=17', '--tpf=' + tpf_rootdir, input_file]
	out, err, exitcode = capture_run_release(params)
	assert exitcode == 2
	assert 'Inconsistent VERSION provided' in out

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("jobs", [1, 0])
@pytest.mark.parametrize("changes,expect_returncode,expect_msg", [
	["UPDATE corr_settings SET corrector='nonsense';", 2, 'Invalid corrector value'],
	["UPDATE diagnostics_corr SET lightcurve='does-not-exists.fits.gz';", 2, 'File not found'],
	["UPDATE todolist SET starid=-1 WHERE priority=1220;", 1, 'STARID'],
	["UPDATE todolist SET sector=-1 WHERE priority=1220;", 1, 'SECTOR'],
	["UPDATE todolist SET camera=-1 WHERE priority=1220;", 1, 'CAMERA'],
	["UPDATE todolist SET ccd=-1 WHERE priority=1220;", 1, 'CCD'],
	["DROP TABLE diagnostics_corr;", 2, 'DIAGNOSTICS_CORR table does not exist'],
	["DROP TABLE datavalidation_corr;", 2, 'DATAVALIDATION_CORR table does not exist'],
	["DELETE FROM datavalidation_corr WHERE priority=1220;", 2, 'DATAVALIDATION_CORR table seems incomplete'],
])
def test_run_release_wrong_db(PRIVATE_INPUT_DIR, jobs, changes, expect_returncode, expect_msg):
	"""
	Try to run package release on different input.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	input_file = os.path.join(PRIVATE_INPUT_DIR, 'ready_for_release', 'todo-cbv.sqlite')
	tpf_rootdir = os.path.dirname(input_file)
	print(input_file)

	with closing(sqlite3.connect(input_file)) as conn:
		conn.row_factory = sqlite3.Row
		cursor = conn.cursor()
		cursor.execute(changes)
		conn.commit()
		cursor.close()

	out, err, exitcode = capture_run_release([
		'--quiet',
		'--jobs={0:d}'.format(jobs),
		'--version=5',
		'--tpf=' + tpf_rootdir,
		input_file
	])
	assert exitcode == expect_returncode
	assert expect_msg in out or expect_msg in err

#--------------------------------------------------------------------------------------------------
def test_check_fits_changes(caplog, PRIVATE_INPUT_DIR):

	# Let's not
	caplog.set_level(logging.ERROR)

	# A random lightcurve file to use for testing:
	fname = os.path.join(PRIVATE_INPUT_DIR, 'ready_for_release', 'ffi', '00004',
		'tess00004207261-s006-1-1-c1800-dr08-v05-tasoc-ens_lc.fits.gz')

	# When nothing changes, it should be an error:
	caplog.clear()
	assert not check_fits_changes(fname, fname)
	assert 'Nothing has changed' in caplog.text

	# When nothing changes, it should be an error, with HDUs:
	caplog.clear()
	with fits.open(fname, mode='readonly') as hdu:
		assert not check_fits_changes(hdu, hdu)
	assert 'Nothing has changed' in caplog.text

	# Delete a HDU:
	with fits.open(fname, mode='readonly') as hdu:
		del hdu[2]
		caplog.clear()
		assert not check_fits_changes(fname, hdu)
		assert 'Different number of HDUs' in caplog.text
		caplog.clear()
		assert not check_fits_changes(hdu, fname)
		assert 'Different number of HDUs' in caplog.text

	# Delte a single header keyword:
	caplog.clear()
	with fits.open(fname, mode='readonly') as hdu:
		del hdu['LIGHTCURVE'].header['TIMEDEL']
		assert not check_fits_changes(fname, hdu) # Missing HDU in modified file
		assert not check_fits_changes(hdu, fname) # Extra HDU in modified file
		# Modified header, it is allowed to change, but not to be missing:
		assert not check_fits_changes(fname, hdu, allow_header_value_changes=['TIMEDEL'])

	# Insert an extra/new keyword:
	caplog.clear()
	with fits.open(fname, mode='readonly') as hdu:
		hdu['LIGHTCURVE'].header['NEWKEY'] = 3.14
		assert not check_fits_changes(fname, hdu) # Key keyword in modified file
		# Modified header, and it is allowed to change:
		assert check_fits_changes(fname, hdu, allow_header_value_changes=['NEWKEY'])

	# Change a single header keyword value:
	caplog.clear()
	with fits.open(fname, mode='readonly') as hdu:
		hdu['LIGHTCURVE'].header['TIMEDEL'] += 3.14
		assert not check_fits_changes(fname, hdu) # Modified header in modified file
		# Modified header, and it is allowed to change:
		assert check_fits_changes(fname, hdu, allow_header_value_changes=['TIMEDEL'])

	# Change a single header keyword comments:
	caplog.clear()
	with fits.open(fname, mode='readonly') as hdu:
		hdu['LIGHTCURVE'].header.comments['TIMEDEL'] = 'completey wrong'
		assert not check_fits_changes(fname, hdu) # Modified header in modified file
		# Modified header, and it is allowed to change:
		assert check_fits_changes(fname, hdu, allow_header_value_changes=['TIMEDEL'])

	# Delete a table column:
	#caplog.clear()
	#with fits.open(fname, mode='readonly') as hdu:
	#	del hdu['LIGHTCURVE'].data['TIMECORR']
	#	assert not check_fits_changes(fname, hdu) # Modified table in modified file
	#	assert not check_fits_changes(hdu, fname) # Modified table in original file

	# Change special Table keywords:
	caplog.clear()
	with fits.open(fname, mode='readonly') as hdu:
		hdu['ENSEMBLE'].header['TDISP2'] = 'E10.4'
		assert not check_fits_changes(fname, hdu) # Modified header in modified file
		assert 'Table header with different values' in caplog.text
		# Modified header, and it is allowed to change:
		assert check_fits_changes(fname, hdu, allow_header_value_changes=['TDISP2'])

	# Change special Table keywords:
	caplog.clear()
	with fits.open(fname, mode='readonly') as hdu:
		hdu['LIGHTCURVE'].header['TUNIT2'] = 'bananas'
		assert not check_fits_changes(fname, hdu) # Modified header in modified file
		assert 'Table header with different values' in caplog.text
		# Modified header, and it is allowed to change:
		assert check_fits_changes(fname, hdu, allow_header_value_changes=['TUNIT2'])

	# Modified data in table:
	caplog.clear()
	with fits.open(fname, mode='readonly') as hdu:
		hdu['LIGHTCURVE'].data['FLUX_RAW'][33] = 1.233556
		assert not check_fits_changes(fname, hdu) # Modified data in modified file
		assert 'Data has been changed' in caplog.text

	# Modified data in image:
	caplog.clear()
	with fits.open(fname, mode='readonly') as hdu:
		hdu['SUMIMAGE'].data[0, 0] = 1.233556
		assert not check_fits_changes(fname, hdu) # Modified data in modified file
		assert 'Data has been changed' in caplog.text

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
