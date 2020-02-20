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
from dataval import DataValidation, DatavalQualityFlags

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("inp,corr", [
	pytest.param('does-not-exist', False, marks=pytest.mark.xfail(raises=FileNotFoundError)),
	('only_raw', False),
	pytest.param('only_raw', True, marks=pytest.mark.xfail(raises=ValueError)),
	('with_corr', False),
	('with_corr', True),
])
def test_dataval(PRIVATE_INPUT_DIR, inp, corr):
	"""
	Try to run DataValidation on different input
	"""

	test_dir = os.path.join(PRIVATE_INPUT_DIR, inp)

	# On this input file it should be possible to run with both
	with DataValidation([test_dir], corr=corr) as dataval:
		# Run validation:
		dataval.Validations()

		# Count the number of stars in to TODOLIST:
		dataval.cursor.execute("SELECT COUNT(*) FROM todolist;")
		num_todo = int(dataval.cursor.fetchone()[0])
		print(num_todo)

		# Count the number of stars in the resulting DATAVAL table:
		dataval.cursor.execute("SELECT COUNT(*) FROM " + dataval.dataval_table + ";")
		num_dataval = int(dataval.cursor.fetchone()[0])
		print(num_dataval)
		assert num_dataval == num_todo, "Not the same number of targets in TODOLIST and DATAVAL-table"

		# Check that the brigh Halo-target has not been rejected because of contamination:
		dataval.cursor.execute("SELECT * FROM " + dataval.dataval_table + " WHERE priority=3;")
		row = dataval.cursor.fetchone()
		print(dict(row))
		dataval_invalid = DatavalQualityFlags.InvalidContamination | DatavalQualityFlags.ContaminationHigh
		assert int(row['dataval']) & dataval_invalid == 0, "Contamination DATAVAL set for Halo target"

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
