#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import sqlite3
import shutil
from contextlib import closing

if __name__ == '__main__':

	input_sqlite = r'D:\TASOC_DR04_INT\S02\todo.sqlite'

	print("Copying input file...")
	shutil.copy(input_sqlite, '../tests/input/with_corr/')

	with closing(sqlite3.connect('../tests/input/with_corr/todo.sqlite')) as conn:
		cursor = conn.cursor()
		cursor.execute("PRAGMA foreign_keys=ON;")

		# List of priorities to include
		pri_total = set()

		print("Finding random samples from file...")
		for datasource in ("datasource='ffi'", "datasource!='ffi'"):
			for b in np.arange(0, 18, 1):
				print(b)

				cursor.execute("SELECT priority FROM todolist WHERE %s AND tmag BETWEEN %s AND %s ORDER BY RANDOM() LIMIT 5000;" % (
					datasource,
					b,
					b+1
				))

				pri = {str(row[0]) for row in cursor.fetchall()}
				pri_total |= pri

		print("Cleaning file...")

		# Delete old datavalidation tables:
		cursor.execute("DROP TABLE IF EXISTS datavalidation;")
		cursor.execute("DROP TABLE IF EXISTS datavalidation_raw;")
		cursor.execute("DROP TABLE IF EXISTS datavalidation_corr;")
		conn.commit()

		# Delete photometry_skipped table, since it is not needed in dataval,
		# and it takes up a lot of space:
		# TODO: But that means we can not test it currently!
		cursor.execute("DROP TABLE IF EXISTS photometry_skipped;")
		conn.commit()

		# Delete all targets that were not picked out above:
		cursor.execute("DELETE FROM todolist WHERE priority NOT IN (%s);" % (
			','.join(list(pri_total))
		))
		conn.commit()

		# Clean the file, to recover the deleted space:
		cursor.execute("ANALYZE;")
		conn.commit()
		conn.isolation_level = None
		cursor.execute("VACUUM;")
		cursor.close()

	# Copy the file to the ONLY_RAW directory:
	print("Copy file for RAW...")
	shutil.copy('../tests/input/with_corr/todo.sqlite', '../tests/input/only_raw/')

	# Open the file and delete the "diagnostics_corr" table.
	print("Cleaning file for RAW...")
	with closing(sqlite3.connect('../tests/input/only_raw/todo.sqlite')) as conn:
		cursor = conn.cursor()
		cursor.execute("PRAGMA foreign_keys=ON;")

		# Clean table for raw:
		cursor.execute("UPDATE todolist SET corr_status=NULL;") # TODO: Ideally we should drop the column
		cursor.execute("DROP TABLE IF EXISTS diagnostics_corr;")
		conn.commit()

		# Clean the file, to recover the deleted space:
		cursor.execute("ANALYZE;")
		conn.commit()
		conn.isolation_level = None
		cursor.execute("VACUUM;")
		cursor.close()