#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of  dataval quality flags.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import conftest # noqa: F401
from dataval import DatavalQualityFlags

#INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')

#----------------------------------------------------------------------
def test_dataval_flags():
	"""
	Simple tests of dataval quality flags
	"""

	flag = DatavalQualityFlags.filter(DatavalQualityFlags.LowPTP, DatavalQualityFlags.LowPTP)
	assert not flag, "Flag should be FALSE"

#----------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
