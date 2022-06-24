#!/usr/bin/env python3
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

	qual = DatavalQualityFlags.LowPTP | DatavalQualityFlags.MagVsFluxHigh

	flag = DatavalQualityFlags.filter(DatavalQualityFlags.LowPTP, DatavalQualityFlags.LowPTP)
	assert not flag, "Flag should be FALSE"

	flag = DatavalQualityFlags.filter(DatavalQualityFlags.LowPTP)
	assert flag, "Flag should be TRUE"

	flag = DatavalQualityFlags.filter(qual, DatavalQualityFlags.LowPTP)
	assert not flag, "Flag should be FALSE"

	flag = DatavalQualityFlags.filter(qual, DatavalQualityFlags.LowRMS)
	assert flag, "Flag should be TRUE"

	assert DatavalQualityFlags.LowPTP.binary_repr == '00000000000000000000000100000000'
	assert qual.binary_repr == '00000000000000000000000100000001'

#----------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
