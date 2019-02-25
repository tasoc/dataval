#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Handling of TESS data quality flags.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
import numpy as np

#------------------------------------------------------------------------------
class QualityFlagsBase(object):

	@classmethod
	def decode(cls, quality):
		"""
		Converts a QUALITY value into a list of human-readable strings.
		This function takes the QUALITY bitstring that can be found for each
		cadence in TESS data files and converts into a list of human-readable
		strings explaining the flags raised (if any).

		Parameters:
			quality (int): Value from the 'QUALITY' column of a TESS data file.

		Returns:
			list of str: List of human-readable strings giving a short
			             description of the quality flags raised.
						 Returns an empty list if no flags raised.
		"""
		result = []
		for flag in cls.STRINGS.keys():
			if quality & flag != 0:
				result.append(cls.STRINGS[flag])
		return result

	@classmethod
	def filter(cls, quality, flags=None):
		"""
		Filter quality flags against a specific set of flags.

		Parameters:
			quality (integer or ndarray): Quality flags.
			flags (integer bitmask): Default=``TESSQualityFlags.DEFAULT_BITMASK``.

		Returns:
			ndarray: ``True`` if quality DOES NOT contain any of the specified ``flags``, ``False`` otherwise.

		"""
		if flags is None: flags = cls.DEFAULT_BITMASK
		return (quality & flags == 0)

	@staticmethod
	def binary_repr(quality):
		"""
		Binary representation of the quality flag.

		Parameters:
			quality (int or ndarray): Quality flag.

		Returns:
			string: Binary representation of quality flag. String will be 32 characters long.

		"""
		if isinstance(quality, (np.ndarray, list, tuple)):
			return np.array([np.binary_repr(q, width=32) for q in quality])
		else:
			return np.binary_repr(quality, width=32)

#------------------------------------------------------------------------------
class DatavalQualityFlags(QualityFlagsBase):

	MagVsFluxHigh = 1
	MagVsFluxLow = 2
	FluxFFIOverTPF = 4
	FluxTPFOverFFI = 8
	MinimalMask = 16
	SmallMask = 32
	LargeMask = 64
	SmallStamp = 128
	LowPTP = 256
	LowRMS = 512
	ContaminationOne = 1024
	ContaminationHigh = 2048
	NegativeFlux = 4096

	# Default bitmask
	DEFAULT_BITMASK = (NegativeFlux)

	# Pretty string descriptions for each flag
	STRINGS = {
		MagVsFluxHigh: "Star has higher flux than given by magnitude relation",
		MagVsFluxLow: "Star has lower flux than given by magnitude relation",
		FluxFFIOverTPF: "Star has higher measured flux in FFIs than TPFs",
		FluxTPFOverFFI: "Star has higher measured flux in TPFs than FFIs",
		MinimalMask: "Star has minimum 4x4 mask",
		SmallMask: "Star has smaller mask than general relation",
		LargeMask: "Star has larger mask than general relation",
		SmallStamp: "Smaller stamp than default",
		LowPTP: "PTP lower than theoretical",
		LowRMS: "RMS lower than theoretical",
		ContaminationOne: "Contamination over 1",
		ContaminationHigh: "Contamination high",
		NegativeFlux: "Negative mean flux"
	}