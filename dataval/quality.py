#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handling of TESS data quality flags.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import enum
import numpy as np

#--------------------------------------------------------------------------------------------------
class QualityFlagsBase(enum.IntFlag):

	@classmethod # noqa: A003
	def filter(cls, quality, flags=None):
		"""
		Filter quality flags against a specific set of flags.

		Parameters:
			quality (integer or ndarray): Quality flags.
			flags (integer bitmask): Default=``TESSQualityFlags.DEFAULT_BITMASK``.

		Returns:
			ndarray: ``True`` if quality DOES NOT contain any of the specified ``flags``, ``False`` otherwise.
		"""
		if flags is None:
			flags = cls.DEFAULT_BITMASK
		return (quality & flags == 0)

	@property
	def binary_repr(self):
		return np.binary_repr(self.value, width=32)

#--------------------------------------------------------------------------------------------------
class DatavalQualityFlags(QualityFlagsBase):
	"""
	Quality flags indicating the status of data validation.
	"""
	MagVsFluxHigh = 1 #: Star has higher flux than given by magnitude relation
	MagVsFluxLow = 2 #: Star has lower flux than given by magnitude relation
	FluxFFIOverTPF = 4 #: Star has higher measured flux in FFIs than TPFs
	FluxTPFOverFFI = 8 #: Star has higher measured flux in TPFs than FFIs
	MinimalMask = 16 #: Star has minimum 4x4 mask
	SmallMask = 32 #: Star has smaller mask than general relation
	LargeMask = 64 #: Star has larger mask than general relation
	SmallStamp = 128 #: Smaller stamp than default
	LowPTP = 256 #: PTP lower than theoretical
	LowRMS = 512 #: RMS lower than theoretical
	InvalidContamination = 1024 #: Invalid contamination
	ContaminationHigh = 2048 #: Contamination high
	InvalidFlux = 4096 #: Invalid mean flux (not finite or negative)
	InvalidNoise = 8192

	# Default bitmask
	DEFAULT_BITMASK = (InvalidFlux | InvalidContamination | SmallMask | LargeMask | InvalidNoise)
