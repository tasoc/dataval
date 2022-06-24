#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Status indicator of the status of the correction.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import enum

#--------------------------------------------------------------------------------------------------
class STATUS(enum.IntEnum):
	"""
	Status indicator of the status of the correction.
	"""
	UNKNOWN = 0 #: The status is unknown. The actual calculation has not started yet.
	STARTED = 6 #: The calculation has started, but not yet finished.
	OK = 1      #: Everything has gone well.
	ERROR = 2   #: Encountered a catastrophic error that I could not recover from.
	WARNING = 3 #: Something is a bit fishy. Maybe we should try again with a different algorithm?
	ABORT = 4   #: The calculation was aborted.
	SKIPPED = 5 #: The target was skipped because the algorithm found that to be the best solution.
