#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 12:33:02 2019

@author: mikkelnl
"""

import six
import logging
import os
import warnings
import numpy as np
from bottleneck import allnan
import matplotlib
matplotlib.use('Agg', warn=False)
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from astropy.visualization import (PercentileInterval, ImageNormalize,
								   SqrtStretch, LogStretch, LinearStretch)


# =============================================================================
# 
# =============================================================================


path = '/media/mikkelnl/Elements/TESS/S01_tests/lightcurves-combined/'

star = '394046358'

starz = star.zfill(11)


folder2 = os.path.join(path, starz[0:6])

file = [f for f in os.listdir(folder2) if star in f]

print(file)