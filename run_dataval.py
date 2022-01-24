#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run TASOC Data Validation Pipeline.

.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import argparse
import logging
import sys
import dataval
from dataval.plots import plt
from dataval.utilities import TqdmLoggingHandler

#--------------------------------------------------------------------------------------------------
def main():

	# All available methods:
	methods = [
		'basic',
		'cleanup',
		'pixvsmag',
		'stampsize',
		'contam',
		'mag2flux',
		'noise',
		'noise_compare',
		'magdist',
		'calctime',
		'waittime',
		'haloswitch',
		'camera_overlap',
		'sumimage',
		'duplicate_headers',
	]

	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Run Data Validation pipeline.')
	parser.add_argument('-c', '--corrected', help='Use corrected or raw values.', action='store_true')
	parser.add_argument('-v', '--validate', help='Store validation.', action='store_true')
	parser.add_argument('-m', '--method', help='Corrector method to run.', action='append', default=[], choices=methods)
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')

	parser.add_argument('--output', type=str, help='Directory in which to place output.', nargs='?', default=None)
	parser.add_argument('todo_file', type=str, help='TODO-file or directory to load from.')

	group = parser.add_argument_group('Plotting settings')
	group.add_argument('-e', '--ext', help='Extension of plots.', default='png', choices=('png','eps','pdf'))
	group.add_argument('-s', '--show', help='Show plots?', action='store_true')
	group.add_argument('-cbs', '--colorbysector', help='Color by sector.', action='store_true')

	group = parser.add_argument_group('Noise model settings')
	group.add_argument('-sn', '--sysnoise', type=float, help='Systematic noise level for noise model.', nargs='?', default=5.0)

	args = parser.parse_args()

	# Set logging level:
	logging_level = logging.INFO
	if args.quiet:
		logging_level = logging.WARNING
	elif args.debug:
		logging_level = logging.DEBUG

	# Setup logging:
	formatter = logging.Formatter('%(asctime)s - %(levelname)-7s - %(funcName)-10.10s - %(message)s')
	console = TqdmLoggingHandler()
	console.setFormatter(formatter)
	logger = logging.getLogger('dataval')
	logger.addHandler(console)
	logger.setLevel(logging_level)

	# Create DataValidation object:
	with dataval.DataValidation(args.todo_file, output_folder=args.output, corr=args.corrected,
		validate=args.validate, colorbysector=args.colorbysector,
		showplots=args.show, ext=args.ext, sysnoise=args.sysnoise) as dval:

		# Run specific methods:
		if 'cleanup' in args.method:
			dval.cleanup()
		if 'basic' in args.method:
			dval.basic()
		if 'mag2flux' in args.method:
			dval.mag2flux()
		if 'pixvsmag' in args.method:
			dval.pixinaperture()
		if 'stampsize' in args.method:
			dval.stampsize()
		if 'magdist' in args.method:
			dval.mag_dist()
		if 'noise' in args.method:
			dval.noise_metrics()
		if 'noise_compare' in args.method:
			dval.compare_noise()
		if 'contam' in args.method:
			dval.contam()
		if 'magdistoverlap' in args.method:
			dval.plot_mag_dist_overlap()
		if 'calctime' in args.method:
			dval.calctime()
			dval.calctime_corrections()
		if 'waittime' in args.method:
			dval.waittime()
		if 'haloswitch' in args.method:
			dval.haloswitch()
		if 'camera_overlap' in args.method:
			dval.camera_overlap()

		# Special methods:
		if 'sumimage' in args.method:
			dataval.special.check_sumimage(dval)
		if 'duplicate_headers' in args.method:
			dataval.special.check_duplicate_headers(dval)

		# Run validation
		if not args.method:
			dval.validate()

		# Get the number of logs (errors, warnings, info) issued during the validations:
		logcounts = dval.logcounts

		# If we were asked to show, actully show figures:
		if dval.show:
			plt.show(block=True)

	# Check the number of errors or warnings issued, and convert these to a return-code:
	if logcounts.get('ERROR', 0) > 0 or logcounts.get('CRITICAL', 0) > 0:
		return 4
	elif logcounts.get('WARNING', 0) > 0:
		return 3
	return 0

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	returncode = main()
	sys.exit(returncode)
