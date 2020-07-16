#!/usr/bin/env python
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

#--------------------------------------------------------------------------------------------------
def main():

	# All available methods:
	methods = [
		'basic',
		'cleanup',
		'pixvsmag',
		'contam',
		'mag2flux',
		'stamp',
		'noise',
		'noise_compare',
		'magdist',
		'waittime',
		'haloswitch',
		'sumimage',
		'camera_overlap'
	]

	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Run Data Validation pipeline.')
	parser.add_argument('-c', '--corrected', help='Use corrected or raw values.', action='store_true')
	parser.add_argument('-v', '--validate', help='Store validation.', action='store_true')
	parser.add_argument('-m', '--method', help='Corrector method to run.', action='append', default=[], choices=methods)
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')

	parser.add_argument('-o', '--output', type=str, help='Directory in which to place output if several input folders are given.', nargs='?', default=None)
	parser.add_argument('input_folders', type=str, help='Directory to load from.', nargs='+')

	group = parser.add_argument_group('Plotting settings')
	group.add_argument('-e', '--ext', help='Extension of plots.', default='png', choices=('png','eps','pdf'))
	group.add_argument('-s', '--show', help='Show plots.', action='store_true')
	group.add_argument('-cbs', '--colorbysector', help='Color by sector.', action='store_true')

	group = parser.add_argument_group('Noise model settings')
	group.add_argument('-sn', '--sysnoise', type=float, help='Systematic noise level for noise model.', nargs='?', default=5.0)

	args = parser.parse_args()

	if args.output is None and len(args.input_folders) > 1:
		parser.error("Please specify an output directory!")

	# Set logging level:
	logging_level = logging.INFO
	if args.quiet:
		logging_level = logging.WARNING
	elif args.debug:
		logging_level = logging.DEBUG

	# Setup logging:
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	console = logging.StreamHandler()
	console.setFormatter(formatter)
	logger = logging.getLogger('dataval')
	logger.addHandler(console)
	logger.setLevel(logging_level)

	# Create DataValidation object:
	with dataval.DataValidation(args.input_folders, output_folder=args.output, corr=args.corrected,
		validate=args.validate, colorbysector=args.colorbysector,
		showplots=args.show, ext=args.ext, sysnoise=args.sysnoise) as dval:

		if 'cleanup' in args.method:
			dval.cleanup()
		if 'basic' in args.method:
			dval.basic()
		if 'mag2flux' in args.method:
			dval.plot_mag2flux()
		if 'pixvsmag' in args.method:
			dval.plot_pixinaperture()
		if 'stamp' in args.method:
			dval.plot_stamp()
		if 'magdist' in args.method:
			dval.plot_mag_dist()
		if 'noise' in args.method:
			dval.plot_noise()
		if 'noise_compare' in args.method:
			dval.compare_noise()
		if 'contam' in args.method:
			dval.plot_contam()
		if 'magdistoverlap' in args.method:
			dval.plot_mag_dist_overlap()
		if 'waittime' in args.method:
			dval.plot_waittime()
		if 'haloswitch' in args.method:
			dval.plot_haloswitch()
		if 'camera_overlap' in args.method:
			dval.camera_overlap()

		# Special methods:
		if 'sumimage' in args.method:
			dataval.special.check_sumimage(dval)

		# Run validation
		if not args.method:
			dval.Validations()

		# Get the number of logs (errors, warnings, info) issued during the validations:
		logcounts = dval.logcounts

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
