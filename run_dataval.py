#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run TASOC Data Validation Pipeline.

.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import argparse
import logging
from dataval import DataValidation

#--------------------------------------------------------------------------------------------------
def main():
	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Run Data Validation pipeline.')
	parser.add_argument('-c', '--corrected', help='Use corrected or raw values.', action='store_true')
	parser.add_argument('-m', '--method', help='Corrector method to use.', action='append', default=[], choices=('basic', 'pixvsmag', 'contam', 'mag2flux', 'stamp', 'noise', 'noise_compare', 'magdist', 'waittime'))
	parser.add_argument('-e', '--ext', help='Extension of plots.', default='png', choices=('png','eps','pdf'))
	parser.add_argument('-s', '--show', help='Show plots.', action='store_true')
	parser.add_argument('-v', '--validate', help='Compute validation (only run is method is "all").', action='store_true')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('-cbs', '--colorbysector', help='Color by sector', action='store_true')
	parser.add_argument('-sn', '--sysnoise', type=float, help='systematic noise level for noise plot.', nargs='?', default=5.0)
	parser.add_argument('-o', '--output', type=str, help='Directory in which to place output if several input folders are given.', nargs='?', default=None)
	parser.add_argument('input_folders', type=str, help='Directory to create catalog files in.', nargs='+')
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
	with DataValidation(args.input_folders, output_folder=args.output, corr=args.corrected,
		validate=args.validate, colorbysector=args.colorbysector,
		showplots=args.show, ext=args.ext, sysnoise=args.sysnoise) as dataval:

		if 'basic' in args.method:
			dataval.basic()
		if 'mag2flux' in args.method:
			dataval.plot_mag2flux()
		if 'pixvsmag' in args.method:
			dataval.plot_pixinaperture()
		if 'stamp' in args.method:
			dataval.plot_stamp()
		if 'magdist' in args.method:
			dataval.plot_mag_dist()
		if 'noise' in args.method:
			dataval.plot_noise()
		if 'noise_compare' in args.method:
			dataval.compare_noise()
		if 'contam' in args.method:
			dataval.plot_contam()
		if 'magdistoverlap' in args.method:
			dataval.plot_mag_dist_overlap()
		if 'waittime' in args.method:
			dataval.plot_waittime()

		# Run validation
		if not args.method:
			dataval.Validations()

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	main()
