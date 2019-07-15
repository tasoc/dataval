#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import argparse
import logging
from dataval import DataValidation

#------------------------------------------------------------------------------
def main():

	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Run Data Validation pipeline.')
	parser.add_argument('-m', '--method', help='Corrector method to use.', default='all', choices=('pixvsmag', 'contam', 'mag2flux', 'stamp', 'noise', 'magdist'))
	parser.add_argument('-c', '--corrected', help='Use corrected or raw values.', action='store_true')
	parser.add_argument('-e', '--ext', help='Extension of plots.', default='png', choices=('png', 'eps'))
	parser.add_argument('-s', '--show', help='Show plots.', action='store_true')
	parser.add_argument('-v', '--validate', help='Compute validation (only run is method is "all").', action='store_true')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('-cbs', '--colorbysector', help='Color by sector', action='store_true')
	parser.add_argument('-sn', '--sysnoise', type=float, help='systematic noise level for noise plot.', nargs='?', default=5.0)
	parser.add_argument('input_folders', type=str, help='Directory to create catalog files in.', nargs='+')
	parser.add_argument('output_folder', type=str, help='Directory in which to place output if several input folders are given.', nargs='?', default=None)
	args = parser.parse_args()

	if args.output_folder is None and len(args.input_folders) > 1:
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
	logger = logging.getLogger(__name__)
	logger.addHandler(console)
	logger.setLevel(logging_level)
	logger_parent = logging.getLogger('dataval')
	logger_parent.addHandler(console)
	logger_parent.setLevel(logging_level)

	logger.info("Loading input data from '%s'", args.input_folders)
	logger.info("Putting output data in '%s'", args.output_folder)

	# Create DataValidation object:
	with DataValidation(args.input_folders, output_folder=args.output_folder,
		validate=args.validate, method=args.method, colorbysector=args.colorbysector,
		showplots=args.show, ext=args.ext, sysnoise=args.sysnoise, corr=args.corrected) as dataval:

		# Run validation
		dataval.Validations()

#------------------------------------------------------------------------------
if __name__ == '__main__':
	main()