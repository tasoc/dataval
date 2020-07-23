********************************
The TASOC Data Validation module
********************************
.. image:: https://travis-ci.org/tasoc/dataval.svg?branch=devel
    :target: https://travis-ci.org/tasoc/dataval
.. image:: https://img.shields.io/codecov/c/github/tasoc/dataval/devel
    :target: https://codecov.io/github/tasoc/dataval
.. image:: https://hitsofcode.com/github/tasoc/dataval?branch=devel
    :alt: Hits-of-Code
    :target: https://hitsofcode.com/view/github/tasoc/dataval?branch=devel
.. image:: https://img.shields.io/github/license/tasoc/dataval.svg
    :alt: license
    :target: https://github.com/tasoc/dataval/blob/devel/LICENSE

This module provides the data-validation setup for the TESS Asteroseismic Science Operations Center (TASOC) pipeline.

The code is available through our GitHub organization (https://github.com/tasoc/dataval) and full documentation for this code can be found on https://tasoc.dk/code/.

Installation instructions
=========================
* Start by making sure that you have `Git Large File Storage (LFS) <https://git-lfs.github.com/>`_ installed. You can verify that is installed by running the command:

  >>> git lfs version

* Go to the directory where you want the Python code to be installed and simply download it or clone it via *git* as::

  >>> git clone https://github.com/tasoc/dataval.git .

* All dependencies can be installed using the following command. It is recommended to do this in a dedicated `virtualenv <https://virtualenv.pypa.io/en/stable/>`_ or similar:

  >>> pip install -r requirements.txt

How to run tests
================
You can test your installation by going to the root directory where you cloned the repository and run the command::

>>> pytest

Running the data validation
===========================
Soon to come...
