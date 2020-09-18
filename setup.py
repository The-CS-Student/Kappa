#!/usr/bin/env python

from setuptools import setup

setup(name='kappa',
      version='1.0',
      # list folders, not files
      packages=['kappa',],
      scripts=['kappa/regression.py',
               'kappa/metrics.py',
               'kappa/error.py',
               'kappa/preprocessing.py']
      )