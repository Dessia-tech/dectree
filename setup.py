#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 12:21:31 2018

@author: jezequel
"""
from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

# Import dtree
setup(name='dtree',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      description="Tools for decision trees",
      long_description='',
      keywords='',
      url='',
#      cmdclass['register']=None,
      author='Steven Masfaraud',
      author_email='masfaraud@dessia.tech',
      packages=['dtree'],
      install_requires=['networkx'])