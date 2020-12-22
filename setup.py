#!/usr/bin/env python

from setuptools import setup, find_packages
setup(
    name='gpder',
    version='0.1.0',
    description="Package for calculating gaussian process with derivative observations.",
    author="Alexis Romero",
    author_email='alexir2@uci.edu',
    classifiers=['license :: OSI Approved :: MIT License',
                   'Programming Language :: Python :: 3.6'],
    license="MIT license",
    include_package_data=True
)
