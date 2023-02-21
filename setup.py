#! /usr/bin/env python
#
# Copyright (C) 2007-2009 Cournapeau David <cournape@gmail.com>
#               2010 Fabian Pedregosa <fabian.pedregosa@inria.fr>
# License: 3-clause BSD

from setuptools import setup

from Cython.Build import cythonize
import numpy as np

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="purf",
    version="0.0.1",
    author="Renee Ti Chou",
    author_email="rtchou3@gmail.com",
    description="Positive-unlabeled random forest",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://gitlab.umiacs.umd.edu/rchou/purf",
    packages=[
            'purf', 'purf.pu_ensemble', 'purf.pu_tree'
    ],
    ext_modules=cythonize(["purf/pu_tree/_pu_criterion.pyx"],
                          compiler_directives={'language_level' : "3"}),
    include_dirs=np.get_include(),
    install_requires=[
        "scikit-learn>=0.22"
    ]
)

