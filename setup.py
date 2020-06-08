#!/usr/bin/env python3

#
# Copyright 2020 NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


"""Python setuptools setup."""

import os

from setuptools import find_packages, setup


def get_verified_absolute_path(path):
    """Verify and return absolute path of argument.

    Args:
        path : Relative/absolute path

    Returns:
        Absolute path
    """
    installed_path = os.path.abspath(path)
    if not os.path.exists(installed_path):
        raise RuntimeError("No valid path for requested component exists")
    return installed_path


def get_installation_requirments(file_path):
    """Parse pip requirements file.

    Args:
        file_path : path to pip requirements file

    Returns:
        list of requirement strings
    """
    with open(file_path, 'r') as file:
        requirements_file_content = \
            [line.strip() for line in file if
             line.strip() and not line.lstrip().startswith('#')]
    return requirements_file_content


# Get current dir (pyclaragenomics folder is copied into a temp directory
# created by pip)
current_dir = os.path.dirname(os.path.realpath(__file__))

# Classifiers for PyPI
pyaw_classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Natural Language :: English",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9"
]

setup(name='variantworks',
      version='0.1.0',
      description='NVIDIA genomics python libraries and utiliites',
      author='NVIDIA Corporation',
      url="https://github.com/clara-parabricks/VariantWorks",
      include_package_data=True,
      install_requires=[get_installation_requirments(
          get_verified_absolute_path(
              os.path.join(current_dir, 'requirements.txt')))
      ],
      packages=find_packages(where=current_dir,
                             include=['variantworks']),
      python_requires='>=3.7',
      long_description='Python libraries and utilities for manipulating '
                       'genomics data',
      classifiers=pyaw_classifiers,
      platforms=['any'],
      )
