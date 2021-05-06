#!/bin/bash

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

################################################################################
# VariantWorks CPU/GPU conda build script for CI
################################################################################

set -e

START_TIME=$(date +%s)

export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
# Set home to the job's workspace
export HOME=$WORKSPACE
cd "${WORKSPACE}"

################################################################################
# Init
################################################################################

source ci/utilities/logger.sh

logger "Calling prep-init-env"
source ci/utilities/prepare_env.sh

################################################################################
# VariantWorks tests
################################################################################

logger "Install VariantWorks external dependencies"
python -m pip install -r requirements.txt

logger "Install varaintworks in editable mode"
python -m pip install -e .

logger "Test VariantWorks"
source ci/tests/test_variantworks.sh

logger "Remove variantworks (editable mode)"
pip uninstall -y variantworks

################################################################################
# Create & Test Wheel Package for VariantWorks
################################################################################
logger "Create Wheel package for VariantWorks"
if [ "${COMMIT_HASH}" != "master" ]; then
  python ci/release/update_configuration.py --configuration_file ./setup.cfg --append-nightly-version-suffix
fi
python3 -m pip wheel . --global-option sdist --wheel-dir "${WORKSPACE}"/variantworks_wheel --no-deps

logger "Insalling VariantWorks from wheel..."
pip install --ignore-installed "${WORKSPACE}"/variantworks_wheel/*

logger "Test VariantWorks"
source ci/tests/test_variantworks.sh

################################################################################
# Upload VariantWorks to PyPI
################################################################################
logger "Upload Wheel to PyPI..."
source ci/release/pypi_uploader.sh

logger "Done"
