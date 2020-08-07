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

# Ignore errors and set path
set -e

# Logger function for build status output
START_TIME=$(date +%s)
. ci/utilities/logger.sh

################################################################################
# Init
################################################################################

PATH=/conda/bin:$PATH

# Set home to the job's workspace
export HOME=$WORKSPACE

cd "${WORKSPACE}"

source ./ci/utilities/prepare_env.sh "${WORKSPACE}"

################################################################################
# SDK style check
################################################################################

# Run copyright header check
logger "Run Copyright header check..."
./ci/checks/check_copyright.py

logger "Run Python formatting check..."
python -m pip install -r ./python-style-requirements.txt
source ./style_check

logger "Run documentation generation..."
./docs/generate-html-docs.sh
