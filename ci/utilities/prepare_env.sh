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

# Cleanup local git
git clean -xdf

logger "Get env..."
env

logger "Activate anaconda enviroment..."
CONDA_NEW_ACTIVATION_CMD_VERSION="4.4"
CONDA_VERSION=$(conda --version | awk '{print $2}')
if [ "$CONDA_NEW_ACTIVATION_CMD_VERSION" == "$(echo -e "$CONDA_VERSION\n$CONDA_NEW_ACTIVATION_CMD_VERSION" | sort -V | head -1)" ]; then
  logger "Version is higer than ${CONDA_NEW_ACTIVATION_CMD_VERSION}, using conda activate"
  source /conda/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME}"
else
  logger "Version is lower than ${CONDA_NEW_ACTIVATION_CMD_VERSION}, using source activate"
  source activate "${CONDA_ENV_NAME}"
fi
conda info --envs

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

logger "Check Python version..."
python --version


