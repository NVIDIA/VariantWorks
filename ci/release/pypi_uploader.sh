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

set -e

# Skip upload if CI is executed locally
if [[ ${RUNNING_CI_LOCALLY} = true  ]]; then
    echo "Skipping PyPi upload - running ci locally"
    return 0
fi

# Skip upload if the merging branch is not master
if [ "${COMMIT_HASH}" != "master" ]; then
    echo "Skipping PyPI upload - merge branch is not master"
    return 0
fi

for f in "${WORKSPACE}"/variantworks_wheel/*.whl; do
    if [ ! -e "${f}" ]; then
        echo "VariantWorks wheel file does not exist"
        exit 1
    else
        conda install -c conda-forge twine
        echo "Uploading file name ${f} to PYPI"
        # Perform Upload
        python3 -m twine upload --skip-existing "${WORKSPACE}"/variantworks_wheel/*
    fi
done
