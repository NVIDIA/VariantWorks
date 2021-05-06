#!/bin/bash

#
# Copyright 2020-2021 NVIDIA CORPORATION.
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

# Utility function which extracts the last logged loss value from an output file and compares it to a threshold value
function check_loss_convergence() {
  LOSS_THRESHOLD=${2}
  LAST_EVAL=$(tac "${1}" | grep "Evaluation Loss"  | head -1)
  if [[ $(awk -v a="${LAST_EVAL##*:}" -v b="${LOSS_THRESHOLD}" 'BEGIN{print(a<b)}') != 1 ]]; then
    echo "ERROR: Evaluation loss ${1}" 1>&2
    exit 1
  fi
}
