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

function check_convergence() {
  # Extract last evaluation loss
  LOSS_THRESHOLD=0.0001
  LAST_EVAL=$(tac "$1" | grep "Evaluation Loss"  | head -1)
  if [[ $(awk -v a="${LAST_EVAL##*:}" -v b="${LOSS_THRESHOLD}" 'BEGIN{print(a<b)}') != 1 ]]; then
    echo "ERROR: Evaluation loss ${1}" 1>&2
    exit 1
  fi
}

TEST_OUTPUT="./samples/simple_consensus_caller/test_output_$(date "+%Y-%m-%d_%H-%M-%S")"
mkdir "$TEST_OUTPUT"

python ./samples/simple_consensus_caller/pileup_hdf5_generator.py \
--single-dir ./samples/simple_consensus_caller/data/samples/1 \
-o "${TEST_OUTPUT}/one_sample.hdf" \
-t 4
echo "${TEST_OUTPUT}/one_sample.hdf was created"

python ./samples/simple_consensus_caller/pileup_hdf5_generator.py \
--data-dir ./samples/simple_consensus_caller/data/samples \
-o "${TEST_OUTPUT}/several_samples.hdf" \
-t 4
echo "${TEST_OUTPUT}/several_samples.hdf was created"

python ./samples/simple_consensus_caller/consensus_trainer.py \
--train-hdf "${TEST_OUTPUT}/several_samples.hdf" \
--eval-hdf "${TEST_OUTPUT}/several_samples.hdf" \
--model-dir "${TEST_OUTPUT}/model_1" \
--epochs 200 --lr 0.01 \
--model 'rnn' 2>&1 | tee "${TEST_OUTPUT}/output_model_1.txt"
check_convergence "${TEST_OUTPUT}/output_model_1.txt"
echo "consensus_trainer finished ${TEST_OUTPUT}/output_model_1.txt"

python ./samples/simple_consensus_caller/consensus_trainer.py \
--train-hdf "${TEST_OUTPUT}/one_sample.hdf" \
--eval-hdf "${TEST_OUTPUT}/one_sample.hdf" \
--model-dir "${TEST_OUTPUT}/model_2" \
--epochs 300 --lr 0.01 \
--model 'cnn' 2>&1 | tee "${TEST_OUTPUT}/output_model_2.txt"
check_convergence "${TEST_OUTPUT}/output_model_2.txt"
echo "consensus_trainer finished ${TEST_OUTPUT}/output_model_2.txt"

python ./samples/simple_consensus_caller/consensus_infer.py \
--infer-hdf "${TEST_OUTPUT}"/one_sample.hdf \
--model-dir "${TEST_OUTPUT}"/model_2 \
--model 'cnn' \
--out-file "${TEST_OUTPUT}"/consensus_inferred

echo "Simple Consensus Caller CI testing finished successfully"
