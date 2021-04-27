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

python ./samples/simple_consensus_caller/pileup_hdf5_generator.py \
--single-dir ./samples/simple_consensus_caller/data/samples/1 \
-o infer_one.hdf \
-t 4

python ./samples/simple_consensus_caller/pileup_hdf5_generator.py \
--data-dir ./samples/simple_consensus_caller/data/samples \
-o train_several.hdf \
-t 4

python ./samples/simple_consensus_caller/consensus_trainer.py \
--train-hdf ./train_several.hdf \
--eval-hdf ./train_several.hdf \
--model-dir ./model_1

python ./samples/simple_consensus_caller/consensus_infer.py \
--infer-hdf ./infer_one.hdf \
--model-dir ./model_1 \
--out-file ./consensus_inferred