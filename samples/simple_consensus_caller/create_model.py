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
"""Common model creation module."""

from variantworks.networks import ConsensusRNN


def create_rnn_model(input_feature_size,
                     num_output_logits,
                     gru_size,
                     gru_layers):
    """Return neural network to train."""
    # Neural Network
    rnn = ConsensusRNN(input_feature_size=input_feature_size,
                       num_output_logits=num_output_logits,
                       gru_size=gru_size,
                       gru_layers=gru_layers,
                       apply_softmax=True)

    return rnn
