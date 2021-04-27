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

from variantworks.networks import ConsensusRNN, ConsensusCNN


def create_model(model,
                 input_feature_size,
                 num_output_logits,
                 gru_size,
                 gru_layers=None,
                 kernel_size=None,
                 is_training=True):
    """Return neural network to train.

    Args:
        model : Model architecture can be 'rnn' or 'cnn'
        input_feature_size : Length of input feature set
        num_output_logits : Number of output classes of classifier
        gru_size : Number of units in RNN
        gru_layers : Number of layers in RNN
        kernel_size : Kernel size for conv layers (only for 'cnn')
        is_training : True if the model is be used training, False for inferring
    Returns:
        Instance of ConsensusRNN or ConsensusCNN.
    """
    if model == 'rnn':
        model = ConsensusRNN(input_feature_size=input_feature_size,
                             num_output_logits=num_output_logits,
                             gru_size=gru_size,
                             gru_layers=gru_layers,
                             apply_softmax=not is_training)
    elif model == 'cnn':
        model = ConsensusCNN(input_feature_size=input_feature_size,
                             gru_size=gru_size,
                             kernel_size=kernel_size,
                             num_output_logits=num_output_logits,
                             apply_softmax=not is_training)
    return model
