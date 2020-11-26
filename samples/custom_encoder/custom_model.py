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
"""A simple MLP model for the sample."""

import torch
import torch.nn as nn

from nemo.backends.pytorch.nm import TrainableNM
from nemo.utils.decorators import add_port_docs
from nemo.core.neural_types import NeuralType, LogitsType, VoidType
from nemo.core.neural_factory import DeviceType


class MLP(TrainableNM):
    """A Neural Module for an MLP."""

    @property
    @add_port_docs()
    def input_ports(self):
        """Return definitions of module input ports.

        Returns:
            Module input ports.
        """
        return {
            "encoding": NeuralType(('B', 'D'), VoidType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Return definitions of module output ports.

        Returns:
            Module output ports.
        """
        return {
            # Variant type
            'output_logit': NeuralType(('B', 'D'), LogitsType()),
        }

    def __init__(self, num_input_nodes, num_hidden_nodes, num_output_logits, apply_softmax=False):
        """Construct an AlexNet NeMo instance.

        Args:
            num_input_nodes : Number of input nodes.
            num_hidden_nodes : Size of hidden nodes.
            num_output_logits : Number of output logits of classifier.
            apply_softmax : Flag to optionally apply softmax to last layer's output.

        Returns:
            Instance of class.
        """
        super().__init__()
        self._num_input_nodes = num_input_nodes
        self._num_hidden_nodes = num_hidden_nodes
        self._num_output_logits = num_output_logits
        self._apply_softmax = apply_softmax

        self._fc1 = nn.Linear(self._num_input_nodes, self._num_hidden_nodes)
        self._relu = nn.ReLU()
        self._fc2 = nn.Linear(self._num_hidden_nodes, self._num_output_logits)
        self._softmax = nn.Softmax(dim=1)

        self._device = torch.device(
            "cuda" if self.placement == DeviceType.GPU else "cpu")
        self.to(self._device)

    def forward(self, encoding):
        """Abstract function to run the network.

        Args:
            encoding : Input to run network on.

        Returns:
            Output of forward pass.
        """
        output = self._fc1(encoding)
        output = self._relu(output)
        output = self._fc2(output)
        if self._apply_softmax:
            output = self._softmax(output)
        return output
