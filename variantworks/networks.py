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
"""Common implementations of networks as Neural Modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.backends.pytorch.nm import TrainableNM
from nemo.utils.decorators import add_port_docs
from nemo.core.neural_types import NeuralType, ChannelType, LogitsType
from nemo.core.neural_factory import DeviceType


class AlexNet(TrainableNM):
    """A Neural Module for AlexNet."""

    @property
    @add_port_docs()
    def input_ports(self):
        """Return definitions of module input ports.

        Returns:
            Module input ports.
        """
        return {
            "encoding": NeuralType(('B', 'C', 'H', 'W'), ChannelType()),
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

    def __init__(self, num_input_channels, num_output_logits):
        """Construct an AlexNet NeMo instance.

        Args:
            num_input_channels : Number of input channels in image.
            num_output_logits : Number of output logits of classifier.

        Returns:
            Instance of class.
        """
        super().__init__()
        self.num_output_logits = num_output_logits
        self.num_input_channels = num_input_channels

        self.features = nn.Sequential(
            nn.Conv2d(self.num_input_channels, 64,
                      kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.common_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(4096, self.num_output_logits)

        self._device = torch.device(
            "cuda" if self.placement == DeviceType.GPU else "cpu")
        self.to(self._device)

    def forward(self, encoding):
        """Abstract function to run the network.

        Args:
            encoding : Input image to run network on.

        Returns:
            Output of forward pass.
        """
        encoding = self.features(encoding)
        encoding = self.avgpool(encoding)
        encoding = torch.flatten(encoding, 1)
        encoding = self.common_classifier(encoding)
        vz = self.classifier(encoding)
        return vz


class ConsensusRNN(TrainableNM):
    """A Neural Module for training a Consensus RNN."""

    @property
    @add_port_docs()
    def input_ports(self):
        """Return definitions of module input ports.

        Returns:
            Module input ports.
        """
        return {
            "encoding": NeuralType(('B', 'W', 'C'), ChannelType()),
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
            'output_logit': NeuralType(('B', 'W', 'D'), LogitsType()),
        }

    def __init__(self, input_feature_size, num_output_logits,
                 gru_size=128, gru_layers=2, apply_softmax=False):
        """Construct an Consensus RNN NeMo instance.

        Args:
            input_feature_size : Length of input feature set.
            num_output_logits : Number of output classes of classifier.
            gru_size : Number of units in RNN
            gru_layers : Number of layers in RNN
            apply_softmax : Apply softmax to the output of the classifier.

        Returns:
            Instance of class.
        """
        super().__init__()
        self.num_output_logits = num_output_logits
        self.apply_softmax = apply_softmax
        self.gru_size = gru_size
        self.gru_layers = gru_layers

        self.gru = nn.GRU(input_feature_size, gru_size, gru_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(2 * gru_size, self.num_output_logits)  # 2* for bidirectional

        self._device = torch.device(
            "cuda" if self.placement == DeviceType.GPU else "cpu")
        self.to(self._device)

    def forward(self, encoding):
        """Abstract function to run the network.

        Args:
            encoding : Input sequence to run network on.

        Returns:
            Output of forward pass.
        """
        encoding, h_n = self.gru(encoding)
        encoding = self.classifier(encoding)
        if self.apply_softmax:
            encoding = F.softmax(encoding, dim=2)
        return encoding
