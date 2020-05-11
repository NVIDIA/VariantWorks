import torch
import torch.nn as nn

from nemo.backends.pytorch.nm import TrainableNM
from nemo.utils.decorators import add_port_docs
from nemo.core.neural_types import *
from nemo.core.neural_factory import DeviceType

from claragenomics.variantworks.neural_types import VariantEncodingType

class AlexNet(TrainableNM):
    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            "encoding": NeuralType(('B', 'C', 'H', 'W'), VariantEncodingType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            'log_probs_vz': NeuralType(('B', 'D'), LogitsType()), # Variant type
        }

    def __init__(self, num_input_channels, num_vz):
        super().__init__()
        self.num_vz = num_vz
        self.num_input_channels = num_input_channels

        self.features = nn.Sequential(
            nn.Conv2d(self.num_input_channels, 64, kernel_size=11, stride=4, padding=2),
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
        self.vz_classifier = nn.Linear(4096, self.num_vz)

        self._device = torch.device("cuda" if self.placement == DeviceType.GPU else "cpu")
        self.to(self._device)

    def forward(self, encoding):
        encoding = self.features(encoding)
        encoding = self.avgpool(encoding)
        encoding = torch.flatten(encoding, 1)
        encoding = self.common_classifier(encoding)
        vz = self.vz_classifier(encoding)
        return vz
