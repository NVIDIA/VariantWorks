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
            "encoding": NeuralType(axes=(
                                   AxisType(kind=AxisKind.Batch, size=None, is_list=False),
                                   AxisType(kind=AxisKind.Channel, size=self.num_input_channels, is_list=False),
                                   AxisType(kind=AxisKind.Height, size=None, is_list=False),
                                   AxisType(kind=AxisKind.Width, size=None, is_list=False),
                                   ), elements_type=VariantEncodingType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            'log_probs_vt': NeuralType(('B', 'D'), LogitsType()), # Variant type
            'log_probs_va': NeuralType(('B', 'D'), LogitsType()), # Variant allele
        }

    def __init__(self, num_input_channels, num_vt, num_alleles):
        super().__init__()
        self.num_vt = num_vt
        self.num_input_channels = num_input_channels
        self.num_alleles = num_alleles

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
            #nn.Linear(4096, self.num_vt),
        )
        self.vt_classifier = nn.Linear(4096, self.num_vt)
        self.va_classifier = nn.Linear(4096, self.num_alleles)

        self._device = torch.device("cuda" if self.placement == DeviceType.GPU else "cpu")
        self.to(self._device)

    def forward(self, encoding):
        encoding = self.features(encoding)
        encoding = self.avgpool(encoding)
        encoding = torch.flatten(encoding, 1)
        encoding = self.common_classifier(encoding)
        vt = self.vt_classifier(encoding)
        va = self.va_classifier(encoding)
        return vt, va
