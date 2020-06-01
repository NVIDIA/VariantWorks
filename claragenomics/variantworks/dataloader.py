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

# Abstract class for creating a dataset from BAM and VCF files

from enum import Enum
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader
import vcf

from nemo.backends.pytorch.nm import DataLayerNM
from nemo.utils.decorators import add_port_docs
from nemo.core.neural_types import *

from claragenomics.variantworks.sample_encoder import PileupEncoder, ZygosityLabelEncoder
from claragenomics.variantworks.neural_types import ReadPileupNeuralType, VariantZygosityNeuralType
from claragenomics.variantworks.types import VariantZygosity


class ReadPileupDataLoader(DataLayerNM):
    """Data layer that outputs (variant type, variant allele, variant position) tuples.

    Args:
        sample_loader : Label loader object
        batch_size : batch size for dataset [32]
        shuffle : shuffle dataset [True]
        num_workers : numbers of parallel data loader threads [4]
        sample_encoder : Encoder for variant input
        label_encoder : An encoder for labels
    """

    class Type(Enum):
        """Type of data loader.
        """
        TRAIN = 0
        EVAL = 1
        TEST = 2

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        if self.data_loader_type == ReadPileupDataLoader.Type.TEST:
            return {
                "encoding": NeuralType(('B', 'C', 'H', 'W'), ReadPileupNeuralType()),
            }
        else:
            return {
                "label": NeuralType(tuple('B'), VariantZygosityNeuralType()),
                "encoding": NeuralType(('B', 'C', 'H', 'W'), ReadPileupNeuralType()),
            }

    def __init__(self, data_loader_type, sample_loader, batch_size=32, shuffle=True, num_workers=4, sample_encoder=None, label_encoder=None):
        super().__init__()

        self.data_loader_type = data_loader_type
        self.sample_loader = sample_loader
        self.sample_encoder = sample_encoder if sample_encoder is not None else PileupEncoder(window_size=50, max_reads=50, layers=[PileupEncoder.Layer.READ])
        self.label_encoder = label_encoder if label_encoder is not None else ZygosityLabelEncoder()

        class DatasetWrapper(TorchDataset):
            def __init__(self, data_loader_type, sample_encoder, sample_loader, label_encoder):
                super().__init__()
                self.sample_loader = sample_loader
                self.label_encoder = label_encoder
                self.sample_encoder = sample_encoder
                self.data_loader_type = data_loader_type

            def __len__(self):
                return len(self.sample_loader)

            def __getitem__(self, idx):
                sample = self.sample_loader[idx]

                if self.data_loader_type == ReadPileupDataLoader.Type.TEST:
                    sample = self.sample_encoder(sample)

                    return sample
                else:
                    encoding = self.sample_encoder(sample)
                    label = self.label_encoder(sample)

                    return label, encoding

        dataset = DatasetWrapper(data_loader_type, self.sample_encoder, self.sample_loader, self.label_encoder)
        self.dataloader = TorchDataLoader(dataset,
                                          batch_size=batch_size, shuffle=shuffle,
                                          num_workers=num_workers)

    def __len__(self):
        return len(self.dataloader)

    @property
    def data_iterator(self):
        return self.dataloader

    @property
    def dataset(self):
        return None
