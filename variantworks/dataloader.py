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

"""A class for creating a dataset from BAM and VCF files."""

from enum import Enum
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader

import h5py
from nemo.backends.pytorch.nm import DataLayerNM
from nemo.utils.decorators import add_port_docs
from nemo.core.neural_types import NeuralType
import torch

from variantworks.encoders import PileupEncoder, ZygosityLabelEncoder
from variantworks.neural_types import ReadPileupNeuralType, VariantZygosityNeuralType


class HDFPileupDataLoader(DataLayerNM):
    """Dataloader class to load pileup encodings and zygosity labels from HDF5 file."""

    class Type(Enum):
        """Type of data loader."""

        TRAIN = 0
        EVAL = 1
        TEST = 2

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        Returns:
            NeMo output port.
        """
        if self.data_loader_type == HDFPileupDataLoader.Type.TEST:
            return {
                "encoding": NeuralType(('B', 'C', 'H', 'W'), ReadPileupNeuralType()),
            }
        else:
            return {
                "label": NeuralType(tuple('B'), VariantZygosityNeuralType()),
                "encoding": NeuralType(('B', 'C', 'H', 'W'), ReadPileupNeuralType()),
            }

    def __init__(self, data_loader_type, hdf_file, batch_size=32, shuffle=True,
                 num_workers=4, encoding_dtype=torch.float32, label_dtype=torch.int64,
                 hdf_encoding_key="encodings", hdf_label_key="labels"):
        """Constructor for data loader.

        Args:
            data_loader_type : Type of data loader (HDFPileupDataLoader.Type.TRAIN/EVAL/TEST)
            hdf_file : Path to HDF file with pileup encodings
            batch_size : batch size for data loader [32]
            shuffle : shuffle dataset [True]
            num_workers : numbers of parallel data loader threads [4]
            encoding_dtype : Torch data type for encoding [torch.float32]
            label_dtype : Torch data type for label [torch.int64]
            hdf_encoding_key : HDF5 key for encodings. [encodings]
            hdf_label_key : HDF5 key for labels. [labels]

        Returns:
            Instance of class.
        """
        super().__init__()
        self.data_loader_type = data_loader_type
        self.hdf_file = hdf_file

        class DatasetWrapper(TorchDataset):
            """A wrapper around Torch dataset class to generate individual samples."""

            def __init__(self, data_loader_type, hdf_file, encoding_dtype, label_dtype,
                         hdf_encoding_key, hdf_label_key):
                """Constructor for dataset wrapper.

                Args:
                    data_loader_type : Type of data loader.
                    hdf_file : Path to HDF5 file.
                    encoding_dtype : Torch type for encoding.
                    label_dtype : Torch type for label.
                    hdf_encoding_key : HDF5 key for encodings.
                    hdf_label_key : HDF5 key for labels.

                Returns:
                    Instance of class.
                """
                super().__init__()
                self.data_loader_type = data_loader_type
                self.hdf_file = hdf_file
                self.encoding_dtype = encoding_dtype
                self.label_dtype = label_dtype
                self.hdf_encoding_key = hdf_encoding_key
                self.hdf_label_key = hdf_label_key

            def __len__(self):
                hdf = h5py.File(self.hdf_file, "r")
                return len(hdf.get(self.hdf_encoding_key))

            def __getitem__(self, idx):
                hdf = h5py.File(self.hdf_file, "r")

                if self.data_loader_type == HDFPileupDataLoader.Type.TEST:
                    encoding = hdf.get(self.hdf_encoding_key)[idx]

                    return torch.tensor(encoding, dtype=self.encoding_dtype)
                else:
                    encoding_data = hdf.get(self.hdf_encoding_key)
                    label_data = hdf.get(self.hdf_label_key)

                    encoding = torch.tensor(
                        encoding_data[idx], dtype=self.encoding_dtype)
                    label = torch.tensor(
                        label_data[idx], dtype=self.label_dtype)

                    return label, encoding

        dataset = DatasetWrapper(
            data_loader_type, self.hdf_file, encoding_dtype, label_dtype,
            hdf_encoding_key, hdf_label_key)
        self.dataloader = TorchDataLoader(dataset,
                                          batch_size=batch_size, shuffle=shuffle,
                                          num_workers=num_workers)

    def __len__(self):
        """Return length of data loader."""
        return len(self.dataloader)

    @property
    def data_iterator(self):
        """Return Torch dataloader instance."""
        return self.dataloader

    @property
    def dataset(self):
        """Not used."""
        return None


class ReadPileupDataLoader(DataLayerNM):
    """Data loader class to train zygosity predictions from variant pileup encodings."""

    class Type(Enum):
        """Type of data loader."""

        TRAIN = 0
        EVAL = 1
        TEST = 2

    @property
    @add_port_docs()
    def output_ports(self):
        """Return definitions of module output ports.

        Returns:
            NeMo output port.
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

    def __init__(self, data_loader_type, variant_loaders, batch_size=32, shuffle=True, num_workers=4,
                 sample_encoder=PileupEncoder(window_size=100, max_reads=100, layers=[PileupEncoder.Layer.READ]),
                 label_encoder=ZygosityLabelEncoder()):
        """Construct a data loader.

        Args:
            data_loader_type : Type of data loader (ReadPileupDataLoader.Type.TRAIN/EVAL/TEST)
            variant_loaders : A list of loader classes for variants
            batch_size : batch size for data loader [32]
            shuffle : shuffle dataset [True]
            num_workers : numbers of parallel data loader threads [4]
            sample_encoder : Custom pileup encoder for variant [READ pileup encoding, window size 100]
            label_encoder : Custom label encoder for variant [ZygosityLabelEncoder] (Only applicable
            when type=TRAIN/EVAL)

        Returns:
            Instance of class.
        """
        super().__init__()
        self.data_loader_type = data_loader_type
        self.variant_loaders = variant_loaders
        self.sample_encoder = sample_encoder
        self.label_encoder = label_encoder

        class DatasetWrapper(TorchDataset):
            """A wrapper around Torch dataset class to generate individual samples."""

            def __init__(self, data_loader_type, sample_encoder, variant_loaders, label_encoder):
                """Construct a dataset wrapper.

                Args:
                    data_loader_type : Type of data loader
                    sample_encoder : Custom pileup encoder for variant
                    variant_loaders : A list of loader classes for variants
                    label_encoder : Custom label encoder for variant

                Returns:
                    Instance of class.
                """
                super().__init__()
                self.variant_loaders = variant_loaders
                self.label_encoder = label_encoder
                self.sample_encoder = sample_encoder
                self.data_loader_type = data_loader_type

                self._len = sum([len(loader) for loader in self.variant_loaders])

            def _map_idx_to_sample(self, sample_idx):
                file_idx = 0
                while(file_idx < len(self.variant_loaders)):
                    if sample_idx < len(self.variant_loaders[file_idx]):
                        return self.variant_loaders[file_idx][sample_idx]
                    else:
                        sample_idx -= len(self.variant_loaders[file_idx])
                        file_idx += 1
                raise RuntimeError("Could not map sample index to file. This is a bug.")

            def __len__(self):
                return self._len

            def __getitem__(self, idx):
                sample = self._map_idx_to_sample(idx)

                if self.data_loader_type == ReadPileupDataLoader.Type.TEST:
                    sample = self.sample_encoder(sample)

                    return sample
                else:
                    encoding = self.sample_encoder(sample)
                    label = self.label_encoder(sample)

                    return label, encoding

        dataset = DatasetWrapper(
            data_loader_type, self.sample_encoder, self.variant_loaders, self.label_encoder)
        self.dataloader = TorchDataLoader(dataset,
                                          batch_size=batch_size, shuffle=shuffle,
                                          num_workers=num_workers)

    def __len__(self):
        """Return number of items in dataloader instance."""
        return len(self.dataloader)

    @property
    def data_iterator(self):
        """Return Torch dataloader instance."""
        return self.dataloader

    @property
    def dataset(self):
        """Not used."""
        return None
