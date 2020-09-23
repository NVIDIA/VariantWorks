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
from nemo.core import DeviceType
import torch

from variantworks.encoders import PileupEncoder, ZygosityLabelEncoder
from variantworks.neural_types import ReadPileupNeuralType, VariantZygosityNeuralType


class HDFDataLoader(DataLayerNM):
    """Dataloader class to load pileup encodings and zygosity labels from HDF5 file."""

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        Returns:
            NeMo output port.
        """
        # Generate output ports from requested tensors.
        port_dict = {}
        for i, key in enumerate(self.tensor_keys):
            port_dict[key] = NeuralType(self.tensor_dims[i], self.tensor_neural_types[i])
        return port_dict

    def __init__(self, hdf_file, batch_size=32, shuffle=True,
                 num_workers=4,
                 tensor_keys=["encodings", "labels"],
                 tensor_dtypes=[torch.float32, torch.int64],
                 tensor_dims=[('B', 'C', 'W', 'H'), tuple('B')],
                 tensor_neural_types=[ReadPileupNeuralType(), VariantZygosityNeuralType()],
                 ):
        """Constructor for data loader.

        Args:
            hdf_file : Path to HDF file with pileup encodings
            batch_size : batch size for data loader [32]
            shuffle : shuffle dataset [True]
            num_workers : numbers of parallel data loader threads [4]
            tensor_keys : List with keys of tensors to load. ["encodings", "labels"]
            tensor_dtypes : torch data types for tensor. [torch.float32, torch.int64]
            tensor_dims : NeuralModule axes for tensors. [('B', 'C', 'W', 'H'), ('B')]
            tensor_neural_types : NeuralTypes for tensors. [SummaryPileupNeuralType(), HaploidNeuralType()]

        Returns:
            Instance of class.
        """
        super().__init__()
        self.hdf_file = hdf_file
        self.tensor_keys = tensor_keys
        self.tensor_dtypes = tensor_dtypes
        self.tensor_dims = tensor_dims
        self.tensor_neural_types = tensor_neural_types

        class DatasetWrapper(TorchDataset):
            """A wrapper around Torch dataset class to generate individual samples."""

            def __init__(self, hdf_file, tensor_dtypes, tensor_keys):
                """Constructor for dataset wrapper.

                Args:
                    hdf_file : Path to HDF5 file.
                    tensor_keys : List with keys of tensors to load.
                    tensor_dtypes : torch data types for tensor.

                Returns:
                    Instance of class.
                """
                super().__init__()
                self.hdf_file = hdf_file
                self.tensor_dtypes = tensor_dtypes
                self.tensor_keys = tensor_keys
                with h5py.File(self.hdf_file, "r") as hdf:
                    self.len = len(hdf.get(self.tensor_keys[0]))
                self._h5_gen = None

            def __len__(self):
                return self.len

            def __getitem__(self, idx):
                # Using generator to keep the file handle to HDF5
                # file open during the life of the process.
                if self._h5_gen is None:
                    self._h5_gen = self._get_generator()
                    next(self._h5_gen)
                return self._h5_gen.send(idx)

            def _get_generator(self):
                hrecs = {}
                hdf = h5py.File(self.hdf_file, "r")
                for key in hdf.keys():
                    hrecs[key] = hdf.get(key)

                idx = yield
                while True:
                    outputs = []
                    for i, key in enumerate(self.tensor_keys):
                        data = hrecs[key]
                        tensor = torch.tensor(data[idx], dtype=self.tensor_dtypes[i])
                        outputs.append(tensor)
                    idx = yield tuple(outputs)

        dataset = DatasetWrapper(self.hdf_file, self.tensor_dtypes, self.tensor_keys)

        sampler = None
        if self._placement == DeviceType.AllGpu:
            sampler = torch.utils.data.distributed.DistributedSampler(self._dataset)

        self.dataloader = TorchDataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=shuffle if sampler is None else False,
                                          num_workers=num_workers,
                                          pin_memory=True,
                                          sampler=sampler)

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

        sampler = None
        if self._placement == DeviceType.AllGpu:
            sampler = torch.utils.data.distributed.DistributedSampler(self._dataset)

        self.dataloader = TorchDataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=shuffle if sampler is None else False,
                                          num_workers=num_workers,
                                          pin_memory=True,
                                          sampler=sampler)

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
