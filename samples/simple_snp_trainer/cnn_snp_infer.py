#!/usr/bin/env python
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
"""A sample program highlighting usage of VariantWorks SDK to write a simple SNP variant caller using a CNN."""

import argparse

import nemo

from variantworks.dataloader import HDFDataLoader
from variantworks.networks import AlexNet
from variantworks.neural_types import ReadPileupNeuralType, VariantZygosityNeuralType


def create_model():
    """Return neural network to test."""
    # Neural Network
    alexnet = AlexNet(num_input_channels=2, num_output_logits=3)

    return alexnet


def infer(parsed_args):
    """Infer a sample model."""
    # Create neural factory as per NeMo requirements.
    nf = nemo.core.NeuralModuleFactory(
        placement=nemo.core.neural_factory.DeviceType.GPU, checkpoint_dir=parsed_args.model_dir)

    model = create_model()

    # Create test DAG
    test_dataset = HDFDataLoader(args.test_hdf, batch_size=32,
                                 shuffle=True, num_workers=args.threads,
                                 tensor_keys=["encodings", "labels"],
                                 tensor_dims=[('B', 'C', 'H', 'W'), tuple('B')],
                                 tensor_neural_types=[ReadPileupNeuralType(), VariantZygosityNeuralType()])
    encoding, vz_labels = test_dataset()

    vz = model(encoding=encoding)

    nf.infer([vz], checkpoint_dir=parsed_args.model_dir, verbose=True)


def build_parser():
    """Build parser object with options for sample."""
    import multiprocessing

    parser = argparse.ArgumentParser(
        description="Simple model inference SNP caller based on VariantWorks.")
    parser.add_argument("--test-hdf",
                        help="HDF with examples for testing.",
                        required=True)
    parser.add_argument("-t", "--threads", type=int,
                        help="Threads to use for parallel loading.",
                        required=False, default=multiprocessing.cpu_count())
    parser.add_argument("--model-dir", type=str,
                        help="Directory for loading saved trained model checkpoints.",
                        required=False, default="./models")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    infer(args)
