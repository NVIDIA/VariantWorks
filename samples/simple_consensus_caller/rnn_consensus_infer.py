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
"""A sample program highlighting usage of VariantWorks SDK to write a simple consensus inference tool."""

import argparse

import nemo

from variantworks.dataloader import HDFDataLoader
from variantworks.networks import ConsensusRNN
from variantworks.neural_types import SummaryPileupNeuralType, HaploidNeuralType
from variantworks.utils.stitcher import stitch, decode_consensus


def create_model():
    """Return neural network to train."""
    # Neural Network
    rnn = ConsensusRNN(sequence_length=1000, input_feature_size=10, num_output_logits=5)

    return rnn


def infer(args):
    """Train a sample model with test data."""
    # Create neural factory as per NeMo requirements.
    nf = nemo.core.NeuralModuleFactory(
        placement=nemo.core.neural_factory.DeviceType.GPU)

    model = create_model()

    # Create train DAG
    infer_dataset = HDFDataLoader(args.infer_hdf, batch_size=32,
                                  shuffle=False, num_workers=1,
                                  tensor_keys=["features", "positions"],
                                  tensor_dims=[('B', 'W', 'C'), ('B', 'C')],
                                  tensor_neural_types=[SummaryPileupNeuralType(), HaploidNeuralType()],
                                  )
    encoding, positions = infer_dataset()
    vz = model(encoding=encoding)

    results = nf.infer([vz, positions], checkpoint_dir=args.model_dir, verbose=True)

    prediction = results[0]
    position = results[1]
    assert(len(prediction) == len(position))
    all_preds = []
    all_pos = []
    for pred, pos in zip(prediction, position):
        all_preds += pred
        all_pos += pos

    # Generate consensus sequence.
    consensus = stitch(all_preds, all_pos, decode_consensus)

    # Write out fasta sequence.
    with open(args.out_file, "w+") as fh:
        fh.write(">dl_consensus\n")
        fh.write(consensus)


def build_parser():
    """Build parser object with options for sample."""
    parser = argparse.ArgumentParser(
        description="Simple SNP caller based on VariantWorks.")

    parser.add_argument("--infer_hdf", type=str,
                        help="HDF with molecule encodings to infer on. Please use one HDF per molecule.",
                        required=True)
    parser.add_argument("--model_dir", type=str,
                        help="Directory for storing trained model checkpoints. Stored after every eppoch of training.",
                        required=False, default="./models")
    parser.add_argument("-o", "--out_file", type=str,
                        help="Output file name for inferred consensus.",
                        required=True)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    infer(args)
