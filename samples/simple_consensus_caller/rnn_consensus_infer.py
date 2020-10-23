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
import itertools

import nemo

from variantworks.dataloader import HDFDataLoader
from variantworks.io import fastxio
from variantworks.networks import ConsensusRNN
from variantworks.neural_types import SummaryPileupNeuralType, HaploidNeuralType
from variantworks.utils.stitcher import stitch, decode_consensus


def create_model():
    """Return neural network to train."""
    # Neural Network
    rnn = ConsensusRNN(sequence_length=1000, input_feature_size=10, num_output_logits=5, apply_softmax=True)

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

    # DATA DESCRIPTION prediction:
    # print("len(prediction)",len(prediction)) # len(prediction) 1
    # print("prediction[0].shape",prediction[0].shape) # prediction[0].shape torch.Size([20, 1000, 5])
    # print("prediction[0:10]", prediction[0:10])
    # prediction[0:10] [tensor([[[4.5658e-02, 2.4074e-02, 3.7805e-03, 9.1837e-01, 8.1153e-03],

    position = results[1]
    assert(len(prediction) == len(position))

    all_preds = []
    all_pos = []
    for pred, pos in zip(prediction, position):
        all_preds += pred
        all_pos += pos

    # Generate a lists of stitched consensus sequences.
    stitched_consensus_seq_parts = stitch(all_preds, all_pos, decode_consensus)

    # unpack the list of tuples into two lists
    nucleotides_sequence, nucleotides_certainty = map(list, zip(*stitched_consensus_seq_parts))
    nucleotides_sequence = "".join(nucleotides_sequence)
    nucleotides_certainty = list(itertools.chain.from_iterable(nucleotides_certainty))

    # Write out FASTQ sequence.
    with fastxio.FastqWriter(output_path=args.out_file, mode='w') as fastq_file:
        fastq_file.write_output("dl_consensus", nucleotides_sequence, nucleotides_certainty)


def build_parser():
    """Build parser object with options for sample."""
    parser = argparse.ArgumentParser(
        description="Simple SNP caller based on VariantWorks.")

    parser.add_argument("--infer-hdf", type=str,
                        help="HDF with molecule encodings to infer on. Please use one HDF per molecule.",
                        required=True)
    parser.add_argument("--model-dir", type=str,
                        help="Directory for storing trained model checkpoints. Stored after every eppoch of training.",
                        required=False, default="./models")
    parser.add_argument("-o", "--out-file", type=str,
                        help="Output file name for inferred consensus.",
                        required=True)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    infer(args)
