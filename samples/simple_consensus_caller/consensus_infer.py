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

import h5py
import nemo
import numpy as np

from variantworks.dataloader import HDFDataLoader
from variantworks.io import fastxio
from variantworks.neural_types import SummaryPileupNeuralType, HaploidNeuralType
from variantworks.utils.stitcher import stitch, decode_consensus

from create_model import create_model


def infer(args):
    """Train a sample model with test data."""
    # Create neural factory as per NeMo requirements.
    nf = nemo.core.NeuralModuleFactory(
        placement=nemo.core.neural_factory.DeviceType.GPU)

    model = create_model(model=args.model,
                         input_feature_size=args.input_feature_size,
                         num_output_logits=args.num_output_logits,
                         gru_size=args.gru_size,
                         gru_layers=args.gru_layers,
                         kernel_size=args.kernel_size)

    # Create train DAG
    infer_dataset = HDFDataLoader(args.infer_hdf, batch_size=256,
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

    # This loop flattens the NeMo output that's grouped by batches into a flat list.
    all_preds = []
    all_pos = []
    for pred, pos in zip(prediction, position):
        all_preds += pred
        all_pos += pos

    # Get list of read_ids from hdf to calculate where read windows begin and end.
    hdf = h5py.File(args.infer_hdf, "r")
    read_ids = hdf["read_ids"]

    # Track read id per window since original np array has a read id per position per window.
    read_ids = [read_ids[window_id, 0] for window_id in range(read_ids.shape[0])]

    # Group consecutive read id windows.
    read_group_lengths = [0]
    read_group_lengths.extend([len(list(g)) for k, g in itertools.groupby(read_ids)])

    # Calculate begin and end boundaries for same read id windows.
    read_boundaries = np.cumsum(read_group_lengths)

    # Convert boundaries into intervals.
    read_intervals = [(read_boundaries[idx], read_boundaries[idx + 1]) for idx in range(len(read_boundaries)-1)]

    with fastxio.FastxWriter(output_path=args.out_file, mode='w') as fastq_file:
        for begin, end in read_intervals:
            read_id = read_ids[begin].decode("utf-8")

            # Generate a lists of stitched consensus sequences.
            stitched_consensus_seq_parts = stitch(all_preds[begin:end], all_pos[begin:end], decode_consensus)

            # unpack the list of tuples into two lists
            nucleotides_sequence, nucleotides_certainty = map(list, zip(*stitched_consensus_seq_parts))
            nucleotides_sequence = "".join(nucleotides_sequence)
            nucleotides_certainty = list(itertools.chain.from_iterable(nucleotides_certainty))

            # Write out FASTQ sequence.
            fastq_file.write_output(read_id, nucleotides_sequence,
                                    description="Generated consensus sequence by NVIDIA VariantWorks",
                                    record_quality=nucleotides_certainty)


def build_parser():
    """Build parser object with options for sample."""
    parser = argparse.ArgumentParser(
        description="Read consensus caller based on VariantWorks.")

    parser.add_argument("--infer-hdf", type=str,
                        help="HDF with read encodings to infer on.",
                        required=True)
    parser.add_argument("--model-dir", type=str,
                        help="Directory for storing trained model checkpoints. Stored after every eppoch of training.",
                        required=False, default="./models")
    parser.add_argument("-o", "--out-file", type=str,
                        help="Output file name for inferred consensus.",
                        required=True)
    parser.add_argument("--input_feature_size", type=int, default=10)
    parser.add_argument("--num_output_logits", type=int, default=5)
    parser.add_argument("--gru_size", help="Number of units in RNN", type=int, default=128)
    parser.add_argument("--gru_layers", help="Number of layers in RNN", type=int, default=2)
    parser.add_argument("--kernel_size", help="Kernel size for CNN", type=int, default=1)
    parser.add_argument("--model", help="Model", type=str,
                        choices=('cnn', 'rnn'), default='cnn')

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    infer(args)
