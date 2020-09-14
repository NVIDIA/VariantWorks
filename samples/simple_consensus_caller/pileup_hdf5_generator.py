#!/usr/bin/python
# -*- coding: utf-8 -*-
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

"""Generation of HDF5 datasets with summary pileup encodings."""

import argparse
from functools import partial
import multiprocessing as mp

import h5py
import numpy as np

from variantworks.encoders import SummaryEncoder, HaploidLabelEncoder
from variantworks.types import FileRegion
from variantworks.utils.encoders import sliding_window


def encode(sample_encoder, label_encoder, region):
    """Generate sample and label encoding for variant."""
    encoding = sample_encoder(region)
    label = label_encoder(region)
    encoding_chunks = sliding_window(encoding, window=CHUNK_LEN, step=CHUNK_LEN - CHUNK_OVLP)
    label_chunks = sliding_window(label, window=CHUNK_LEN,
                                  step=CHUNK_LEN - CHUNK_OVLP)
    return (encoding_chunks, label_chunks)


def generate_hdf5(args):
    """Generate encodings in multiprocess loop and save tensors to HDF5."""
    file_regions = []
    for pileup_file in args.pileup_files:
        file_regions.append(FileRegion(start_pos=0, end_pos=None,
                            file_path=pileup_file))

    # Setup encoder for samples and labels.

    global CHUNK_LEN
    global CHUNK_OVLP
    CHUNK_LEN = args.chunk_len
    CHUNK_OVLP = args.chunk_ovlp

    sample_encoder = SummaryEncoder()
    label_encoder = HaploidLabelEncoder()

    encode_func = partial(encode, sample_encoder, label_encoder)

    pool = mp.Pool(args.threads)
    features = []
    labels = []
    print('Serializing {} pileup files...'.format(len(file_regions)))
    for out in pool.imap(encode_func, file_regions):
        label_idx = 0
        if label_idx % 100 == 0:
            print('Saved {} pileups'.format(label_idx))
        (encoding_chunks, label_chunks) = out
        if encoding_chunks.shape[0] == CHUNK_LEN and label_chunks.shape[0] == CHUNK_LEN:
            features.append(encoding_chunks)
            labels.append(label_chunks)
        label_idx += 1
    print('Saved {} pileup files'.format(len(file_regions)))
    features = np.stack(features, axis=0)
    labels = np.stack(labels, axis=0)
    h5_file = h5py.File(args.output_file, 'w')
    h5_file.create_dataset('features', data=features)
    h5_file.create_dataset('labels', data=labels)
    h5_file.close()


def build_parser():
    """Setup option parsing for sample."""
    parser = \
        argparse.ArgumentParser(description='Store encoded data in HDF5 format.'
                                )
    parser.add_argument('--pileup-files', nargs='+',
                        help='List of pileup files.', default=[])
    parser.add_argument('-o', '--output_file', type=str,
                        help='Path to output HDF5 file.')
    parser.add_argument('-t', '--threads', type=int,
                        help='Threads to parallelize over.',
                        default=mp.cpu_count())
    parser.add_argument('-c', '--chunk_len', type=int,
                        help='Length of chunks to be created from pileups.', default=1000)
    parser.add_argument('--chunk_ovlp', type=int,
                        help='Length of overlaps between chunks.', default=200)
    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    generate_hdf5(args)
