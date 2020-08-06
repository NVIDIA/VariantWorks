#!/usr/bin/env python3

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
"""Sample showcasing the generation of HDF5 datasets with variant encodings."""

import argparse
from functools import partial
import multiprocessing as mp

import h5py
import numpy as np

from variantworks.sample_encoder import PileupEncoder, ZygosityLabelEncoder
from variantworks.io.vcfio import VCFReader


def encode(sample_encoder, label_encoder, variant):
    """Generate sample and label encoding for variant."""
    encoding = sample_encoder(variant)
    label = label_encoder(variant)
    return (encoding, label)


def generate_hdf5(args):
    """Serialize encodings to HDF5.

    Generate encodings in multiprocess loop and save tensors to HDF5.
    """
    # Get list of files from arguments
    # and generate the variant entries using VCF reader.
    bam = args.bam
    vcf_readers = []
    for tp_file in args.tp_files:
        vcf_readers.append(VCFReader(vcf=tp_file, bams=[bam], is_fp=False))
    for fp_file in args.fp_files:
        vcf_readers.append(VCFReader(vcf=fp_file, bams=[bam], is_fp=True))
    total_labels = sum([len(reader) for reader in vcf_readers])

    # Setup encoder for samples and labels.
    sample_encoder = PileupEncoder(window_size=100, max_reads=100,
                                   layers=[PileupEncoder.Layer.READ, PileupEncoder.Layer.BASE_QUALITY])
    label_encoder = ZygosityLabelEncoder()

    encode_func = partial(encode, sample_encoder, label_encoder)

    # Create HDF5 datasets.
    h5_file = h5py.File(args.output_file, "w")
    encoded_data = h5_file.create_dataset("encodings",
                                          shape=(total_labels, sample_encoder.depth,
                                                 sample_encoder.height, sample_encoder.width),
                                          dtype=np.float32, fillvalue=0)
    label_data = h5_file.create_dataset("labels",
                                        shape=(total_labels,), dtype=np.int64, fillvalue=0)

    pool = mp.Pool(args.threads)
    print("Serializing {} entries...".format(total_labels))
    for vcf_reader in vcf_readers:
        label_idx = 0
        for out in pool.imap(encode_func, vcf_reader):
            if label_idx % 1000 == 0:
                print("Saved {} entries".format(label_idx))
            encoding, label = out
            encoded_data[label_idx] = encoding
            label_data[label_idx] = label
            label_idx += 1
    print("Saved {} entries".format(total_labels))

    h5_file.close()


def build_parser():
    """Setup option parsing for sample."""
    parser = argparse.ArgumentParser(
        description="Store encoded data in HDF5 format.")
    parser.add_argument("--tp-files", nargs="+",
                        help="List of true positive files.", default=[])
    parser.add_argument("--fp-files", nargs="+",
                        help="List of false positive files. For inference test set, use this option.", default=[])
    parser.add_argument("--bam", type=str,
                        help="BAM file with reads.", required=True)
    parser.add_argument("-o", "--output_file", type=str,
                        help="Path to output HDF5 file.")
    parser.add_argument("-t", "--threads", type=int,
                        help="Threads to parallelize over.", default=mp.cpu_count())
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    generate_hdf5(args)
