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
    # Get list of files from arguments.
    bam = args.bam
    file_list = []
    for tp_file in args.tp_files:
        file_list.append(VCFReader.VcfBamPath(
            vcf=tp_file, bam=bam, is_fp=False))
    for fp_file in args.fp_files:
        file_list.append(VCFReader.VcfBamPath(
            vcf=fp_file, bam=bam, is_fp=True))

    # Generate the variant entries using VCF reader.
    vcf_reader = VCFReader(file_list)
    for variant in vcf_reader:
        print(variant)
    exit(0)

    # Setup encoder for samples and labels.
    sample_encoder = PileupEncoder(window_size=100, max_reads=100,
                                   layers=[PileupEncoder.Layer.READ, PileupEncoder.Layer.BASE_QUALITY])
    label_encoder = ZygosityLabelEncoder()

    encode_func = partial(encode, sample_encoder, label_encoder)

    # Create HDF5 datasets.
    h5_file = h5py.File(args.output_file, "w")
    encoded_data = h5_file.create_dataset("encodings",
                                          shape=(len(vcf_reader), sample_encoder.depth,
                                                 sample_encoder.height, sample_encoder.width),
                                          dtype=np.float32, fillvalue=0)
    label_data = h5_file.create_dataset("labels",
                                        shape=(len(vcf_reader),), dtype=np.int64, fillvalue=0)

    pool = mp.Pool(args.threads)
    print("Serializing {} entries...".format(len(vcf_reader)))
    for i, out in enumerate(pool.imap(encode_func, vcf_reader)):
        if i % 1000 == 0:
            print("Saved {} entries".format(i))
        encoding, label = out
        encoded_data[i] = encoding
        label_data[i] = label
    print("Saved {} entries".format(len(vcf_reader)))

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
