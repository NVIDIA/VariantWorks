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

import os
import subprocess
import h5py
import numpy as np
import glob
import pandas as pd

from variantworks.encoders import SummaryEncoder, HaploidLabelEncoder
from variantworks.types import FileRegion
from variantworks.utils.encoders import sliding_window


def validate_data_dirs(data_dirs):
    """Ensure that each data directory contains subreads, draft, and truth."""
    for directory in data_dirs:
        if (not os.path.exists(directory + "/subreads.fa")):
            raise RuntimeError("subreads.fa not present in all data folders.")
        if (not os.path.exists(directory + "/draft.fa")):
            raise RuntimeError("draft.fa not present in all data folders.")
        if (not os.path.exists(directory + "/truth.fa")):
            raise RuntimeError("truth.fa not present in all data folders.")


def create_pileup(data_dir):
    """Create a pileup file from subreads, draft, and truth."""
    subreads_file = data_dir + "subreads.fa"
    draft_file = data_dir + "draft.fa"
    truth_file = data_dir + "truth.fa"
    suffix = data_dir.split("/")[-2]

    subreads_align_cmd = [
        "minimap2",
        "-x",
        "map-pb",
        "-t",
        "1",
        str(draft_file),
        str(subreads_file),
        "--MD",
        "-a",
        "-o",
        "subreads2draft"+str(suffix)+".bam"]
    subprocess.check_call(subreads_align_cmd)

    subreads_sort_cmd = [
        "samtools",
        "sort",
        "subreads2draft"+str(suffix)+".bam",
        "-o",
        "subreads2draft"+str(suffix)+".sorted.bam"]
    subreads_idx_cmd = [
        "samtools", "index", "subreads2draft"+str(suffix)+".sorted.bam"]
    subprocess.check_call(subreads_sort_cmd)
    subprocess.check_call(subreads_idx_cmd)

    truth_align_cmd = [
        "minimap2",
        "-x",
        "map-pb",
        "-t",
        "1",
        str(draft_file),
        str(truth_file),
        "--MD",
        "-a",
        "-o",
        "truth2draft"+str(suffix)+".bam"]
    subprocess.check_call(truth_align_cmd)

    truth_sort_cmd = [
        "samtools",
        "sort",
        "truth2draft"+str(suffix)+".bam",
        "-o",
        "truth2draft"+str(suffix)+".sorted.bam"]
    truth_idx_cmd = ["samtools", "index", "truth2draft"+str(suffix)+".sorted.bam"]
    subprocess.check_call(truth_sort_cmd)
    subprocess.check_call(truth_idx_cmd)

    pileup = "subreads_and_truth"+str(suffix)+".pileup"
    pileup_cmd = ["samtools", "mpileup", "subreads2draft"+str(suffix)+".sorted.bam",
                  "truth2draft"+str(suffix)+".sorted.bam", "-s", "--reverse-del", "-o", pileup]
    subprocess.check_call(pileup_cmd)

    return FileRegion(start_pos=0, end_pos=None, file_path=pileup)


def encode(sample_encoder, label_encoder, data_dir):
    """Generate sample and label encoding for variant."""
    region = create_pileup(data_dir)
    try:
        encoding = sample_encoder(region)
        label = label_encoder(region)
    except pd.errors.ParserError:
        return ([], [])
    encoding_chunks = sliding_window(encoding, window=CHUNK_LEN, step=CHUNK_LEN - CHUNK_OVLP)
    label_chunks = sliding_window(label, window=CHUNK_LEN,
                                  step=CHUNK_LEN - CHUNK_OVLP)

    # Delete created files by samtools and minimap2
    os.system("rm " + str(region.file_path))
    suffix = data_dir.split("/")[-2]
    os.system("rm subreads2draft"+str(suffix)+".bam")
    os.system("rm subreads2draft"+str(suffix)+".sorted.bam")
    os.system("rm subreads2draft"+str(suffix)+".sorted.bam.bai")
    os.system("rm truth2draft"+str(suffix)+".bam")
    os.system("rm truth2draft"+str(suffix)+".sorted.bam")
    os.system("rm truth2draft"+str(suffix)+".sorted.bam.bai")

    return (encoding_chunks, label_chunks)


def generate_hdf5(args):
    """Generate encodings in multiprocess loop and save tensors to HDF5."""
    if (args.single_dir is None):
        data_dirs = glob.glob(args.data_dir[0] + "/*/")
        if len(data_dirs) == 0:
            raise RuntimeError("Could not find any folders within data directory.")
        validate_data_dirs(data_dirs)
    else:
        data_dirs = [args.single_dir[0]]
        validate_data_dirs(data_dirs)

    print(data_dirs)
    global CHUNK_LEN
    global CHUNK_OVLP
    global N_DIRS
    CHUNK_LEN = args.chunk_len
    CHUNK_OVLP = args.chunk_ovlp
    N_DIRS = len(data_dirs)

    # Setup encoder for samples and labels.
    sample_encoder = SummaryEncoder()
    label_encoder = HaploidLabelEncoder()
    encode_func = partial(encode, sample_encoder, label_encoder)

    # Multi-processing
    pool = mp.Pool(args.threads)
    features = []
    labels = []
    print('Serializing {} pileup files...'.format(len(data_dirs)))
    label_idx = 0
    for out in pool.imap(encode_func, data_dirs):
        if label_idx % 100 == 0:
            print('Saved {} pileups'.format(label_idx))
        (encoding_chunks, label_chunks) = out
        if len(encoding_chunks) > 0 and len(label_chunks) > 0:
            if encoding_chunks[0].shape[0] == CHUNK_LEN and label_chunks[0].shape[0] == CHUNK_LEN:
                features += (encoding_chunks)
                labels += (label_chunks)
        label_idx += 1
    print('Saved {} pileup files'.format(len(data_dirs)))
    features = np.stack(features, axis=0)
    labels = np.stack(labels, axis=0)
    h5_file = h5py.File(args.output_file, 'w')
    h5_file.create_dataset('features', data=features)
    h5_file.create_dataset('labels', data=labels)
    h5_file.close()

    # cleanup
    os.system("rm *.bam")
    os.system("rm *.bai")
    os.system("rm *.pileup")


def build_parser():
    """Setup option parsing for sample."""
    parser = \
        argparse.ArgumentParser(description='Store encoded data in HDF5 format.'
                                )
    parser.add_argument('-d', '--data_dir', nargs='+',
                        help='Directory with folders containing subreads, draft, truth.', default=[])
    parser.add_argument('-r', '--single_dir', nargs='+',
                        help='Directory containing subreads, draft, truth.', default=None)
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
