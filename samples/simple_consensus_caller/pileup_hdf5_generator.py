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

from variantworks.encoders import SummaryEncoder, HaploidLabelEncoder
from variantworks.types import FileRegion
from variantworks.utils.encoders import sliding_window


def validate_data_dirs(data_dirs):
    """Ensure that each data directory contains subreads, draft, and truth."""
    for directory in data_dirs:
        if (not os.path.exists(os.path.join(directory, "subreads.fa"))):
            raise RuntimeError("subreads.fa not present in all data folders.")
        if (not os.path.exists(os.path.join(directory, "draft.fa"))):
            raise RuntimeError("draft.fa not present in all data folders.")
        if (not os.path.exists(os.path.join(directory, "truth.fa"))):
            raise RuntimeError("truth.fa not present in all data folders.")


def create_pileup(data_dir):
    """Create a pileup file from subreads, draft, and truth."""
    subreads_file = os.path.join(data_dir, "subreads.fa")
    draft_file = os.path.join(data_dir, "draft.fa")
    truth_file = os.path.join(data_dir, "truth.fa")
    suffix = os.path.basename(os.path.normpath(data_dir))

    subreads_draft_bam = "{}_{}.bam".format("subreads2draft", suffix)
    subreads_align_cmd = [
        "minimap2",
        "-x",
        "map-pb",
        "-t",
        "1",
        draft_file,
        subreads_file,
        "--MD",
        "-a",
        "-o",
        subreads_draft_bam]
    subprocess.check_call(subreads_align_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    subreads_draft_sorted_bam = "{}_{}.sorted.bam".format("subreads2draft", suffix)
    subreads_sort_cmd = [
        "samtools",
        "sort",
        subreads_draft_bam,
        "-o",
        subreads_draft_sorted_bam]
    subprocess.check_call(subreads_sort_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    subreads_idx_cmd = [
        "samtools", "index", subreads_draft_sorted_bam]
    subprocess.check_call(subreads_idx_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    truth_draft_bam = "{}_{}.bam".format("truth2draft", suffix)
    truth_align_cmd = [
        "minimap2",
        "-x",
        "map-pb",
        "-t",
        "1",
        draft_file,
        truth_file,
        "--MD",
        "-a",
        "-o",
        truth_draft_bam]
    subprocess.check_call(truth_align_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    truth_draft_sorted_bam = "{}_{}.sorted.bam".format("truth2draft", suffix)
    truth_sort_cmd = [
        "samtools",
        "sort",
        truth_draft_bam,
        "-o",
        truth_draft_sorted_bam]
    subprocess.check_call(truth_sort_cmd)

    truth_idx_cmd = ["samtools", "index", truth_draft_sorted_bam]
    subprocess.check_call(truth_idx_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    mpileup_file = "subreads_and_truth_{}.pileup".format(suffix)
    pileup_cmd = ["samtools", "mpileup", subreads_draft_sorted_bam,
                  truth_draft_sorted_bam, "-s", "--reverse-del", "-o", mpileup_file]
    subprocess.check_call(pileup_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Remove intermediate files
    files = glob.glob("*_{}.*bam*".format(suffix))
    for f in files:
        os.remove(f)

    return FileRegion(start_pos=0, end_pos=None, file_path=mpileup_file)


def encode(sample_encoder, label_encoder, chunk_len, chunk_ovlp, data_dir):
    """Generate sample and label encoding for variant."""
    region = create_pileup(data_dir)

    # Generate matrix and label encoding.
    try:
        encoding, encoding_positions = sample_encoder(region)
        label, label_positions = label_encoder(region)
        assert(len(encoding) == len(label)), print("Encoding and label dimensions not as expected:",
                                                   encoding.shape,
                                                   label.shape,
                                                   region)

        os.remove(region.file_path)
        encoding_chunks = sliding_window(encoding, chunk_len, step=chunk_len - chunk_ovlp)
        label_chunks = sliding_window(label, chunk_len,
                                      step=chunk_len - chunk_ovlp)
        return (encoding_chunks, label_chunks)
    except Exception:
        os.remove(region.file_path)
        return ([], [])


def generate_hdf5(args):
    """Generate encodings in multiprocess loop and save tensors to HDF5."""
    data_dirs = []
    for data_dir in args.data_dir:
        for subdir in os.listdir(data_dir):
            subdir = os.path.abspath(os.path.join(data_dir, subdir))
            if os.path.isdir(subdir):
                data_dirs.append(subdir)

    for subdir in args.single_dir:
        data_dirs.append(subdir)

    # Validate directories
    validate_data_dirs(data_dirs)

    # Setup encoder for samples and labels.
    sample_encoder = SummaryEncoder(exclude_no_coverage_positions=True)
    label_encoder = HaploidLabelEncoder(exclude_no_coverage_positions=True)
    encode_func = partial(encode, sample_encoder, label_encoder, args.chunk_len, args.chunk_ovlp)

    # Multi-processing
    pool = mp.Pool(args.threads)
    features = []
    labels = []
    print('Serializing {} pileup files...'.format(len(data_dirs)))
    label_idx = 0
    for out in pool.imap(encode_func, data_dirs):
        if (label_idx + 1) % 100 == 0:
            print('Generated {} pileups'.format(label_idx + 1))
        (encoding_chunks, label_chunks) = out
        if len(encoding_chunks) > 0 and len(label_chunks) > 0:
            if encoding_chunks[0].shape[0] == args.chunk_len and label_chunks[0].shape[0] == args.chunk_len:
                features += (encoding_chunks)
                labels += (label_chunks)
        label_idx += 1
    print('Generated {} pileup files'.format(len(data_dirs)))
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
    parser.add_argument('-d', '--data_dir', nargs='+',
                        help='Directory with folders containing subreads, draft, truth.', default=[])
    parser.add_argument('-r', '--single_dir', nargs='+',
                        help='Directory containing subreads, draft, truth.', default=[])
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
