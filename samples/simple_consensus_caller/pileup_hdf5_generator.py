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
from datetime import datetime
from functools import partial
import glob
import h5py
import multiprocessing as mp
import numpy as np
import os
import pysam
import shutil
import subprocess
import tempfile
import warnings

from variantworks.encoders import SummaryEncoder, HaploidLabelEncoder
from variantworks.io.fastxio import FastxWriter
from variantworks.types import FileRegion
from variantworks.utils.encoders import sliding_window


def validate_data_dir(directory):
    """Ensure that each data directory contains subreads, draft, and truth."""
    if not os.path.exists(os.path.join(directory, "subreads.fa")):
        raise RuntimeError("subreads.fa not present in all data folders.")
    if not os.path.exists(os.path.join(directory, "draft.fa")):
        raise RuntimeError("draft.fa not present in all data folders.")
    if not os.path.exists(os.path.join(directory, "truth.fa")):
        raise RuntimeError("truth.fa not present in all data folders.")


def align_sequences(target, query, output_file):
    """Align query to target using minimap2."""
    query_to_target_align_cmd = [
        "minimap2", "-x", "map-pb", "-t", "1", target, query, "--MD", "-a", "--secondary=no", "-o", output_file
    ]
    subprocess.check_call(query_to_target_align_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def create_pileup(data_dir):
    """Create a pileup file from subreads, draft, and truth."""

    def sort_bam(intput_bam, output_bam):
        sort_cmd = [
            "samtools", "sort", intput_bam, "-o", output_bam
        ]
        subprocess.check_call(sort_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def index_bam(input_bam):
        index_bam_cmd = [
            "samtools", "index", input_bam
        ]
        subprocess.check_call(index_bam_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    subreads_file = os.path.join(data_dir, "subreads.fa")
    draft_file = os.path.join(data_dir, "draft.fa")
    truth_file = os.path.join(data_dir, "truth.fa")
    suffix = os.path.basename(os.path.normpath(data_dir))

    # subreads to draft
    subreads_draft_bam = os.path.join(data_dir, "{}_{}.bam".format("subreads2draft", suffix))
    subreads_draft_sorted_bam = os.path.join(data_dir, "{}_{}.sorted.bam".format("subreads2draft", suffix))
    align_sequences(draft_file, subreads_file, subreads_draft_bam)
    sort_bam(subreads_draft_bam, subreads_draft_sorted_bam)
    index_bam(subreads_draft_sorted_bam)

    # truth to draft
    truth_draft_bam = os.path.join(data_dir, "{}_{}.bam".format("truth2draft", suffix))
    truth_draft_sorted_bam = os.path.join(data_dir, "{}_{}.sorted.bam".format("truth2draft", suffix))
    align_sequences(draft_file, truth_file, truth_draft_bam)
    sort_bam(truth_draft_bam, truth_draft_sorted_bam)
    index_bam(truth_draft_sorted_bam)

    mpileup_file = os.path.join(data_dir, "subreads_and_truth_{}.pileup".format(suffix))
    pileup_cmd = ["samtools", "mpileup", subreads_draft_sorted_bam,
                  truth_draft_sorted_bam, "-s", "--reverse-del", "-o", mpileup_file]
    subprocess.check_call(pileup_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Remove intermediate files
    files = glob.glob(os.path.join(data_dir, "*_{}.*bam*".format(suffix)))
    for f in files:
        os.remove(f)

    return FileRegion(start_pos=0, end_pos=None, file_path=mpileup_file)


def encode(sample_encoder, label_encoder, chunk_len, chunk_ovlp, data_dir, remove_data_dir=False):
    """Generate sample and label encoding for variant."""
    if remove_data_dir:
        validate_data_dir(data_dir)

    region = create_pileup(data_dir)

    # Generate matrix and label encoding.
    try:
        encoding, encoding_positions = sample_encoder(region)
        label, label_positions = label_encoder(region)

        # Generate read id per folder to help with bulk inference.
        read_id = data_dir.split("/")[-1]
        read_ids = np.array([read_id]*len(encoding))

        assert(len(encoding) == len(label)), print("Encoding and label dimensions not as expected:",
                                                   encoding.shape,
                                                   label.shape,
                                                   region)
        assert(len(encoding_positions) == len(encoding)), print("Encoding and positions not as expected:",
                                                                encoding.shape,
                                                                encoding_positions.shape,
                                                                region)
        if remove_data_dir:
            shutil.rmtree(data_dir)
        else:
            os.remove(region.file_path)  # remove mpileup file
        encoding_chunks = sliding_window(encoding, chunk_len, step=chunk_len - chunk_ovlp)
        position_chunks = sliding_window(encoding_positions, chunk_len, step=chunk_len - chunk_ovlp)
        label_chunks = sliding_window(label, chunk_len, step=chunk_len - chunk_ovlp)
        read_id_chunks = sliding_window(read_ids, chunk_len, step=chunk_len - chunk_ovlp)
        return encoding_chunks, position_chunks, label_chunks, read_id_chunks
    except Exception as err:
        warnings.warn(
            "An Exception occurred while processing: {}\n{}".format(region.file_path, str(err)), RuntimeWarning)
        return [], [], [], []


def extract_bam_into_subfolders(draft_path, subreads_path, draft_to_ref_path, out_root_folder):
    """A generator which returns a data folder with extracted draft.fa, subreads.fa & truth.fa files."""

    def validate_same_dataset(dataset_1, dataset_2, dataset_3):
        assert dataset_1 == dataset_2 and dataset_2 == dataset_3,\
            "datasets in draft, dubreads & draft2ref files: ({}, {}, {}) don't have the same name".format(
                dataset_1, dataset_2, dataset_3
            )

    drafts_intput_file = pysam.AlignmentFile(draft_path, "rb", check_sq=False)
    subreads_intput_file = pysam.AlignmentFile(subreads_path, "rb", check_sq=False)
    draft2ref_intput_file = pysam.AlignmentFile(draft_to_ref_path, "rb", check_sq=False)
    drafts_iter = drafts_intput_file.fetch(until_eof=True)
    subreads_iter = subreads_intput_file.fetch(until_eof=True)
    draft2ref_iter = draft2ref_intput_file.fetch(until_eof=True)
    draft = next(drafts_iter, None)
    subread = next(subreads_iter, None)
    draf2ref_aln = next(draft2ref_iter, None)

    while draft and subread and draf2ref_aln:
        dataset_draft, molecule_draft, _ = draft.query_name.split('/')
        dataset_subread, molecule_subread, pos_subread = subread.query_name.split('/')
        dataset_draft2ref, molecule_draft2ref, _ = draf2ref_aln.query_name.split('/')
        validate_same_dataset(dataset_draft, dataset_subread, dataset_draft2ref)
        assert molecule_draft == molecule_draft2ref, \
            "There is a mismatch in the entries order in the draft and draft2ref file: draft:{}, draft2ref:{}".format(
                molecule_draft, molecule_draft2ref
            )
        assert molecule_draft >= molecule_subread, \
            "There is a draft that has no corresponding entries in the subreads " \
            "input file: draft:{} subread:{}".format(molecule_draft, molecule_subread)
        # skip subreads that have no corresponding draft in the input draft BAM file
        while subread and int(molecule_draft) > int(molecule_subread):
            subread = next(subreads_iter, None)
            if subread:
                dataset_subread, molecule_subread, pos_subread = subread.query_name.split('/')
        # check for EOF subreads input
        if not subread:
            break
        # In case no subreads are present for some draft
        if int(molecule_draft) == int(molecule_subread):
            # Create molecule output folder
            out_folder = os.path.join(out_root_folder, dataset_draft, molecule_draft)
            os.makedirs(out_folder, exist_ok=False)
            # Write truth file
            truth_file_path = os.path.join(out_folder, 'truth.fa')
            loc = (draf2ref_aln.reference_name + ":" +
                   str(draf2ref_aln.reference_start) + "-" + str(draf2ref_aln.reference_end))  # chrom:start:end
            assert not os.path.isfile(truth_file_path),\
                "A truth.fa file already exists under {}".format(out_folder)
            with FastxWriter(truth_file_path, 'w') as truth_output_handle:
                truth_output_handle.write_output(record_id=loc,
                                                 record_sequence=draf2ref_aln.get_reference_sequence(),
                                                 record_name=loc)
            # Write draft file output from draft2ref
            draft_file_path = os.path.join(out_folder, 'draft.fa')
            draft_read = draf2ref_aln.query_sequence
            draft_seqid = draf2ref_aln.query_name
            assert not os.path.isfile(draft_file_path),\
                "A draft.fa file already exists under {}".format(out_folder)
            with FastxWriter(draft_file_path, "w") as draft_output_handle:
                draft_output_handle.write_output(record_id=draft_seqid,
                                                 record_sequence=draft_read,
                                                 record_name=draft_seqid)
            # Write subreads file output
            while subread and int(molecule_draft) == int(molecule_subread):
                validate_same_dataset(dataset_draft, dataset_subread, dataset_draft2ref)
                with FastxWriter(os.path.join(out_folder, 'subreads.fa'), "a") as subreads_output_handle:
                    subreads_output_handle.write_output(record_id=molecule_subread + '/' + pos_subread,
                                                        record_sequence=subread.seq,
                                                        record_name=molecule_subread + '/' + pos_subread)
                subread = next(subreads_iter, None)
                if subread:
                    dataset_subread, molecule_subread, pos_subread = subread.query_name.split('/')
            yield out_folder
            # move to the next draft/draft2ref entry
            draft = next(drafts_iter, None)
            draf2ref_aln = next(draft2ref_iter, None)
    drafts_intput_file.close()
    subreads_intput_file.close()


def create_draft_to_ref_file(draft_file_path, reference_file_path, working_dir):
    """Align draft to reference."""
    def bam_to_fasta(input_path, output_path):
        bam_to_fasta_cmd = "samtools bam2fq {} | seqtk seq -A > {}".format(input_path, output_path)
        subprocess.check_call(bam_to_fasta_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def filter_alignments(input_bam, output_bam):
        filter_cmd = [
            "samtools", "view", "-bS", "-F", "2308", input_bam, "-o", output_bam
        ]
        subprocess.check_call(filter_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    draft_fasta_path = os.path.join(working_dir, "draft_all.fa")
    draft_to_ref_unfiltered_bam_path = os.path.join(working_dir, "draft2ref_all_unfiltered.bam")
    draft_to_ref_bam_path = os.path.join(working_dir, "draft2ref.bam")
    bam_to_fasta(draft_file_path, output_path=draft_fasta_path)
    align_sequences(reference_file_path, draft_fasta_path, output_file=draft_to_ref_unfiltered_bam_path)
    os.remove(draft_fasta_path)
    filter_alignments(input_bam=draft_to_ref_unfiltered_bam_path, output_bam=draft_to_ref_bam_path)
    os.remove(draft_to_ref_unfiltered_bam_path)
    return draft_to_ref_bam_path


def get_validated_list_input_folders(input_data_dirs, input_single_dirs):
    """Concatenate input data directories under one list and validate them."""
    data_dirs = []
    for data_dir in input_data_dirs:
        for subdir in os.listdir(data_dir):
            subdir = os.path.abspath(os.path.join(data_dir, subdir))
            if os.path.isdir(subdir):
                data_dirs.append(subdir)
    for subdir in input_single_dirs:
        data_dirs.append(subdir)
    # Validate directories
    map(validate_data_dir, data_dirs)
    return data_dirs


def generate_hdf5(args):
    """Generate encodings in multiprocess loop and save tensors to HDF5."""
    if args.data_dir or args.single_dir:
        folders_to_encode = get_validated_list_input_folders(args.data_dir, args.single_dir)
        print('Serializing {} pileup files...'.format(len(folders_to_encode)))
        to_remove_data_dir = False
    else:
        working_dir = tempfile.mkdtemp(
            prefix="variantworks_ccs_sample_pileup_hdf5_{}_".format(datetime.now().strftime("%m.%d.%Y-%H:%M:%S")))
        draft_to_ref_path = create_draft_to_ref_file(args.draft_file, args.reference, working_dir)
        # Folders generator function
        folders_to_encode = extract_bam_into_subfolders(
            args.draft_file, args.subreads_file, draft_to_ref_path, working_dir
        )
        to_remove_data_dir = True

    # Setup encoder for samples and labels.
    sample_encoder = SummaryEncoder(exclude_no_coverage_positions=True)
    label_encoder = HaploidLabelEncoder(exclude_no_coverage_positions=True)
    encode_func = partial(encode, sample_encoder, label_encoder,
                          args.chunk_len, args.chunk_ovlp, remove_data_dir=to_remove_data_dir)

    # output data
    features = []    # features in column
    labels = []      # correct labeling
    positions = []   # track match/insert for stitching
    read_ids = []    # track folder name and windows

    label_idx = 0
    pool = mp.Pool(args.threads)
    for out in pool.imap(encode_func, folders_to_encode):
        if (label_idx + 1) % 100 == 0:
            print('Generated {} pileups'.format(label_idx + 1))
        (encoding_chunks, position_chunks, label_chunks, read_id_chunks) = out
        if encoding_chunks and position_chunks and label_chunks:
            if encoding_chunks[0].shape[0] == args.chunk_len \
                    and label_chunks[0].shape[0] == args.chunk_len \
                    and position_chunks[0].shape[0] == args.chunk_len:
                features += (encoding_chunks)
                labels += (label_chunks)
                positions += (position_chunks)
                read_ids += (read_id_chunks)
                label_idx += 1
    print('Generated {} pileup files'.format(label_idx))
    features = np.stack(features, axis=0)
    labels = np.stack(labels, axis=0)
    positions = np.stack(positions, axis=0)
    h5_file = h5py.File(args.output_file, 'w')
    h5_file.create_dataset('features', data=features)
    h5_file.create_dataset('positions', data=positions)
    h5_file.create_dataset('labels', data=labels)
    h5_file.create_dataset('read_ids', data=np.string_(read_ids))
    h5_file.close()


def build_parser():
    """Setup option parsing for sample."""
    parser = argparse.ArgumentParser(description='Store encoded data in HDF5 format.')
    draft_file_arg =\
        parser.add_argument('--draft-file', type=str,
                            help='drafts BAM file path')
    subreads_file_arg =\
        parser.add_argument('--subreads-file', type=str,
                            help='subreads BAM filepath')
    reference_arg =\
        parser.add_argument('--reference', type=str,
                            help='reference genome')
    data_dir_arg =\
        parser.add_argument('-d', '--data-dir', nargs='+',
                            help='Directory with folders containing subreads, draft, truth.', default=[])
    single_dir_arg =\
        parser.add_argument('-r', '--single-dir', nargs='+',
                            help='Directory containing subreads, draft, truth.', default=[])
    parser.add_argument('-o', '--output-file', type=str, help='Path to output HDF5 file.')
    parser.add_argument('-t', '--threads', type=int,
                        help='Threads to parallelize over.',
                        default=mp.cpu_count())
    parser.add_argument('-c', '--chunk-len', type=int,
                        help='Length of chunks to be created from pileups.', default=1000)
    parser.add_argument('--chunk-ovlp', type=int,
                        help='Length of overlaps between chunks.', default=200)

    args = parser.parse_args()

    if (args.draft_file or args.subreads_file or args.reference) and (args.data_dir or args.single_dir):
        raise parser.error(
            "{} and {} can not be set together with either: {} {} {}".format(
                data_dir_arg.dest, single_dir_arg.dest,
                draft_file_arg.dest, subreads_file_arg.dest, reference_arg.dest
            ))
    return args


if __name__ == '__main__':
    parsed_args = build_parser()
    generate_hdf5(parsed_args)
