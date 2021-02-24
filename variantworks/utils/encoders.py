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
"""Utilities for summary encoder."""

import numpy as np
import torch


def reencode_base_pileup(ref_base, pileup_str):
    """Re-encodes mpileup output of list of special characters to list of nucleotides.

    Args:
        ref_base : Reference nucleotide
        pileup_str : mpileup encoding of pileup bases relative to reference
    Returns:
        A pileup string of special characters replaced with nucleotides.
    """
    pileup = []
    for c in pileup_str:
        if c == ".":
            pileup.append(ref_base)
        elif c == ",":
            pileup.append(ref_base.lower())
        else:
            pileup.append(c)
    return "".join(pileup)


def find_insertions(base_pileup):
    """Finds all of the insertions in a given base's pileup string.

    Args:
        base_pileup: Single base's pileup string output from samtools mpileup
    Returns:
        insertions: list of all insertions in pileup string
        next_to_del: whether insertion is next to deletion symbol (should be ignored)
    """
    insertions = []
    idx = 0
    next_to_del = []
    while idx < len(base_pileup):
        if base_pileup[idx] == "+":
            end_of_number = False
            insertion_bases_start_idx = idx+1
            while not end_of_number:
                if base_pileup[insertion_bases_start_idx].isdigit():
                    insertion_bases_start_idx += 1
                else:
                    end_of_number = True
            insertion_length = int(base_pileup[idx:insertion_bases_start_idx])
            inserted_bases = base_pileup[insertion_bases_start_idx:insertion_bases_start_idx+insertion_length]
            insertions.append(inserted_bases)
            next_to_del.append(True if base_pileup[idx - 1] in '*#' else False)
            idx = insertion_bases_start_idx + insertion_length + 1  # skip the consecutive base after insertion
        else:
            idx += 1
    return insertions, next_to_del


def normalize_counts(pileup_counts, positions):
    """Normalizes pileup counts based on depth of positions.

    Args:
        pileup_counts: torch tensor containing counts for each pileup column
        positions: structured numpy array containing major and minor positions
    Returns:
        norm_counts: Depth normalized pileup counts
    """
    # Calculate depth across all pileup columns
    depth = torch.sum(pileup_counts, axis=1)
    # Update depths of insert columns with the corresponding ref columns
    prev_ref_pos = None
    cur_ref_depth = 0
    for idx in range(len(positions)):
        if positions[idx][0] != prev_ref_pos:
            prev_ref_pos = positions[idx][0]
            cur_ref_depth = depth[idx]
        else:
            depth[idx] = cur_ref_depth
    # Normalize each column
    norm_counts = pileup_counts / np.maximum(1, depth).reshape((-1, 1))
    return norm_counts


def calculate_positions(start_pos, end_pos, subreads, truth_coverage, exclude_no_coverage_positions=True):
    """Calculates positions array from read pileup columns.

    Args:
        start_pos: Starting index of pileup columns
        end_pos: Ending index of pileup columns
        subreads: Array of subread strings from pileup file
        truth_coverage: Array of integers specifying coverage at each pileup column
        exclude_no_coverage_positions: Boolean specifying whether to include 0 coverage
        positions during position calculation
    Returns:
        positions: Array of tuples containing major/minor positions of pileup
    """
    positions = []
    # Calculate ref and insert positions
    for i in range(start_pos, end_pos):
        if exclude_no_coverage_positions and truth_coverage[i] == 0:
            continue

        base_pileup = subreads[i].strip("^]").strip("$")

        # Get all insertions in pileup
        insertions, next_to_del = find_insertions(base_pileup)

        # Find length of maximum insertion
        longest_insertion = len(max(insertions, key=len)) if insertions else 0

        # Keep track of ref and insert positions in the pileup and the insertions
        # in the pileup.
        ref_insert_pos = []  # ref position for ref base pos in pileup, insert for additional inserted bases
        ref_insert_pos.append([i, 0])
        for j in range(longest_insertion):
            ref_insert_pos.append([i, j+1])
        positions += ref_insert_pos
    return positions


def sliding_window(array, window, step=1, axis=0):
    """Generate chunks for encoding and labels.

    Args:
        array: Numpy array with the pileup counts
        window: Length of output chunks
        step: window minus chunk overlap
        axis: defaults to 0
    Returns:
        Iterator with chunks
    """
    chunk = [slice(None)] * array.ndim
    end = 0
    chunks = []
    for start in range(0, array.shape[axis] - window + 1, step):
        end = start + window
        chunk[axis] = slice(start, end)
        chunks.append(array[tuple(chunk)])
    if array.shape[axis] > end:
        start = array.shape[axis] - window
        chunk[axis] = slice(start, array.shape[axis])
        chunks.append(array[tuple(chunk)])
    return chunks
