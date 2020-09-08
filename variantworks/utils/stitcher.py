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
"""Stitcher Utilities.

Combine chunk predictions into a sequence.
"""

import numpy as np


def decode_consensus(probs, label_symbols):
    """Decode probabilities into sequence.

    Returns:
        seq: sequence output from probabilities
    """
    seq = ''
    for i in range(len(probs)):
        base = probs[i, :]
        mp = np.argmax(base)
        seq += label_symbols[mp]
    seq = seq.replace('*', '')
    return seq


def overlap_indices(first_positions_chunk, second_positions_chunk):
    """Calculate overlap indices given two chunks.

    Args:
        first_positions_chunk: First positions chunk
        second_positions_chunk: Second positions chunk
    Returns:
        padded_first_chunk_end_idx: End index of the current chunk
        padded_second_chunk_start_idx: Start index of the next chunk
    """
    first_chunk_overlap_start_idx = np.searchsorted(first_positions_chunk, second_positions_chunk[0])
    second_chunk_overlap_end_idx = np.searchsorted(second_positions_chunk, first_positions_chunk[-1], side='right')
    first_chunk_overlap_values = first_positions_chunk[first_chunk_overlap_start_idx:]
    second_chunk_overlap_values = second_positions_chunk[0:second_chunk_overlap_end_idx]
    if first_chunk_overlap_values.size != 0 and second_chunk_overlap_values.size != 0 and \
            np.array_equal(first_chunk_overlap_values['inserted_pos'], second_chunk_overlap_values['inserted_pos']):
        first_chunk_padding_size = round(len(first_chunk_overlap_values) / 2)
        padded_first_chunk_end_idx = first_chunk_overlap_start_idx + first_chunk_padding_size
        padded_second_chunk_start_idx = second_chunk_overlap_end_idx - (
                len(first_chunk_overlap_values) - first_chunk_padding_size)
        if all(np.concatenate([first_positions_chunk[first_chunk_overlap_start_idx:padded_first_chunk_end_idx],
                               second_positions_chunk[padded_second_chunk_start_idx:second_chunk_overlap_end_idx]])
               == first_chunk_overlap_values):
            return padded_first_chunk_end_idx, padded_second_chunk_start_idx
    raise ValueError("Can not Stitch {} {}".format(first_positions_chunk, second_positions_chunk))


def stitch(probs, positions, label_symbols, chunk_len=1000):
    """Stitch predictions on chunks into a contiguous sequence.

    taking into account window size and overlap
    Returns:
        seq: Stitched consensus sequence
    """
    if hasattr(label_symbols, "__getitem__"):
        raise TypeError("label_symbols does not ")
    seq_parts = []
    start_1 = 0
    for i in range(1, len(positions), 1):
        probabilities_chunk = probs[i - 1]
        first_positions_chunk = positions[i - 1]
        second_positions_chunk = positions[i]
        # end1 and start2 are the new breaking points between two consecutive overlaps
        # found by the overlap_indices function.
        end_1, start_2 = overlap_indices(first_positions_chunk, second_positions_chunk)
        new_seq = decode_consensus(probabilities_chunk[start_1:end_1], label_symbols)
        seq_parts.append(new_seq)
        if i == len(positions) - 1:
            new_seq = decode_consensus(probs[i][chunk_len - end_1:], label_symbols)
            seq_parts.append(new_seq)
        start_1 = start_2
    return "".join(seq_parts)
