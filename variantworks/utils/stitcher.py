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

from collections import namedtuple
import numpy as np

NucleotideCertainty = namedtuple('NucleotideCertainty', ['nucleotide_literal', 'nucleotide_certainty'])


def decode_consensus(probs, include_certainty_score=True):
    """Decode probabilities into sequence by choosing the Nucleotide base with the highest probability.

    Returns:
        seq: sequence output from probabilities
    """
    label_symbols = ["*", "A", "C", "G", "T"]  # Corresponding labels for each network output channel
    seq = ''
    seq_quality = list()
    for i in range(len(probs)):
        base = probs[i, :]
        mp = np.argmax(base).item()
        nuc = label_symbols[mp]
        if nuc != '*':
            seq += nuc
            if include_certainty_score:
                seq_quality.append(base[mp].item())
    return seq, seq_quality if include_certainty_score else list()


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


def stitch(probs, positions, decode_consensus_func):
    """Stitch predictions on chunks into a contiguous sequence.

    Args:
        probs: 3D array of predicted probabilities. no. of chunks X  no. of positions in chunk X no. of bases.
        positions: Corresponding list of position array for each chunk in probs.
        decode_consensus_func: A function which decodes each chunk from probs into label_symbols.

    Returns:
        seq: Stitched consensus sequence
    """
    decoded_sequece_parts = list()
    first_start_idx = 0
    for i in range(1, len(positions), 1):
        probabilities_chunk = probs[i - 1]
        # Convert positions tensor into np.darray format expected for overlap_indices function
        first_positions_chunk = np.array([(pos[0], pos[1]) for pos in positions[i - 1]],
                                         dtype=[('reference_pos', '<i8'), ('inserted_pos', '<i8')])
        second_positions_chunk = np.array([(pos[0], pos[1]) for pos in positions[i]],
                                          dtype=[('reference_pos', '<i8'), ('inserted_pos', '<i8')])
        # first_end_idx and second_start_idx are the new breaking points between two consecutive overlaps
        # found by the overlap_indices function.
        first_end_idx, second_start_idx = overlap_indices(first_positions_chunk, second_positions_chunk)
        # Decoding chunk in i-1 position and adding to sequence
        prev_decoded_seq = decode_consensus_func(probabilities_chunk[first_start_idx:first_end_idx])
        decoded_sequece_parts.append(prev_decoded_seq)
        # Handling last sequence
        if i == len(positions) - 1:
            current_decoded_seq = decode_consensus_func(probs[i][second_start_idx:])
            decoded_sequece_parts.append(current_decoded_seq)
        first_start_idx = second_start_idx
    return decoded_sequece_parts
