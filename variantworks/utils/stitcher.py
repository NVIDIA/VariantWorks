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
"""Stitcher Utilities."""

import numpy as np


class Stitcher:
    """Combine chunk predictions into a sequence."""

    def __init__(self, probs, positions, label_symbols=None):
        """Calculate overlap indices given two chunks.

        Args:
            probs: chunked overlapping probabilities of predicted summary encoded samples
            positions: chucked overlapping positions of the summary encoded samples
            label_symbols(optional): overwrite nucleotide bases
        """
        self.probs = probs
        self.positions = np.array(positions, dtype=[('reference_pos', '<i8'), ('inserted_pos', '<i8')])
        if label_symbols is None:
            self.label_symbols = ["*", "A", "C", "G", "T"]

    def _decode_consensus(self, probs):
        """Decode probabilities into sequence.

        Returns:
            seq: sequence output from probabilities
        """
        seq = ''
        for i in range(len(probs)):
            base = probs[i, :]
            mp = np.argmax(base)
            seq += self.label_symbols[mp]
        seq = seq.replace('*', '')
        return seq

    @staticmethod
    def _overlap_indices(first_positions_chunk, second_positions_chunk):
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

    def stitch(self, chunk_len=1000):
        """Stitch predictions on chunks into a contiguous sequence.

        taking into account window size and overlap
        Returns:
            seq: Stitched consensus sequence
        """
        seq_parts = []
        start_1 = 0
        for i in range(1, len(self.positions), 1):
            probabilities_chunk = self.probs[i - 1]
            first_positions_chunk = self.positions[i - 1]
            second_positions_chunk = self.positions[i]
            # end1 and start2 are the new breaking points between two consecutive overlaps
            # found by the overlap_indices function.
            end_1, start_2 = self._overlap_indices(first_positions_chunk, second_positions_chunk)
            new_seq = self._decode_consensus(probabilities_chunk[start_1:end_1])
            seq_parts.append(new_seq)
            if i == len(self.positions) - 1:
                new_seq = self._decode_consensus(self.probs[i][chunk_len - end_1:])
                seq_parts.append(new_seq)
            start_1 = start_2
        return "".join(seq_parts)
