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

import numpy as np
import pytest

from variantworks.utils.stitcher import overlap_indices, decode_consensus, stitch


def test_decode_consensus():
    probs_to_decode = np.array(
        [[1.11062377e-06, 9.99998808e-01, 9.45534850e-08, 5.90062443e-10, 2.44629739e-08],
         [1.52412838e-08, 2.44717496e-10, 1.00000000e+00, 6.66738956e-11, 1.94615679e-09],
         [5.04629094e-09, 2.82073245e-08, 2.82073245e-05, 4.63095291e-11, 2.10598863e-08],
         [3.60693385e-11, 2.04000684e-12, 3.61853776e-11, 2.19558313e-11, 1.00000000e+00],
         [1.00000000e+00, 5.60136160e-10, 2.11448139e-08, 9.51203560e-11, 1.02282349e-09],
         [4.10584065e-08, 7.19157733e-10, 1.00000000e+00, 3.22154911e-12, 1.89948821e-11],
         [6.04322314e-10, 1.00000000e+00, 9.59179264e-12, 5.67443835e-11, 4.77851786e-12],
         [3.24753627e-11, 1.76283585e-10, 1.00000000e+00, 4.86621091e-11, 4.08135250e-13],
         [1.85279381e-09, 4.08135250e-13, 4.57124269e-11, 9.99999881e-01, 4.97050274e-12],
         [9.99999881e-01, 6.09803052e-08, 8.86788643e-10, 4.56723086e-08, 1.30385489e-08]], dtype=float
    )
    decoder, certainty_score = decode_consensus(probs_to_decode, include_certainty_score=False)
    assert decoder == 'ACCTCACG' and not certainty_score

    decoder, certainty_score = decode_consensus(probs_to_decode)
    assert \
        decoder == 'ACCTCACG' and \
        np.array_equal(certainty_score, [0.999998808, 1.0, 2.82073245e-05, 1.0, 1.0, 1.0, 1.0, 0.999999881])


@pytest.mark.parametrize(
    "pos_chunk1,pos_chunk2,output",
    [
        (np.array(
            [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
             (5, 0), (6, 0), (7, 0), (8, 0), (9, 0),
             (10, 0), (11, 0), (12, 0)], dtype=[('reference_pos', '<i8'), ('inserted_pos', '<i8')]),
         np.array(
             [(9, 0), (10, 0), (11, 0), (12, 0), (13, 0),
              (13, 1), (13, 2), (14, 0), (14, 1), (14, 2),
              (15, 0), (16, 0), (17, 0)], dtype=[('reference_pos', '<i8'), ('inserted_pos', '<i8')]),
         (11, 2))
    ],
)
def test_overlap_stitch(pos_chunk1, pos_chunk2, output):
    first_end_index, second_start_index = overlap_indices(pos_chunk1, pos_chunk2)
    assert first_end_index == output[0] and second_start_index == output[1]


# Test 3 chunks of length 6 with overlap = 2
@pytest.mark.parametrize(
    "probs,positions,decode_consensus_function,expected_output",
    [
        (np.array([
            [[1.09477956e-08, 1.00000000e+00, 5.52753621e-10, 2.30751862e-10, 4.01220918e-10],  # A
             [7.66537767e-09, 4.54994592e-12, 4.86722260e-11, 1.00000000e+00, 1.23715330e-10],  # G
             [8.02772432e-11, 5.73730646e-12, 1.00000000e+00, 5.13983604e-13, 1.73759864e-14],  # C
             [7.33906136e-09, 1.00000000e+00, 3.07665049e-09, 8.15870538e-10, 1.75069598e-10],  # A
             [6.39799893e-08, 1.48147379e-08, 1.48587995e-08, 9.99999881e-01, 2.30921460e-09],  # G
             [6.21890495e-09, 3.26714868e-08, 6.37951345e-08, 9.99999881e-01, 1.45624020e-08]],  # G

            [[1.27002045e-06, 1.05614718e-10, 7.13807957e-10, 9.99998569e-01, 1.05759014e-07],  # G
             [8.24138358e-09, 1.74944434e-10, 6.70461686e-10, 1.00000000e+00, 3.03972780e-09],  # G
             [4.20199342e-07, 1.22673005e-09, 9.78943371e-10, 5.50214452e-09, 9.99999523e-01],  # T
             [4.69269878e-07, 2.79435799e-08, 9.99999523e-01, 1.78214741e-08, 6.20921647e-10],  # C
             [6.08718665e-06, 9.99993801e-01, 8.25351236e-08, 8.10780687e-09, 2.69226899e-08],  # A
             [2.13696563e-08, 9.99999285e-01, 3.68703439e-07, 1.91942135e-07, 1.13494544e-07]],  # A

            [[2.4675433e-07, 9.9999976e-01, 1.1541250e-08, 6.2487366e-09, 4.9959055e-09],  # A
             [7.5940561e-09, 1.0000000e+00, 5.2648463e-09, 1.7200236e-10, 1.0723046e-09],  # A
             [6.7918995e-09, 3.8870717e-12, 1.2175504e-10, 5.3161930e-11, 1.0000000e+00],  # T
             [8.6349354e-09, 1.3653510e-10, 1.0000000e+00, 2.6004756e-11, 1.2342026e-10],  # C
             [4.6234220e-08, 9.9999988e-01, 9.0397293e-08, 8.5086944e-11, 1.7032700e-09],  # A
             [1.4115821e-07, 1.5209878e-07, 1.0382157e-08, 2.9765086e-08, 9.9999964e-01]]  # T
        ]),
         [
             np.array([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)],
                      dtype=[('reference_pos', '<i8'), ('inserted_pos', '<i8')]),
             np.array([(4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0)],
                      dtype=[('reference_pos', '<i8'), ('inserted_pos', '<i8')]),
             np.array([(8, 0), (9, 0), (10, 0), (11, 0), (12, 0), (13, 0)],
                      dtype=[('reference_pos', '<i8'), ('inserted_pos', '<i8')])
         ],
         decode_consensus,
         ("".join(['AGCAG', 'GTCA', 'ATCAT']),
          [1.0, 1.0, 1.0, 1.0, 0.999999881,
           1.0, 0.999999523, 0.999999523, 0.999993801,
           1.0, 1.0, 1.0, 0.99999988, 0.99999964])
        )
    ],
)
def test_stitch(probs, positions, decode_consensus_function, expected_output):
    stitched_seq, seq_certainty = stitch(probs, positions, decode_consensus_function)
    assert expected_output[0] == stitched_seq and np.array_equal(expected_output[1], seq_certainty)
