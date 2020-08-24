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

from variantworks.utils.stitcher import Stitcher


def test_decode_consensus():
    stitcher = Stitcher(None, np.array([[]]))
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
    decoder = stitcher._decode_consensus(probs_to_decode)
    assert decoder == 'ACCTCACG'


@pytest.mark.parametrize(
    "pos_chunk1,pos_chunk2,ouput",
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
def test_overlap_stitch(pos_chunk1, pos_chunk2, ouput):
    sticher = Stitcher(None, np.array([[]]))
    first_end_index, second_start_index = sticher._overlap_indices(pos_chunk1, pos_chunk2)
    assert first_end_index == ouput[0] and second_start_index == ouput[1]
