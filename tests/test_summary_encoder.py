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

import os
import numpy as np
import pytest

from variantworks.types import FileRegion
from variantworks.encoders import SummaryEncoder
from test_utils import get_data_folder


@pytest.mark.parametrize(
    'base_string,expected_output',
    [
        ('AAA+1TCA+1TAAaaaaaaaAaa', [['T', 'T'], [False, False]]),
        ('A+1T+1TAa+1Ga', [['T', 'T', 'G'], [False, False, False]]),
        ('A+1Ta*+1TAa+1Ga', [['T', 'T', 'G'], [False, True, False]]),
    ],
)
def test_find_insertion(base_string, expected_output):
    output = SummaryEncoder._find_insertions(base_string)
    assert all([x == y for x, y in zip(output, expected_output)])


def test_counts_correctness():
    region = FileRegion(start_pos=0,
                        end_pos=14460,
                        file_path=os.path.join(get_data_folder(), "subreads_and_truth.pileup"))
    encoder = SummaryEncoder(exclude_no_coverage_positions=False, normalize_counts=True)
    pileup_counts = encoder(region)
    correct_counts = np.load(os.path.join(get_data_folder(), "sample_counts.npy"))
    assert(pileup_counts.shape == correct_counts.shape)
    assert(np.allclose(pileup_counts, correct_counts))


@pytest.mark.parametrize(
    "start_pos,end_pos,shape,pileup_file",
    [
        (0, 1, (1, 10), os.path.join(get_data_folder(), "subreads_and_truth.pileup")),
        (1, 4, (3, 10), os.path.join(get_data_folder(), "subreads_and_truth.pileup")),
        (14459, 14460, (1, 10), os.path.join(get_data_folder(), "subreads_and_truth.pileup")),
        (5, 6, (2, 10), os.path.join(get_data_folder(), "subreads_and_truth.pileup"))
    ],
)
def test_encoder_region_bounds(start_pos, end_pos, shape, pileup_file):
    encoder = SummaryEncoder(exclude_no_coverage_positions=False, normalize_counts=True)
    # Loop through multiple ranges from checked in test file
    region = FileRegion(start_pos=start_pos,
                        end_pos=end_pos,
                        file_path=pileup_file)
    pileup_counts = encoder(region)
    assert(pileup_counts.shape == shape), "Pileup shape inconsistent with input."
