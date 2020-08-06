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

from variantworks.sample_encoder import SummaryEncoder
from test_utils import get_data_folder


class TestRegion(object):
    def __init__(self):
        self.start_pos = 0
        self.end_pos = 14460
        self.pileup = os.path.join(get_data_folder(), "subreads_and_truth.pileup")


def test_counts_correctness():
    region = TestRegion()
    encoder = SummaryEncoder(exclude_no_coverage_positions=False, normalize_counts=True)
    pileup_counts, positions = encoder(region)
    correct_counts = np.load(os.path.join(get_data_folder(), "sample_counts.npy"))
    assert(pileup_counts.shape == correct_counts.shape)
    assert(np.allclose(pileup_counts, correct_counts))


def test_positions_correctness():
    region = TestRegion()
    encoder = SummaryEncoder(exclude_no_coverage_positions=False)
    pileup_counts, positions = encoder(region)
    correct_positions = np.load(os.path.join(get_data_folder(), "sample_positions.npy"))
    assert(positions.shape == correct_positions.shape)
    all_equal = True
    for i in range(len(positions)):
        if (positions[i] != correct_positions[i]):
            all_equal = False
            break
    assert(all_equal)
