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

from variantworks.types import FileRegion
from variantworks.encoders import HaploidLabelEncoder
from test_utils import get_data_folder


def test_labels_correctness():
    region = FileRegion(start_pos=0,
                        end_pos=14460,
                        file_path=os.path.join(get_data_folder(), "subreads_and_truth.pileup"))
    encoder = HaploidLabelEncoder(exclude_no_coverage_positions=False)
    haploid_labels = encoder(region)
    correct_labels = np.load(os.path.join(get_data_folder(), "sample_haploid_labels.npy"))
    assert(haploid_labels.shape == correct_labels.shape)
    assert(np.allclose(haploid_labels, correct_labels))
