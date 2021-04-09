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

import pytest

from variantworks.io.bedio import BEDReader

from test_utils import get_data_folder


def test_bedpe_reader():
    """Test correct parsing of BEDPE format."""
    sample_bedpe = os.path.join(get_data_folder(), "sample_bedpe.txt")
    reader = BEDReader(sample_bedpe, BEDReader.BEDType.BEDPE)
    assert(len(reader) == 4)
    assert(reader[1].start1 == 67685907)
    assert(reader[2].id == "DUP00000116")
    assert(reader[3].svtype == "TRA")


def test_unknown_bed_type():
    """Test failure to parse for unknown BED type."""
    sample_bedpe = os.path.join(get_data_folder(), "sample_bedpe.txt")
    with pytest.raises(AssertionError):
        BEDReader(sample_bedpe, "blah")
