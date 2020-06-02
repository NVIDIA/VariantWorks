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

import pytest

from variantworks.base_encoder import base_enum_encoder


def test_base_encoder():
    seq = "ATCGNTCGA"
    encoded_seq = [str(base_enum_encoder[n]) for n in seq]
    encoded_seq = "".join(encoded_seq)
    assert encoded_seq == "123452341"
