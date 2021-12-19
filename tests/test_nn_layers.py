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

import torch

from variantworks.layers.attention import Attention


def test_attention_layer():
    input_tensor = torch.zeros((10, 10, 5), dtype=torch.float32)
    attn_layer = Attention(5)
    out, _ = attn_layer(input_tensor, input_tensor)
    assert(torch.all(input_tensor.eq(out)))
