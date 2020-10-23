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
"""Custom encoder that converts VCF scalar values into a torch tensor."""

import torch

from variantworks.encoders import Encoder


class CustomEncoder(Encoder):
    """An encoder that converts scalar VCF format values into a flattened tensor."""

    def __init__(self, vcf_format_keys=[]):
        """Constructor for the encoder.

        Args:
            vcf_format_keys : A list of format keys to process for the encoding.

        Returns:
            Instance of class.
        """
        self._vcf_format_keys = vcf_format_keys

    def __call__(self, variant):
        """Virtual function that implements the actual encoding.

        Returns:
            VCF values in torch tensor.
        """
        data = []
        for key in self._vcf_format_keys:
            idx = variant.format.index(key)
            val = variant.samples[0][idx]
            if isinstance(val, list):
                data.extend(val)
            else:
                data.append(val)
        tensor = torch.FloatTensor(data)
        return tensor
