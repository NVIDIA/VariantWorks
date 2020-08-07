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
import torch

from variantworks.types import Variant, VariantZygosity, VariantType
from variantworks.encoders import ZygosityLabelEncoder

from test_utils import get_data_folder


def test_zygosity_encoder():
    encoder = ZygosityLabelEncoder()

    bam = os.path.join(get_data_folder(), "small_bam.bam")
    variant = Variant(chrom="1", pos=240000, id="GL000235", ref='T', allele='A', quality=60, filter=None,
                      info={'DP': 35, 'AF': 0.0185714}, format=['GT', 'GQ'], samples=[['1/1', '50']],
                      zygosity=[VariantZygosity.HOMOZYGOUS], type=VariantType.SNP, vcf='null.vcf', bams=[bam])
    encoding = encoder(variant)
    # Since it should return a scalar
    assert(encoding.size() == torch.Size([]))
    assert(encoding == 1)

    variant = Variant(chrom="1", pos=240000, id="GL000235", ref='T', allele='A', quality=60, filter=None,
                      info={'DP': 35, 'AF': 0.0185714}, format=['GT', 'GQ'], samples=[['0/0', '50']],
                      zygosity=[VariantZygosity.NO_VARIANT], type=VariantType.SNP, vcf='null.vcf', bams=[bam])
    encoding = encoder(variant)
    assert(encoding == 0)

    variant = Variant(chrom="1", pos=240000, id="GL000235", ref='T', allele='A', quality=60, filter=None,
                      info={'DP': 35, 'AF': 0.0185714}, format=['GT', 'GQ'], samples=[['0/1', '50']],
                      zygosity=[VariantZygosity.HETEROZYGOUS], type=VariantType.SNP, vcf='null.vcf', bams=[bam])
    encoding = encoder(variant)
    assert(encoding == 2)
