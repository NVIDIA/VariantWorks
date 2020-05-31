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
import torch

from claragenomics.variantworks.base_encoder import base_enum_encoder
from claragenomics.variantworks.types import Variant, VariantZygosity, VariantType
from claragenomics.variantworks.sample_encoder import PileupEncoder

from test_utils import get_data_folder


@pytest.fixture
def snp_variant():
    bam = os.path.join(get_data_folder(), "small_bam.bam")
    variant = Variant(chrom="1", pos=240000, id="GL000235", ref='T', allele='A',
                      quality=60, filter=None, info={'DP': 35, 'AF': 0.0185714}, format=['GT', 'GQ'],
                      samples=[['1/1', '50']], zygosity=VariantZygosity.HOMOZYGOUS,
                      type=VariantType.SNP, vcf='null.vcf', bam=bam)
    return variant


def test_snp_encoder_basic(snp_variant):
    max_reads = 100
    window_size = 5
    width = 2 * window_size + 1
    height = max_reads
    layers = [PileupEncoder.Layer.READ]

    encoder = PileupEncoder(window_size=window_size, max_reads = max_reads, layers=layers)
    assert(encoder.size == (len(layers), height, width))

    variant = snp_variant

    encoding = encoder(variant)
    assert(encoding.size() == torch.Size([len(layers), height, width]))


def test_snp_ref_encoding(snp_variant):
    max_reads = 1
    window_size = 5
    layers = [PileupEncoder.Layer.REFERENCE]

    encoder = PileupEncoder(window_size=window_size, max_reads = max_reads, layers=layers)

    variant = snp_variant
    encoding = encoder(variant)
    assert(encoding[0, 0, window_size] == base_enum_encoder[variant.ref])


def test_snp_allele_encoding(snp_variant):
    max_reads = 1
    window_size = 5
    layers = [PileupEncoder.Layer.ALLELE]

    encoder = PileupEncoder(window_size=window_size, max_reads=max_reads, layers=layers)

    variant = snp_variant
    encoding = encoder(variant)
    assert(encoding[0, 0, window_size] == base_enum_encoder[variant.allele])


def test_pileup_unknown_layer():
    try:
        max_reads = 100
        window_size = 5
        width = 2 * window_size + 1
        height = max_reads
        layers = [PileupEncoder.Layer.BLAH]
        encoder = PileupEncoder(window_size=window_size, max_reads=max_reads, layers=layers)
    except:
        assert(True) # Should reach here because an unknown layer is being passed in
