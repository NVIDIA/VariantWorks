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
import shutil
import tempfile
import torch

from variantworks.base_encoder import base_char_value_encoder
from variantworks.types import Variant, VariantZygosity, VariantType
from variantworks.sample_encoder import BaseEnumEncoder, PileupEncoder

from test_utils import get_data_folder


@pytest.fixture
def snp_variant():
    bam = os.path.join(get_data_folder(), "some_indels.bam")
    variant = Variant(chrom="1", pos=10106775, id="rs12406448", ref='T', allele='C',
                      quality=50, filter=[], info={}, format=['GT', 'PS', 'DP', 'ADALL', 'AD', 'GQ'],
                      samples=[['0/1', None, 638, [149, 142], [175, 174], 1079]], zygosity=VariantZygosity.HETEROZYGOUS,
                      type=VariantType.SNP, vcf='null.vcf', bam=bam)
    return variant


@pytest.fixture
def insertion_variant():
    bam = os.path.join(get_data_folder(), "some_indels.bam")
    variant = Variant(chrom="1", pos=10122622, id="rs57037935", ref='T', allele='TG',
                      quality=50, filter=[], info={}, format=['GT', 'PS', 'DP', 'ADALL', 'AD', 'GQ'],
                      samples=[['1/1', None, 546, [0, 246], [25, 25], 330]], zygosity=VariantZygosity.HOMOZYGOUS,
                      type=VariantType.INSERTION, vcf='null.vcf', bam=bam)
    return variant


@pytest.fixture
def deletion_variant():
    bam = os.path.join(get_data_folder(), "some_indels.bam")
    variant = Variant(chrom="1", pos=10163457, id=None, ref='CTTTA', allele='C',
                      quality=50, filter=[], info={}, format=['GT', 'PS', 'DP', 'ADALL', 'AD', 'GQ'],
                      samples=[['1/0', None, 177, [0, 0, 0], [0, 0, 0], 160]], zygosity=VariantZygosity.HETEROZYGOUS,
                      type=VariantType.DELETION, vcf='null.vcf', bam=bam)
    return variant


def test_snp_encoder_basic(snp_variant):
    max_reads = 100
    window_size = 10
    width = 2 * window_size + 1
    height = max_reads
    layers = [PileupEncoder.Layer.READ]

    encoder = PileupEncoder(window_size=window_size,
                            max_reads=max_reads, layers=layers)

    variant = snp_variant

    encoding = encoder(variant)
    assert(encoding.size() == torch.Size([len(layers), height, width]))


def test_snp_ref_encoding(snp_variant):
    max_reads = 1
    window_size = 5
    layers = [PileupEncoder.Layer.REFERENCE]

    encoder = PileupEncoder(window_size=window_size,
                            max_reads=max_reads, layers=layers)

    variant = snp_variant
    encoding = encoder(variant)
    assert(encoding[0, 0, window_size] == BaseEnumEncoder()(variant.ref))


def test_snp_allele_encoding(snp_variant):
    max_reads = 1
    window_size = 5
    layers = [PileupEncoder.Layer.ALLELE]

    encoder = PileupEncoder(window_size=window_size,
                            max_reads=max_reads, layers=layers)

    variant = snp_variant
    encoding = encoder(variant)
    assert(encoding[0, 0, window_size] == BaseEnumEncoder()(variant.allele))


def test_snp_encoder_base_quality(snp_variant):
    max_reads = 100
    window_size = 5
    width = 2 * window_size + 1
    height = max_reads
    layers = [PileupEncoder.Layer.BASE_QUALITY]

    encoder = PileupEncoder(window_size=window_size,
                            max_reads=max_reads, layers=layers)

    variant = snp_variant

    encoding = encoder(variant)
    assert(encoding.size() == torch.Size([len(layers), height, width]))

    # Verify that all elements are <= 1 by first outputing a bool tensor
    # and then converting it to a long tensor and summing up all elements to match
    # against total size.
    all_lt_1 = (encoding <= 1.0).long()
    assert(torch.sum(all_lt_1) == (height * width))


def test_snp_encoder_mapping_quality(snp_variant):
    max_reads = 100
    window_size = 5
    width = 2 * window_size + 1
    height = max_reads
    layers = [PileupEncoder.Layer.MAPPING_QUALITY]

    encoder = PileupEncoder(window_size=window_size,
                            max_reads=max_reads, layers=layers)

    variant = snp_variant

    encoding = encoder(variant)
    assert(encoding.size() == torch.Size([len(layers), height, width]))

    # Verify that all elements are <= 1 by first outputing a bool tensor
    # and then converting it to a long tensor and summing up all elements to match
    # against total size.
    all_lt_1 = (encoding <= 1.0).long()
    assert(torch.sum(all_lt_1) == (height * width))


def test_insertion_read_encoding(insertion_variant):
    max_reads = 100
    window_size = 30
    width = 2 * window_size + 1
    height = max_reads
    layers = [PileupEncoder.Layer.READ, PileupEncoder.Layer.REFERENCE, PileupEncoder.Layer.ALLELE]

    encoder = PileupEncoder(window_size=window_size,
                            max_reads=max_reads, layers=layers)

    variant = insertion_variant

    encoding = encoder(variant)
    assert(encoding.size() == torch.Size([len(layers), height, width]))


def test_deletion_read_encoding(deletion_variant):
    max_reads = 100
    window_size = 10
    width = 2 * window_size + 1
    height = max_reads
    layers = [PileupEncoder.Layer.READ, PileupEncoder.Layer.REFERENCE, PileupEncoder.Layer.ALLELE]

    encoder = PileupEncoder(window_size=window_size,
                            max_reads=max_reads, layers=layers)

    variant = deletion_variant

    encoding = encoder(variant)
    assert(encoding.size() == torch.Size([len(layers), height, width]))


def test_pileup_unknown_layer():
    max_reads = 100
    window_size = 5
    with pytest.raises(AttributeError):
        layers = [PileupEncoder.Layer.BLAH]
        PileupEncoder(window_size=window_size, max_reads=max_reads, layers=layers)


def test_pileup_visualization(snp_variant):
    output_folder = tempfile.mkdtemp(prefix='vw_test_output_')
    encoder = PileupEncoder(
        layers=[PileupEncoder.Layer.READ, PileupEncoder.Layer.ALLELE, PileupEncoder.Layer.REFERENCE,
                PileupEncoder.Layer.BASE_QUALITY, PileupEncoder.Layer.MAPPING_QUALITY],
        base_encoder=base_char_value_encoder
    )
    encoder.visualize(snp_variant, save_to_path=output_folder, max_subplots_per_line=2)
    assert len([name for name in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, name))]) == 1
    shutil.rmtree(output_folder)
