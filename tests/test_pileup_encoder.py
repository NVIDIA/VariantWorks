import os
import pytest
import torch

from claragenomics.variantworks.types import Variant, VariantZygosity, VariantType
from claragenomics.variantworks.variant_encoder import PileupEncoder

from test_utils import get_data_folder

def test_snp_encoder():
    max_reads = 100
    window_size = 5
    width = 2 * window_size + 1
    height = max_reads
    layers = [PileupEncoder.Layer.READ]

    encoder = PileupEncoder(window_size = window_size, max_reads = max_reads, layers = layers)
    assert(encoder.size == (len(layers), height, width))

    bam = os.path.join(get_data_folder(), "small_bam.bam")
    variant = Variant(chrom="1", pos=240000, ref='T', allele='A', zygosity=VariantZygosity.HOMOZYGOUS, vcf='null.vcf', type=VariantType.SNP, bam=bam)
    encoding = encoder(variant)
    assert(encoding.size() == torch.Size([len(layers), height, width]))
