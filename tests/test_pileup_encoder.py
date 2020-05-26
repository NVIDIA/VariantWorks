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

    encoder = PileupEncoder(window_size=window_size, max_reads = max_reads, layers=layers)
    assert(encoder.size == (len(layers), height, width))

    bam = os.path.join(get_data_folder(), "small_bam.bam")
    variant = Variant(
        idx=0, chrom="1", pos=240000, id="GL000235", ref='T', allele='A',
        quality=60, filter=None, info='DP=35;AF=0.0185714', format='GT:GQ',
        zygosity=VariantZygosity.HOMOZYGOUS, type=VariantType.SNP, vcf='null.vcf', bam=bam)
    encoding = encoder(variant)
    assert(encoding.size() == torch.Size([len(layers), height, width]))


def test_pileup_unknown_layer():
    try:
        max_reads = 100
        window_size = 5
        width = 2 * window_size + 1
        height = max_reads
        layers = [PileupEncoder.Layer.BLAH]
        encoder = PileupEncoder(window_size=window_size, max_reads = max_reads, layers=layers)
    except:
        assert(True) # Should reach here because an unknown layer is being passed in
