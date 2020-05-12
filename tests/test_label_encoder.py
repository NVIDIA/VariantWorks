import os
import pytest
import torch

from claragenomics.variantworks.types import Variant, VariantZygosity, VariantType
from claragenomics.variantworks.variant_encoder import ZygosityLabelEncoder

from test_utils import get_data_folder

def test_zygosity_encoder():
    encoder = ZygosityLabelEncoder()
    assert(encoder.size() == 1)

    bam = os.path.join(get_data_folder(), "small_bam.bam")
    variant = Variant(chrom="1", pos=240000, ref='T', allele='A', zygosity=VariantZygosity.HOMOZYGOUS, vcf='null.vcf', type=VariantType.SNP, bam=bam)
    encoding = encoder(variant)
    assert(encoding.size() == torch.Size([])) # Since it should return a scalar
    assert(encoding == 1)

    variant = Variant(chrom="1", pos=240000, ref='T', allele='A', zygosity=VariantZygosity.NO_VARIANT , vcf='null.vcf', type=VariantType.SNP, bam=bam)
    encoding = encoder(variant)
    assert(encoding == 0)

    variant = Variant(chrom="1", pos=240000, ref='T', allele='A', zygosity=VariantZygosity.HETEROZYGOUS, vcf='null.vcf', type=VariantType.SNP, bam=bam)
    encoding = encoder(variant)
    assert(encoding == 2)
