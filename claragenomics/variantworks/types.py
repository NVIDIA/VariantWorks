#Shared enums and types acrosss VariantWorks

from collections import namedtuple
from enum import Enum


class VariantZygosity(Enum):
    NO_VARIANT = 0
    HOMOZYGOUS = 1
    HETEROZYGOUS = 2


class VariantType(Enum):
    SNP = 0
    INSERTION = 1
    DELETION = 2


Variant = namedtuple('Variant', ['chrom', 'pos', 'ref', 'zygosity', 'type', 'allele', 'vcf', 'bam'])

VcfBamPaths = namedtuple('VcfBamPaths', ['vcf', 'bam', 'is_fp'], defaults=[False])
