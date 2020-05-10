#Shared enums and types acrosss VariantWorks

from collections import namedtuple
from enum import Enum

class VariantZygosity(Enum):
    NO_VARIANT = 0
    HOMOZYGOUS = 1
    HETEROZYGOUS = 2

Variant = namedtuple('Variant', ['chrom', 'pos', 'ref', 'zygosity', 'allele'])

