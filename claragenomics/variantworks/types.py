# Shared enums and types acrosss VariantWorks

from dataclasses import dataclass
from enum import Enum


class VariantZygosity(Enum):
    NO_VARIANT = 0
    HOMOZYGOUS = 1
    HETEROZYGOUS = 2


class VariantType(Enum):
    SNP = 0
    INSERTION = 1
    DELETION = 2


@dataclass
class Variant:
    idx: int
    chrom: str
    pos: str
    ref: str
    zygosity: VariantZygosity
    type: VariantType
    allele: str
    vcf: str
    bam: str
