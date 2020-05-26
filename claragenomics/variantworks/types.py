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
    pos: int
    id: str
    ref: str
    allele: str
    quality: int
    filter: str
    info: str
    format: str
    zygosity: VariantZygosity
    type: VariantType
    vcf: str
    bam: str
