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
    chrom: str
    pos: str
    ref: str
    zygosity: VariantZygosity
    type: VariantType
    allele: str
    vcf: str
    bam: str
