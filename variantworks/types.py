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
"""Shared enums and types across VariantWorks."""

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Dict


class VariantZygosity(IntEnum):
    """An enum defining zygosity of variant."""

    NONE = -1
    NO_VARIANT = 0
    HOMOZYGOUS = 1
    HETEROZYGOUS = 2


class VariantType(IntEnum):
    """An enum defining type of variant."""

    NONE = -1
    SNP = 0
    INSERTION = 1
    DELETION = 2


@dataclass
class Variant:
    """A dataclass encapsulating a variant."""

    chrom: str
    pos: int
    id: str
    ref: str
    allele: str
    quality: int
    filter: str
    info: Dict
    format: List[List]
    samples: List[List]
    zygosity: List[VariantZygosity]
    type: VariantType
    vcf: str
    bams: List[str]


@dataclass
class FileRegion:
    """A dataclass encapsulating a region of positions within a file."""

    start_pos: int
    end_pos: int
    file_path: str


class BEDEntry:
    """A dataclass encapsulating a BED entry."""

    def __init__(self, kv_dict):
        """Constructor for generic BED Entry.

        Args:
            kv_dict : A dictionary with key-value pairs per column in BED.

        Returns:
            Instance of class.
        """
        self._dict = kv_dict

    def __getattr__(self, attr):
        """Accessor method for BED column."""
        return self._dict[attr]

    def __str__(self):
        """String serializer for BED entry."""
        return str(self._dict)
