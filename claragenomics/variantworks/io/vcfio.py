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

# Implementation clases for VCF IO.
from abc import ABC, abstractmethod
from collections import namedtuple
import vcf
import warnings

from claragenomics.variantworks.types import VariantZygosity, VariantType, Variant


class VCFReaderIterator:
    def __init__(self, vcf_reader):
        assert(isinstance(vcf_reader, VCFReader))
        self._vcf_reader = vcf_reader
        self._index = 0

    def __next__(self):
        if self._index < len(self._vcf_reader):
            result = self._vcf_reader[self._index]
            self._index += 1
            return result
        raise StopIteration


class VCFReader():
    """VCF based label loader for true and false positive example files.
    """

    VcfBamPaths = namedtuple(
        'VcfBamPaths', ['vcf', 'bam', 'is_fp'], defaults=[False])

    def __init__(self, vcf_bam_list):
        super().__init__()
        self._labels = []
        for elem in vcf_bam_list:
            assert (elem.vcf is not None and elem.bam is not None and type(
                elem.is_fp) is bool)
            self._parse_vcf(elem.vcf, elem.bam, self._labels, elem.is_fp)

    def __getitem__(self, idx):
        return self._labels[idx]

    def __len__(self):
        return len(self._labels)

    def __iter__(self):
        return VCFReaderIterator(self)

    @staticmethod
    def _get_variant_zygosity(record, is_fp=False):
        """Determine variant type from pyvcf record.
        """
        if is_fp:
            return VariantZygosity.NO_VARIANT
        if record.num_het > 0:
            return VariantZygosity.HETEROZYGOUS
        elif record.num_hom_alt > 0:
            return VariantZygosity.HOMOZYGOUS
        raise ValueError("Unexpected variant zygosity - {}, num_het - {}, num_hom_alt - {}".format(record, record.num_het, record.num_hom_alt))

    @staticmethod
    def _get_variant_type(record):
        """Determine variant type.
        """
        if record.is_snp:
            return VariantType.SNP
        elif record.is_indel:
            if record.is_deletion:
                return VariantType.DELETION
            else:
                return VariantType.INSERTION
        raise ValueError("Unexpected variant type - {}".format(record))

    def _create_variant_tuple_from_record(self, record, vcf_file, bam, is_fp):
        var_zyg = self._get_variant_zygosity(record, is_fp)
        var_type = self._get_variant_type(record)
        # Split multi alleles into multiple entries
        for alt in record.ALT:
            var_allele = alt.sequence
            try:
                yield Variant(chrom=record.CHROM, pos=record.POS, id=record.ID, ref=record.REF,
                              allele=var_allele, quality=record.QUAL, filter=record.FILTER,
                              info=record.INFO, format=record.FORMAT.split(':'),
                              samples=[[field_value for field_value in sample.data]
                                       for sample in record.samples],
                              zygosity=var_zyg, type=var_type, vcf=vcf_file, bam=bam)
            except:
                raise RuntimeError("Could not parse variant from entry - {}".format(record))

    def _parse_vcf(self, vcf_file, bam, labels, is_fp=False):
        """Parse VCF file and retain labels after they have passed filters.
        """
        assert(
            vcf_file[-3:] == ".gz"), "VCF file needs to be compressed and indexed"  # Check for compressed file
        vcf_reader = vcf.Reader(filename=vcf_file)
        if len(vcf_reader.samples) != 1:
            raise RuntimeError(
                "Can not parse: {}. VariantWorks currently only supports single sample VCF files".format(vcf_file))
        for record in vcf_reader:
            if record.num_called < len(vcf_reader.samples):
                raise RuntimeError(
                    "Can not parse record %s in %s,  all samples must be called" % (record, vcf_file))
            if not record.is_snp:
                #warnings.warn("%s is filtered - not an SNP record" % record)
                continue
            if len(record.ALT) > 1:
                warnings.warn(
                    "%s is filtered - multiallele recrods are not supported" % record)
                continue
            for variant in self._create_variant_tuple_from_record(record, vcf_file, bam, is_fp):
                labels.append(variant)
