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
"""Classes for reading and writing VCFs."""


from collections import defaultdict
from dataclasses import dataclass
import vcf
import pandas as pd

from variantworks.io.baseio import BaseReader
from variantworks.types import VariantZygosity, VariantType, Variant
from variantworks.utils import extend_exception


class VCFReader(BaseReader):
    """Reader for VCF files."""

    @dataclass
    class VcfBamPath:
        """Data class encapsulating paired VCF and BAM inputs."""
        vcf: str
        bam: str
        is_fp: bool = False
        require_genotype: bool = True

    def __init__(self, vcf_bam_list):
        """Parse and extract variants from a vcf/bam tuple.

        Args:
            vcf_bam_list: A list of VcfBamPath namedtuple specifying VCF file and corresponding BAM file.
                           The VCF file must be bgzip compressed and indexed.

        Returns:
           Instance of class.
        """
        super().__init__()
        self._labels = []
        for elem in vcf_bam_list:
            assert (elem.vcf is not None and elem.bam is not None and type(
                elem.is_fp) is bool)
            self._parse_vcf(elem.vcf, elem.bam, self._labels, elem.is_fp, elem.require_genotype)
        self._dataframe = None  # None at init time, only generate when requested.

    def __getitem__(self, idx):
        """Get Variant instance in location.

        Args:
            idx: Variant index
        Returns:
            Variant instance
        """
        return self._labels[idx]

    def __len__(self):
        """Return number of Varint objects."""
        return len(self._labels)

    @property
    def df(self):
        """Get variant list as a CPU pandas dataframe.

        Each row in the returned dataframe represents a variant entry.
        For each variant entry, the following metrics are currently tracked -
        1. chrom - Chromosome
        2. start_pos - Start position of variant (inclusive)
        3. end_pos - End position of variant (exclusive)
        4. ref - Reference base(s)
        5. alt - Alternate base(s)
        6. variant_type - VariantType enum specifying SNP/INSERTION/DELETION

        This dataframe can be easily converted to cuDF for large
        variant processing.

        Returns:
            Parsed variants as pandas DataFrame.
        """
        if not self._dataframe:
            self._dataframe = self._create_dataframe()
        return self._dataframe

    @staticmethod
    def _get_variant_zygosity(call, is_fp, require_genotype):
        """Determine variant type from pyvcf record.

        False positive variants are considered NO_VARIANT entries.

        Args:
            call : a pyVCF call.
            is_fp : is the call a false positive variant.
            require_genotype : does the sample require genotype to be called.

        Returns:
            A variant type
        """
        if is_fp:
            return VariantZygosity.NO_VARIANT

        if require_genotype and call.is_het is None:
            raise RuntimeError(
                "Can not parse call %s, all samples must be called in VCF file" % (call)
            )

        if not call.is_het:
            return None
        elif call.is_het > 0:
            return VariantZygosity.HETEROZYGOUS
        else:
            return VariantZygosity.HOMOZYGOUS

    @staticmethod
    def _get_variant_type(record):
        """Determine variant type.

        Args:
            record : pyVCF entry.

        Returns:
            Type of variant - SNP, INSERTION or DELETION
        """
        if record.is_snp:
            return VariantType.SNP
        elif record.is_indel:
            if record.is_deletion:
                return VariantType.DELETION
            else:
                return VariantType.INSERTION
        raise ValueError("Unexpected variant type - {}".format(record))

    def _create_variant_tuple_from_record(self, record, vcf_file, bam, is_fp, require_genotype):
        """Create a variant record from pyVCF record.

        Args:
            record : pyVCF record
            vcf_file : Path to VCF file
            bam : Path to corresponding BAM file
            is_fp : Boolean indicating whether entry is a false positive variant or not.
            require_genotype : Boolean to indicate if VCF calls require genotype information.

        Returns:
           Variant dataclass record.
        """
        var_type = self._get_variant_type(record)
        # Split multi alleles into multiple entries
        for alt in record.ALT:
            var_allele = alt.sequence
            try:
                var_format = record.FORMAT.split(':')
            except AttributeError:
                if is_fp:
                    var_format = []
                else:
                    raise RuntimeError("Could not parse format field for entry - {}".format(record))

            try:
                yield Variant(chrom=record.CHROM, pos=record.POS, id=record.ID, ref=record.REF,
                              allele=var_allele, quality=record.QUAL, filter=record.FILTER,
                              info=record.INFO, format=var_format,
                              samples=[[field_value for field_value in sample.data]
                                       for sample in record.samples],
                              zygosity=[self._get_variant_zygosity(sample, is_fp, require_genotype)
                                        for sample in record.samples],
                              type=var_type, vcf=vcf_file, bam=bam)
            except Exception as e:
                raise extend_exception(e, "Could not parse variant from entry - {}".format(record)) from None

    @staticmethod
    def _get_file_reader(vcf_file_object=None, vcf_file_path=None):
        """Create VCF file reader from file object or file path.

        Args:
            vcf_file_object: VCF file object
            vcf_file_path: VCF file path

        Returns:
            pyVCF Reader iterator
        """
        if not (vcf_file_object or vcf_file_path):
            raise RuntimeError('You must provide at least one - file object or file path to the vcf reader')
        if vcf_file_path:
            # Check for compressed file
            assert (vcf_file_path[-3:] == ".gz"), "VCF file needs to be compressed and indexed"
        return vcf.Reader(vcf_file_object, vcf_file_path)

    def _parse_vcf(self, vcf_file, bam, labels, is_fp=False, require_genotype=True):
        """Parse VCF file and retain labels after they have passed filters.

        Args:
            vcf_file : Path to VCF file.
            bam : Path to BAM file for VCF.
            labels : List to store parsed variant records.
            is_fp : Boolean to indicate if file is for false positive variants.
            require_genotype : Boolean to indicate if VCF calls require genotype information.
        """
        try:
            vcf_reader = self._get_file_reader(vcf_file_path=vcf_file)
            for record in vcf_reader:
                for variant in self._create_variant_tuple_from_record(record, vcf_file, bam, is_fp, require_genotype):
                    labels.append(variant)
        except Exception as e:
            # raise from None is used to clear the context of the parent exception.
            # Detail description at https://bit.ly/2CoEbHu
            raise extend_exception(e, "VCF file {}".format(vcf_file)) from None

    def _create_dataframe(self):
        """Generate a pandas dataframe with all parsed variant entries.

        Returns:
            Dataframe with variant data.
        """
        df_dict = defaultdict(list)
        for variant in self._labels:
            df_dict["chrom"].append(variant.chrom)
            df_dict["start_pos"].append(variant.pos)
            df_dict["end_pos"].append(variant.pos + 1)
            df_dict["ref"].append(variant.ref)
            df_dict["alt"].append(variant.allele)
            df_dict["variant_type"].append(variant.type)
        return pd.DataFrame(df_dict)
