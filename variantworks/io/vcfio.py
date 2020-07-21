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
import vcf
import pandas as pd

from variantworks.io.baseio import BaseReader
from variantworks.types import VariantZygosity, VariantType, Variant
from variantworks.utils import extend_exception


class VCFReader(BaseReader):
    """Reader for VCF files."""

    def __init__(self, vcf, bams=[], is_fp=False, require_genotype=True, tags=[]):
        """Parse and extract variants from a vcf/bam tuple.

        Note -VCFReader splits multi-allelic entries into separate variant
        entries.

        Args:
            vcf : Path to VCF file.
            bams : List of BAMs corresponding to the VCF. BAM ordering should match sample
                   ordering in VCF.
            is_fp : Is the VCF for false positive variants.
            require_genotype : If all samples need genotype called.
            tags : List of strings to tag VCF data frame with.

        Returns:
           Instance of class.
        """
        super().__init__()
        self._vcf = vcf
        self._bams = bams
        self._is_fp = is_fp
        self._require_genotype = require_genotype
        self._tags = tags
        self._labels = []
        self._dataframe = None
        self._parse_vcf()

    def __getitem__(self, idx):
        """Get Variant instance in location.

        Args:
            idx: Variant index
        Returns:
            Variant instance
        """
        row = self._dataframe.iloc[idx]
        samples = []
        zygosities = []
        for i in range(row["num_samples"]):
            samples.append(row["sample_{}_call".format(i)])
            zygosities.append(row["sample_{}_zyg".format(i)])
        variant = Variant(chrom=row["chrom"],
                          pos=row["start_pos"],
                          id=row["id"],
                          ref=row["ref"],
                          allele=row["alt"],
                          quality=row["quality"],
                          filter=row["filter"],
                          info=row["info"],
                          format=row["format"],
                          type=row["variant_type"],
                          samples=samples,
                          zygosity=zygosities,
                          vcf=self._vcf,
                          bams=self._bams)
        return variant

    def __len__(self):
        """Return number of Varint objects."""
        return len(self._dataframe)

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
        7. quality - Quality of variant call
        8. filter - VCF FILTER column
        9. info - VCF INFO column
        10. format - VCF FORMAT column
        11. num_samples - Number of samples
        12. sample_{idx}_call - Call information for sample idx.
        13. sample_{idx}_zyg - Zygosity enum for sample idx.

        This dataframe can be easily converted to cuDF for large
        variant processing.

        Returns:
            Parsed variants as pandas DataFrame.
        """
        if self._dataframe is None:
            raise RuntimeError("VCF data frame should be available.")
        return self._dataframe

    def _get_variant_zygosity(self, call):
        """Determine variant type from pyvcf record.

        False positive variants are considered NO_VARIANT entries.

        Args:
            call : a pyVCF call.

        Returns:
            A variant type
        """
        if self._is_fp:
            return VariantZygosity.NO_VARIANT

        if self._require_genotype and call.is_het is None:
            raise RuntimeError(
                "Can not parse call %s, all samples must be called in VCF file" % (call)
            )

        if call.is_het is None:
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

    def _add_variant_to_dict(self, df_dict, record):
        """Create a variant record from pyVCF record.

        Args:
            df_dict : Python dictionary to keep parsed record data in
            record : pyVCF record
        """
        var_type = self._get_variant_type(record)
        # Split multi alleles into multiple entries
        for alt in record.ALT:
            var_allele = alt.sequence
            try:
                var_format = record.FORMAT.split(':')
            except AttributeError:
                if self._is_fp:
                    var_format = []
                else:
                    raise RuntimeError("Could not parse format field for entry - {}".format(record))

            try:
                df_dict["chrom"].append(record.CHROM)
                df_dict["start_pos"].append(record.POS)
                df_dict["end_pos"].append(record.POS + 1)
                df_dict["id"].append(record.ID)
                df_dict["ref"].append(record.REF)
                df_dict["alt"].append(var_allele)
                df_dict["variant_type"].append(var_type)
                df_dict["quality"].append(record.QUAL)
                df_dict["filter"].append(record.FILTER)
                df_dict["info"].append(record.INFO)
                df_dict["format"].append(var_format)
                df_dict["num_samples"].append(len(record.samples))
                for i, sample in enumerate(record.samples):
                    df_dict["sample_{}_call".format(i)].append(
                        [field_value for field_value in sample.data]
                    )
                    df_dict["sample_{}_zyg".format(i)].append(
                        self._get_variant_zygosity(sample)
                    )
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

    def _parse_vcf(self):
        """Parse VCF file and retain labels after they have passed filters."""
        try:
            vcf_reader = self._get_file_reader(vcf_file_path=self._vcf)
            df_dict = defaultdict(list)
            for record in vcf_reader:
                self._add_variant_to_dict(df_dict, record)

            self._dataframe = pd.DataFrame(df_dict)
            for tag in self._tags:
                self._dataframe[tag] = 1

        except Exception as e:
            # raise from None is used to clear the context of the parent exception.
            # Detail description at https://bit.ly/2CoEbHu
            raise extend_exception(e, "VCF file {}".format(self._vcf)) from None
