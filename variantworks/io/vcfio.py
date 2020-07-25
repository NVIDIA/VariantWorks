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

    def __init__(self, vcf, bams=[], is_fp=False, require_genotype=True, tag="caller"):
        """Parse and extract variants from a vcf/bam tuple.

        Note -VCFReader splits multi-allelic entries into separate variant
        entries.

        Args:
            vcf : Path to VCF file.
            bams : List of BAMs corresponding to the VCF. BAM ordering should match sample
                   ordering in VCF.
            is_fp : Is the VCF for false positive variants.
            require_genotype : If all samples need genotype called.
            tag : Tag VCF data frame with.

        Returns:
           Instance of class.
        """
        super().__init__()
        self._vcf = vcf
        self._bams = bams
        self._is_fp = is_fp
        self._require_genotype = require_genotype
        self._tag = tag
        self._labels = []
        self._dataframe = None

        self._filter_df_keys = set()
        self._filter_vcf_keys = set()

        self._call_df_keys = set()
        self._call_vcf_keys = dict()
        self._call_names = set()

        self._info_df_keys = set()
        self._info_vcf_keys = dict()

        self._parse_vcf()


    def __getitem__(self, idx):
        """Get Variant instance in location.

        Args:
            idx: Variant index
        Returns:
            Variant instance
        """
        row = self._dataframe.iloc[idx]

        # Build sample data
        samples = []
        zygosities = []
        format_keys = sorted(self._call_vcf_keys.keys())
        for call in self._call_names:
            call_data = []
            for k in format_keys:
                count = self._call_vcf_keys[k]
                if count == 1:
                    call_data.append(row["{}_{}_{}".format(self._tag, call, k)])
                else:
                    for i in range(count):
                        call_data.append(row["{}_{}_{}_{}".format(self._tag, call, k, i)])
            samples.append(call_data)
            zygosities.append(row["{}_{}_zyg".format(self._tag, call)])

        # Build filter data
        var_filter = []
        for k in self._filter_vcf_keys:
            if row["{}_FILTER_{}".format(self._tag, k)]:
                var_filter.append(k)

        # Build info data
        info = {}
        for k, count in self._info_vcf_keys.items():
            if count == 1:
                info[k] = row["{}_INFO_{}".format(self._tag, k)]
            else:
                vals = []
                for i in range(count):
                    vals.append(row["{}_INFO_{}_{}".format(self._tag, k, i)])
                info[k] = vals

        variant = Variant(chrom=row["chrom"],
                          pos=row["start_pos"],
                          id=row["id"],
                          ref=row["ref"],
                          allele=row["alt"],
                          quality=row["{}_quality".format(self._tag)],
                          filter=(var_filter if var_filter else None),
                          info=info,
                          format=format_keys,
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

    @staticmethod
    def _extend_list(l, expected_length, default_value):
        if len(l) != expected_length:
            l += [default_value] * (expected_length - len(l))

    def _finalize_filters(self, df_dict, total_entries):
        for k in self._filter_df_keys:
            self._extend_list(df_dict[k], total_entries, False)

    def _add_filter(self, df_dict, vcf_filter, idx):
        keys = []
        values = []
        if vcf_filter is None:
            pass
        elif vcf_filter == []:
            keys.append("{}_FILTER_PASS".format(self._tag))
            values.append(True)

            self._filter_vcf_keys.add("PASS")
        else:
            for k in vcf_filter:
                keys.append("{}_FILTER_{}".format(self._tag, k))
                values.append(True)

                self._filter_vcf_keys.add(k)

        for k, v in zip(keys, values):
            self._extend_list(df_dict[k], idx, False)
            df_dict[k].append(v)

            self._filter_df_keys.add(k)

    def _finalize_calls(self, df_dict, total_entries):
        for k in self._call_df_keys:
            self._extend_list(df_dict[k], total_entries, None)

    def _add_call(self, df_dict, format_str, sample, idx):
        name = sample.sample

        self._call_names.add(name)

        keys = []
        values = []

        for col, data in zip(format_str, sample.data):
            if isinstance(data, list):
                for i, val in enumerate(data):
                    keys.append("{}_{}_{}_{}".format(self._tag, name, col, i))
                    values.append(val)

                self._call_vcf_keys[col] = len(data)
            else:
                keys.append("{}_{}_{}".format(self._tag, name, col))
                values.append(data)

                self._call_vcf_keys[col] = 1

        if self._require_genotype:
            keys.append("{}_{}_zyg".format(self._tag, name))
            values.append(self._get_variant_zygosity(sample))

        for k, v in zip(keys, values):
            self._extend_list(df_dict[k], idx, None)
            df_dict[k].append(v)

            self._call_df_keys.add(k)

    def _finalize_info(self, df_dict, total_entries):
        for k in self._info_df_keys:
            self._extend_list(df_dict[k], total_entries, None)

    def _add_info(self, df_dict, info, idx):
        keys = []
        values = []
        for k, v in info.items():
            if isinstance(v, list) and len(v) > 1:
                for i in range(len(v)):
                    keys.append("{}_INFO_{}_{}".format(self._tag, k, i))
                    values.append(v[i])

                self._info_vcf_keys[k] = len(v)
            else:
                keys.append("{}_INFO_{}".format(self._tag, k))
                values.append(v)

                self._info_vcf_keys[k] = 1

        for k, v in zip(keys, values):
            self._extend_list(df_dict[k], idx, None)
            df_dict[k].append(v)

            self._info_df_keys.add(k)

    def _add_variant_to_dict(self, df_dict, record, idx):
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
                # Common columns
                df_dict["chrom"].append(record.CHROM)
                df_dict["start_pos"].append(record.POS)
                df_dict["end_pos"].append(record.POS + 1)
                df_dict["id"].append(record.ID)
                df_dict["ref"].append(record.REF)
                df_dict["alt"].append(var_allele)
                df_dict["variant_type"].append(var_type)
                df_dict["{}_quality".format(self._tag)].append(record.QUAL)
                # Columns that vary by VCF
                self._add_info(df_dict, record.INFO, idx)
                self._add_filter(df_dict, record.FILTER, idx)
                #df_dict["info"].append(record.INFO)
                for sample in record.samples:
                    self._add_call(df_dict, var_format, sample, idx)
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
            record_counter = 0
            for idx, record in enumerate(vcf_reader):
                self._add_variant_to_dict(df_dict, record, idx)
                #if record_counter == 400:
                #    record_counter += 1
                #    break
                record_counter += 1
            self._finalize_filters(df_dict, record_counter)
            self._finalize_calls(df_dict, record_counter)
            self._finalize_info(df_dict, record_counter)

            self._dataframe = pd.DataFrame(df_dict)
            self._dataframe[self._tag] = 1

            #print(self._dataframe)
            #print(self._info_df_keys)

        except Exception as e:
            # raise from None is used to clear the context of the parent exception.
            # Detail description at https://bit.ly/2CoEbHu
            raise extend_exception(e, "VCF file {}".format(self._vcf)) from None
