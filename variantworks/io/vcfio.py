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
import cyvcf2

import multiprocessing as mp
from functools import partial

from variantworks.io.baseio import BaseReader
from variantworks.types import VariantZygosity, VariantType, Variant
from variantworks.utils import extend_exception


class VCFReader(BaseReader):
    """Reader for VCF files."""

    def __init__(self, vcf, bams=[], is_fp=False, require_genotype=True, tag="caller", info_keys=[], filter_keys=[], format_keys=["*"], regions=None, num_threads=mp.cpu_count(), chunksize=5000):
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

        #self._parse_vcf()

        self._info_vcf_keys = info_keys
        self._info_vcf_key_counts = dict()
        self._filter_vcf_keys = filter_keys
        self._format_vcf_keys = format_keys
        self._format_vcf_key_counts = dict()

        self._regions = regions
        self._num_threads = num_threads
        self._chunksize = chunksize

        #self._parse_vcf_cyvcf()
        self._parallel_parse_vcf()


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
        format_keys = sorted(self._format_vcf_key_counts.keys())
        for call in self._call_names:
            call_data = []
            for k in format_keys:
                count = self._format_vcf_key_counts[k]
                if count == 1:
                    call_data.append(row["{}_{}".format(call, k)])
                else:
                    for i in range(count):
                        call_data.append(row["{}_{}_{}".format(call, k, i)])
            samples.append(call_data)
            zygosities.append(row["{}_zyg".format(call)])

        # Build filter data
        var_filter = []
        for k in self._filter_vcf_keys:
            if row["FILTER_{}".format(k)]:
                var_filter.append(k)

        # Build info data
        info = {}
        for k, count in self._info_vcf_key_counts.items():
            if count == 1:
                info[k] = row["INFO_{}".format(k)]
            else:
                vals = []
                for i in range(count):
                    vals.append(row["INFO_{}_{}".format(k, i)])
                info[k] = vals

        variant = Variant(chrom=row["chrom"],
                          pos=row["start_pos"],
                          id=row["id"],
                          ref=row["ref"],
                          allele=row["alt"],
                          quality=row["quality".format(self._tag)],
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

    def _detect_variant_type(self, ref, alt):
        if len(ref) == len(alt):
            return VariantType.SNP
        elif len(ref) < len(alt):
            return VariantType.INSERTION
        else:
            return VariantType.DELETION

    def _detect_zyg(self, gt):
        if self._is_fp:
            return VariantZygosity.NO_VARIANT

        if gt[0] == -1 or gt[1] == -1:
            return None
        elif gt[0] == gt[1]:
            if gt[0] == 0:
                return VariantZygosity.NO_VARIANT
            else:
                return VariantZygosity.HOMOZYGOUS
        else:
            return VariantZygosity.HETEROZYGOUS

    def _get_normalized_count(self, header_number, num_alts):
        if header_number == "A":
            return num_alts
        elif header_number == "R":
            return (num_alts + 1)
        elif header_number.isdigit():
            return int(header_number)
        elif header_number == ".":
            return 1

    def _create_df(self, vcf, variant_list):
        df_dict = defaultdict(list)

        samples = vcf.samples
        #print(samples)

        for variant in variant_list:
            alts = variant.ALT
            for alt_idx, alt in enumerate(alts):
                df_dict["chrom"].append(variant.CHROM)
                df_dict["start_pos"].append(variant.start)
                df_dict["end_pos"].append(variant.end)
                df_dict["id"].append(variant.ID)
                df_dict["ref"].append(variant.REF)
                df_dict["alt"].append(alt)
                df_dict["variant_type"].append(self._detect_variant_type(variant.REF, alt))
                df_dict["quality"].append(variant.QUAL)

                # Process variant filter
                variant_filter = "PASS" if variant.FILTER is None else variant.FILTER
                filter_set = set(variant_filter.split(";"))
                for filter_col in self._filter_vcf_keys:
                    df_dict["FILTER_" + filter_col].append(filter_col in filter_set)

                # Process info columns
                for info_col in self._info_vcf_keys:
                    # Get header type
                    header_number = vcf.get_header_type(info_col)['Number']

                    if info_col in variant.INFO:
                        val = variant.INFO[info_col]
                        # Make value a tuple to reduce special case handling later.
                        if not isinstance(val, tuple):
                            val = tuple((val,))
                    else:
                        val = [None] * self._get_normalized_count(header_number, len(alts))

                    #print(info_col, val)
                    if header_number == "A":
                        df_dict["INFO_" + info_col].append(val[alt_idx])
                    elif header_number == "R":
                        df_dict["INFO_" + info_col + "_REF"].append(val[0])
                        df_dict["INFO_" + info_col + "_ALT"].append(val[alt_idx + 1])
                    elif header_number.isdigit():
                        header_number = int(header_number)
                        if header_number == 1:
                            df_dict["INFO_" + info_col].append(val[0])
                        else:
                            for i in range(int(header_number)):
                                df_dict["INFO_" + info_col + "_" + str(i)].append(val[i])
                    elif header_number == ".":
                        df_dict["INFO_" + info_col].append(",".join([str(v) for v in val]))

                # Process format columns
                for format_col in self._format_vcf_keys:
                    #print(format_col, variant.format(format_col))
                    for sample_idx, sample_name in enumerate(samples):
                        if format_col == "GT":
                            def fix_gt(gt_alt_id, loop_alt_id):
                                if gt_alt_id == loop_alt_id:
                                    return 1
                                elif gt_alt_id != 0:
                                    return -1
                                else:
                                    return 0
                            # Handle GT column specially
                            gt = variant.genotypes[sample_idx]
                            # Fixup haplotype number based on multi allele split
                            alt_id = alt_idx + 1
                            gt[0] = fix_gt(gt[0], alt_id)
                            gt[1] = fix_gt(gt[1], alt_id)
                            if gt[0] == -1 or gt[1] == -1:
                                gt[0] = gt[1] = -1
                            df_dict["{}_zyg".format(sample_name)].append(self._detect_zyg(gt))
                            df_dict["{}_GT".format(sample_name)].append("{}/{}".format(gt[0], gt[1]))
                        else:
                            # Get header type
                            header_number = vcf.get_header_type(format_col)['Number']

                            val = variant.format(format_col)
                            if val is not None:
                                val = val[sample_idx]
                            else:
                                val = [None] * self._get_normalized_count(header_number, len(alts))


                            #print(format_col, val)
                            if header_number == "A":
                                df_dict[sample_name + "_" + format_col].append(val[alt_idx])
                            elif header_number == "R":
                                df_dict[sample_name + "_" + format_col + "_REF"].append(val[0])
                                df_dict[sample_name + "_" + format_col + "_ALT"].append(val[alt_idx + 1])
                            elif header_number.isdigit():
                                header_number = int(header_number)
                                if header_number == 1:
                                    df_dict[sample_name + "_" + format_col].append(val[0])
                                else:
                                    #print(header_number, format_col, val)
                                    for i in range(int(header_number)):
                                        df_dict[sample_name + "_" + format_col + "_" + str(i)].append(val[i])
                            elif header_number == ".":
                                df_dict[sample_name + "_" + format_col].append(",".join([str(v) for v in val]))


        df = pd.DataFrame.from_dict(df_dict)
        #print(df)
        return df


    def _parse_vcf_cyvcf(self, thread_id, chunksize, total_threads):
        vcf = cyvcf2.VCF(self._vcf)

        # Go through variants and add to list
        df_dict = defaultdict(list)
        variant_list = []
        df_list = []
        generator = vcf(self._regions) if self._regions else vcf
        for idx, variant in enumerate(generator):
            if ((idx // chunksize) % total_threads == thread_id):
                variant_list.append(variant)
                if idx % chunksize == 0:
                    df_list.append(self._create_df(vcf, variant_list))
                    variant_list = []
                    print("added", idx)
        if variant_list:
            df_list.append(self._create_df(vcf, variant_list))

        #self._dataframe = pd.concat(df_list)
        if df_list:
            return pd.concat(df_list, ignore_index=True)
        else:
            return pd.DataFrame()

    def _parallel_parse_vcf(self):
        vcf = cyvcf2.VCF(self._vcf)

        # Populate column keys if all are requested
        if "*" in self._info_vcf_keys:
            self._info_vcf_keys = []
            for h in vcf.header_iter():
                if h['HeaderType'] == 'INFO':
                    self._info_vcf_keys.append(h['ID'])
        for k in self._info_vcf_keys:
            header_number = vcf.get_header_type(k)['Number']
            self._info_vcf_key_counts[k] = self._get_normalized_count(header_number, 1)


        if "*" in self._filter_vcf_keys:
            self._filter_vcf_keys = []
            for h in vcf.header_iter():
                if h['HeaderType'] == 'FILTER':
                    self._filter_vcf_keys.append(h['ID'])

        if "*" in self._format_vcf_keys:
            self._format_vcf_keys = []
            for h in vcf.header_iter():
                if h['HeaderType'] == 'FORMAT':
                    self._format_vcf_keys.append(h['ID'])
        for k in self._format_vcf_keys:
            header_number = vcf.get_header_type(k)['Number']
            self._format_vcf_key_counts[k] = self._get_normalized_count(header_number, 1)


        for sample in vcf.samples:
            self._call_names.add(sample)

        threads = self._num_threads
        pool = mp.Pool(threads)
        df_list = []
        func = partial(self._parse_vcf_cyvcf, chunksize=self._chunksize, total_threads=threads)
        for df in pool.imap(func, range(threads)):
            df_list.append(df)
        #self._parse_vcf_cyvcf(1, 10000, 3)
        self._dataframe = pd.concat(df_list, ignore_index=True)
