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
import pandas as pd
import cyvcf2
import numpy as np

import multiprocessing as mp
from functools import partial

from variantworks.io.baseio import BaseReader
from variantworks.types import VariantZygosity, VariantType, Variant
from variantworks.utils.exceptions import extend_exception


def _df_row_to_variant(dataframe, idx, format_vcf_key_counts, sample_names, filter_vcf_keys, info_vcf_key_counts, header_type, vcf, bams):
    # Convert row Series into dict to improve access performance.
    row = dataframe.iloc[idx].to_dict()

    # Build sample data by iterating through FORMAT columns.
    samples = []
    zygosities = []
    format_keys = sorted(format_vcf_key_counts.keys())
    for call in sample_names:
        call_data = []
        for k in format_keys:
            count = format_vcf_key_counts[k]
            if count == 1:
                call_data.append(header_type[k](row["{}_{}".format(call, k)]))
            else:
                vals = []
                for i in range(count):
                    vals.append(header_type[k](row["{}_{}_{}".format(call, k, i)]))
                call_data.append(vals)
        samples.append(call_data)
        zyg_col = "{}_zyg".format(call)
        if zyg_col in row:
            zygosities.append(VariantZygosity(row[zyg_col]))

    # Build filter data by iterating over all saved FILTER keys.
    var_filter = []
    for k in filter_vcf_keys:
        if row["FILTER_{}".format(k)]:
            var_filter.append(k)

    # Build info data by iterating over all saved INFO keys.
    info = {}
    for k, count in info_vcf_key_counts.items():
        if count == 1:
            val = header_type[k](row["{}".format(k)])
            info[k] = val
        else:
            vals = []
            for i in range(count):
                vals.append(header_type[k](row["{}_{}".format(k, i)]))
            info[k] = vals

    variant = Variant(chrom=row["chrom"],
                      pos=row["start_pos"],
                      id=row["id"],
                      ref=row["ref"],
                      allele=row["alt"],
                      quality=row["quality"],
                      filter=(var_filter if var_filter else None),
                      info=info,
                      format=format_keys,
                      type=VariantType(row["variant_type"]),
                      samples=samples,
                      zygosity=zygosities,
                      vcf=vcf,
                      bams=bams)
    return variant


class VCFReader(BaseReader):
    """Reader for VCF files."""

    def __init__(self,
                 vcf,
                 bams=[],
                 is_fp=False,
                 require_genotype=True,
                 tags={},
                 info_keys=[],
                 filter_keys=[],
                 format_keys=["*"],
                 regions=[],
                 num_threads=mp.cpu_count(),
                 chunksize=5000,
                 sort=False,
                 unbounded_val_max_col=3):
        """Parse and extract variants from a vcf/bam tuple.

        Note VCFReader splits multi-allelic entries into separate variant
        entries. VCFReader also doesn't maintain the original ordering of variants
        from the VCF becasue of its multi threaded nature for parsing.

        Args:
            vcf : Path to VCF file.
            bams : List of BAMs corresponding to the VCF. BAM ordering should match sample ordering in VCF.
            is_fp : Is the VCF for false positive variants.
            require_genotype : If all samples need genotype called.
            tags : A dictionary with custom tags for VCF dataframe.\
                Format = {"tag1" : 1, "tag2" : False}. [{} by default]
            info_keys : List of INFO columns to parse. For all columns, pass "*". [Empty by default].
            filter_keys : List of FILTER columns to parse. For all columns, pass "*". [Empty be default].
            format_keys : List of FORMAT columns to parse. For all columns, pass "*". [Empty by default].
            regions : Region of VCF to parse. Needs tabix index (vcf name + .tbi).\
                Format = ["chr:start-end", "chr2:s-e",...]. [None by default]
            num_threads : Number of threads to use for parallel parsing of VCF. [CPU count by default]
            chunksize : Number of VCF rows to parse in a single threads. [5000 by default].
            sort : Order DataFrame by chr and start position.
            unbounded_val_max_col : Number of values to assume for unbounded VCF keys. [3 by default]

        Returns:
           Instance of class.
        """
        super().__init__()
        self._vcf = vcf
        self._bams = bams
        self._is_fp = is_fp
        self._require_genotype = require_genotype
        self._tags = tags
        self._regions = regions if regions else [None]
        self._num_threads = num_threads
        self._chunksize = chunksize
        self._sort = sort
        self._unbounded_val_max_col = unbounded_val_max_col

        self._dataframe = None
        self._sample_names = list()

        # Keep track of metadata per column
        self._info_vcf_keys = info_keys
        self._info_vcf_key_counts = dict()

        self._filter_vcf_keys = filter_keys

        self._format_vcf_keys = format_keys
        self._format_vcf_key_counts = dict()

        self._header_number = {}
        self._header_type = {}

        # Parse the VCF
        self._parallel_parse_vcf()

    @property
    def samples(self):
        """Get list of samples names in VCF file."""
        return self._sample_names

    def __getitem__(self, idx):
        """Get Variant instance in location.

        Args:
            idx: Variant index

        Returns:
            Variant instance
        """
        return _df_row_to_variant(self._dataframe,
                                  idx,
                                  self._format_vcf_key_counts,
                                  self._sample_names,
                                  self._filter_vcf_keys,
                                  self._info_vcf_key_counts,
                                  self._header_type,
                                  self._vcf,
                                  self._bams)

    def __len__(self):
        """Return number of Varint objects."""
        return len(self._dataframe)

    @property
    def dataframe(self):
        """Get variant list as a CPU pandas dataframe.

        Each row in the returned dataframe represents a variant entry.
        For each variant entry, the following metrics are currently tracked -
        1. chrom - Chromosome
        2. start_pos - Start position of variant (inclusive)
        3. end_pos - End position of variant (exclusive)
        4. ref - Reference base(s)
        5. alt - Alternate base(s)
        6. variant_type - VariantType enum specifying SNP/INSERTION/DELETION as an int
        7. {tag}_quality - Quality of variant call
        8. filter - VCF FILTER column [if requested]
        9. info - VCF INFO column [if requested]
        10. format - VCF FORMAT column [if requested]
        11. {sample_name}_zyg - VariantZygosity enum for sample idx as an int

        This dataframe can be easily converted to cuDF for large scale
        variant processing.

        Returns:
            Parsed variants as pandas DataFrame.
        """
        if self._dataframe is None:
            extend_exception(RuntimeError, "VCF data frame should be available.")
        return self._dataframe

    def _detect_variant_type(self, ref, alt):
        """Get variant type enum.

        Given ref and alt alleles, determine type of variant.

        Args:
            ref : ref bases
            alt : alt bases

        Returns:
            VariantType enum
        """
        if len(ref) == len(alt):
            return VariantType.SNP
        elif len(ref) < len(alt):
            return VariantType.INSERTION
        else:
            return VariantType.DELETION

    def _detect_zygosity(self, gt):
        """Get variant zygosity as enum.

        Given a diploid genotype type with genotype information per
        haploid, determine the zygosity of the sample. If the variant is known
        false positive, returns NO_VARIANT. If any of the alts are -1, then returns
        NONE (e.g. in multi allele cases that were split).

        Args:
            gt : Diploid genotype in the format [haploid 1 alt num, haploid 2 alt num]

        Returns:
            Relevant VariantZygosity enum.
        """
        if self._is_fp:
            return VariantZygosity.NO_VARIANT

        if gt[0] == -1 or gt[1] == -1:
            return VariantZygosity.NONE
        elif gt[0] == gt[1]:
            if gt[0] == 0:
                return VariantZygosity.NO_VARIANT
            else:
                return VariantZygosity.HOMOZYGOUS
        else:
            return VariantZygosity.HETEROZYGOUS

    def _get_normalized_count(self, header_number, num_alts, num_samples):
        """Calculate number of values for a VCF key based.

        Determine number of values based on header number, alt count and sample
        count.

        Args:
            header_number : VCF header number
            num_alts : Number of alt alleles for variant
            num_samples : Number of samples in VCF

        Returns:
            Integer number of values.
        """
        if header_number == "A":
            return num_alts
        elif header_number == "R":
            return (num_alts + 1)
        elif header_number == "G":
            return num_samples
        elif header_number.isdigit():
            # Number = 0 implies one entry of boolean. So force min number to be 1.
            return max(1, int(header_number))
        elif header_number == ".":
            return self._unbounded_val_max_col

    def _get_python_header_type(self, header_type):
        """Convert VCF header type to python type.

        Args:
            header_type : Header type string from VCF.

        Returns:
            Python data type corresponding to string.
        """
        if header_type == "Integer":
            return int
        elif header_type == "Float":
            return float
        elif header_type == "Flag":
            return bool
        else:
            return str

    def _get_default_val(self, header_type):
        """Get default value for VCF header type.

        Args:
            header_type : Header type string from VCF.

        Returns:
            Default value.
        """
        if header_type == int:
            return np.int32(0)
        elif header_type == float:
            return np.float32(float("NaN"))
        elif header_type == bool:
            return np.bool_(False)
        else:
            return ""

    def _create_df(self, vcf, variant_list, num_variants):
        """Create dataframe from list of cyvcf2.Variant objects.

        Process each cyvcf2.Variant object in the variant list and generate a dict
        with all the user specified VCF columns and their values. To simplify downstream
        processing, also split multi alleles into separate rows and update relevant info/format
        columns to account for the multi allele change.

        Args:
            vcf : cyvcf2 object for VCF
            variant_list : List of cyvcf2.Variant objects to convert to dataframe
            num_variants : Number of variants in list. This is needed because variant_list
                           is pre-allocated for perf reasons and some elements may be None

        Returns:
            DataFrame for entries in variant_list
        """
        df_dict = defaultdict(list)

        samples = vcf.samples

        # Iterate over all variants in variant list.
        for var_idx in range(num_variants):
            variant = variant_list[var_idx]
            # Iterate over each allele in variant to split up multi alleles.
            alts = variant.ALT
            for alt_idx, alt in enumerate(alts):
                # Add standard DF entries for each variant.
                df_dict["chrom"].append(variant.CHROM)
                df_dict["start_pos"].append(variant.start)
                df_dict["end_pos"].append(variant.end)
                df_dict["id"].append(variant.ID)
                df_dict["ref"].append(variant.REF)
                df_dict["alt"].append(alt)
                df_dict["variant_type"].append(np.int32(self._detect_variant_type(variant.REF, alt)))
                df_dict["quality"].append(variant.QUAL)

                # Process variant filter columns. If filter is present in entry, store True else False.
                variant_filter = "PASS" if variant.FILTER is None else variant.FILTER
                filter_set = set(variant_filter.split(";"))
                for filter_col in self._filter_vcf_keys:
                    df_dict["FILTER_" + filter_col].append(filter_col in filter_set)

                # Process INFO columns. INFO column values need to be handled specially based on header number
                # and header type. Since multi alleles are split up, the right filter values need to go to the
                # right variant row.
                for info_col in self._info_vcf_keys:
                    # Get header type
                    header_number = self._header_number[info_col]

                    if variant.INFO.get(info_col) is not None:
                        val = variant.INFO[info_col]
                        # Make value a tuple to reduce special case handling later.
                        if not isinstance(val, tuple):
                            val = tuple((val,))
                    else:
                        default_val = self._get_default_val(self._header_type[info_col])
                        num_vals = self._get_normalized_count(header_number, len(alts), len(samples))
                        val = [default_val] * num_vals

                    df_key = info_col

                    if header_number == "A":
                        df_dict[df_key].append(val[alt_idx])
                    elif header_number == "R":
                        df_dict[df_key + "_REF"].append(val[0])
                        df_dict[df_key + "_ALT"].append(val[alt_idx + 1])
                    elif header_number.isdigit():
                        header_number = int(header_number)
                        if header_number <= 1:
                            df_dict[df_key].append(val[0])
                        else:
                            for i in range(int(header_number)):
                                df_dict[df_key + "_" + str(i)].append(val[i])
                    elif header_number == ".":
                        num_vals = self._get_normalized_count(header_number, len(alts), len(samples))
                        if len(val) < num_vals:
                            temp_val = [self._get_default_val(self._header_type[info_col])] * num_vals
                            for i in range(len(val)):
                                temp_val[i] = val[i]
                            val = temp_val
                        for i in range(num_vals):
                            df_dict[df_key + "_" + str(i)].append(val[i])

                # Process format columns. Handle GT specially, and the rest can be handled like INFO columns.
                for format_col in self._format_vcf_keys:
                    for sample_idx, sample_name in enumerate(samples):
                        if format_col == "GT":
                            def fix_gt(gt_alt_id, loop_alt_id):
                                """Fix up genotype.

                                If gt alt id and loop alt id are the same, return 1 for alt.
                                If they're not 0, then it represents a split multi allele that's not handled in
                                the current loop.
                                If gt is 0, then return 0 as ref.

                                Args:
                                    gt_alt_id : ID of alt allele
                                    loop_alt_id : ID of current alt in loop

                                Returns:
                                    Fixed up ID of alt.
                                """
                                if gt_alt_id == loop_alt_id:
                                    return 1
                                elif gt_alt_id != 0:
                                    return -1
                                else:
                                    return 0
                            # Handle GT column specially
                            gt = variant.genotypes[sample_idx]
                            # Fixup haplotype number based on multi allele split.
                            alt_id = alt_idx + 1
                            new_gt = [gt[0], gt[1]]
                            new_gt[0] = fix_gt(gt[0], alt_id)
                            new_gt[1] = fix_gt(gt[1], alt_id)
                            if new_gt[0] == -1 or new_gt[1] == -1:
                                new_gt[0] = new_gt[1] = -1
                            df_dict["{}_zyg".format(sample_name)].append(np.int32(self._detect_zygosity(new_gt)))
                            df_dict["{}_GT".format(sample_name)].append("{}/{}".format(new_gt[0], new_gt[1]))
                        else:
                            # Get header type
                            header_number = self._header_number[format_col]

                            val = variant.format(format_col)
                            if val is not None:
                                val = val[sample_idx]
                            else:
                                default_val = self._get_default_val(self._header_type[format_col])
                                num_vals = self._get_normalized_count(header_number, len(alts), len(samples))
                                val = [default_val] * num_vals

                            df_key = sample_name + "_" + format_col

                            if header_number == "A":
                                df_dict[df_key].append(val[alt_idx])
                            elif header_number == "R":
                                df_dict[df_key + "_REF"].append(val[0])
                                df_dict[df_key + "_ALT"].append(val[alt_idx + 1])
                            elif header_number.isdigit():
                                header_number = int(header_number)
                                if header_number <= 1:
                                    df_dict[df_key].append(val[0])
                                else:
                                    for i in range(int(header_number)):
                                        df_dict[df_key + "_" + str(i)].append(val[i])
                            elif header_number == ".":
                                num_vals = self._get_normalized_count(header_number, len(alts), len(samples))
                                if len(val) < num_vals:
                                    temp_val = [self._get_default_val(self._header_type[format_col])] * num_vals
                                    for i in range(len(val)):
                                        temp_val[i] = val[i]
                                    val = temp_val
                                for i in range(num_vals):
                                    df_dict[df_key + "_" + str(i)].append(val[i])

        # Downcast all dataframe types to 32bit.
        for k in df_dict.keys():
            if isinstance(df_dict[k][0], np.float32) or isinstance(df_dict[k][0], float):
                df_dict[k] = np.array(df_dict[k], dtype=np.float32)
            elif isinstance(df_dict[k][0], np.int32) or isinstance(df_dict[k][0], int):
                df_dict[k] = np.array(df_dict[k], dtype=np.int32)
        # Convert local dictionary of k/v to DataFrame.
        df = pd.DataFrame.from_dict(df_dict)

        # Add custom tags to DataFrame
        for col, val in self._tags.items():
            df[col] = val

        return df

    def _parse_vcf_cyvcf(self, thread_id):
        """Parse portions of a VCF file as determined by chunk size and thread id.

        Based on the thread ID, split up a VCF file into equal sized chunks and distribute
        chunks in a round robin fashion to various threads. Each thread only processes
        variants that occur within its chunk. This distribution is handled within each
        thread indepdently. Each thread goes through all variants, and only processes the
        ones that fall in the variant index range determined by its thread id.

        Args:
           thread_id : Thead ID of VCF processing thread.

        Returns:
           List of DataFrames for chunks assigned to thread.
        """
        df_list = []

        # Global is to track which variants are to be processed
        # by current thread id across all regions
        global_variant_idx = 0

        # Go through all regions assigned to reader
        for region in self._regions:
            # Go through variants and add to list
            vcf = cyvcf2.VCF(self._vcf)
            generator = vcf(region) if region else vcf

            # Pre-allocate variant list to chunk size to prevent list extension
            variant_list = [None] * self._chunksize

            # Local idx is to track number of variants currently in the
            # variant list
            local_variant_idx = 0

            # Loop through all variants in cyvcf2 object.
            for variant in generator:
                # Check if a variant maps to this thread.
                if ((global_variant_idx // self._chunksize) % self._num_threads == thread_id):
                    variant_list[local_variant_idx] = variant
                    local_variant_idx += 1
                    if global_variant_idx % self._chunksize == 0:
                        df_list.append(self._create_df(vcf, variant_list, local_variant_idx))
                        variant_list = [None] * self._chunksize  # Reset list
                        local_variant_idx = 0
                global_variant_idx += 1

            if variant_list:
                df_list.append(self._create_df(vcf, variant_list, local_variant_idx))

        return df_list

    def _parallel_parse_vcf(self):
        """Parse VCF file in multi threaded fashion.

        Split the work of parsing a VCF file among multiple threads each of which
        generate a DataFrame with subsections of the main VCF, and then concatenate them
        together to form the final VCF.

        The final DataFrame does now guarantee any ordering of the variants. It only guarantees
        the presence of all variants from the VCF.
        """
        vcf = cyvcf2.VCF(self._vcf)

        # Populate column keys and the number of values for them. Do this for INFO, FILTER
        # and FORMAT keys.
        if "*" in self._info_vcf_keys:
            self._info_vcf_keys = []
            for h in vcf.header_iter():
                if h['HeaderType'] == 'INFO':
                    self._info_vcf_keys.append(h['ID'])
        for k in self._info_vcf_keys:
            header_number = vcf.get_header_type(k)['Number']
            self._info_vcf_key_counts[k] = self._get_normalized_count(header_number, 1, len(vcf.samples))
            self._header_number[k] = header_number
            self._header_type[k] = self._get_python_header_type(vcf.get_header_type(k)['Type'])

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

        # Make sure format columns have "GT" if genotype requested
        if self._require_genotype and "GT" not in self._format_vcf_keys:
            for h in vcf.header_iter():
                if h['HeaderType'] == 'FORMAT' and h['ID'] == "GT":
                    self._format_vcf_keys.append(h['ID'])
            if "GT" not in self._format_vcf_keys:
                raise RuntimeError("Required genotype key GT not available in VCF file.")

        for k in self._format_vcf_keys:
            header_number = vcf.get_header_type(k)['Number']
            self._format_vcf_key_counts[k] = self._get_normalized_count(header_number, 1, len(vcf.samples))
            self._header_number[k] = header_number
            self._header_type[k] = self._get_python_header_type(vcf.get_header_type(k)['Type'])

        # Store name of samples in VCF.
        for sample in vcf.samples:
            self._sample_names.append(sample)

        # Create a pool of threads and distribute parsing to multiple threads.
        pool = mp.Pool(self._num_threads)
        df_lists = []
        func = partial(self._parse_vcf_cyvcf)
        for df in pool.imap(func, range(self._num_threads)):
            df_lists.append(df)

        df_list = [item for sublist in df_lists for item in sublist]

        # Generate final DataFrame from intermediate DataFrames computed by
        # individual threads.
        self._dataframe = pd.concat(df_list, ignore_index=True)

        # Manually set strict data types of specific fields
        self._dataframe = self._dataframe.infer_objects()
        if not self._dataframe.empty:
            self._dataframe["chrom"] = self._dataframe["chrom"].astype('object')
            self._dataframe["start_pos"] = self._dataframe["start_pos"].astype('int32')
            self._dataframe["end_pos"] = self._dataframe["end_pos"].astype('int32')
            self._dataframe["variant_type"] = self._dataframe["variant_type"].astype('int32')

        # Sort dataframe by chromosome and start_position if requested
        if self._sort:
            self._dataframe.sort_values(["chrom", "start_pos"], axis='index', inplace=True, ignore_index=True)
