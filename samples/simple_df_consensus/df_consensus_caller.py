#!/usr/bin/env python3
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
"""A sample program highlighting usage of VariantWorks I/O dataframe APIs."""

import pandas as pd

from variantworks.io.vcfio import VCFReader

df_1 = "/home/jdaw/s3-parliament-somatic-variant-calling/Set1/Mutect/set1.mutect.filtered.vcf.gz"
df_2 = "/home/jdaw/s3-parliament-somatic-variant-calling/Set1/Strelka/strelka/variants/somatic.vcf.gz"

vcf_reader_strelka = VCFReader([VCFReader.VcfBamPath(vcf=df_2, bam="", is_fp=False, require_genotype=False)])
vcf_reader_mutect = VCFReader([VCFReader.VcfBamPath(vcf=df_1, bam="", is_fp=False, require_genotype=False)])

mutect_df = vcf_reader_mutect.df
strelka_df = vcf_reader_strelka.df

print(len(mutect_df))
print(len(strelka_df))

intersection = pd.merge(mutect_df, strelka_df,
                        how='inner',
                        on=['chrom', 'start_pos', 'end_pos', 'ref', 'alt', 'variant_type'])
print(len(intersection))
