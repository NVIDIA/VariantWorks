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

# import pandas as pd

from variantworks.io.vcfio import VCFReader

df_1 = "/home/jdaw/s3-parliament-somatic-variant-calling/Set1/Lancet/set1.lancet.vcf.gz"
df_2 = "/home/jdaw/s3-parliament-somatic-variant-calling/Set1/Muse/set1.muse.vcf.gz"

vcf_reader1 = VCFReader(vcf=df_2, bams=[], is_fp=False, require_genotype=False, tags=["muse"])
vcf_reader2 = VCFReader(vcf=df_1, bams=[], is_fp=False, require_genotype=False, tags=["lancet"])

df1 = vcf_reader1.df
print(len(df1))
print(df1)
df2 = vcf_reader2.df
print(len(df2))
print(df2)
for i, v in enumerate(vcf_reader1):
    print(v)
    if i == 10:
        break


# df["caller_count"] = df[["muse", "lancet"]].sum(axis=1)
# count_consensus = df[df["caller_count"] > 1]
# print(count_consensus)
