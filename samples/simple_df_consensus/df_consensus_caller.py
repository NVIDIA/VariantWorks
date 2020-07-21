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

import time

from variantworks.io.vcfio import VCFReader
from variantworks.consensus import MajorityConsensus

df_1 = "/home/jdaw/s3-parliament-somatic-variant-calling/Set1/Lancet/set1.lancet.vcf.gz"
df_2 = "/home/jdaw/s3-parliament-somatic-variant-calling/Set1/Muse/set1.muse.vcf.gz"
df_3 = "/home/jdaw/s3-parliament-somatic-variant-calling/Set1/Strelka/strelka/variants/somatic.vcf.gz"

t1 = time.time()
vcf_reader1 = VCFReader(vcf=df_1, bams=[], is_fp=False, require_genotype=False, tags=["lancet"])
print("Loaded 1", len(vcf_reader1), time.time() - t1)
t1 = time.time()
vcf_reader2 = VCFReader(vcf=df_2, bams=[], is_fp=False, require_genotype=False, tags=["muse"])
print("Loaded 2", len(vcf_reader2), time.time() - t1)
t1 = time.time()
vcf_reader3 = VCFReader(vcf=df_3, bams=[], is_fp=False, require_genotype=False, tags=["strelka"])
print("Loaded 3", len(vcf_reader3), time.time() - t1)

t1 = time.time()
con_caller = MajorityConsensus([vcf_reader1, vcf_reader2, vcf_reader3], ["muse", "lancet", "strelka"], min_callers=3)
df = con_caller.generate_consensus()
print("Generate consensus", time.time() - t1)
print(df)
