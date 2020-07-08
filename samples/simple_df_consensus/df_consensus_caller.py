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

import argparse
import pandas as pd

from variantworks.io.vcfio import VCFReader

df_1 = "/ssd/VariantWorks/end_to_end_workflow_sample_files/tp_vcf_1m_giab.vcf.gz"
df_2 = "/ssd/VariantWorks/end_to_end_workflow_sample_files/test_tp_vcf_100k_giab.vcf.gz"

vcf_reader_mutect = VCFReader([VCFReader.VcfBamPath(vcf=df_1, bam="", is_fp=False)])
vcf_reader_strelka = VCFReader([VCFReader.VcfBamPath(vcf=df_2, bam="", is_fp=False)])

mutect_df = vcf_reader_mutect.df
strelka_df = vcf_reader_strelka.df

print(len(mutect_df))
print(len(strelka_df))

intersection = pd.merge(mutect_df, strelka_df, how='inner', on=['chrom', 'start_pos', 'end_pos', 'ref', 'alt', 'variant_type'])
print(len(intersection))
