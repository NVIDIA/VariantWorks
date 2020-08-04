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

import pandas as pd
from variantworks.io.vcfio import VCFReader
import time

pd.set_option('max_columns', 100)

t= time.time()
#reader = VCFReader(vcf="/home/jdaw/s3-parliament-somatic-variant-calling/Set1/Lancet/set1.lancet.vcf.gz", bams=[], tag="lancet")
#reader = VCFReader(vcf="/home/jdaw/tijyojwad/VariantWorks/tests/data/candidates.vcf", bams=[], tag="lancet")
#reader = VCFReader(vcf="/ssd/VariantWorks/end_to_end_workflow_sample_files/tp_vcf_1m_giab.vcf.gz", bams=[], tag="lancet")
reader = VCFReader(vcf="/home/jdaw/s3-parliament-somatic-variant-calling/Set1/Strelka/strelka/variants/somatic.vcf.gz", bams=[], require_genotype=False, tag="strelka", info_keys=["*"], filter_keys=["*"], format_keys=["*"], regions=None, num_threads=16, chunksize=10000)
#reader = VCFReader(vcf="/home/jdaw/s3-parliament-somatic-variant-calling/HG002_CCS_pepper_hp_snp01_indel_05.TRUTH.vcf.gz", bams=[], require_genotype=False)
#reader = VCFReader(vcf="/home/jdaw/s3-parliament-somatic-variant-calling/Set1/Sniper/set1.sniper.vcf.gz", bams=[], require_genotype=False)
#reader = VCFReader(vcf="/home/jdaw/temp.vcf.gz", bams=[], require_genotype=False)
#reader = VCFReader(vcf="/home/jdaw/Downloads/ALL.wgs.phase3_shapeit2_mvncall_integrated_v5b.20130502.sites.vcf.gz", bams=[], require_genotype=False, info_keys=["*"], filter_keys=[], format_keys=[], regions=None)
print(reader.df)
print("elapsed ", time.time() - t)
