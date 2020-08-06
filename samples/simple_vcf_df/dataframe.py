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
"""Sample showing utilization of VCFReader to generate dataframe."""

import os
import pandas as pd
import time

from variantworks.io.vcfio import VCFReader

pd.set_option('max_columns', 100)

sample_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
repo_root_folder = os.path.dirname(sample_folder)
tests_data_folder = os.path.join(repo_root_folder, "tests", "data")
test_vcf_file = os.path.join(tests_data_folder, "candidates_multisample.vcf.gz")

t = time.time()
reader = VCFReader(test_vcf_file,
                   bams=[],
                   tags={"custom_tag": 1},
                   info_keys=["AF"],
                   filter_keys=[],
                   format_keys=[],
                   num_threads=4,
                   regions=[],
                   require_genotype=True,
                   sort=True)
read_time = time.time() - t
print(reader.dataframe)

print("Elapsed time for reading VCF (seconds): ", read_time)
