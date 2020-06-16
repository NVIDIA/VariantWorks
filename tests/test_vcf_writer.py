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

import os
import pytest
import shutil
import vcf

from variantworks.io.vcfio import VCFReader
from variantworks.types import VariantZygosity
from variantworks.result_writer import VCFResultWriter

from data.vcf_file_mock import mock_small_filtered_file_input, get_created_vcf_tabix_files


@pytest.mark.parametrize('get_created_vcf_tabix_files', [mock_small_filtered_file_input()], indirect=True)
def test_vcf_outputting(get_created_vcf_tabix_files):
    """Write inference output into vcf files
    """
    vcf_file_path, tabix_file_path = get_created_vcf_tabix_files
    first_vcf_bam_tuple = VCFReader.VcfBamPath(vcf=vcf_file_path, bam=tabix_file_path, is_fp=False)
    second_vcf_bam_tuple = VCFReader.VcfBamPath(vcf=vcf_file_path, bam=tabix_file_path, is_fp=False)
    vcf_loader = VCFReader([first_vcf_bam_tuple, second_vcf_bam_tuple])

    inferred_results = [VariantZygosity.HOMOZYGOUS, VariantZygosity.HOMOZYGOUS, VariantZygosity.HETEROZYGOUS,
                        VariantZygosity.HETEROZYGOUS, VariantZygosity.HOMOZYGOUS, VariantZygosity.HETEROZYGOUS]
    assert (len(inferred_results) == len(vcf_loader))

    result_writer = VCFResultWriter(vcf_loader, inferred_results)
    result_writer.write_output()

    # Validate output files format and make sure the outputted genotype for each record matches to the network output
    i = 0
    for f in ['{}_{}.{}'.format("inferred", vcf_base_name, 'vcf'), 'inferred_path2.vcf']:
        vcf_reader = vcf.Reader(filename=os.path.join(
            result_writer.output_location, f))
        for record in vcf_reader:
            assert(record.samples[0]['GT'] == result_writer.zygosity_to_vcf_genotype[inferred_results[i]])
            i += 1
    assert (i == 6)
    # Clean up files
    shutil.rmtree(result_writer.output_location)
