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
import shutil
import vcf

from claragenomics.variantworks.io.vcfio import VCFReader
from claragenomics.variantworks.types import VariantZygosity
from claragenomics.variantworks.result_writer import VCFResultWriter

from data.vcf_file_mock import mock_vcf_file_reader_input


class MockPyVCFReader:
    original_pyvcf_reader_init_function = vcf.Reader.__init__

    @staticmethod
    def new_vcf_reader_init(self, *args, **kargs):
        if 'filename' not in kargs:  # Reader must be initiated using `filename`
            raise RuntimeError('Please use `filename` to initiate vcf.Reader')
        MockPyVCFReader.original_pyvcf_reader_init_function(self, mock_vcf_file_reader_input(kargs['filename']))


def test_vcf_outputting(monkeypatch):
    """Write inference output into vcf files
    """
    first_vcf_bam_tuple = VCFReader.VcfBamPaths(vcf="/dummy/path1.gz", bam="temp.bam", is_fp=False)
    second_vcf_bam_tuple = VCFReader.VcfBamPaths(vcf="/dummy/path2.gz", bam="temp.bam", is_fp=False)
    with monkeypatch.context() as mp:
        mp.setattr(vcf.Reader, "__init__", MockPyVCFReader.new_vcf_reader_init)
        vcf_loader = VCFReader([first_vcf_bam_tuple, second_vcf_bam_tuple])
    inferred_results = [VariantZygosity.HOMOZYGOUS, VariantZygosity.HOMOZYGOUS, VariantZygosity.HETEROZYGOUS,
                        VariantZygosity.HETEROZYGOUS, VariantZygosity.HOMOZYGOUS, VariantZygosity.HETEROZYGOUS]
    assert (len(inferred_results) == len(vcf_loader))
    with monkeypatch.context() as mp:
        mp.setattr(vcf.Reader, "__init__", MockPyVCFReader.new_vcf_reader_init)
        result_writer = VCFResultWriter(vcf_loader, inferred_results)
        result_writer.write_output()
    # Validate output files format and make sure the outputted genotype for each record matches to the network output
    i = 0
    for f in ['path1.gz.vcf', 'path2.gz.vcf']:
        vcf_reader = vcf.Reader(filename=os.path.join(result_writer.output_location, f))
        for record in vcf_reader:
            assert(record.samples[0]['GT'] == result_writer.zygosity_to_vcf_genotype[inferred_results[i]])
            i += 1
    assert (i == 6)
    # Clean up files
    shutil.rmtree(result_writer.output_location)
