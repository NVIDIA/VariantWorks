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
from tempfile import mkdtemp
import shutil
import vcf

from claragenomics.variantworks.label_loader import VCFLabelLoader
from claragenomics.variantworks.types import VariantZygosity
from claragenomics.variantworks.result_writer import VCFResultWriter
from test_utils import get_data_folder

from data.vcf_file_mock import mock_vcf_file_reader_input


class MockPyVCFReader:
    original_pyvcf_reader_init_function = vcf.Reader.__init__
    tmp_folder_location = None

    @staticmethod
    def new_vcf_reader_init(self, *args, **kargs):
        if 'filename' not in kargs:
            raise RuntimeError('Please use `filename` to initiate vcf.Reader')  # Reader was not initiated using `fsock`
        tmp_vcf_file_path, _ = \
            mock_vcf_file_reader_input(kargs['filename'], MockPyVCFReader.tmp_folder_location)
        MockPyVCFReader.original_pyvcf_reader_init_function(self, open(tmp_vcf_file_path, 'rb'))


def test_vcf_outputting(monkeypatch):
    """Write inference output into vcf files
    """
    MockPyVCFReader.tmp_folder_location = mkdtemp(prefix=get_data_folder() + "/")
    first_vcf_bam_tuple = VCFLabelLoader.VcfBamPaths(vcf="/dummy/path1.gz", bam="temp.bam", is_fp=False)
    second_vcf_bam_tuple = VCFLabelLoader.VcfBamPaths(vcf="/dummy/path2.gz", bam="temp.bam", is_fp=False)
    with monkeypatch.context() as mp:
        mp.setattr(vcf.Reader, "__init__", MockPyVCFReader.new_vcf_reader_init)
        vcf_loader = VCFLabelLoader([first_vcf_bam_tuple, second_vcf_bam_tuple])
    infered_results = [VariantZygosity.HETEROZYGOUS, VariantZygosity.HOMOZYGOUS, VariantZygosity.HETEROZYGOUS,
                       VariantZygosity.HETEROZYGOUS, VariantZygosity.HOMOZYGOUS, VariantZygosity.HOMOZYGOUS]
    assert (len(infered_results) == len(vcf_loader))
    with monkeypatch.context() as mp:
        mp.setattr(vcf.Reader, "__init__", MockPyVCFReader.new_vcf_reader_init)
        result_writer = VCFResultWriter(vcf_loader, infered_results)
        result_writer.write_output()
    # Test valid vcf files
    i = 0
    for f in ['path1.gz.vcf', 'path2.gz.vcf']:
        vcf_reader = vcf.Reader(filename=os.path.join(result_writer.output_location, f))
        for record in vcf_reader:
            assert(record.INFO['IZ'] == result_writer.zygosity_to_vcf_genotype[infered_results[i]])
            i += 1
    assert (i == 6)
    # Clean up files
    shutil.rmtree(MockPyVCFReader.tmp_folder_location)
