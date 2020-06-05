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

from variantworks.io.vcfio import VCFReader
from variantworks.types import VariantZygosity
from variantworks.result_writer import VCFResultWriter

from data.vcf_file_mock import MockPyVCFReader, mock_small_filtered_file_input


def get_headers_from_file_for_writer(*args, **kargs):
    return [line.strip() for line in mock_small_filtered_file_input() if line.startswith('##')], \
           ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT'], \
           ['CALLED']


def test_vcf_outputting(monkeypatch):
    """Write inference output into vcf files
    """
    first_vcf_bam_tuple = VCFReader.VcfBamPath(
        vcf="/dummy/path1.gz", bam="temp.bam", is_fp=False)
    second_vcf_bam_tuple = VCFReader.VcfBamPath(
        vcf="/dummy/path2.gz", bam="temp.bam", is_fp=False)
    vcf_loader = MockPyVCFReader.get_reader(
        monkeypatch,
        [first_vcf_bam_tuple, second_vcf_bam_tuple],
        content_type=MockPyVCFReader.ContentType.SMALL_FILTERED
    )

    inferred_results = [VariantZygosity.HOMOZYGOUS, VariantZygosity.HOMOZYGOUS, VariantZygosity.HETEROZYGOUS,
                        VariantZygosity.HETEROZYGOUS, VariantZygosity.HOMOZYGOUS, VariantZygosity.HETEROZYGOUS]
    assert (len(inferred_results) == len(vcf_loader))

    result_writer = VCFResultWriter(vcf_loader, inferred_results)

    with monkeypatch.context() as mp:
        mp.setattr(VCFResultWriter, "_get_original_headers_from_vcf_reader", get_headers_from_file_for_writer)
        result_writer.write_output()

    # Validate output files format and make sure the outputted genotype for each record matches to the network output
    i = 0
    for f in ['inferred_path1.vcf', 'inferred_path2.vcf']:
        vcf_reader = vcf.Reader(filename=os.path.join(
            result_writer.output_location, f))
        for record in vcf_reader:
            assert(record.samples[0]['GT'] ==
                   result_writer.zygosity_to_vcf_genotype[inferred_results[i]])
            i += 1
    assert (i == 6)
    # Clean up files
    shutil.rmtree(result_writer.output_location)
