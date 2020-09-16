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

from variantworks.io.vcfio import VCFReader, VCFWriter
from variantworks.types import VariantZygosity, Variant

from data.vcf_file_mock import mock_file_input, mock_small_filtered_file_input


def test_vcf_reader(get_created_vcf_tabix_files):
    """Get all variants from mocked file stream, filter SNPs, multi allele & multi samples
    """
    vcf_file_path, tabix_file_path = get_created_vcf_tabix_files(mock_file_input())
    vcf_reader = VCFReader(vcf_file_path, bams=[], is_fp=False)
    assert(len(vcf_reader) == 17)


def test_vcf_fetch_variant(get_created_vcf_tabix_files):
    """Get first variant from mocked VCF file stream.
    """
    vcf_file_path, tabix_file_path = get_created_vcf_tabix_files(mock_file_input())
    vcf_reader = VCFReader(vcf=vcf_file_path, bams=[], is_fp=False)
    try:
        assert (type(vcf_reader[0]) == Variant)
    except IndexError:
        pytest.fail("Can not retrieve first element from VCFReader")


def test_vcf_load_fp(get_created_vcf_tabix_files):
    """Get first variant from false positive mocked VCF file stream and check zygosity.
    """
    vcf_file_path, tabix_file_path = get_created_vcf_tabix_files(mock_file_input())
    vcf_reader = VCFReader(vcf=vcf_file_path, bams=[], is_fp=True, format_keys=["GT"])
    for v in vcf_reader:
        for i in range(len(v.samples)):
            assert(v.zygosity[i] == VariantZygosity.NO_VARIANT)


def test_vcf_load_variant_from_multiple_files(get_created_vcf_tabix_files):
    """Get variants from multiple mocked VCF files.
    """
    vcf_file_path, tabix_file_path = get_created_vcf_tabix_files(mock_file_input())
    vcf_reader = VCFReader(vcf=vcf_file_path, bams=[], is_fp=False)
    vcf_reader_2x = VCFReader(vcf=vcf_file_path, bams=[], is_fp=False)
    assert (len(vcf_reader) == len(vcf_reader_2x))


def test_vcf_reader_to_df(get_created_vcf_tabix_files):
    """Get all variants from parsed file into dataframe.
    """
    vcf_file_path, tabix_file_path = get_created_vcf_tabix_files(mock_file_input())
    vcf_reader = VCFReader(vcf=vcf_file_path, bams=[], is_fp=False)
    df = vcf_reader.dataframe
    assert(len(vcf_reader) == len(df))


def test_vcf_outputting(get_created_vcf_tabix_files):
    """Write inference output into vcf files
    """
    orig_vcf_file_path, orig_vcf_tabix = get_created_vcf_tabix_files(mock_small_filtered_file_input())
    vcf_reader = VCFReader(orig_vcf_file_path,
                           bams=[],
                           is_fp=False,
                           format_keys=["*"],
                           info_keys=["*"],
                           filter_keys=["*"],
                           sort=True)

    inferred_results = [int(VariantZygosity.NO_VARIANT),
                        int(VariantZygosity.NO_VARIANT),
                        int(VariantZygosity.NO_VARIANT)]
    assert (len(inferred_results) == len(vcf_reader))

    input_vcf_df = vcf_reader.dataframe
    gt_col = "{}_GT".format(vcf_reader.samples[0])
    assert(gt_col in input_vcf_df)

    # Update GT column data
    input_vcf_df[gt_col] = inferred_results

    output_path = '{}_{}.{}'.format("inferred", "".join(os.path.basename(orig_vcf_file_path).split('.')[0:-2]), 'vcf')
    vcf_writer = VCFWriter(input_vcf_df, output_path=output_path, sample_names=vcf_reader.samples)
    vcf_writer.write_output(input_vcf_df)

    # Tabix index output file
    with open(output_path, "rb") as in_file:
        data = in_file.read()
    indexed_output_file_path, _ = get_created_vcf_tabix_files(data)

    # Validate output files format and make sure the outputted genotype for each record matches to the network output
    vcf_reader_updated = VCFReader(indexed_output_file_path,
                                   is_fp=False,
                                   format_keys=["*"],
                                   info_keys=["*"],
                                   filter_keys=["*"],
                                   sort=True)
    assert(len(vcf_reader) == len(vcf_reader_updated))
    for i, record in enumerate(vcf_reader_updated):
        assert(record.zygosity[0] == inferred_results[i])

    # Clean up files
    os.remove(output_path)
