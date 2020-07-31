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

import pytest

from variantworks.io.vcfio import VCFReader
from variantworks.types import VariantZygosity, Variant

from data.vcf_file_mock import mock_file_input, mock_invalid_file_input


def test_vcf_loader(get_created_vcf_tabix_files):
    """Get all variants from mocked file stream, filter SNPs, multi allele & multi samples
    """
    vcf_file_path, tabix_file_path = get_created_vcf_tabix_files(mock_file_input())
    vcf_loader = VCFReader(vcf_file_path, bams=[], is_fp=False)
    assert(len(vcf_loader) == 17)


def test_vcf_fetch_variant(get_created_vcf_tabix_files):
    """Get first variant from mocked VCF file stream.
    """
    vcf_file_path, tabix_file_path = get_created_vcf_tabix_files(mock_file_input())
    vcf_loader = VCFReader(vcf=vcf_file_path, bams=[], is_fp=False)
    try:
        assert (type(vcf_loader[0]) == Variant)
    except IndexError:
        pytest.fail("Can not retrieve first element from VCFReader")


def test_vcf_load_fp(get_created_vcf_tabix_files):
    """Get first variant from false positive mocked VCF file stream and check zygosity.
    """
    vcf_file_path, tabix_file_path = get_created_vcf_tabix_files(mock_file_input())
    vcf_loader = VCFReader(vcf=vcf_file_path, bams=[], is_fp=True)
    for v in vcf_loader:
        for i in range(len(v.samples)):
            assert(v.zygosity[i] == VariantZygosity.NO_VARIANT)


def test_vcf_load_variant_from_multiple_files(get_created_vcf_tabix_files):
    """Get variants from multiple mocked VCF files.
    """
    vcf_file_path, tabix_file_path = get_created_vcf_tabix_files(mock_file_input())
    vcf_loader = VCFReader(vcf=vcf_file_path, bams=[], is_fp=False)
    vcf_loader_2x = VCFReader(vcf=vcf_file_path, bams=[], is_fp=False)
    assert (len(vcf_loader) == len(vcf_loader_2x))


def test_load_vcf_content_with_wrong_format(get_created_vcf_tabix_files):
    """ parse vcf file with wrong format
    """
    vcf_file_path, tabix_file_path = get_created_vcf_tabix_files(mock_invalid_file_input())
    print(vcf_file_path, tabix_file_path)
    with pytest.raises(Exception):
        reader = VCFReader(vcf=vcf_file_path, bams=[], is_fp=False)
        print(reader.df)


def test_vcf_loader_to_df(get_created_vcf_tabix_files):
    """Get all variants from parsed file into dataframe.
    """
    vcf_file_path, tabix_file_path = get_created_vcf_tabix_files(mock_file_input())
    vcf_loader = VCFReader(vcf=vcf_file_path, bams=[], is_fp=False)
    df = vcf_loader.df
    assert(len(vcf_loader) == len(df))
