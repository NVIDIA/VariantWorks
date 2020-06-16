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

from variantworks.io.vcfio import VCFReader
from variantworks.types import VariantZygosity, Variant

from data.vcf_file_mock import mock_file_input, mock_invalid_file_input, get_created_vcf_tabix_files


@pytest.mark.parametrize('get_created_vcf_tabix_files', [mock_file_input()], indirect=True)
def test_vcf_loader_snps(get_created_vcf_tabix_files):
    """Get all variants from mocked file stream, filter SNPs, multi allele & multi samples
    """
    vcf_file_path, tabix_file_path = get_created_vcf_tabix_files
    vcf_bam_tuple = VCFReader.VcfBamPath(vcf=vcf_file_path, bam=tabix_file_path, is_fp=False)
    vcf_loader = VCFReader([vcf_bam_tuple])
    assert(len(vcf_loader) == 13)


@pytest.mark.parametrize('get_created_vcf_tabix_files', [mock_file_input()], indirect=True)
def test_vcf_fetch_variant(get_created_vcf_tabix_files):
    """Get first variant from mocked VCF file stream.
    """
    vcf_file_path, tabix_file_path = get_created_vcf_tabix_files
    vcf_bam_tuple = VCFReader.VcfBamPath(vcf=vcf_file_path, bam=tabix_file_path, is_fp=False)
    vcf_loader = VCFReader([vcf_bam_tuple])
    try:
        assert (type(vcf_loader[0]) == Variant)
    except IndexError:
        pytest.fail("Can not retrieve first element from VCFReader")


@pytest.mark.parametrize('get_created_vcf_tabix_files', [mock_file_input()], indirect=True)
def test_vcf_load_fp(get_created_vcf_tabix_files):
    """Get first variant from false positive mocked VCF file stream and check zygosity.
    """
    vcf_file_path, tabix_file_path = get_created_vcf_tabix_files
    vcf_bam_tuple = VCFReader.VcfBamPath(vcf=vcf_file_path, bam=tabix_file_path, is_fp=True)
    vcf_loader = VCFReader([vcf_bam_tuple])
    for v in vcf_loader:
        assert(v.zygosity == VariantZygosity.NO_VARIANT)


@pytest.mark.parametrize('get_created_vcf_tabix_files', [mock_file_input()], indirect=True)
def test_vcf_load_variant_from_multiple_files(get_created_vcf_tabix_files):
    """Get variants from multiple mocked VCF files.
    """
    vcf_file_path, tabix_file_path = get_created_vcf_tabix_files
    first_vcf_bam_tuple = VCFReader.VcfBamPath(vcf=vcf_file_path, bam=tabix_file_path, is_fp=False)
    second_vcf_bam_tuple = VCFReader.VcfBamPath(vcf=vcf_file_path, bam=tabix_file_path, is_fp=False)
    vcf_loader = VCFReader([first_vcf_bam_tuple])
    vcf_loader_2x = VCFReader([first_vcf_bam_tuple, second_vcf_bam_tuple])
    assert (2 * len(vcf_loader) == len(vcf_loader_2x))


@pytest.mark.parametrize('get_created_vcf_tabix_files', [mock_invalid_file_input()], indirect=True)
def test_load_vcf_content_with_wrong_format(get_created_vcf_tabix_files):
    """ parse vcf file with wrong format
    """
    vcf_file_path, tabix_file_path = get_created_vcf_tabix_files
    vcf_bam_tuple = VCFReader.VcfBamPath(vcf=vcf_file_path, bam=tabix_file_path, is_fp=False)
    with pytest.raises(RuntimeError):
        VCFReader([vcf_bam_tuple])
