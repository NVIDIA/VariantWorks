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

from claragenomics.variantworks.io.vcfio import VCFReader
from claragenomics.variantworks.types import VariantZygosity

from data.vcf_file_mock import MockPyVCFReader


def test_vcf_loader_snps(monkeypatch):
    """Get all variants from mocked file stream, filter SNPs, multi allele & multi samples
    """
    vcf_bam_tuple = VCFReader.VcfBamPaths(
        vcf="/dummy/path.gz", bam="temp.bam", is_fp=False)
    vcf_loader = \
        MockPyVCFReader.get_reader(monkeypatch, [vcf_bam_tuple], content_type=MockPyVCFReader.ContentType.UNFILTERED)
    assert(len(vcf_loader) == 13)


def test_vcf_fetch_variant(monkeypatch):
    """Get first variant from mocked VCF file stream.
    """
    vcf_bam_tuple = VCFReader.VcfBamPaths(
        vcf="/dummy/path.gz", bam="temp.bam", is_fp=False)
    vcf_loader = \
        MockPyVCFReader.get_reader(monkeypatch, [vcf_bam_tuple], content_type=MockPyVCFReader.ContentType.UNFILTERED)
    try:
        entry = vcf_loader[0]
    except IndexError:
        raise


def test_vcf_load_fp(monkeypatch):
    """Get first variant from false positive mocked VCF file stream and check zygosity.
    """
    vcf_bam_tuple = VCFReader.VcfBamPaths(
        vcf="/dummy/path.gz", bam="temp.bam", is_fp=True)
    vcf_loader = \
        MockPyVCFReader.get_reader(monkeypatch, [vcf_bam_tuple], content_type=MockPyVCFReader.ContentType.UNFILTERED)
    for v in vcf_loader:
        assert(v.zygosity == VariantZygosity.NO_VARIANT)


def test_vcf_load_variant_from_multiple_files(monkeypatch):
    """Get variants from multiple mocked VCF files.
    """
    first_vcf_bam_tuple = VCFReader.VcfBamPaths(
        vcf="/dummy/path.gz", bam="temp.bam", is_fp=False)
    second_vcf_bam_tuple = VCFReader.VcfBamPaths(
        vcf="/dummy/path.gz", bam="temp.bam", is_fp=False)
    vcf_loader = MockPyVCFReader.get_reader(
        monkeypatch, [first_vcf_bam_tuple], content_type=MockPyVCFReader.ContentType.UNFILTERED)
    vcf_loader_2x = MockPyVCFReader.get_reader(
        monkeypatch, [first_vcf_bam_tuple, second_vcf_bam_tuple], content_type=MockPyVCFReader.ContentType.UNFILTERED)
    assert (2 * len(vcf_loader) == len(vcf_loader_2x))


def test_load_vcf_content_with_wrong_format(monkeypatch):
    """ parse vcf file with wrong format
    """
    vcf_bam_tuple = VCFReader.VcfBamPaths(
        vcf="/dummy/path.gz", bam="temp.bam", is_fp=False)
    with pytest.raises(RuntimeError):
        vcf_loader = \
            MockPyVCFReader.get_reader(monkeypatch, [vcf_bam_tuple], content_type=MockPyVCFReader.ContentType.INVALID)
