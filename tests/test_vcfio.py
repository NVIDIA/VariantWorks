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
import vcf

from variantworks.io.vcfio import VCFReader
from variantworks.types import VariantZygosity
from data.vcf_file_mock import mock_file_input, mock_invalid_file_input


class MockPyVCFReader:
    original_pyvcf_reader_init_function = vcf.Reader.__init__

    @staticmethod
    def new_vcf_reader_init(self, *args, **kargs):
        MockPyVCFReader.original_pyvcf_reader_init_function(
            self, mock_file_input())

    @staticmethod
    def new_bad_vcf_reader_init(self, *args, **kargs):
        MockPyVCFReader.original_pyvcf_reader_init_function(
            self, mock_invalid_file_input())

    @staticmethod
    def get_vcf(mp, vcf_bam_list):
        with mp.context() as m:
            # Mock vcf.Reader.__init__() return value
            m.setattr(vcf.Reader, "__init__",
                      MockPyVCFReader.new_vcf_reader_init)
            vcf_loader = VCFReader(vcf_bam_list)
        return vcf_loader

    @staticmethod
    def get_invalid_vcf(mp, vcf_bam_list):
        with mp.context() as m:
            # Mock vcf.Reader.__init__() return value
            m.setattr(vcf.Reader, "__init__",
                      MockPyVCFReader.new_bad_vcf_reader_init)
            vcf_loader = VCFReader(vcf_bam_list)
        return vcf_loader


def test_vcf_loader_snps(monkeypatch):
    """Get all variants from mocked file stream, filter SNPs, multi allele & multi samples
    """
    vcf_bam_tuple = VCFReader.VcfBamPath(
        vcf="/dummy/path.gz", bam="temp.bam", is_fp=False)
    vcf_loader = MockPyVCFReader.get_vcf(monkeypatch, [vcf_bam_tuple])
    assert(len(vcf_loader) == 13)


def test_vcf_fetch_variant(monkeypatch):
    """Get first variant from mocked VCF file stream.
    """
    vcf_bam_tuple = VCFReader.VcfBamPath(
        vcf="/dummy/path.gz", bam="temp.bam", is_fp=False)
    vcf_loader = MockPyVCFReader.get_vcf(monkeypatch, [vcf_bam_tuple])
    try:
        entry = vcf_loader[0]
    except IndexError:
        raise


def test_vcf_load_fp(monkeypatch):
    """Get first variant from false positive mocked VCF file stream and check zygosity.
    """
    vcf_bam_tuple = VCFReader.VcfBamPath(
        vcf="/dummy/path.gz", bam="temp.bam", is_fp=True)
    vcf_loader = MockPyVCFReader.get_vcf(monkeypatch, [vcf_bam_tuple])
    for v in vcf_loader:
        assert(v.zygosity == VariantZygosity.NO_VARIANT)


def test_vcf_load_variant_from_multiple_files(monkeypatch):
    """Get variants from multiple mocked VCF files.
    """
    first_vcf_bam_tuple = VCFReader.VcfBamPath(
        vcf="/dummy/path.gz", bam="temp.bam", is_fp=False)
    second_vcf_bam_tuple = VCFReader.VcfBamPath(
        vcf="/dummy/path.gz", bam="temp.bam", is_fp=False)
    vcf_loader = MockPyVCFReader.get_vcf(monkeypatch, [first_vcf_bam_tuple])
    vcf_loader_2x = MockPyVCFReader.get_vcf(
        monkeypatch, [first_vcf_bam_tuple, second_vcf_bam_tuple])
    assert (2 * len(vcf_loader) == len(vcf_loader_2x))


def test_load_vcf_content_with_wrong_format(monkeypatch):
    """ parse vcf file with wrong format
    """
    vcf_bam_tuple = VCFReader.VcfBamPath(
        vcf="/dummy/path.gz", bam="temp.bam", is_fp=False)
    with pytest.raises(RuntimeError):
        vcf_loader = MockPyVCFReader.get_invalid_vcf(
            monkeypatch, [vcf_bam_tuple])
