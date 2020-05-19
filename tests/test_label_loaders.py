import os
import builtins

from claragenomics.variantworks.label_loader import VCFLabelLoader
from claragenomics.variantworks.types import VariantZygosity
from data.vcf_file_mock import mock_file_input


def get_vcf_label_loader_using_mocked_input(mp, vcf_bam_list):
    with mp.context() as m:
        m.setattr(builtins, "open", mock_file_input)  # mock file open return value
        vcf_loader = VCFLabelLoader(vcf_bam_list)
    return vcf_loader


def test_vcf_loader_snps(monkeypatch):
    """Get all variants from mocked file stream, filter SNPs, multi allele & multi samples
    """
    vcf_bam_tuple = VCFLabelLoader.VcfBamPaths(vcf="/dummy/path.gz", bam="temp.bam", is_fp=False)
    vcf_loader = get_vcf_label_loader_using_mocked_input(monkeypatch, [vcf_bam_tuple])
    assert(len(vcf_loader) == 13)


def test_vcf_fetch_variant(monkeypatch):
    """Get first variant from mocked VCF file stream.
    """
    vcf_bam_tuple = VCFLabelLoader.VcfBamPaths(vcf="/dummy/path.gz", bam="temp.bam", is_fp=False)
    vcf_loader = get_vcf_label_loader_using_mocked_input(monkeypatch, [vcf_bam_tuple])
    try:
        entry = vcf_loader[0]
    except IndexError:
        raise


def test_vcf_load_fp(monkeypatch):
    """Get first variant from false positive mocked VCF file stream and check zygosity.
    """
    vcf_bam_tuple = VCFLabelLoader.VcfBamPaths(vcf="/dummy/path.gz", bam="temp.bam", is_fp=True)
    vcf_loader = get_vcf_label_loader_using_mocked_input(monkeypatch, [vcf_bam_tuple])
    for v in vcf_loader:
        assert(v.zygosity == VariantZygosity.NO_VARIANT)


def test_vcf_load_variant_from_multiple_files(monkeypatch):
    """Get variants from multiple mocked VCF files.
    """
    first_vcf_bam_tuple = VCFLabelLoader.VcfBamPaths(vcf="/dummy/path.gz", bam="temp.bam", is_fp=False)
    second_vcf_bam_tuple = VCFLabelLoader.VcfBamPaths(vcf="/dummy/path.gz", bam="temp.bam", is_fp=False)
    vcf_loader = get_vcf_label_loader_using_mocked_input(monkeypatch, [first_vcf_bam_tuple])
    vcf_loader_2x = get_vcf_label_loader_using_mocked_input(monkeypatch, [first_vcf_bam_tuple, second_vcf_bam_tuple])
    assert (2 * len(vcf_loader) == len(vcf_loader_2x))
