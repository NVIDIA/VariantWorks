import os
import builtins

from claragenomics.variantworks.label_loader import VCFLabelLoader
from claragenomics.variantworks.types import VariantZygosity
from data.vcf_file_mock import mock_file_input
from test_utils import get_data_folder


def test_vcf_loader_snps_from_file():
    """Get all the variants from the file.
    """
    labels = os.path.join(get_data_folder(), "candidates.vcf.gz")
    vcf_bam_tuple = VCFLabelLoader.VcfBamPaths(vcf=labels, bam="temp.bam")
    vcf_loader = VCFLabelLoader([vcf_bam_tuple])
    assert (len(vcf_loader) == 13)


def test_vcf_load_variant_from_multiple_files():
    """Get variants from multiple VCF files.
    """
    labels = os.path.join(get_data_folder(), "candidates.vcf.gz")
    first_vcf_bam_tuple = VCFLabelLoader.VcfBamPaths(vcf=labels, bam="temp.bam", is_fp=False)
    second_vcf_bam_tuple = VCFLabelLoader.VcfBamPaths(vcf=labels, bam="temp.bam", is_fp=False)
    vcf_loader = VCFLabelLoader([first_vcf_bam_tuple])
    vcf_loader_2x = VCFLabelLoader([first_vcf_bam_tuple, second_vcf_bam_tuple])
    assert (2 * len(vcf_loader) == len(vcf_loader_2x))


def test_vcf_loader_snps_mocked_file_content(monkeypatch):
    """Get all variants from mocked file stream, filter  SNPs, multi allele & multi samples
    """
    vcf_bam_tuple = VCFLabelLoader.VcfBamPaths(vcf="/dummy/path.gz", bam="temp.bam", is_fp=False)
    with monkeypatch.context() as m:
        m.setattr(builtins, "open", mock_file_input)  # mock file open return value
        vcf_loader = VCFLabelLoader([vcf_bam_tuple])
    assert(len(vcf_loader) == 13)


def test_vcf_fetch_variant(monkeypatch):
    """Get first variant from mocked VCF file stream.
    """
    vcf_bam_tuple = VCFLabelLoader.VcfBamPaths(vcf="/dummy/path.gz", bam="temp.bam", is_fp=False)
    with monkeypatch.context() as m:
        m.setattr(builtins, "open", mock_file_input)  # mock file open return value
        vcf_loader = VCFLabelLoader([vcf_bam_tuple])
    try:
        entry = vcf_loader[0]
    except IndexError:
        raise


def test_vcf_load_fp(monkeypatch):
    """Get first variant from false positive mocked VCF file stream and check zygosity.
    """
    vcf_bam_tuple = VCFLabelLoader.VcfBamPaths(vcf="/dummy/path.gz", bam="temp.bam", is_fp=True)
    with monkeypatch.context() as m:
        m.setattr(builtins, "open", mock_file_input)  # mock file open return value
        vcf_loader = VCFLabelLoader([vcf_bam_tuple])

    for v in vcf_loader:
        assert(v.zygosity == VariantZygosity.NO_VARIANT)
