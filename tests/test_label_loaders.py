import os
import pytest

from claragenomics.variantworks.label_loader import VCFLabelLoader
from claragenomics.variantworks.types import VariantZygosity

from test_utils import get_data_folder


def test_vcf_loader_snps():
    """Get all the variants from the file.
    """
    labels = os.path.join(get_data_folder(), "candidates.vcf.gz")
    vcf_bam_tuple = VCFLabelLoader.VcfBamPaths(vcf=labels, bam="temp.bam", is_fp=False)
    vcf_loader = VCFLabelLoader([vcf_bam_tuple], allow_snps=True, allow_multiallele=False)
    assert(len(vcf_loader) == 13)

    # Testing is_fp field default value
    vcf_bam_tuple = VCFLabelLoader.VcfBamPaths(vcf=labels, bam="temp.bam")
    vcf_loader = VCFLabelLoader([vcf_bam_tuple], allow_snps=True, allow_multiallele=False)
    assert (len(vcf_loader) == 13)


def test_vcf_loader_np_snps():
    """Get none of the variants from the file since they're all SNPs.
    """
    labels = os.path.join(get_data_folder(), "candidates.vcf.gz")
    vcf_bam_tuple = VCFLabelLoader.VcfBamPaths(vcf=labels, bam="temp.bam", is_fp=False)
    vcf_loader = VCFLabelLoader([vcf_bam_tuple], allow_snps=False, allow_multiallele=False)

    assert(len(vcf_loader) == 0)


def test_vcf_fetch_variant():
    """Get first variant from VCF.
    """
    labels = os.path.join(get_data_folder(), "candidates.vcf.gz")
    vcf_bam_tuple = VCFLabelLoader.VcfBamPaths(vcf=labels, bam="temp.bam", is_fp=False)
    vcf_loader = VCFLabelLoader([vcf_bam_tuple], allow_snps=True, allow_multiallele=False)

    try:
        entry = vcf_loader[0]
    except:
        assert(False)


def test_vcf_load_fp():
    """Get first variant from false positive VCF and check zygosity.
    """
    labels = os.path.join(get_data_folder(), "candidates.vcf.gz")
    vcf_bam_tuple = VCFLabelLoader.VcfBamPaths(vcf=labels, bam="temp.bam", is_fp=True)
    vcf_loader = VCFLabelLoader([vcf_bam_tuple], allow_snps=True, allow_multiallele=False)

    for v in vcf_loader:
        assert(v.zygosity == VariantZygosity.NO_VARIANT)


def test_vcf_fetch_variant():
    """Ge variants from multiple VCF files.
    """
    labels = os.path.join(get_data_folder(), "candidates.vcf.gz")
    first_vcf_bam_tuple = VCFLabelLoader.VcfBamPaths(vcf=labels, bam="temp.bam", is_fp=False)
    second_vcf_bam_tuple = VCFLabelLoader.VcfBamPaths(vcf=labels, bam="temp.bam", is_fp=False)
    vcf_loader = VCFLabelLoader([first_vcf_bam_tuple], allow_snps=True, allow_multiallele=False)
    vcf_loader_2x = VCFLabelLoader(
        [first_vcf_bam_tuple, second_vcf_bam_tuple], allow_snps=True, allow_multiallele=False)

    assert(2 * len(vcf_loader) == len(vcf_loader_2x))
