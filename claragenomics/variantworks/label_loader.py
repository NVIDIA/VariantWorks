# Abstract and implementation clases for label loaders.
from collections import namedtuple
import vcf

from claragenomics.variantworks.types import VariantZygosity, VariantType, Variant


class LabelLoaderIterator():
    def __init__(self, label_loader):
        assert(isinstance(label_loader, BaseLabelLoader))
        self._label_loader = label_loader
        self._index = 0

    def __next__(self):
        if (self._index < len(self._label_loader)):
            result = self._label_loader[self._index]
            self._index += 1
            return result
        raise StopIteration


class BaseLabelLoader():
    def __init__(self, allow_snps=True, allow_multiallele=True, allow_multisample=False):
        """Base class label loader that sotres variant filters and implements indexing
        and length methods.
        """
        self._allow_snps = allow_snps
        self._allow_multiallele = allow_multiallele
        self._allow_multisample = allow_multisample
        self._labels = []

    def __getitem__(self, idx):
        return self._labels[idx]

    def __len__(self):
        return len(self._labels)

    def __iter__(self):
        return LabelLoaderIterator(self)


class VCFLabelLoader(BaseLabelLoader):
    """VCF based label loader for true and false positive example files.
    """

    VcfBamPaths = namedtuple('VcfBamPaths', ['vcf', 'bam', 'is_fp'], defaults=[False])

    def __init__(self, vcf_bam_list, **kwargs):
        super().__init__(**kwargs)

        for elem in vcf_bam_list:
            assert (elem.vcf is not None and elem.bam is not None and type(elem.is_fp) is bool)
            self._parse_vcf(elem.vcf, elem.bam, self._labels, elem.is_fp)

    def _get_variant_zygosity(self, record, is_fp=False):
        """Determine variant type from pyvcf record.
        """
        if is_fp:
            return VariantZygosity.NO_VARIANT
        if record.num_het > 0:
            return VariantZygosity.HETEROZYGOUS
        elif record.num_hom_alt > 0:
            return VariantZygosity.HOMOZYGOUS
        assert(False), "Unexpected variant zygosity - {}".format(record)

    def _get_variant_type(self, record):
        """Determine variant type.
        """
        if record.is_snp:
            return VariantType.SNP
        elif record.is_indel:
            if record.is_deletion:
                return VariantType.DELETION
            else:
                return VariantType.INSERTION
        assert(False), "Unexpected variant type - {}".format(record)

    def _parse_vcf(self, vcf_file, bam, labels, is_fp=False):
        """Parse VCF file and retain labels after they have passed filters.
        """
        assert(vcf_file[-3:] == ".gz"), "VCF file needs to be compressed and indexed" # Check for compressed file
        vcf_reader = vcf.Reader(open(vcf_file, "rb"))
        for record in vcf_reader:
            if (not(self._allow_snps and record.is_snp)):
                continue
            if (not self._allow_multisample and record.num_called > 1):
                continue
            if (not self._allow_multiallele and len(record.ALT) > 1):
                continue
            chrom = record.CHROM
            pos = record.POS
            ref = record.REF
            var_zyg = self._get_variant_zygosity(record, is_fp)
            var_type = self._get_variant_type(record)
            # Split multi alleles into multiple entries
            for alt in record.ALT:
                var_allele = alt.sequence
                labels.append(Variant(chrom, pos, ref, var_zyg, var_type, var_allele, vcf_file, bam))
