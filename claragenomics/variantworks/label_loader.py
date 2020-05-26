# Abstract and implementation clases for label loaders.
from abc import ABC, abstractmethod
from collections import namedtuple
import vcf
import warnings

from claragenomics.variantworks.types import VariantZygosity, VariantType, Variant


class LabelLoaderIterator:
    def __init__(self, label_loader):
        assert(isinstance(label_loader, BaseLabelLoader))
        self._label_loader = label_loader
        self._index = 0

    def __next__(self):
        if self._index < len(self._label_loader):
            result = self._label_loader[self._index]
            self._index += 1
            return result
        raise StopIteration


class BaseLabelLoader(ABC):
    @abstractmethod
    def __init__(self):
        """Base class label loader that sotres variant filters and implements indexing
        and length methods.
        """
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

    def __init__(self, vcf_bam_list):
        super().__init__()
        for elem in vcf_bam_list:
            assert (elem.vcf is not None and elem.bam is not None and type(elem.is_fp) is bool)
            self._parse_vcf(elem.vcf, elem.bam, self._labels, elem.is_fp)

    @staticmethod
    def _get_variant_zygosity(record, is_fp=False):
        """Determine variant type from pyvcf record.
        """
        if is_fp:
            return VariantZygosity.NO_VARIANT
        if record.num_het > 0:
            return VariantZygosity.HETEROZYGOUS
        elif record.num_hom_alt > 0:
            return VariantZygosity.HOMOZYGOUS
        raise ValueError("Unexpected variant zygosity - {}".format(record))

    @staticmethod
    def _get_variant_type(record):
        """Determine variant type.
        """
        if record.is_snp:
            return VariantType.SNP
        elif record.is_indel:
            if record.is_deletion:
                return VariantType.DELETION
            else:
                return VariantType.INSERTION
        raise ValueError("Unexpected variant type - {}".format(record))

    def _create_variant_tuple_from_record(self, record, vcf_file, bam, is_fp):
        var_zyg = self._get_variant_zygosity(record, is_fp)
        var_type = self._get_variant_type(record)
        # Split multi alleles into multiple entries
        for alt in record.ALT:
            var_allele = alt.sequence
            yield Variant(idx=len(self), chrom=record.CHROM, pos=record.POS, id=record.ID, ref=record.REF,
                          allele=var_allele, quality=record.QUAL, filter=record.FILTER, info=record.INFO,
                          format=record.FORMAT, zygosity=var_zyg, type=var_type, vcf=vcf_file, bam=bam)

    def _parse_vcf(self, vcf_file, bam, labels, is_fp=False):
        """Parse VCF file and retain labels after they have passed filters.
        """
        assert(vcf_file[-3:] == ".gz"), "VCF file needs to be compressed and indexed"  # Check for compressed file
        vcf_reader = vcf.Reader(filename=vcf_file)
        for record in vcf_reader:
            if not record.is_snp:
                warnings.warn("%s is filtered - not an SNP record" % record)
                continue
            if record.num_called > 1:
                warnings.warn("%s is filtered - multisample records are not supported" % record)
                continue
            if len(record.ALT) > 1:
                warnings.warn("%s is filtered - multiallele recrods are not supported" % record)
                continue
            for variant in self._create_variant_tuple_from_record(record, vcf_file, bam, is_fp):
                labels.append(variant)
